import json
import joblib
import numpy as np
import pytest
from sklearn.datasets import load_wine
from experiment import build_pipeline, run, predict

@pytest.mark.parametrize('method', ['baseline', 'pca', 'nca', 'pca_nca'])
def test_train_only_preprocessing_and_inference(method):
    data = load_wine()
    X, y = data.data[:140].copy(), data.target[:140]
    X[0, 0] = np.nan
    model = build_pipeline(method).fit(X, y)
    # A large held-out shift must not alter the fitted training statistics.
    before = model.named_steps['scale'].mean_.copy()
    model.predict(data.data[140:] + 1000)
    assert np.allclose(model.named_steps['scale'].mean_, before)
    expected = np.mean(model.named_steps['impute'].transform(X), axis=0)
    np.testing.assert_allclose(before, expected)
    assert set(model.predict(data.data[140:])).issubset(set(y))


def test_experiment_artifacts_and_cv_selection(tmp_path):
    report = run(tmp_path)
    assert set(report['train_indices']).isdisjoint(report['test_indices'])
    assert len(report['train_indices']) + len(report['test_indices']) == 178
    expected = max(report['results'], key=lambda k: report['results'][k]['cv_macro_f1'])
    assert report['selected_by_training_cv'] == expected
    for metrics in report['results'].values():
        assert np.asarray(metrics['confusion_matrix']).sum() == 45
        assert 0 <= metrics['test_macro_f1'] <= 1
    X = load_wine().data[:2]
    inputs = tmp_path / 'input.json'
    inputs.write_text(json.dumps(X.tolist()))
    assert len(predict(tmp_path / 'model.joblib', inputs)) == 2
    restored = joblib.load(tmp_path / 'model.joblib')['pipeline']
    np.testing.assert_array_equal(restored.predict(X), joblib.load(tmp_path / 'model.joblib')['pipeline'].predict(X))
    inputs.write_text('[[1,2]]')
    with pytest.raises(ValueError, match='13 columns'):
        predict(tmp_path / 'model.joblib', inputs)


def test_invalid_method():
    with pytest.raises(ValueError):
        build_pipeline('unknown')
