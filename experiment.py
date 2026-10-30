"""Reproducible, training-only PCA/NCA classification on the public Wine dataset."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def build_pipeline(method: str, seed: int = SEED) -> Pipeline:
    """Each CV fold fits its own imputer, scaler and feature transformation."""
    steps = [('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]
    if method in ('pca', 'pca_nca'):
        steps.append(('pca', PCA(n_components=6, svd_solver='full')))
    if method in ('nca', 'pca_nca'):
        steps.append(('nca', NeighborhoodComponentsAnalysis(
            n_components=2, init='pca', max_iter=100, random_state=seed)))
    if method not in ('baseline', 'pca', 'nca', 'pca_nca'):
        raise ValueError(f'Unknown method: {method}')
    steps.append(('classifier', KNeighborsClassifier()))
    return Pipeline(steps)


def tune(method, X_train, y_train, seed=SEED):
    """No test features or labels enter hyperparameter selection."""
    grid = {'classifier__n_neighbors': [3, 7]}
    if method == 'pca':
        grid['pca__n_components'] = [2, 6]
    elif method in ('nca', 'pca_nca'):
        grid['nca__n_components'] = [2, 6]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(build_pipeline(method, seed), grid, cv=cv,
                        scoring='f1_macro', n_jobs=1, error_score='raise', refit=True).fit(X_train, y_train)


def run(output: Path, seed: int = SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_wine()
    train_idx, test_idx = train_test_split(np.arange(len(data.target)), test_size=.25,
                                         random_state=seed, stratify=data.target)
    X_train, X_test = data.data[train_idx], data.data[test_idx]
    y_train, y_test = data.target[train_idx], data.target[test_idx]
    # Complete model selection before looking at any held-out predictions.
    searches, timing = {}, {}
    for method in ('baseline', 'pca', 'nca', 'pca_nca'):
        start = perf_counter()
        searches[method] = tune(method, X_train, y_train, seed)
        timing[method] = perf_counter() - start
    winner = max(searches, key=lambda name: searches[name].best_score_)
    results = {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (method, search) in zip(axes, searches.items()):
        model = search.best_estimator_
        pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        results[method] = {
            'cv_macro_f1': float(search.best_score_),
            'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
            'test_accuracy': float(accuracy_score(y_test, pred)),
            'test_macro_f1': float(f1_score(y_test, pred, average='macro')),
            'test_roc_auc_ovr': float(roc_auc_score(y_test, probs, multi_class='ovr')),
            'search_seconds': timing[method], 'best_params': search.best_params_,
            'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
            'classification_report': classification_report(y_test, pred, target_names=data.target_names,
                                                           output_dict=True, zero_division=0)}
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(method)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrices.png', dpi=150)
    plt.close(fig)
    selected = searches[winner].best_estimator_
    joblib.dump({'pipeline': selected, 'feature_names': list(data.feature_names),
                 'target_names': list(data.target_names), 'method': winner}, output / 'model.joblib')
    report = {'dataset': 'sklearn Wine (UCI Wine recognition)', 'seed': seed,
              'python': platform.python_version(), 'sklearn': sklearn.__version__,
              'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(),
              'selected_by_training_cv': winner, 'results': results}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = ['# Wine classification results', '',
            'Selected by training CV macro-F1: **' + winner + '**. Test set: 45 rows; training: 133 rows.', '',
            '| Pipeline | CV macro-F1 | Test accuracy | Test macro-F1 | Test ROC-AUC (OvR) |',
            '|---|---:|---:|---:|---:|']
    for method, r in results.items():
        rows.append(f"| {method} | {r['cv_macro_f1']:.4f} | {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | {r['test_roc_auc_ovr']:.4f} |")
    rows += ['', 'One seeded split on a small public dataset. These are demonstration results, not evidence of production performance.',
             '', '![Held-out confusion matrices](confusion_matrices.png)', '']
    (output / 'README.md').write_text('\n'.join(rows))
    return report


def predict(model_path: Path, input_path: Path):
    # Only load a trusted local artifact; joblib files can execute Python code.
    bundle = joblib.load(model_path)
    rows = np.asarray(json.loads(input_path.read_text()), dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(bundle['feature_names']) or len(rows) == 0:
        raise ValueError('Input must be a nonempty matrix with 13 columns in Wine feature order.')
    if np.isinf(rows).any():
        raise ValueError('Infinite feature values are not supported.')
    labels = bundle['pipeline'].predict(rows)
    return [bundle['target_names'][int(label)] for label in labels]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--output', type=Path, default=Path('artifacts'))
    train.add_argument('--seed', type=int, default=SEED)
    pred = commands.add_parser('predict')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('--input', type=Path, required=True, help='JSON matrix, Wine feature order')
    args = parser.parse_args()
    if args.command == 'train':
        print(json.dumps(run(args.output, args.seed), indent=2))
    else:
        print(json.dumps(predict(args.model, args.input)))


if __name__ == '__main__':
    main()
