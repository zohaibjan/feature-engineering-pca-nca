# Feature Engineering: PCA, NCA and Classification

A reproducible machine-learning project comparing **baseline, PCA, NCA and PCA → NCA** feature pipelines with a k-nearest-neighbour classifier on the public Wine recognition dataset.

This demonstrates feature preprocessing, supervised representation learning, model selection and held-out evaluation. It is a new public reference implementation; it does not reproduce private research datasets or claim production outcomes.

## Run it

Python 3.12 is tested. No account, GPU, API key or dataset download is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python experiment.py train --output artifacts
python experiment.py predict --model artifacts/model.joblib --input examples/wine_samples.json
```

Training writes `metrics.json`, a fitted `model.joblib`, a results README and confusion-matrix plots. The model bundle includes the complete imputation/scaling/feature/classifier pipeline, input feature names and output labels. Only load model files from trusted sources.

## What the methods mean

| Pipeline | Feature engineering | Classifier |
|---|---|---|
| Baseline | Training median imputation + standardisation | kNN |
| PCA | Baseline preprocessing + unsupervised principal components | kNN |
| NCA | Baseline preprocessing + supervised neighbourhood component projection | kNN |
| PCA → NCA | PCA compression followed by supervised NCA | kNN |

PCA creates combinations of features that preserve variance; it does not rank original columns by predictive value. Scikit-learn NCA learns a linear distance transformation using training labels. MATLAB `fscnca`, used in the linked research projects, instead learns weights for selecting original features. These approaches are related but are not interchangeable implementations.

## Evaluation design

1. Split 178 Wine samples into 133 training and 45 held-out test rows using seed 42 and stratification.
2. Fit every preprocessing and feature-learning step **inside** each training CV fold using `Pipeline`.
3. Use three-fold stratified CV macro-F1 to select k (3 or 7) and feature dimension (2 or 6). The combined pipeline fixes the PCA stage at six dimensions and tunes the NCA stage.
4. Select the winning pipeline from training CV scores, then evaluate every fitted candidate on the same untouched test rows. The saved model is the CV winner, even if another candidate has a higher test score.
5. Report accuracy, macro-F1, multiclass one-vs-rest ROC-AUC, per-class precision/recall and confusion matrices. Record split indices, library version, seed, best parameters and search time.

NCA optimises pairwise relationships and can become expensive on large datasets. This small benchmark keeps it practical on a CPU. A single small test split has substantial uncertainty; 100% on 45 rows is not a general performance guarantee. Repeated/nested CV or an external dataset would be needed for stronger comparisons. NCA is supervised: never fit it on the full dataset before splitting.

## Measured results

The committed [results report](results/README.md) and [machine-readable metrics](results/metrics.json) were produced by running this code locally. PCA won training CV; NCA scored highest on this particular test split. We retain the CV-selected model to avoid choosing a winner using test results.

![Held-out confusion matrices](results/confusion_matrices.png)

## Connection to the MATLAB research repositories

A source review of seven MATLAB repositories found an NCA configuration flag in two:

- [Ensemble optimisation experiments](https://github.com/zohaibjan/ensemble_optimised_PSO_GA_MOEA): an unsafe legacy NCA threshold path had already been disabled. It is now replaced with an inner-training-only fitted feature transform.
- [Classifier selection experiments](https://github.com/zohaibjan/ACM_TIST_CSME): the NCA flag existed but was not connected to the training path. It now uses the same fitted-transform interface.

Both now support optional **PCA components or top-k NCA feature selection**, frozen train-only scaling, validation/test transformation, and synthetic regression tests. PCA and the safe transform helpers are additions in this update; the existing source did not establish historical PCA use. The original ensemble-selection policy remains in place. Updated runs must not be compared directly with historical CSVs without rerunning the experiments.

The review also checked `IEEE_TKDE`, `IEEE_Access`, `IEEE_TNNLS`, `ensemble_works_TKDD` and `PatternRecognition`; their checked-out `.m` source did not contain PCA/NCA calls. No unsupported feature-method claims were added to those repositories.

## Project layout

- `experiment.py`: reproducible training, evaluation, persistence and prediction CLI.
- `tests/`: train-only preprocessing checks, inference validation and end-to-end artifacts.
- `results/`: a recorded demonstration run; model binary is generated locally instead of committed.
- `.github/workflows/tests.yml`: CPU tests, training and downloadable CI artifacts.

## Dataset and technical references

The bundled scikit-learn loader provides 178 samples, 13 continuous chemical measurements and three wine cultivar classes. Input JSON rows must follow `load_wine().feature_names`; `examples/feature_names.json` records the order. This is an educational chemical classification dataset, not an industrial failure benchmark.

- [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), S. Aeberhard and M. Forina, DOI: 10.24432/C5PC7J. Dataset terms remain those of its source.
- [Scikit-learn Wine loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).
- [Scikit-learn NCA](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html).
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- [MATLAB NCA feature selection](https://www.mathworks.com/help/stats/fscnca.html).

## Historical NCA implementation

The [original July 2020 MATLAB training source](https://github.com/zohaibjan/ensemble_optimised_PSO_GA_MOEA/blob/0758eaa810d0b9ba521d5df27356e324c753b841/runTraining.m) contains an explicit `fscnca` feature-selection implementation. This verifies that NCA was part of the earlier research code, beyond a configuration flag. Its threshold used held-out test loss; the current training-only replacement removes that evaluation leakage. This source evidence does not identify a specific client dataset or establish historical PCA use.
