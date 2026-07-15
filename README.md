# Pediatric MASLD stacking model

This repository contains the analysis code accompanying a study of machine-learning models for pediatric metabolic dysfunction-associated steatotic liver disease (MASLD). It is provided to make the model-development and validation workflow transparent.

## Data availability and privacy

This is a **code-only repository**. No raw data, participant-level data, imputed datasets, fitted model objects, prediction files, figures, tables, or manuscript files are included. These files are excluded by `.gitignore`. The clinical datasets cannot be made publicly available because they contain individual-level health information and are subject to ethics and institutional data-use restrictions.

Consequently, the exact numerical results cannot be reproduced from this repository alone. Authorized users may place the required development and external-validation datasets in a local directory and set the `MASLD_BASE_DIR` environment variable to that directory. Analysis outputs can be redirected with `MASLD_OUTPUT_DIR`.

## Stacking workflow

The development analysis used a shuffled 10-fold cross-validation split with `random_state=42`.

1. In each fold, every base learner was fitted on the other nine folds.
2. The fitted base learner generated predictions only for the held-out fold. After all ten folds, each participant therefore had one out-of-fold (OOF) prediction from each base learner.
3. The ten OOF class predictions were concatenated with the original clinical predictors to form the inputs to the CatBoost meta-learner.
4. CatBoost was evaluated using the same 10-fold partition. In-fold base-learner predictions were not used as meta-features.
5. For external validation, each saved fold-specific model generated a probability for every external participant. The current external-validation script uses the patient-level mean of the ten probabilities as the final score.
6. Confidence intervals for performance metrics were obtained by nonparametric patient-level bootstrap resampling.

The base learners passed to the stacker are XGBoost, support vector machine, multilayer perceptron, k-nearest neighbors, random forest, AdaBoost, LightGBM, logistic regression, Gaussian naïve Bayes, and Extra Tree. A standalone Decision Tree is not included as a stacking input; a depth-3 decision tree is used only as the weak learner within AdaBoost.

## Hyperparameters

The manuscript analysis used fixed hyperparameter settings coded directly in `src/train_stacking_model.py`; no separate grid, random, or Bayesian hyperparameter search was run within the cross-validation loop. The principal non-default settings are summarized below.

| Model | Principal fixed settings |
|---|---|
| XGBoost | `learning_rate=0.008`, `n_estimators=800`, `max_depth=5`, `subsample=0.75`, `colsample_bytree=0.8`, `gamma=0`, `reg_alpha=5`, `reg_lambda=200` |
| SVM | RBF kernel, `C=0.8`, `gamma=0.002`, `probability=True` |
| MLP | `hidden_layer_sizes=(55,)`, `max_iter=1000`, `alpha=0.1`, Adam, `learning_rate_init=0.08`, `random_state=42` |
| KNN | `n_neighbors=180`, uniform weights, Manhattan distance (`p=1`) |
| Random forest | `n_estimators=485`, `max_depth=11`, `min_samples_leaf=5`, `min_samples_split=10`, `random_state=42` |
| AdaBoost | depth-3 decision-tree weak learner, `n_estimators=200`, `learning_rate=0.3`, `random_state=42` |
| LightGBM | `feature_fraction=0.6`, `learning_rate=0.005`, `max_depth=5`, `num_leaves=6`, `subsample=0.8`, `n_estimators=800`, `min_child_weight=1`, `lambda_l1=5`, `lambda_l2=200` |
| Logistic regression | L1 penalty, `C=0.02`, `solver='liblinear'`, `random_state=42` |
| Gaussian NB | `var_smoothing=1e-5` |
| Extra Tree | `criterion='log_loss'`, `max_features=20`, `max_depth=10`, `min_samples_leaf=20`, `min_samples_split=50`, `random_state=42` |
| CatBoost meta-learner | `iterations=1250`, `learning_rate=0.004`, `depth=4`, Logloss, `l2_leaf_reg=300`, `eval_metric='Accuracy'`, `subsample=0.75`, depthwise growth, `min_data_in_leaf=50`, gradient leaf estimation |

## Repository contents

- `src/train_stacking_model.py`: primary training script, including 10-fold OOF stacking, model saving, and internal performance evaluation.
- `src/validate_external_cohort.py`: external validation using the mean fold-specific probability per participant.
- `src/bootstrap_metrics.py`: patient-level bootstrap confidence intervals for discrimination and classification metrics.
- `analysis/evaluate_internal_oof.py`: reconstruction and evaluation of internal OOF probabilities, including paired DeLong analyses and calibration utilities.
- `analysis/evaluate_auc_calibration.py`: AUC comparison and calibration analysis using unmodified predicted probabilities.
- `analysis/complete_case_sensitivity_analysis.py`: complete-case sensitivity analysis using the same fixed model settings and 10-fold stacking workflow as the primary analysis; rows with any missing predictor or outcome are excluded.

## Running the code

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Then place the restricted datasets in a local directory, set `MASLD_BASE_DIR` to that directory, and run the required analysis. If `MASLD_BASE_DIR` is not set, the scripts use the repository root. The main script is a research script rather than a packaged command-line application; the order of columns in the supplied datasets must match that used during model development.

## Complete-case sensitivity analysis

The complete-case sensitivity analysis changes only the missing-data strategy: participants with any missing value are excluded, while the prespecified model hyperparameters, 10-fold assignments, OOF stacking protocol, external-validation aggregation, and patient-level bootstrap procedure are retained.

Run it with:

```bash
python analysis/complete_case_sensitivity_analysis.py
```

The script expects the same restricted development and external-validation workbooks used by the primary analysis. Set `MASLD_BASE_DIR` and, if desired, `MASLD_OUTPUT_DIR` as described above. Generated workbooks and participant-level predictions remain local and are not included in this repository.

## Methodological note

The repository reflects the analysis used for the manuscript. It documents OOF transfer from base learners to CatBoost, but it is not a nested cross-validation implementation. Preprocessing and model settings should therefore be interpreted exactly as coded.

