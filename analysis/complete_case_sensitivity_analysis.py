from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


RANDOM_STATE = 42
OUTER_SPLITS = 10
BOOTSTRAP_REPLICATES = 2000

MODEL_NAMES = [
    "XGB",
    "SVM",
    "MLP",
    "KNN",
    "RF",
    "AdaBoost",
    "LGBM",
    "LR",
    "GNB",
    "ET",
]
STACKING_NAME = "Stacking"
ALL_MODEL_NAMES = MODEL_NAMES + [STACKING_NAME]

DROP_SHARED = [
    "Large platelet ratio",
    "Red blood cell distribution width sd",
    "Creatine kinase isoenzyme",
    "chlorine",
    "Cystatin _c",
    "potassium",
    "sodium",
    "phosphorus",
    "Retinol binding protein",
]
DROP_DEVELOPMENT_ONLY = [
    "black",
    "tyg",
    "tg/hdl",
    "tc/hdl",
    "nohdl",
    "lhr",
    "homa-ir",
]


def load_complete_case_data(base_dir: Path):
    development_1 = pd.read_excel(base_dir / "obesity-5936-0716.xlsx")
    development_2 = pd.read_excel(base_dir / "obesity-1555-0815.xlsx").drop(
        columns="ID"
    )
    external = pd.read_excel(
        base_dir / "External verification-354-0815.xlsx"
    ).drop(columns="ID")

    development_1 = development_1.drop(columns=DROP_SHARED + DROP_DEVELOPMENT_ONLY)
    development_2 = development_2.drop(columns=DROP_SHARED)
    external = external.drop(columns=DROP_SHARED)

    development = pd.concat(
        [development_1, development_2], ignore_index=True
    )
    development = development.loc[~(development["bmiz"] < 1)].reset_index(drop=True)
    external = external.loc[~(external["bmiz"] < 1)].reset_index(drop=True)

    if list(development.columns) != list(external.columns):
        raise ValueError("Development and external columns do not match.")

    development_before = len(development)
    external_before = len(external)
    development = development.dropna(axis=0, how="any").reset_index(drop=True)
    external = external.dropna(axis=0, how="any").reset_index(drop=True)

    features = [column for column in development.columns if column != "group"]
    X = development[features].copy()
    y = development["group"].astype(int).copy()
    X_external = external[features].copy()
    y_external = external["group"].astype(int).copy()

    X["gender"] = X["gender"].astype(int)
    X_external["gender"] = X_external["gender"].astype(int)

    counts = {
        "development_before_complete_case": development_before,
        "development_complete_cases": len(X),
        "development_events": int(y.sum()),
        "external_before_complete_case": external_before,
        "external_complete_cases": len(X_external),
        "external_events": int(y_external.sum()),
        "predictors": len(features),
    }
    return X, y, X_external, y_external, counts


def fixed_base_model(name: str):
    """Return the prespecified base learner used in the primary analysis."""
    if name == "XGB":
        return xgb.XGBClassifier(learning_rate=0.008, n_estimators=800, max_depth=5, subsample=0.75, colsample_bytree=0.8, gamma=0, reg_alpha=5, reg_lambda=200, eval_metric="logloss", n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", C=0.8, gamma=0.002, probability=True)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(55,), max_iter=1000, alpha=0.1, solver="adam", random_state=42, learning_rate_init=0.08)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=180, weights="uniform", algorithm="auto", p=1, n_jobs=-1)
    if name == "RF":
        return RandomForestClassifier(max_depth=11, min_samples_leaf=5, min_samples_split=10, n_estimators=485, random_state=42, n_jobs=-1)
    if name == "AdaBoost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.3, random_state=42)
    if name == "LGBM":
        return LGBMClassifier(feature_fraction=0.6, learning_rate=0.005, max_depth=5, num_leaves=6, subsample=0.8, verbosity=-1, n_estimators=800, min_child_weight=1, lambda_l1=5, lambda_l2=200, n_jobs=-1)
    if name == "LR":
        return LogisticRegression(penalty="l1", C=0.02, random_state=42, solver="liblinear", max_iter=2000)
    if name == "GNB":
        return GaussianNB(var_smoothing=1e-5)
    if name == "ET":
        return ExtraTreeClassifier(criterion="log_loss", max_features=20, random_state=42, max_depth=10, min_samples_leaf=20, min_samples_split=50)
    raise KeyError(name)


def fixed_meta_model(categorical):
    """Return the prespecified CatBoost meta-learner used in the primary analysis."""
    return CatBoostClassifier(
        iterations=1250, learning_rate=0.004, depth=4,
        loss_function="Logloss", l2_leaf_reg=300, eval_metric="Accuracy",
        subsample=0.75, grow_policy="Depthwise", min_data_in_leaf=50,
        leaf_estimation_method="Gradient", cat_features=categorical,
        random_seed=0, verbose=False, allow_writing_files=False, thread_count=-1,
    )


def continuous_score(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def make_meta_frame(X: pd.DataFrame, base_predictions: np.ndarray):
    frame = X.reset_index(drop=True).copy()
    for index, name in enumerate(MODEL_NAMES):
        frame[f"pred_{name}"] = base_predictions[:, index].astype(int)
    categorical = ["gender"] + [f"pred_{name}" for name in MODEL_NAMES]
    frame[categorical] = frame[categorical].astype(int)
    return frame, categorical


def metric_values(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": npv,
    }


def bootstrap_intervals(y_true, predictions, scores, seed, fold_ids=None):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_names = list(metric_values(y_true, predictions[:, 0], scores[:, 0]))
    values = {
        (model_name, metric_name): []
        for model_name in ALL_MODEL_NAMES
        for metric_name in metric_names
    }
    completed = 0
    attempts = 0
    while completed < BOOTSTRAP_REPLICATES and attempts < BOOTSTRAP_REPLICATES * 3:
        attempts += 1
        if fold_ids is None:
            sampled_indices = [rng.integers(0, n, n)]
        else:
            sampled_indices = []
            for fold in np.unique(fold_ids):
                fold_pool = np.flatnonzero(fold_ids == fold)
                sampled_indices.append(
                    rng.choice(fold_pool, len(fold_pool), replace=True)
                )
        if any(np.unique(y_true[index]).size < 2 for index in sampled_indices):
            continue
        for model_index, model_name in enumerate(ALL_MODEL_NAMES):
            fold_metrics = [
                metric_values(
                    y_true[index],
                    predictions[index, model_index],
                    scores[index, model_index],
                )
                for index in sampled_indices
            ]
            sample_metrics = {
                metric_name: float(
                    np.nanmean([row[metric_name] for row in fold_metrics])
                )
                for metric_name in metric_names
            }
            for metric_name, value in sample_metrics.items():
                if np.isfinite(value):
                    values[(model_name, metric_name)].append(value)
        completed += 1
    if completed < BOOTSTRAP_REPLICATES:
        raise RuntimeError("Too few valid bootstrap samples.")
    intervals = {}
    for key, sample_values in values.items():
        if sample_values:
            intervals[key] = tuple(np.percentile(sample_values, [2.5, 97.5]))
        else:
            intervals[key] = (np.nan, np.nan)
    return intervals
def fold_metric_means(y, predictions, scores, fold_ids):
    means = {}
    for model_index, model_name in enumerate(ALL_MODEL_NAMES):
        fold_rows = []
        for fold in range(OUTER_SPLITS):
            mask = fold_ids == fold
            fold_rows.append(
                metric_values(y[mask], predictions[mask, model_index], scores[mask, model_index])
            )
        means[model_name] = {
            metric: float(np.mean([row[metric] for row in fold_rows]))
            for metric in fold_rows[0]
        }
    return means


def external_metric_values(y, predictions, scores):
    return {
        model_name: metric_values(y, predictions[:, index], scores[:, index])
        for index, model_name in enumerate(ALL_MODEL_NAMES)
    }


def build_tables(point_estimates, intervals, cohort):
    raw_rows = []
    formatted_rows = []
    for model_name in ALL_MODEL_NAMES:
        raw_row = {"Cohort": cohort, "Model": model_name}
        formatted_row = {"Cohort": cohort, "Model": model_name}
        for metric_name, point in point_estimates[model_name].items():
            lower, upper = intervals[(model_name, metric_name)]
            raw_row[f"{metric_name}_point"] = point
            raw_row[f"{metric_name}_ci_lower"] = lower
            raw_row[f"{metric_name}_ci_upper"] = upper
            formatted_row[metric_name] = f"{point:.3f} ({lower:.3f}-{upper:.3f})"
        raw_rows.append(raw_row)
        formatted_rows.append(formatted_row)
    return pd.DataFrame(raw_rows), pd.DataFrame(formatted_rows)


def main():
    start = time.time()
    script_dir = Path(__file__).resolve().parent
    default_base_dir = (
        script_dir.parent if script_dir.name in {"analysis", "src"} else script_dir
    )
    base_dir = Path(os.environ.get("MASLD_BASE_DIR", default_base_dir))
    output_dir = Path(
        os.environ.get(
            "MASLD_OUTPUT_DIR", base_dir / "complete_case_sensitivity_outputs"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X, y_series, X_external, y_external_series, counts = load_complete_case_data(base_dir)
    y = y_series.to_numpy()
    y_external = y_external_series.to_numpy()
    print(json.dumps(counts, ensure_ascii=False, indent=2))

    standardized_models = {"SVM", "MLP", "KNN", "LR"}
    numeric_features = [column for column in X.columns if column != "gender"]
    X_standardized = X.copy()
    X_external_standardized = X_external.copy()
    X_standardized[numeric_features] = StandardScaler().fit_transform(
        X[numeric_features]
    )
    X_external_standardized[numeric_features] = StandardScaler().fit_transform(
        X_external[numeric_features]
    )

    n_models = len(ALL_MODEL_NAMES)
    internal_predictions = np.zeros((len(X), n_models), dtype=int)
    internal_scores = np.zeros((len(X), n_models), dtype=float)
    external_fold_predictions = np.zeros(
        (OUTER_SPLITS, len(X_external), n_models), dtype=int
    )
    external_fold_scores = np.zeros(
        (OUTER_SPLITS, len(X_external), n_models), dtype=float
    )
    fold_ids = np.full(len(X), -1, dtype=int)

    outer_cv = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(outer_cv.split(X))
    for fold, (_, test_index) in enumerate(splits):
        fold_ids[test_index] = fold

    for model_index, model_name in enumerate(MODEL_NAMES):
        model_start = time.time()
        X_model = X_standardized if model_name in standardized_models else X
        X_external_model = (
            X_external_standardized
            if model_name in standardized_models
            else X_external
        )
        for fold, (train_index, test_index) in enumerate(splits):
            model = fixed_base_model(model_name)
            model.fit(X_model.iloc[train_index], y[train_index])

            internal_predictions[test_index, model_index] = np.asarray(
                model.predict(X_model.iloc[test_index]), dtype=int
            )
            internal_scores[test_index, model_index] = continuous_score(
                model, X_model.iloc[test_index]
            )
            external_fold_predictions[fold, :, model_index] = np.asarray(
                model.predict(X_external_model), dtype=int
            )
            external_fold_scores[fold, :, model_index] = continuous_score(
                model, X_external_model
            )
        print(
            f"{model_name}: 10 fixed-parameter folds completed; "
            f"time={(time.time() - model_start) / 60:.1f} min"
        )

    X_meta, categorical = make_meta_frame(
        X, internal_predictions[:, : len(MODEL_NAMES)]
    )
    stack_index = len(MODEL_NAMES)
    for fold, (train_index, test_index) in enumerate(splits):
        fold_start = time.time()
        X_meta_external, _ = make_meta_frame(
            X_external, external_fold_predictions[fold, :, : len(MODEL_NAMES)]
        )
        meta_model = fixed_meta_model(categorical)
        meta_model.fit(X_meta.iloc[train_index], y[train_index])

        internal_predictions[test_index, stack_index] = np.asarray(
            meta_model.predict(X_meta.iloc[test_index]), dtype=int
        ).reshape(-1)
        internal_scores[test_index, stack_index] = meta_model.predict_proba(
            X_meta.iloc[test_index]
        )[:, 1]
        external_fold_predictions[fold, :, stack_index] = np.asarray(
            meta_model.predict(X_meta_external), dtype=int
        ).reshape(-1)
        external_fold_scores[fold, :, stack_index] = meta_model.predict_proba(
            X_meta_external
        )[:, 1]
        print(
            f"{STACKING_NAME} fold {fold + 1}/{OUTER_SPLITS} completed; "
            f"time={(time.time() - fold_start) / 60:.1f} min"
        )
    if np.any(fold_ids < 0):
        raise RuntimeError("Some development participants did not receive OOF predictions.")

    external_scores = external_fold_scores.mean(axis=0)
    external_predictions = (
        external_fold_predictions.sum(axis=0) >= (OUTER_SPLITS / 2)
    ).astype(int)
    external_predictions[:, -1] = (external_scores[:, -1] >= 0.5).astype(int)

    internal_points = fold_metric_means(
        y, internal_predictions, internal_scores, fold_ids
    )
    external_points = external_metric_values(
        y_external, external_predictions, external_scores
    )
    internal_intervals = bootstrap_intervals(
        y,
        internal_predictions,
        internal_scores,
        RANDOM_STATE + 20000,
        fold_ids=fold_ids,
    )
    external_intervals = bootstrap_intervals(
        y_external,
        external_predictions,
        external_scores,
        RANDOM_STATE + 30000,
    )

    internal_raw, internal_formatted = build_tables(
        internal_points, internal_intervals, "Internal OOF"
    )
    external_raw, external_formatted = build_tables(
        external_points, external_intervals, "External validation"
    )
    raw_table = pd.concat([internal_raw, external_raw], ignore_index=True)
    formatted_table = pd.concat(
        [internal_formatted, external_formatted], ignore_index=True
    )

    output_file = output_dir / "complete_case_sensitivity_performance.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        formatted_table.to_excel(writer, sheet_name="Formatted results", index=False)
        raw_table.to_excel(writer, sheet_name="Numeric results", index=False)

    prediction_columns = [f"{name}_score" for name in ALL_MODEL_NAMES]
    pd.DataFrame(internal_scores, columns=prediction_columns).assign(
        outcome=y, fold=fold_ids + 1
    ).to_csv(output_dir / "internal_oof_scores.csv", index=False)
    pd.DataFrame(external_scores, columns=prediction_columns).assign(
        outcome=y_external
    ).to_csv(output_dir / "external_scores.csv", index=False)
    print("\nComplete-case sensitivity-analysis results")
    print(formatted_table.to_string(index=False))
    print(f"\nSaved: {output_file}")
    print(f"Total runtime: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()


