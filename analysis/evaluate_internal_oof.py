"""Post-hoc internal validation analyses for the pediatric MASLD models.

This script reconstructs patient-level out-of-fold (OOF) predictions from the
10 fold-specific models saved by ``train_stacking_model.py``. It then performs:

1. pooled OOF AUC estimation with DeLong confidence intervals;
2. paired DeLong comparisons of Stacking versus every base model;
3. calibration-in-the-large, calibration intercept/slope, O:E and Brier score;
4. bootstrap confidence intervals and calibration plots.

The source workbooks and model files are read-only. All outputs are written to
the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 10
EPS = 1e-6

BASE_MODELS = [
    ("XGB", "XGBClassifier", False),
    ("SVM", "SVC", True),
    ("MLP", "MLPClassifier", True),
    ("KNN", "KNeighborsClassifier", True),
    ("RandomForest", "RandomForestClassifier", False),
    ("AdaBoost", "AdaBoostClassifier", False),
    ("LGBM", "LGBMClassifier", False),
    ("LogisticRegression", "LogisticRegression", True),
    ("GaussianNB", "GaussianNB", False),
    ("ExtraTree", "ExtraTreeClassifier", False),
]

STACK_FEATURE_NAMES = [
    "XGB",
    "SVM",
    "MLP",
    "KNN",
    "RF",
    "ADB",
    "LGBM",
    "LR",
    "NB",
    "ET",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"D:\Python code\mechine_learning_stroke"),
        help="Directory containing the imputed workbook and NAFLD_allmodel.",
    )
    parser.add_argument(
        "--data-file",
        default="obesity_NAFLD-5936+1555-0816插值.xlsx",
        help="Internal imputed dataset produced by the original training script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "internal_calibration_auc_results",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of stratified bootstrap replicates for calibration CIs.",
    )
    return parser.parse_args()


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midranks
    return result


def fast_delong(
    predictions_sorted_transposed: np.ndarray, label_1_count: int
) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for row in range(k):
        tx[row] = compute_midrank(positive[row])
        ty[row] = compute_midrank(negative[row])
        tz[row] = compute_midrank(predictions_sorted_transposed[row])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_stats(
    y_true: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.atleast_2d(np.asarray(predictions, dtype=float))
    if predictions.shape[1] != len(y_true):
        raise ValueError("Predictions must have shape (models, patients).")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("DeLong requires a binary outcome encoded as 0/1.")
    order = np.argsort(-y_true)
    return fast_delong(predictions[:, order], int(y_true.sum()))


def auc_ci(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    aucs, covariance = delong_stats(y_true, pred)
    auc_value = float(aucs[0])
    se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    lower = max(0.0, auc_value - norm.ppf(0.975) * se)
    upper = min(1.0, auc_value + norm.ppf(0.975) * se)
    return auc_value, lower, upper


def paired_delong(
    y_true: np.ndarray, pred_stacking: np.ndarray, pred_comparator: np.ndarray
) -> dict[str, float]:
    aucs, covariance = delong_stats(
        y_true, np.vstack([pred_stacking, pred_comparator])
    )
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    se = math.sqrt(max(variance, 0.0))
    difference = float(aucs[0] - aucs[1])
    if se == 0:
        p_value = 1.0 if difference == 0 else 0.0
        z_value = 0.0 if difference == 0 else math.copysign(math.inf, difference)
    else:
        z_value = float(difference / se)
        p_value = float(2.0 * norm.sf(abs(z_value)))
    z = norm.ppf(0.975)
    return {
        "auc_stacking": float(aucs[0]),
        "auc_comparator": float(aucs[1]),
        "auc_difference": difference,
        "difference_ci_lower": difference - z * se,
        "difference_ci_upper": difference + z * se,
        "z_value": z_value,
        "p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def logistic_recalibration(
    y_true: np.ndarray, pred: np.ndarray
) -> tuple[float, float, float]:
    """Return joint intercept, slope, and calibration-in-the-large."""
    y = np.asarray(y_true, dtype=float)
    z = np.log(np.clip(pred, EPS, 1.0 - EPS) / np.clip(1.0 - pred, EPS, 1.0))
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        mu = expit(design @ beta)
        weights = np.clip(mu * (1.0 - mu), 1e-10, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y - mu)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break

    citl = 0.0
    for _ in range(100):
        mu = expit(z + citl)
        denominator = np.sum(mu * (1.0 - mu))
        if denominator <= 1e-12:
            break
        step = np.sum(y - mu) / denominator
        citl += step
        if abs(step) < 1e-10:
            break
    return float(beta[0]), float(beta[1]), float(citl)


def calibration_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    intercept, slope, citl = logistic_recalibration(y_true, pred)
    observed = float(np.sum(y_true))
    expected = float(np.sum(pred))
    auc_value, auc_lower, auc_upper = auc_ci(y_true, pred)
    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "mean_predicted_risk": float(np.mean(pred)),
        "auc": auc_value,
        "auc_ci_lower": auc_lower,
        "auc_ci_upper": auc_upper,
        "brier_score": float(np.mean((pred - y_true) ** 2)),
        "observed_expected_ratio": observed / expected if expected > 0 else np.nan,
        "calibration_in_the_large": citl,
        "recalibration_intercept": intercept,
        "calibration_slope": slope,
    }


def stratified_bootstrap_calibration(
    y_true: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    records: list[dict[str, float]] = []
    betas: list[tuple[float, float]] = []
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        y_b = y_true[sample]
        p_b = pred[sample]
        intercept, slope, citl = logistic_recalibration(y_b, p_b)
        records.append(
            {
                "brier_score": float(np.mean((p_b - y_b) ** 2)),
                "observed_expected_ratio": float(np.sum(y_b) / np.sum(p_b)),
                "calibration_in_the_large": citl,
                "recalibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        betas.append((intercept, slope))
    return pd.DataFrame(records), np.asarray(betas)


def wilson_interval(events: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(0.975)
    proportion = events / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def calibration_bins(y_true: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y_true, "prediction": pred})
    frame["bin"] = pd.qcut(frame["prediction"], q=n_bins, duplicates="drop")
    rows = []
    for interval, group in frame.groupby("bin", observed=True):
        events = int(group["outcome"].sum())
        total = len(group)
        lower, upper = wilson_interval(events, total)
        rows.append(
            {
                "bin": str(interval),
                "n": total,
                "events": events,
                "mean_predicted": float(group["prediction"].mean()),
                "observed_rate": float(group["outcome"].mean()),
                "observed_ci_lower": lower,
                "observed_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def reconstruct_oof_predictions(
    data_path: Path, model_dir: Path
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_excel(data_path)
    if "group" not in data.columns or "gender" not in data.columns:
        raise ValueError("The imputed dataset must contain group and gender columns.")
    y = data["group"].astype(int).to_numpy()
    x = data.drop(columns=["group"]).copy()
    x["gender"] = x["gender"].round().astype(int)

    x_values = x.to_numpy(dtype=float)
    scaler = StandardScaler()
    x_std = np.insert(
        scaler.fit_transform(x_values[:, 1:]), 0, x_values[:, 0], axis=1
    )

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(kfold.split(y))
    n_models = len(BASE_MODELS)
    base_probability = np.zeros((len(y), n_models), dtype=float)
    base_label = np.zeros((len(y), n_models), dtype=int)
    fold_number = np.zeros(len(y), dtype=int)

    for fold_idx, (_, test_index) in enumerate(folds):
        fold_number[test_index] = fold_idx + 1
        for model_idx, (_, file_prefix, use_standardized) in enumerate(BASE_MODELS):
            model_path = model_dir / f"{file_prefix}{fold_idx}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = joblib.load(model_path)
            fold_x = x_std[test_index] if use_standardized else x.iloc[test_index]
            probability = model.predict_proba(fold_x)[:, 1]
            base_probability[test_index, model_idx] = probability
            base_label[test_index, model_idx] = model.predict(fold_x).astype(int)

    stack_labels = pd.DataFrame(base_label, columns=STACK_FEATURE_NAMES, index=x.index)
    x_stack = pd.concat([x, stack_labels], axis=1)
    categorical_features = STACK_FEATURE_NAMES + ["gender"]
    x_stack[categorical_features] = x_stack[categorical_features].astype(int)
    stacking_probability = np.zeros(len(y), dtype=float)

    for fold_idx, (_, test_index) in enumerate(folds):
        model_path = model_dir / f"CatBoostClassifier{fold_idx}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = joblib.load(model_path)
        test_pool = Pool(
            x_stack.iloc[test_index],
            y[test_index],
            cat_features=categorical_features,
        )
        stacking_probability[test_index] = model.predict_proba(test_pool)[:, 1]

    return y, x, base_probability, stacking_probability, fold_number


def plot_calibration_panels(
    output_dir: Path,
    y_true: np.ndarray,
    prediction_map: dict[str, np.ndarray],
    metrics_by_model: dict[str, dict[str, float]],
    bootstrap_betas: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, len(prediction_map), figsize=(12.5, 5.6), sharex=True, sharey=True)
    if len(prediction_map) == 1:
        axes = [axes]
    grid = np.linspace(0.001, 0.999, 300)
    grid_logit = np.log(grid / (1.0 - grid))
    for ax, (model_name, pred) in zip(axes, prediction_map.items()):
        bins = calibration_bins(y_true, pred)
        yerr = np.vstack(
            [
                bins["observed_rate"] - bins["observed_ci_lower"],
                bins["observed_ci_upper"] - bins["observed_rate"],
            ]
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Ideal")
        ax.errorbar(
            bins["mean_predicted"],
            bins["observed_rate"],
            yerr=yerr,
            fmt="o",
            markersize=5,
            capsize=2,
            color="#1f77b4",
            alpha=0.9,
            label="Risk deciles (95% CI)",
        )
        metric = metrics_by_model[model_name]
        curve = expit(
            metric["recalibration_intercept"]
            + metric["calibration_slope"] * grid_logit
        )
        ax.plot(grid, curve, color="#d62728", linewidth=2, label="Logistic calibration")
        betas = bootstrap_betas[model_name]
        boot_curves = expit(
            betas[:, 0, None] + betas[:, 1, None] * grid_logit[None, :]
        )
        lower, upper = np.quantile(boot_curves, [0.025, 0.975], axis=0)
        ax.fill_between(grid, lower, upper, color="#d62728", alpha=0.14, linewidth=0)
        ax.set_title(model_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.grid(alpha=0.15)
        ax.text(
            0.04,
            0.96,
            f"Slope = {metric['calibration_slope']:.3f}\n"
            f"CITL = {metric['calibration_in_the_large']:.3f}\n"
            f"Brier = {metric['brier_score']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
    axes[0].set_ylabel("Observed outcome proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Internal 10-fold out-of-fold calibration", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_dir / "internal_calibration_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "internal_calibration_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    data_path = base_dir / args.data_file
    model_dir = base_dir / "NAFLD_allmodel"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y, _, base_probability, stacking_probability, fold_number = reconstruct_oof_predictions(
        data_path, model_dir
    )
    model_probabilities = {
        name: base_probability[:, idx] for idx, (name, _, _) in enumerate(BASE_MODELS)
    }
    model_probabilities["Stacking"] = stacking_probability

    prediction_frame = pd.DataFrame(
        {"outcome": y, "fold": fold_number, **model_probabilities}
    )
    prediction_frame.to_csv(output_dir / "internal_oof_predictions.csv", index=False)

    auc_rows = []
    for name, probability in model_probabilities.items():
        auc_value, lower, upper = auc_ci(y, probability)
        auc_rows.append(
            {
                "model": name,
                "pooled_oof_auc": auc_value,
                "auc_ci_lower": lower,
                "auc_ci_upper": upper,
            }
        )
    auc_estimates = pd.DataFrame(auc_rows)

    comparison_rows = []
    for name, probability in model_probabilities.items():
        if name == "Stacking":
            continue
        comparison = paired_delong(y, stacking_probability, probability)
        comparison_rows.append({"comparator": name, **comparison})
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["p_value_holm"] = holm_adjust(comparisons["p_value"].tolist())
    comparisons = comparisons.merge(
        auc_estimates[["model", "auc_ci_lower", "auc_ci_upper"]],
        left_on="comparator",
        right_on="model",
        how="left",
    ).drop(columns="model")
    stacking_ci = auc_estimates.loc[auc_estimates["model"] == "Stacking"].iloc[0]
    comparisons["stacking_auc_ci_lower"] = stacking_ci["auc_ci_lower"]
    comparisons["stacking_auc_ci_upper"] = stacking_ci["auc_ci_upper"]
    comparisons.to_csv(output_dir / "internal_delong_comparisons.csv", index=False)
    auc_estimates.to_csv(output_dir / "internal_auc_estimates.csv", index=False)

    calibration_models = {
        "AdaBoost": model_probabilities["AdaBoost"],
        "Stacking": stacking_probability,
    }
    metrics_by_model: dict[str, dict[str, float]] = {}
    bootstrap_betas: dict[str, np.ndarray] = {}
    metric_rows = []
    for model_index, (name, probability) in enumerate(calibration_models.items()):
        metrics = calibration_metrics(y, probability)
        bootstrap, betas = stratified_bootstrap_calibration(
            y, probability, args.bootstrap, RANDOM_STATE + model_index
        )
        metrics_by_model[name] = metrics
        bootstrap_betas[name] = betas
        row: dict[str, float | int | str] = {"model": name, **metrics}
        for metric_name in [
            "brier_score",
            "observed_expected_ratio",
            "calibration_in_the_large",
            "recalibration_intercept",
            "calibration_slope",
        ]:
            lower, upper = bootstrap[metric_name].quantile([0.025, 0.975])
            row[f"{metric_name}_ci_lower"] = float(lower)
            row[f"{metric_name}_ci_upper"] = float(upper)
        metric_rows.append(row)
        calibration_bins(y, probability).to_csv(
            output_dir / f"internal_calibration_bins_{name.lower()}.csv", index=False
        )
    calibration_table = pd.DataFrame(metric_rows)
    calibration_table.to_csv(output_dir / "internal_calibration_metrics.csv", index=False)

    plot_calibration_panels(
        output_dir, y, calibration_models, metrics_by_model, bootstrap_betas
    )

    primary = comparisons.loc[comparisons["comparator"] == "AdaBoost"].iloc[0]
    fold_auc_rows = []
    for fold in range(1, N_SPLITS + 1):
        mask = fold_number == fold
        ada_auc = auc_ci(y[mask], model_probabilities["AdaBoost"][mask])[0]
        stack_auc = auc_ci(y[mask], stacking_probability[mask])[0]
        fold_auc_rows.append({"fold": fold, "AdaBoost": ada_auc, "Stacking": stack_auc})
    fold_auc = pd.DataFrame(fold_auc_rows)
    fold_auc.to_csv(output_dir / "internal_fold_auc_check.csv", index=False)

    summary = {
        "cohort": {
            "n": int(len(y)),
            "events": int(y.sum()),
            "non_events": int(len(y) - y.sum()),
            "event_rate": float(y.mean()),
        },
        "reported_style_fold_mean_check": {
            "AdaBoost_mean_fold_auc": float(fold_auc["AdaBoost"].mean()),
            "Stacking_mean_fold_auc": float(fold_auc["Stacking"].mean()),
        },
        "primary_pooled_oof_delong": primary.to_dict(),
        "calibration": calibration_table.set_index("model").to_dict(orient="index"),
        "notes": [
            "Paired DeLong uses one OOF prediction per patient, not the mean of 10 fold AUCs.",
            "Calibration is estimated on pooled OOF predictions from the original saved fold models.",
            "The original preprocessing was reproduced for numerical comparability; it was fitted before CV and can yield optimistic internal estimates.",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
