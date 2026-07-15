"""Final AUC comparison and raw-probability calibration analysis.

This script supersedes the earlier post-hoc validation scripts. It:

1. reconstructs one internal out-of-fold (OOF) prediction per patient;
2. compares Stacking with AdaBoost internally using paired DeLong inference;
3. creates one ensemble prediction per external patient by averaging the ten
   fold-specific probabilities;
4. compares Stacking with Random Forest (and all other base models) in the
   external cohort using paired DeLong inference;
5. evaluates calibration from the models' unmodified raw probabilities; and
6. draws risk-decile calibration plots without applying or displaying any
   recalibration transformation.

The external standardized features are transformed using a StandardScaler
fitted on the development cohort. The external cohort is never used to fit a
scaler or a probability recalibration model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.preprocessing import StandardScaler

import evaluate_internal_oof as core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RF_INDEX = next(
    index for index, (name, _, _) in enumerate(core.BASE_MODELS) if name == "RandomForest"
)
ADABOOST_INDEX = next(
    index for index, (name, _, _) in enumerate(core.BASE_MODELS) if name == "AdaBoost"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"D:\Python code\mechine_learning_stroke"),
    )
    parser.add_argument(
        "--internal-data-file",
        default="obesity_NAFLD-5936+1555-0816插值.xlsx",
    )
    parser.add_argument(
        "--external-data-file",
        default="obesity_NAFLD-外部验证353-0816插值.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "auc_calibration_raw_probability_results",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Patient-level bootstrap replicates for calibration metric CIs.",
    )
    return parser.parse_args()


def load_xy(data_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = pd.read_excel(data_path)
    if "group" not in data.columns or "gender" not in data.columns:
        raise ValueError(f"Missing group/gender columns in {data_path}")
    y = data["group"].astype(int).to_numpy()
    x = data.drop(columns=["group"]).copy()
    x["gender"] = x["gender"].round().astype(int)
    return y, x


def development_scaled_external(
    development_x: pd.DataFrame, external_x: pd.DataFrame
) -> np.ndarray:
    """Transform external continuous features on the development scale."""
    development_values = development_x.to_numpy(dtype=float)
    external_values = external_x.to_numpy(dtype=float)
    scaler = StandardScaler().fit(development_values[:, 1:])
    return np.insert(
        scaler.transform(external_values[:, 1:]),
        0,
        external_values[:, 0],
        axis=1,
    )


def external_ensemble_predictions(
    internal_data_path: Path,
    external_data_path: Path,
    model_dir: Path,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return one raw probability per external patient for every model.

    Each base-model probability is the mean of the ten fold-specific model
    probabilities. Stacking uses majority-vote base labels as meta-features,
    matching the original external prediction rule, and the mean of the ten
    fold-specific CatBoost probabilities as its final patient-level score.
    """
    _, development_x = load_xy(internal_data_path)
    y, external_x = load_xy(external_data_path)
    external_x_std = development_scaled_external(development_x, external_x)

    n_patients = len(y)
    n_models = len(core.BASE_MODELS)
    base_fold_labels = np.zeros(
        (core.N_SPLITS, n_patients, n_models), dtype=int
    )
    base_fold_probabilities = np.zeros(
        (core.N_SPLITS, n_patients, n_models), dtype=float
    )

    for fold_idx in range(core.N_SPLITS):
        for model_idx, (_, prefix, use_standardized) in enumerate(core.BASE_MODELS):
            model_path = model_dir / f"{prefix}{fold_idx}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = joblib.load(model_path)
            model_x = external_x_std if use_standardized else external_x
            base_fold_labels[fold_idx, :, model_idx] = model.predict(model_x).astype(
                int
            )
            base_fold_probabilities[fold_idx, :, model_idx] = model.predict_proba(
                model_x
            )[:, 1]

    majority_labels = (
        base_fold_labels.sum(axis=0) >= (core.N_SPLITS / 2)
    ).astype(int)
    meta_labels = pd.DataFrame(
        majority_labels,
        columns=core.STACK_FEATURE_NAMES,
        index=external_x.index,
    )
    external_stack_x = pd.concat([external_x, meta_labels], axis=1)
    categorical_features = core.STACK_FEATURE_NAMES + ["gender"]
    external_stack_x[categorical_features] = external_stack_x[
        categorical_features
    ].astype(int)

    stacking_fold_probabilities = np.zeros(
        (n_patients, core.N_SPLITS), dtype=float
    )
    external_pool = Pool(
        external_stack_x,
        y,
        cat_features=categorical_features,
    )
    for fold_idx in range(core.N_SPLITS):
        model_path = model_dir / f"CatBoostClassifier{fold_idx}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = joblib.load(model_path)
        stacking_fold_probabilities[:, fold_idx] = model.predict_proba(
            external_pool
        )[:, 1]

    base_mean_probabilities = {
        name: base_fold_probabilities[:, :, model_idx].mean(axis=0)
        for model_idx, (name, _, _) in enumerate(core.BASE_MODELS)
    }
    stacking_mean_probability = stacking_fold_probabilities.mean(axis=1)
    return (
        y,
        base_mean_probabilities,
        stacking_mean_probability,
        base_fold_probabilities,
        stacking_fold_probabilities,
    )


def calibration_point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    metrics = core.calibration_metrics(y, p)
    metrics["calibration_intercept"] = metrics.pop("recalibration_intercept")
    return metrics


def ordinary_bootstrap_calibration(
    y: np.ndarray,
    p: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Patient-level nonparametric bootstrap for raw-probability metrics."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n = len(y)
    while len(rows) < replicates:
        sample = rng.integers(0, n, size=n)
        y_boot = y[sample]
        if np.unique(y_boot).size != 2:
            continue
        p_boot = p[sample]
        intercept, slope, citl = core.logistic_recalibration(y_boot, p_boot)
        expected = float(p_boot.sum())
        rows.append(
            {
                "brier_score": float(np.mean((p_boot - y_boot) ** 2)),
                "observed_expected_ratio": float(y_boot.sum() / expected),
                "calibration_in_the_large": citl,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def calibration_analysis(
    cohort: str,
    y: np.ndarray,
    p: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    metrics = calibration_point_metrics(y, p)
    bootstrap = ordinary_bootstrap_calibration(y, p, replicates, seed)
    row: dict[str, float | int | str] = {
        "cohort": cohort,
        "probability_source": "raw model predict_proba; no recalibration applied",
        **metrics,
    }
    for metric in [
        "brier_score",
        "observed_expected_ratio",
        "calibration_in_the_large",
        "calibration_intercept",
        "calibration_slope",
    ]:
        lower, upper = bootstrap[metric].quantile([0.025, 0.975])
        row[f"{metric}_ci_lower"] = float(lower)
        row[f"{metric}_ci_upper"] = float(upper)
    return row, core.calibration_bins(y, p)


def raw_probability_calibration_plot(
    output_dir: Path,
    cohorts: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, dict[str, float | int | str]],
    bins: dict[str, pd.DataFrame],
) -> None:
    """Draw observed risk against mean raw predicted probability by decile."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), sharex=True, sharey=True)
    colors = ["#176B87", "#B14A3B"]
    for ax, color, (cohort, _) in zip(axes, colors, cohorts.items()):
        cohort_bins = bins[cohort]
        yerr = np.vstack(
            [
                cohort_bins["observed_rate"]
                - cohort_bins["observed_ci_lower"],
                cohort_bins["observed_ci_upper"]
                - cohort_bins["observed_rate"],
            ]
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.3,
            color="#666666",
            label="Ideal calibration",
        )
        ax.errorbar(
            cohort_bins["mean_predicted"],
            cohort_bins["observed_rate"],
            yerr=yerr,
            fmt="o-",
            markersize=5.5,
            linewidth=1.8,
            capsize=2.5,
            color=color,
            label="Raw probabilities by risk decile (95% CI)",
        )
        metric = metrics[cohort]
        ax.text(
            0.04,
            0.96,
            f"Slope = {float(metric['calibration_slope']):.3f}\n"
            f"CITL = {float(metric['calibration_in_the_large']):.3f}\n"
            f"Brier = {float(metric['brier_score']):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#cccccc",
            },
        )
        ax.set_title(cohort)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean raw predicted probability")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Observed outcome proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Stacking model calibration using unmodified raw probabilities",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(
        output_dir / "stacking_calibration_raw_probabilities.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "stacking_calibration_raw_probabilities.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def auc_table(
    y: np.ndarray, model_probabilities: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for model, probability in model_probabilities.items():
        auc, lower, upper = core.auc_ci(y, probability)
        rows.append(
            {
                "model": model,
                "auc": auc,
                "auc_ci_lower": lower,
                "auc_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def stacking_delong_table(
    y: np.ndarray,
    stacking_probability: np.ndarray,
    comparator_probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for comparator, probability in comparator_probabilities.items():
        rows.append(
            {
                "comparator": comparator,
                **core.paired_delong(y, stacking_probability, probability),
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = core.holm_adjust(result["p_value"].tolist())
    return result


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def reviewer_response(
    internal: dict[str, float],
    external: dict[str, float],
    calibration_rows: dict[str, dict[str, float | int | str]],
) -> str:
    internal_cal = calibration_rows["Internal OOF"]
    external_cal = calibration_rows["External validation"]
    return f"""# Response to reviewer: AUC comparison and calibration

## Comment 6

**Reviewer comment:** The stacking model only marginally outperformed AdaBoost (AUC 0.877 vs 0.874). Statistical comparison of AUCs should be performed to demonstrate meaningful improvement.

**Response:** Thank you for this important comment. We performed a paired DeLong comparison using one out-of-fold prediction per participant, thereby accounting for the correlation between predictions generated for the same individuals. The stacking model had an AUC of {fmt(internal['auc_stacking'])}, compared with {fmt(internal['auc_comparator'])} for AdaBoost. The absolute AUC difference was {fmt(internal['auc_difference'])} (95% CI, {fmt(internal['difference_ci_lower'])} to {fmt(internal['difference_ci_upper'])}; P={fmt_p(internal['p_value'])}). Although the difference was statistically detectable, its absolute magnitude was small. We therefore revised the manuscript to describe the improvement as marginal and removed statements implying a substantial or clinically meaningful superiority.

As an additional external-validation comparison, the stacking model had an AUC of {fmt(external['auc_stacking'])}, compared with {fmt(external['auc_comparator'])} for random forest, the best-performing base model in the external cohort. The AUC difference was {fmt(external['auc_difference'])} (95% CI, {fmt(external['difference_ci_lower'])} to {fmt(external['difference_ci_upper'])}; P={fmt_p(external['p_value'])}). These results have been added to the revised Results section.

## Comment 3

**Reviewer comment:** Provide calibration metrics and calibration plots. AUC alone is insufficient for evaluating clinical prediction models.

**Response:** We agree and have added calibration-in-the-large, calibration intercept, calibration slope, Brier score, and calibration plots for both internal and external validation. All analyses used the models' original, unmodified predicted probabilities; no probability recalibration was fitted or applied. In internal out-of-fold validation, the calibration-in-the-large was {fmt(float(internal_cal['calibration_in_the_large']))}, the calibration intercept was {fmt(float(internal_cal['calibration_intercept']))}, the calibration slope was {fmt(float(internal_cal['calibration_slope']))}, and the Brier score was {fmt(float(internal_cal['brier_score']))}. In external validation, the corresponding values were {fmt(float(external_cal['calibration_in_the_large']))}, {fmt(float(external_cal['calibration_intercept']))}, {fmt(float(external_cal['calibration_slope']))}, and {fmt(float(external_cal['brier_score']))}, respectively. The external results indicate systematic overprediction, which is now explicitly acknowledged in the Results and Discussion. Risk-decile calibration plots with 95% confidence intervals have been added to the revised manuscript.

## Suggested Methods text

Pairwise comparisons of correlated AUCs were performed using the DeLong method based on patient-level predicted probabilities. The stacking model was compared with AdaBoost in internal out-of-fold validation and with random forest, the best-performing base model, in external validation. Calibration was evaluated using the original, unmodified predicted probabilities. Calibration-in-the-large, calibration intercept, calibration slope, and Brier score were reported. Calibration plots compared the mean raw predicted probability with the observed outcome proportion across risk-decile groups; no recalibration model was fitted or applied. Ninety-five percent confidence intervals for calibration metrics were obtained using patient-level bootstrap resampling.

## Suggested Results text

In internal out-of-fold validation, the stacking model achieved an AUC of {fmt(internal['auc_stacking'])}, compared with {fmt(internal['auc_comparator'])} for AdaBoost (difference, {fmt(internal['auc_difference'])}; 95% CI, {fmt(internal['difference_ci_lower'])} to {fmt(internal['difference_ci_upper'])}; paired DeLong P={fmt_p(internal['p_value'])}). Thus, the stacking model showed only a marginal improvement in discrimination. In external validation, the stacking model achieved an AUC of {fmt(external['auc_stacking'])}, compared with {fmt(external['auc_comparator'])} for random forest (difference, {fmt(external['auc_difference'])}; 95% CI, {fmt(external['difference_ci_lower'])} to {fmt(external['difference_ci_upper'])}; paired DeLong P={fmt_p(external['p_value'])}).

Using the original predicted probabilities, the stacking model showed good overall calibration in internal out-of-fold validation (calibration-in-the-large, {fmt(float(internal_cal['calibration_in_the_large']))}; calibration intercept, {fmt(float(internal_cal['calibration_intercept']))}; calibration slope, {fmt(float(internal_cal['calibration_slope']))}; Brier score, {fmt(float(internal_cal['brier_score']))}). In the external cohort, the calibration-in-the-large was {fmt(float(external_cal['calibration_in_the_large']))}, the calibration intercept was {fmt(float(external_cal['calibration_intercept']))}, the calibration slope was {fmt(float(external_cal['calibration_slope']))}, and the Brier score was {fmt(float(external_cal['brier_score']))}. The negative calibration-in-the-large and calibration intercept indicated that the model systematically overestimated absolute risk in the external cohort despite retaining good discrimination.
"""


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    internal_data_path = base_dir / args.internal_data_file
    external_data_path = base_dir / args.external_data_file
    model_dir = base_dir / "NAFLD_allmodel"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_internal, _, base_internal, stack_internal, fold_number = (
        core.reconstruct_oof_predictions(internal_data_path, model_dir)
    )
    ada_internal = base_internal[:, ADABOOST_INDEX]
    internal_delong = core.paired_delong(
        y_internal,
        stack_internal,
        ada_internal,
    )

    (
        y_external,
        base_external,
        stack_external,
        base_external_by_fold,
        stack_external_by_fold,
    ) = external_ensemble_predictions(
        internal_data_path,
        external_data_path,
        model_dir,
    )
    external_probabilities = {**base_external, "Stacking": stack_external}
    external_auc = auc_table(y_external, external_probabilities)
    external_comparisons = stacking_delong_table(
        y_external,
        stack_external,
        base_external,
    )
    external_rf = external_comparisons.loc[
        external_comparisons["comparator"] == "RandomForest"
    ].iloc[0].to_dict()

    cohorts = {
        "Internal OOF": (y_internal, stack_internal),
        "External validation": (y_external, stack_external),
    }
    calibration_rows: list[dict[str, float | int | str]] = []
    metrics_by_cohort: dict[str, dict[str, float | int | str]] = {}
    bins_by_cohort: dict[str, pd.DataFrame] = {}
    for cohort_index, (cohort, (y, probability)) in enumerate(cohorts.items()):
        row, bins = calibration_analysis(
            cohort,
            y,
            probability,
            args.bootstrap,
            core.RANDOM_STATE + cohort_index,
        )
        calibration_rows.append(row)
        metrics_by_cohort[cohort] = row
        bins_by_cohort[cohort] = bins
        bins.to_csv(
            output_dir
            / f"calibration_bins_{cohort.lower().replace(' ', '_')}_raw.csv",
            index=False,
        )

    calibration_table = pd.DataFrame(calibration_rows)
    calibration_table.to_csv(
        output_dir / "calibration_metrics_raw_probabilities.csv",
        index=False,
    )
    raw_probability_calibration_plot(
        output_dir,
        cohorts,
        metrics_by_cohort,
        bins_by_cohort,
    )

    pd.DataFrame([internal_delong]).to_csv(
        output_dir / "internal_stacking_vs_adaboost_delong.csv",
        index=False,
    )
    external_auc.to_csv(output_dir / "external_auc_estimates.csv", index=False)
    external_comparisons.to_csv(
        output_dir / "external_stacking_delong_comparisons.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "outcome": y_internal,
            "fold": fold_number,
            "Stacking_raw_probability": stack_internal,
            "AdaBoost_raw_probability": ada_internal,
        }
    ).to_csv(output_dir / "internal_patient_predictions.csv", index=False)
    pd.DataFrame(
        {
            "outcome": y_external,
            **{
                f"{name}_mean_raw_probability": probability
                for name, probability in external_probabilities.items()
            },
        }
    ).to_csv(output_dir / "external_patient_predictions.csv", index=False)

    response = reviewer_response(
        internal_delong,
        external_rf,
        metrics_by_cohort,
    )
    (output_dir / "reviewer_response_auc_calibration.md").write_text(
        response,
        encoding="utf-8",
    )

    summary = {
        "cohorts": {
            "internal": {
                "n": int(len(y_internal)),
                "events": int(y_internal.sum()),
            },
            "external": {
                "n": int(len(y_external)),
                "events": int(y_external.sum()),
            },
        },
        "internal_stacking_vs_adaboost_paired_delong": internal_delong,
        "external_stacking_vs_random_forest_paired_delong": external_rf,
        "calibration_raw_probabilities": calibration_table.set_index("cohort").to_dict(
            orient="index"
        ),
        "analysis_rules": [
            "One final prediction per patient was used for all AUC inference.",
            "Ten fold-specific external probabilities were averaged before AUC analysis.",
            "External standardization used development-cohort mean and scale only.",
            "Calibration plots used unmodified raw probabilities; no recalibration was applied.",
            "Internal OOF predictions reproduce the preprocessing of the saved original models.",
        ],
        "audit_shapes": {
            "external_base_fold_probabilities": list(base_external_by_fold.shape),
            "external_stacking_fold_probabilities": list(
                stack_external_by_fold.shape
            ),
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
