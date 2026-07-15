"""Patient-level bootstrap confidence intervals for binary prediction metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ("Accuracy", "AUC", "Recall", "F1", "Kappa", "Specificity")


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Specificity": _specificity(y_true, y_pred),
    }


def patient_bootstrap_metric_tables(
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_labels: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260713,
) -> dict[str, pd.DataFrame]:
    """Estimate metrics and percentile CIs by resampling patients.

    The same patient indices are used for every model in each replicate so that
    estimates remain paired across models. Samples containing one outcome class
    are discarded because AUC is undefined for those samples.
    """
    outcome = np.asarray(y_true, dtype=int).reshape(-1)
    predicted_class = np.asarray(y_pred, dtype=int)
    probability = np.asarray(y_proba, dtype=float)
    labels = list(model_labels)

    if predicted_class.ndim == 1:
        predicted_class = predicted_class[:, None]
    if probability.ndim == 1:
        probability = probability[:, None]
    expected_shape = (len(outcome), len(labels))
    if predicted_class.shape != expected_shape or probability.shape != expected_shape:
        raise ValueError(
            "Patient prediction matrices must have shape "
            f"{expected_shape}; got {predicted_class.shape} and {probability.shape}."
        )
    if set(np.unique(outcome)) != {0, 1}:
        raise ValueError("Outcome must contain both binary classes 0 and 1.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Predicted probabilities must be finite and within [0, 1].")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    point = {metric: np.empty(len(labels), dtype=float) for metric in METRIC_ORDER}
    for model_idx in range(len(labels)):
        values = _metric_values(
            outcome,
            predicted_class[:, model_idx],
            probability[:, model_idx],
        )
        for metric in METRIC_ORDER:
            point[metric][model_idx] = values[metric]

    draws = {
        metric: np.empty((n_bootstrap, len(labels)), dtype=float)
        for metric in METRIC_ORDER
    }
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < n_bootstrap:
        sample = rng.integers(0, len(outcome), size=len(outcome))
        sampled_outcome = outcome[sample]
        if np.unique(sampled_outcome).size != 2:
            continue
        for model_idx in range(len(labels)):
            values = _metric_values(
                sampled_outcome,
                predicted_class[sample, model_idx],
                probability[sample, model_idx],
            )
            for metric in METRIC_ORDER:
                draws[metric][completed, model_idx] = values[metric]
        completed += 1

    tables: dict[str, pd.DataFrame] = {}
    for metric in METRIC_ORDER:
        tables[metric] = pd.DataFrame(
            {
                "patient_level_estimate": point[metric],
                "bootstrap_std": np.nanstd(draws[metric], axis=0, ddof=1),
                "ci_lower": np.nanquantile(draws[metric], 0.025, axis=0),
                "ci_upper": np.nanquantile(draws[metric], 0.975, axis=0),
                "bootstrap_replicates": n_bootstrap,
            },
            index=labels,
        )
    return tables


def write_metric_workbook_with_bootstrap(
    fold_metrics: Mapping[str, np.ndarray],
    model_labels: Sequence[str],
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    *,
    n_folds: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 20260713,
) -> dict[str, pd.DataFrame]:
    """Write fold descriptions plus patient-level estimates and bootstrap CIs."""
    tables = patient_bootstrap_metric_tables(
        y_true,
        y_pred,
        y_proba,
        model_labels,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    fold_columns = [f"Fold{i}" for i in range(1, n_folds + 1)]
    long_rows: list[pd.DataFrame] = []
    with pd.ExcelWriter(Path(output_path)) as writer:
        for metric in METRIC_ORDER:
            fold_values = np.asarray(fold_metrics[metric], dtype=float)[:, :n_folds]
            if fold_values.shape != (len(model_labels), n_folds):
                raise ValueError(
                    f"{metric} fold matrix has shape {fold_values.shape}; expected "
                    f"{(len(model_labels), n_folds)}."
                )
            fold_table = pd.DataFrame(
                fold_values,
                index=list(model_labels),
                columns=fold_columns,
            )
            combined = pd.concat([fold_table, tables[metric]], axis=1)
            combined.to_excel(writer, sheet_name=metric, index=True)
            long_table = tables[metric].reset_index(names="model")
            long_table.insert(1, "metric", metric)
            long_rows.append(long_table)
        pd.concat(long_rows, ignore_index=True).to_excel(
            writer,
            sheet_name="Bootstrap summary",
            index=False,
        )
    return tables
