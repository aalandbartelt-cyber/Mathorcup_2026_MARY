"""Reusable continuous and ordered-proxy metrics for Q1 predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


N_LEVELS = 7
METRIC_COLUMNS = [
    "r2", "mae", "rmse", "spearman", "qwk_proxy", "acc_proxy",
    "acc_pm1_proxy", "level_mae_proxy", "level_spearman_proxy",
]


def quantile_levels(
    values: np.ndarray,
    reference: np.ndarray,
    n_levels: int = N_LEVELS,
) -> np.ndarray:
    """Map continuous values to integer levels using reference quantiles."""
    value_array = np.asarray(values, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if value_array.ndim != 1 or reference_array.ndim != 1:
        raise ValueError("values and reference must both be one-dimensional")
    if not np.isfinite(value_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("values and reference must be finite")
    if n_levels < 2:
        raise ValueError("n_levels must be at least two")
    quantiles = np.linspace(0.0, 1.0, n_levels + 1)[1:-1]
    edges = np.unique(np.quantile(reference_array, quantiles))
    return np.digitize(value_array, edges).astype(int)


def ordered_proxy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return BZD-compatible seven-bin diagnostic metrics for one target."""
    true_levels = quantile_levels(y_true, y_true)
    predicted_levels = quantile_levels(y_pred, y_pred)
    difference = np.abs(true_levels - predicted_levels)
    level_spearman = spearmanr(true_levels, predicted_levels).statistic
    return {
        "qwk_proxy": float(
            cohen_kappa_score(true_levels, predicted_levels, weights="quadratic")
        ),
        "acc_proxy": float(np.mean(difference == 0)),
        "acc_pm1_proxy": float(np.mean(difference <= 1)),
        "level_mae_proxy": float(np.mean(difference)),
        "level_spearman_proxy": float(level_spearman),
    }


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return continuous and ordered-proxy metrics for one sensory target."""
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    metrics = {
        "r2": float(r2_score(true_array, pred_array)),
        "mae": float(mean_absolute_error(true_array, pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(true_array, pred_array))),
        "spearman": float(spearmanr(true_array, pred_array).statistic),
    }
    metrics.update(ordered_proxy_metrics(true_array, pred_array))
    return metrics


def matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-average all Q1 metrics over the sensory targets."""
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if true_array.ndim != 2:
        raise ValueError("prediction matrices must be two-dimensional")
    rows = [
        target_metrics(true_array[:, position], pred_array[:, position])
        for position in range(true_array.shape[1])
    ]
    return {
        name: float(np.nanmean([row[name] for row in rows]))
        for name in METRIC_COLUMNS
    }


def evaluate_prediction_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_codes: np.ndarray,
    experiment_id: str,
    evaluation: str,
) -> pd.DataFrame:
    """Evaluate every target and return one tidy row per target."""
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if true_array.ndim != 2 or true_array.shape[1] != len(target_codes):
        raise ValueError("target_codes do not match the prediction matrix")
    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "evaluation": evaluation,
                "target_code": str(code),
                **target_metrics(true_array[:, position], pred_array[:, position]),
            }
            for position, code in enumerate(target_codes)
        ]
    )


def summarise_target_metrics(by_target: pd.DataFrame) -> pd.DataFrame:
    """Macro-average target metrics for each OOF evaluation."""
    summary = (
        by_target.groupby(["experiment_id", "evaluation"], sort=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    summary.insert(2, "n_targets", by_target["target_code"].nunique())
    summary.insert(3, "n_samples", 250)
    return summary
