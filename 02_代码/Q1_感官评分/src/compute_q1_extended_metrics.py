"""Compute continuous and ordered-proxy metrics for the frozen E003 OOF array.

The ordered metrics intentionally reproduce the BZD reference convention:
for every target, the observed values and predictions are independently split
into seven empirical-quantile bins.  These scores are useful only for a
same-definition post-hoc comparison.  They are not the deployable 0--6 scale
mapping used by the Q1 pipeline, because their bin edges use the complete OOF
sample (including validation observations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_contract import RESULT_DIR


EXPERIMENT_ID = "E003"
N_LEVELS = 7
SOURCE = RESULT_DIR / "q1_oof_predictions_E003.npz"


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
    continuous_spearman = spearmanr(y_true, y_pred).statistic
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(continuous_spearman),
    }
    metrics.update(ordered_proxy_metrics(y_true, y_pred))
    return metrics


def evaluate_prediction_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_codes: np.ndarray,
    evaluation: str,
) -> pd.DataFrame:
    """Evaluate every target and return one tidy row per target."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(target_codes):
        raise ValueError("target_codes do not match the prediction matrix")

    rows = []
    for target_position, target_code in enumerate(target_codes):
        row = {
            "experiment_id": EXPERIMENT_ID,
            "evaluation": evaluation,
            "target_code": str(target_code),
        }
        row.update(target_metrics(y_true[:, target_position], y_pred[:, target_position]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_target_metrics(by_target: pd.DataFrame) -> pd.DataFrame:
    """Macro-average target metrics for each OOF evaluation."""
    metric_columns = [
        "r2",
        "mae",
        "rmse",
        "spearman",
        "qwk_proxy",
        "acc_proxy",
        "acc_pm1_proxy",
        "level_mae_proxy",
        "level_spearman_proxy",
    ]
    summary = (
        by_target.groupby(["experiment_id", "evaluation"], sort=False)[metric_columns]
        .mean()
        .reset_index()
    )
    summary.insert(2, "n_targets", by_target["target_code"].nunique())
    summary.insert(3, "n_samples", 250)
    return summary


def main() -> None:
    archive = np.load(SOURCE, allow_pickle=True)
    y_true = np.asarray(archive["y_true"], dtype=float)
    predictions = np.asarray(archive["pred_extra_trees"], dtype=float)
    target_codes = np.asarray(archive["target_code"])
    if predictions.ndim != 3 or predictions.shape[1:] != y_true.shape:
        raise ValueError("unexpected E003 prediction tensor shape")
    if not np.isfinite(predictions).all():
        raise ValueError("E003 OOF predictions contain non-finite values")

    evaluations = [
        evaluate_prediction_matrix(
            y_true,
            predictions[repeat],
            target_codes,
            evaluation=f"repeat_{repeat}",
        )
        for repeat in range(predictions.shape[0])
    ]
    evaluations.append(
        evaluate_prediction_matrix(
            y_true,
            predictions.mean(axis=0),
            target_codes,
            evaluation="mean_oof",
        )
    )
    by_target = pd.concat(evaluations, ignore_index=True)
    summary = summarise_target_metrics(by_target)

    by_target_path = RESULT_DIR / "q1_extended_metrics_by_target_E003.csv"
    summary_path = RESULT_DIR / "q1_extended_metrics_E003.csv"
    manifest_path = RESULT_DIR / "q1_extended_metrics_manifest_E003.json"
    by_target.to_csv(by_target_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "source": str(SOURCE.relative_to(RESULT_DIR.parents[1])),
        "prediction_shape": list(predictions.shape),
        "continuous_metrics": ["R2", "MAE", "RMSE", "Spearman"],
        "ordered_proxy_metrics": [
            "QWK",
            "ACC",
            "ACC_pm1",
            "LevelMAE",
            "level_Spearman",
        ],
        "ordered_proxy_definition": (
            "For each target, observed values and predictions are independently "
            "mapped to integer levels 0..6 using their own full-sample empirical "
            "quantiles; target scores are then macro-averaged."
        ),
        "usage_limit": (
            "Post-hoc BZD-compatible comparison only. This mapping is not used "
            "for final 0--6 inference because it uses the full OOF distribution."
        ),
        "outputs": [str(summary_path), str(by_target_path)],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
