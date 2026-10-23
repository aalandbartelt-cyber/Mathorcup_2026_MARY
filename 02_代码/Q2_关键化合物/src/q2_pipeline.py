"""Shared orchestration utilities for the Q2 outer-fold experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from q2_contract import Q1_RESULT_DIR


@dataclass(frozen=True)
class OuterFold:
    repeat: int
    fold: int
    split_id: str
    train_index: np.ndarray
    validation_index: np.ndarray


def load_frozen_outer_folds(n_samples: int, repeats: int) -> list[OuterFold]:
    """Load the first ``repeats`` from Q1's frozen 10×3 outer split file."""
    if not 1 <= int(repeats) <= 10:
        raise ValueError("repeats must be in 1..10")
    split_path = Q1_RESULT_DIR / "q1_cv_splits.csv"
    table = pd.read_csv(split_path)
    table = table.loc[table["repeat"] < int(repeats)].copy()
    all_index = np.arange(int(n_samples), dtype=int)
    folds = []
    for (repeat, fold, split_id), group in table.groupby(
        ["repeat", "fold", "split_id"], sort=True
    ):
        validation = np.sort(group["row_position"].to_numpy(dtype=int))
        if len(np.unique(validation)) != len(validation):
            raise ValueError(f"duplicate validation rows in {split_id}")
        if (validation < 0).any() or (validation >= n_samples).any():
            raise IndexError(f"validation row outside dataset in {split_id}")
        train = np.setdiff1d(all_index, validation, assume_unique=True)
        folds.append(
            OuterFold(
                repeat=int(repeat),
                fold=int(fold),
                split_id=str(split_id),
                train_index=train,
                validation_index=validation,
            )
        )
    if len(folds) != int(repeats) * 3:
        raise ValueError(f"expected {int(repeats) * 3} folds, got {len(folds)}")
    validation_counts = np.bincount(
        np.concatenate([fold.validation_index for fold in folds]),
        minlength=n_samples,
    )
    if not np.all(validation_counts == int(repeats)):
        raise ValueError("each row must be validated once per repeat")
    return folds


def extend_laplacian(laplacian: np.ndarray, total_features: int) -> np.ndarray:
    """Append zero graph penalties for fold-specific missing indicators."""
    laplacian = np.asarray(laplacian, dtype=float)
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian must be square")
    if int(total_features) < laplacian.shape[0]:
        raise ValueError("total_features cannot be smaller than the graph")
    result = np.zeros((int(total_features), int(total_features)), dtype=float)
    result[: laplacian.shape[0], : laplacian.shape[1]] = laplacian
    return result


def aggregate_fold_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold ranks without mixing score scales across methods."""
    required = {
        "method",
        "target_code",
        "feature_code",
        "rank",
        "score",
        "split_id",
    }
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings missing columns: {sorted(missing)}")

    def q25(series: pd.Series) -> float:
        return float(series.quantile(0.25))

    def q75(series: pd.Series) -> float:
        return float(series.quantile(0.75))

    def median_nonzero(series: pd.Series) -> float:
        selected = series.loc[series.abs() > 1e-12]
        return float(selected.median()) if len(selected) else 0.0

    aggregate = (
        rankings.groupby(["method", "target_code", "feature_code"], sort=False)
        .agg(
            median_rank=("rank", "median"),
            rank_q25=("rank", q25),
            rank_q75=("rank", q75),
            median_score=("score", "median"),
            median_nonzero_score=("score", median_nonzero),
            median_abs_score=("score", lambda x: float(x.abs().median())),
            score_q25=("score", q25),
            score_q75=("score", q75),
            selection_frequency=(
                "score", lambda x: float((x.abs() > 1e-12).mean())
            ),
            positive_fraction=("score", lambda x: float((x > 0).mean())),
            negative_fraction=("score", lambda x: float((x < 0).mean())),
            n_outer_folds=("split_id", "nunique"),
        )
        .reset_index()
    )
    aggregate["rank_iqr"] = aggregate["rank_q75"] - aggregate["rank_q25"]
    aggregate["direction"] = np.select(
        [
            aggregate["median_nonzero_score"] > 0,
            aggregate["median_nonzero_score"] < 0,
        ],
        ["positive", "negative"],
        default="zero",
    )
    aggregate = aggregate.sort_values(
        [
            "method",
            "target_code",
            "selection_frequency",
            "median_abs_score",
            "median_rank",
            "feature_code",
        ],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    aggregate["aggregate_rank"] = (
        aggregate.groupby(["method", "target_code"]).cumcount() + 1
    )
    columns = [
        "method",
        "target_code",
        "feature_code",
        "aggregate_rank",
        "median_rank",
        "rank_q25",
        "rank_q75",
        "rank_iqr",
        "median_score",
        "median_nonzero_score",
        "median_abs_score",
        "score_q25",
        "score_q75",
        "direction",
        "selection_frequency",
        "positive_fraction",
        "negative_fraction",
        "n_outer_folds",
    ]
    return aggregate[columns]


def metrics_by_target(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    target_names: list[str],
    method: str,
) -> pd.DataFrame:
    """Calculate target-wise OOF metrics without clipping negative R²."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_prediction, dtype=float)
    if truth.shape != prediction.shape or truth.shape[1] != len(target_names):
        raise ValueError("target arrays or target names disagree")
    rows = []
    for position, target in enumerate(target_names):
        actual = truth[:, position]
        fitted = prediction[:, position]
        spearman = pd.Series(actual).corr(pd.Series(fitted), method="spearman")
        rows.append(
            {
                "method": method,
                "target_code": target,
                "r2": float(r2_score(actual, fitted)),
                "mae": float(mean_absolute_error(actual, fitted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, fitted))),
                "spearman": float(spearman) if np.isfinite(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Macro-average target metrics for each fitted method."""
    return (
        metrics.groupby("method", sort=False)
        .agg(
            macro_r2=("r2", "mean"),
            macro_mae=("mae", "mean"),
            macro_rmse=("rmse", "mean"),
            macro_spearman=("spearman", "mean"),
            positive_r2_targets=("r2", lambda x: int((x > 0).sum())),
            n_targets=("target_code", "nunique"),
        )
        .reset_index()
    )
