"""Hierarchical module and conditional-residual importance primitives."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def joint_permutation(
    x, columns: Sequence[int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Permute selected module columns with one shared row order."""
    values = np.asarray(x, dtype=float)
    selected = np.asarray(columns, dtype=int)
    if values.ndim != 2 or not len(values):
        raise ValueError("x must be a non-empty two-dimensional matrix")
    if selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError("columns must be a non-empty vector of unique positions")
    if selected.min() < 0 or selected.max() >= values.shape[1]:
        raise IndexError("module column is out of bounds")
    order = rng.permutation(len(values))
    changed = values.copy()
    changed[:, selected] = values[order][:, selected]
    return changed, order


@dataclass
class ConditionalFeatureModel:
    pipeline: Pipeline
    n_conditioners: int

    def predict(self, conditioners, style) -> np.ndarray:
        numeric = np.asarray(conditioners, dtype=float)
        styles = np.asarray(style, dtype=str)
        if numeric.ndim != 2 or numeric.shape[1] != self.n_conditioners:
            raise ValueError("conditioner shape disagrees with fitted model")
        if styles.shape != (len(numeric),):
            raise ValueError("style vector must align with conditioners")
        frame = pd.DataFrame(
            numeric,
            columns=[f"x_{position}" for position in range(self.n_conditioners)],
        )
        frame["style"] = styles
        return np.asarray(self.pipeline.predict(frame), dtype=float)


def fit_conditional_feature_model(
    conditioners, feature, style
) -> ConditionalFeatureModel:
    """Fit compact HGB E[X_j | module peers, style] on training rows only."""
    numeric = np.asarray(conditioners, dtype=float)
    target = np.asarray(feature, dtype=float)
    styles = np.asarray(style, dtype=str)
    if numeric.ndim != 2 or numeric.shape[0] < 20 or numeric.shape[1] < 1:
        raise ValueError("conditioners must contain at least 20 rows and one column")
    if target.shape != (len(numeric),) or styles.shape != (len(numeric),):
        raise ValueError("feature and style must align with conditioners")
    if not np.isfinite(target).all():
        raise ValueError("conditional feature target must be finite")
    frame = pd.DataFrame(
        numeric,
        columns=[f"x_{position}" for position in range(numeric.shape[1])],
    )
    frame["style"] = styles
    numeric_columns = list(frame.columns[:-1])
    transformer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "style",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["style"],
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("transform", transformer),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_leaf_nodes=7,
                    min_samples_leaf=20,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    max_iter=100,
                    random_state=20261025,
                ),
            ),
        ]
    )
    pipeline.fit(frame, target)
    return ConditionalFeatureModel(pipeline, numeric.shape[1])


def residual_permutation(
    conditional_mean, residual, rng: np.random.Generator
) -> np.ndarray:
    """Keep the conditional mean fixed and permute only residual variation."""
    mean = np.asarray(conditional_mean, dtype=float)
    error = np.asarray(residual, dtype=float)
    if mean.ndim != 1 or error.shape != mean.shape or not len(mean):
        raise ValueError("conditional mean and residual must be aligned vectors")
    if not np.isfinite(mean).all() or not np.isfinite(error).all():
        raise ValueError("conditional mean and residual must be finite")
    return mean + error[rng.permutation(len(error))]


def delta_mae(y_true, baseline_prediction, changed_prediction) -> float:
    """Return permutation MAE minus baseline MAE; positive means useful signal."""
    truth = np.asarray(y_true, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    changed = np.asarray(changed_prediction, dtype=float)
    if truth.shape != baseline.shape or truth.shape != changed.shape or not truth.size:
        raise ValueError("truth and predictions must have identical non-empty shapes")
    if not all(np.isfinite(values).all() for values in (truth, baseline, changed)):
        raise ValueError("truth and predictions must be finite")
    return float(
        np.mean(np.abs(truth - changed)) - np.mean(np.abs(truth - baseline))
    )


def aggregate_hierarchical_importance(fold_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate field-level fold importance with stability-first ordering."""
    required = {"target_code", "feature_code", "delta_mae", "fold_rank"}
    missing = required - set(fold_rows.columns)
    if missing:
        raise ValueError(f"hierarchical importance rows miss {sorted(missing)}")
    frame = pd.DataFrame(fold_rows).copy()
    if frame.empty:
        raise ValueError("hierarchical importance rows are empty")
    grouped = frame.groupby(["target_code", "feature_code"], sort=False)
    result = grouped.agg(
        median_delta_mae=("delta_mae", "median"),
        positive_fraction=("delta_mae", lambda value: float((value > 0).mean())),
        fold_top3_frequency=("fold_rank", lambda value: float((value <= 3).mean())),
        fold_top10_frequency=("fold_rank", lambda value: float((value <= 10).mean())),
        median_fold_rank=("fold_rank", "median"),
        n_outer_folds=("fold_rank", "size"),
    ).reset_index()
    result = result.sort_values(
        [
            "target_code",
            "positive_fraction",
            "median_delta_mae",
            "fold_top3_frequency",
            "median_fold_rank",
            "feature_code",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["hierarchical_rank"] = result.groupby("target_code").cumcount() + 1
    return result
