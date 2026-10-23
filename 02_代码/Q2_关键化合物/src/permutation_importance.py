"""Leakage-safe helpers for Q2 conditional and module permutation importance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def fit_quantile_edges(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Fit finite quantile-bin edges from training values only."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least two")
    if not len(finite):
        return np.array([-np.inf, np.inf])
    interior = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    interior = np.unique(np.asarray(interior, dtype=float))
    return np.concatenate(([-np.inf], interior, [np.inf]))


def assign_quantile_groups(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign validation values using already-fitted training quantile edges."""
    values = np.asarray(values, dtype=float).reshape(-1)
    edges = np.asarray(edges, dtype=float).reshape(-1)
    if len(edges) < 2 or not np.isneginf(edges[0]) or not np.isposinf(edges[-1]):
        raise ValueError("edges must start at -inf and end at inf")
    groups = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(values)
    groups[finite] = np.digitize(values[finite], edges[1:-1], right=True)
    return groups


def grouped_permutation_order(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return a row order that shuffles only within each conditioning group."""
    groups = np.asarray(groups).reshape(-1)
    order = np.arange(len(groups), dtype=int)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) > 1:
            order[rows] = rng.permutation(rows)
    return order


def connected_modules(
    correlation: pd.DataFrame,
    *,
    eligible_features: Sequence[str],
    threshold: float = 0.7,
) -> list[list[str]]:
    """Build deterministic connected components from an absolute-correlation graph."""
    features = [str(feature) for feature in eligible_features]
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    missing = sorted(set(features) - set(correlation.index))
    if missing:
        raise ValueError(f"correlation matrix misses features: {missing[:3]}")
    matrix = correlation.loc[features, features].to_numpy(float)
    adjacency = np.abs(matrix) >= float(threshold)
    np.fill_diagonal(adjacency, True)
    n_components, labels = connected_components(
        csr_matrix(adjacency), directed=False, return_labels=True
    )
    modules = []
    for label in range(n_components):
        modules.append(sorted(np.asarray(features, dtype=object)[labels == label].tolist()))
    return sorted(modules, key=lambda members: (members[0], len(members), tuple(members)))


def strongest_neighbor(correlation: pd.DataFrame, feature: str) -> str:
    """Return the feature with largest absolute training-fold correlation."""
    if feature not in correlation.index or feature not in correlation.columns:
        raise ValueError(f"unknown feature: {feature}")
    values = correlation.loc[feature].drop(index=feature).dropna()
    if values.empty:
        raise ValueError(f"feature has no finite neighbor: {feature}")
    table = pd.DataFrame({"feature": values.index.astype(str), "score": values.abs()})
    table = table.sort_values(
        ["score", "feature"], ascending=[False, True], kind="mergesort"
    )
    return str(table.iloc[0]["feature"])


def aggregate_feature_importance(rankings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-local error increases, prioritizing stable positive effects."""
    required = {
        "method",
        "target_code",
        "feature_code",
        "split_id",
        "rank",
        "score",
        "selected_top3",
    }
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"importance rankings miss columns: {sorted(missing)}")

    def q25(values: pd.Series) -> float:
        return float(values.quantile(0.25))

    def q75(values: pd.Series) -> float:
        return float(values.quantile(0.75))

    aggregate = (
        rankings.groupby(["method", "target_code", "feature_code"], sort=False)
        .agg(
            median_score=("score", "median"),
            score_q25=("score", q25),
            score_q75=("score", q75),
            positive_fraction=("score", lambda x: float((x > 0).mean())),
            top3_frequency=("selected_top3", "mean"),
            median_rank=("rank", "median"),
            n_outer_folds=("split_id", "nunique"),
        )
        .reset_index()
    )
    aggregate["score_iqr"] = aggregate["score_q75"] - aggregate["score_q25"]
    aggregate["positive_median"] = aggregate["median_score"] > 0
    aggregate = aggregate.sort_values(
        [
            "method",
            "target_code",
            "positive_median",
            "top3_frequency",
            "positive_fraction",
            "median_score",
            "median_rank",
            "feature_code",
        ],
        ascending=[True, True, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    aggregate["aggregate_rank"] = (
        aggregate.groupby(["method", "target_code"]).cumcount() + 1
    )
    return aggregate[
        [
            "method",
            "target_code",
            "feature_code",
            "aggregate_rank",
            "median_rank",
            "median_score",
            "score_q25",
            "score_q75",
            "score_iqr",
            "positive_fraction",
            "top3_frequency",
            "n_outer_folds",
        ]
    ]
