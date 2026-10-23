"""Low-level deterministic utilities for Q2 bootstrap stability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stratified_bootstrap_indices(
    strata: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw within-stratum bootstrap rows and return their out-of-bag complement."""
    strata = np.asarray(strata)
    if strata.ndim != 1 or not len(strata):
        raise ValueError("strata must be a non-empty one-dimensional array")
    sampled = []
    for value in np.unique(strata):
        rows = np.flatnonzero(strata == value)
        sampled.append(rng.choice(rows, size=len(rows), replace=True))
    sample = np.concatenate(sampled).astype(int)
    rng.shuffle(sample)
    observed = np.zeros(len(strata), dtype=bool)
    observed[np.unique(sample)] = True
    oob = np.flatnonzero(~observed)
    return sample, oob


def rank_scores_descending(
    scores: np.ndarray, feature_codes: Sequence[str]
) -> np.ndarray:
    """Rank feature-by-target scores descending with deterministic name ties."""
    scores = np.asarray(scores, dtype=float)
    codes = np.asarray([str(value) for value in feature_codes], dtype=object)
    if scores.ndim != 2 or scores.shape[0] != len(codes):
        raise ValueError("score rows and feature codes disagree")
    ranks = np.empty(scores.shape, dtype=np.uint16)
    for target in range(scores.shape[1]):
        values = np.nan_to_num(scores[:, target], nan=-np.inf)
        order = np.lexsort((codes, -values))
        ranks[order, target] = np.arange(1, len(codes) + 1, dtype=np.uint16)
    return ranks


def consensus_ranks_from_methods(
    method_ranks: Sequence[np.ndarray],
    feature_codes: Sequence[str],
    *,
    rrf_k: float = 60.0,
) -> np.ndarray:
    """Apply the E106 support-aware reciprocal-rank rule to rank matrices."""
    if len(method_ranks) < 2 or float(rrf_k) <= 0:
        raise ValueError("at least two methods and positive rrf_k are required")
    arrays = [np.asarray(values) for values in method_ranks]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("method rank matrices must have identical shapes")
    n_features, n_targets = arrays[0].shape
    codes = np.asarray([str(value) for value in feature_codes], dtype=object)
    if len(codes) != n_features:
        raise ValueError("feature code count disagrees with rank matrices")
    stack = np.stack(arrays, axis=0).astype(float)
    top3 = np.sum(stack <= 3, axis=0)
    top10 = np.sum(stack <= 10, axis=0)
    rrf = np.sum(1.0 / (float(rrf_k) + stack), axis=0)
    best = np.min(stack, axis=0)
    consensus = np.empty((n_features, n_targets), dtype=np.uint16)
    for target in range(n_targets):
        order = np.lexsort(
            (
                codes,
                best[:, target],
                -rrf[:, target],
                -top3[:, target],
                -top10[:, target],
            )
        )
        consensus[order, target] = np.arange(
            1, n_features + 1, dtype=np.uint16
        )
    return consensus


def update_coclustering(
    counts: np.ndarray,
    modules: Sequence[Sequence[str]],
    feature_codes: Sequence[str],
) -> None:
    """Increment co-membership counts in place for one bootstrap graph."""
    counts = np.asarray(counts)
    codes = [str(value) for value in feature_codes]
    if counts.shape != (len(codes), len(codes)):
        raise ValueError("co-clustering matrix shape disagrees with features")
    positions = {feature: index for index, feature in enumerate(codes)}
    seen: set[str] = set()
    for members in modules:
        member_list = [str(value) for value in members]
        if any(feature not in positions for feature in member_list):
            raise ValueError("module contains an unknown feature")
        if seen.intersection(member_list):
            raise ValueError("modules must be disjoint")
        seen.update(member_list)
        index = np.asarray([positions[feature] for feature in member_list], dtype=int)
        counts[np.ix_(index, index)] += 1
    if seen != set(codes):
        raise ValueError("modules must cover every feature exactly once")
