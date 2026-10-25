"""Complementary-pairs stability utilities for Q2 E117.

These functions are model-agnostic: they construct style-stratified
complementary halves and summarize selection/null-control frequencies without
fitting a model or reading result files.  That makes leakage behavior
unit-testable before a formal E117 run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def stratified_complementary_pairs(
    styles: Sequence[object],
    n_pairs: int,
    random_state: int = 20261025,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return disjoint, exhaustive row-index halves within every style.

    Each pair uses a fresh within-style permutation.  For odd style counts,
    the extra row alternates sides between pairs so one half is not favored.
    Returned values are row positions, not beer identifiers.
    """

    labels = np.asarray(styles, dtype=object)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("styles must be a one-dimensional sequence with >=2 rows")
    if _contains_missing(labels):
        raise ValueError("styles must not contain missing labels")
    if int(n_pairs) != n_pairs or n_pairs < 1:
        raise ValueError("n_pairs must be a positive integer")

    # Preserve first-seen order; np.unique is not reliable for mixed objects.
    groups: dict[object, list[int]] = {}
    for row, label in enumerate(labels.tolist()):
        try:
            groups.setdefault(label, []).append(row)
        except TypeError as exc:
            raise ValueError("style labels must be hashable") from exc

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for pair_id in range(int(n_pairs)):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_state), int(pair_id)])
        )
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for rows in groups.values():
            shuffled = np.asarray(rows, dtype=int)[rng.permutation(len(rows))]
            midpoint = len(shuffled) // 2
            extra_left = int(len(shuffled) % 2 == 1 and pair_id % 2 == 0)
            left_parts.append(shuffled[: midpoint + extra_left])
            right_parts.append(shuffled[midpoint + extra_left :])
        left = np.concatenate(left_parts).astype(int, copy=False)
        right = np.concatenate(right_parts).astype(int, copy=False)
        if len(np.intersect1d(left, right)):
            raise RuntimeError("complementary halves overlap")
        if len(np.union1d(left, right)) != len(labels):
            raise RuntimeError("complementary halves do not cover all rows")
        result.append((left, right))
    return result


def _contains_missing(values: np.ndarray) -> bool:
    for value in values.tolist():
        if value is None:
            return True
        try:
            if bool(np.isnan(value)):
                return True
        except (TypeError, ValueError):
            pass
    return False


def selection_probability(
    selection_sets: Iterable[Iterable[int]], n_features: int
) -> np.ndarray:
    """Count how often each feature is selected across all supplied sets."""

    if int(n_features) != n_features or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    selections = list(selection_sets)
    if not selections:
        raise ValueError("at least one selection set is required")
    count = np.zeros(int(n_features), dtype=float)
    for selected in selections:
        index = np.asarray(sorted(set(selected)), dtype=int)
        if len(index) and (index.min() < 0 or index.max() >= n_features):
            raise ValueError("selection index is out of bounds")
        count[index] += 1.0
    return count / len(selections)


def complementary_pair_jaccard(
    left_sets: Sequence[Iterable[int]], right_sets: Sequence[Iterable[int]]
) -> np.ndarray:
    """Calculate pairwise Jaccard overlap of selections from both halves."""

    if len(left_sets) != len(right_sets) or not left_sets:
        raise ValueError("left and right selections must have equal nonzero length")
    values = []
    for left, right in zip(left_sets, right_sets):
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return np.asarray(values, dtype=float)


def mb_pfer_bound(q: float, pi_threshold: float, p: int) -> float:
    """Return a Meinshausen--Bühlmann-style PFER upper bound."""

    q = float(q)
    pi_threshold = float(pi_threshold)
    if q < 0 or int(p) != p or p <= 0:
        raise ValueError("q must be nonnegative and p must be positive")
    if not 0.5 < pi_threshold <= 1.0:
        raise ValueError("pi_threshold must lie in (0.5, 1]")
    return q * q / ((2.0 * pi_threshold - 1.0) * int(p))


def empirical_fdp(real_selected: int, null_selected: int) -> float:
    """Conservative null-control FDP proxy, clipped to one."""

    if int(real_selected) != real_selected or int(null_selected) != null_selected:
        raise ValueError("selected counts must be integers")
    if real_selected < 0 or null_selected < 0:
        raise ValueError("selected counts must be nonnegative")
    return min(1.0, int(null_selected) / max(int(real_selected), 1))


def confirmed_threshold(
    real_frequency: Sequence[float],
    null_frequency: Sequence[float],
    thresholds: Sequence[float],
    max_fdp: float = 0.10,
) -> dict[str, object]:
    """Choose the lowest registered threshold satisfying the null FDP gate.

    The result is descriptive only: it never replaces forced Top-3 entries.
    If no threshold qualifies, the confirmed set is explicitly empty.
    """

    real = np.asarray(real_frequency, dtype=float)
    null = np.asarray(null_frequency, dtype=float)
    if real.shape != null.shape or real.ndim != 1:
        raise ValueError("frequency vectors must have equal one-dimensional shape")
    if not np.isfinite(real).all() or not np.isfinite(null).all():
        raise ValueError("frequency vectors must be finite")
    if (real < 0).any() or (real > 1).any() or (null < 0).any() or (null > 1).any():
        raise ValueError("frequencies must lie in [0, 1]")
    if not 0 <= float(max_fdp) <= 1:
        raise ValueError("max_fdp must lie in [0, 1]")
    for threshold in sorted({float(value) for value in thresholds}):
        selected = real >= threshold
        null_selected = int((null >= threshold).sum())
        fdp = null_selected / max(int(selected.sum()), 1)
        if int(selected.sum()) > 0 and fdp <= float(max_fdp):
            return {
                "threshold": threshold,
                "selected_indices": np.flatnonzero(selected).astype(int),
                "real_selected": int(selected.sum()),
                "null_selected": null_selected,
                "empirical_fdp": float(fdp),
            }
    return {
        "threshold": None,
        "selected_indices": np.asarray([], dtype=int),
        "real_selected": 0,
        "null_selected": 0,
        "empirical_fdp": None,
    }
