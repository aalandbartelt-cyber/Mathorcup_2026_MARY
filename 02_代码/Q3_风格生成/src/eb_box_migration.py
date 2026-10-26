"""Four-field two-sided Mahalanobis projection for strict Q3 migration."""

from __future__ import annotations

from itertools import product

import numpy as np


def mahalanobis_box_projection(anchor, covariance, indices, lower, upper, *, tolerance=1e-9):
    """Project onto direct-field intervals with deterministic KKT enumeration."""
    anchor = np.asarray(anchor, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    indices = np.asarray(indices, dtype=int)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if anchor.ndim != 1 or covariance.shape != (len(anchor), len(anchor)):
        raise ValueError("aligned anchor and covariance required")
    if indices.ndim != 1 or lower.shape != indices.shape or upper.shape != indices.shape or len(indices) > 4:
        raise ValueError("at most four aligned direct-field intervals required")
    if len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= len(anchor)).any():
        raise ValueError("unique in-range direct-field indices required")
    if not np.isfinite(anchor).all() or not np.isfinite(covariance).all() or np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any():
        raise ValueError("finite anchor/covariance and valid non-NaN interval bounds required")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("finite nonnegative numerical tolerance required")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0) or np.linalg.eigvalsh(covariance).min() <= 0.0:
        raise ValueError("symmetric positive-definite covariance required")
    if len(indices) == 0:
        return anchor.copy(), np.empty(0)
    if ((anchor[indices] >= lower - tolerance) & (anchor[indices] <= upper + tolerance)).all():
        return anchor.copy(), np.zeros(len(indices))

    sub = covariance[np.ix_(indices, indices)]
    for states in product((0, -1, 1), repeat=len(indices)):
        active = np.flatnonzero(np.asarray(states) != 0)
        if len(active) == 0:
            continue
        bounds = np.array([lower[j] if states[j] == -1 else upper[j] for j in active], dtype=float)
        if not np.isfinite(bounds).all():
            continue
        signed_dual = np.zeros(len(indices))
        signed_dual[active] = np.linalg.solve(sub[np.ix_(active, active)], anchor[indices[active]] - bounds)
        if any(states[j] == -1 and signed_dual[j] > tolerance for j in active):
            continue
        if any(states[j] == 1 and signed_dual[j] < -tolerance for j in active):
            continue
        projected = anchor - covariance[:, indices] @ signed_dual
        if ((projected[indices] >= lower - tolerance) & (projected[indices] <= upper + tolerance)).all() and np.allclose(projected[indices[active]], bounds, atol=tolerance, rtol=0.0):
            return projected, signed_dual
    raise ValueError("no numerically feasible two-sided KKT active set")


def project_stored_box(geometry, anchor, target_style, indices, lower_stored, upper_stored, *, tolerance=1e-9):
    """Apply the covariance metric in standardized chemical coordinates."""
    anchor = np.asarray(anchor, dtype=float)
    indices = np.asarray(indices, dtype=int)
    lower_stored = np.asarray(lower_stored, dtype=float)
    upper_stored = np.asarray(upper_stored, dtype=float)
    if anchor.shape != geometry.global_mean.shape:
        raise ValueError("chemical anchor must match fitted feature order")
    standardized = (anchor - geometry.global_mean) / geometry.global_scale
    lower = (lower_stored - geometry.global_mean[indices]) / geometry.global_scale[indices]
    upper = (upper_stored - geometry.global_mean[indices]) / geometry.global_scale[indices]
    projected, dual = mahalanobis_box_projection(
        standardized, geometry.covariance(target_style), indices, lower, upper, tolerance=tolerance
    )
    stored = projected * geometry.global_scale + geometry.global_mean
    # Snap only round-off sized deviations at mathematically active bounds.
    # This avoids reporting -1e-15 mg/L as a negative concentration while
    # preserving the covariance projection away from the exact boundary.
    for position, column in enumerate(indices):
        if np.isfinite(lower_stored[position]) and abs(stored[column] - lower_stored[position]) <= 1e-10:
            stored[column] = lower_stored[position]
        if np.isfinite(upper_stored[position]) and abs(stored[column] - upper_stored[position]) <= 1e-10:
            stored[column] = upper_stored[position]
    return stored, dual
