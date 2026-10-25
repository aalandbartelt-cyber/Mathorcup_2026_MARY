"""Distribution diagnostics without training, tuning or significance claims."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, wasserstein_distance


def distribution_diagnostics(real, generated, training, scale, *, knn_k: int = 1,
                             duplicate_tolerance: float = 1e-6) -> dict:
    arrays = [np.asarray(a, float) for a in [real, generated, training]]
    if any(a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite matrices with at least two rows required")
    real, generated, training = arrays
    p = real.shape[1]
    if not p or generated.shape[1] != p or training.shape[1] != p or len(real) != len(generated):
        raise ValueError("matched real/generated sample counts and field dimensions required")
    scale = np.asarray(scale, float)
    if scale.shape != (p,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("positive finite training-derived scale required")
    if (not isinstance(knn_k, (int, np.integer)) or not 1 <= knn_k < len(real)
            or not np.isfinite(duplicate_tolerance) or duplicate_tolerance < 0):
        raise ValueError("invalid neighbor count or duplicate tolerance")
    r, g, train = real / scale, generated / scale, training / scale
    rms_factor = np.sqrt(p)
    rg, rr, gg = [a / rms_factor for a in [cdist(r, g), cdist(r, r), cdist(g, g)]]
    w1 = [float(wasserstein_distance(r[:, j], g[:, j])) for j in range(p)]
    nonconstant = (np.ptp(r, axis=0) > 0) & (np.ptp(g, axis=0) > 0)
    n_cols = int(nonconstant.sum())
    n_pairs = n_cols * (n_cols - 1) // 2
    correlation_error = np.nan
    if n_pairs:
        cr = np.corrcoef(rankdata(r[:, nonconstant], axis=0), rowvar=False)
        cg = np.corrcoef(rankdata(g[:, nonconstant], axis=0), rowvar=False)
        upper = np.triu_indices(n_cols, k=1)
        correlation_error = float(np.sqrt(np.mean((cr[upper] - cg[upper])**2)))
    rr_no_self = rr.copy()
    np.fill_diagonal(rr_no_self, np.inf)
    radii = np.partition(rr_no_self, knn_k - 1, axis=1)[:, knn_k - 1]
    nearest_train = cdist(g, train).min(axis=1) / rms_factor
    return {
        "n_real": len(r), "n_generated": len(g), "n_training": len(train),
        "n_features": p, "knn_k": int(knn_k),
        "mean_w1": float(np.mean(w1)), "max_w1": float(np.max(w1)), "w1_by_feature": w1,
        "energy_v": float(max(2 * rg.mean() - rr.mean() - gg.mean(), 0.)),
        "spearman_pair_rmse": correlation_error, "n_spearman_pairs": n_pairs,
        "n_constant_real": int((np.ptp(r, axis=0) == 0).sum()),
        "n_constant_generated": int((np.ptp(g, axis=0) == 0).sum()),
        "coverage": float(np.mean(rg.min(axis=1) <= radii)),
        "near_duplicate_fraction": float(np.mean(nearest_train <= duplicate_tolerance)),
        "median_nearest_training_rms_distance": float(np.median(nearest_train)),
        "duplicate_tolerance": float(duplicate_tolerance),
        "scope": "descriptive_distribution_diagnostics_no_hypothesis_test",
    }
