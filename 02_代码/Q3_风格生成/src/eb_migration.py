"""Empirical-Bayes style translation and deterministic Mahalanobis projection."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.covariance import LedoitWolf


def _positive_definite(matrix, *, floor=1e-8):
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    result = (eigenvectors * np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    return (result + result.T) / 2.0


def mahalanobis_upper_projection(anchor, covariance, indices, upper, *, tolerance=1e-9):
    """Project onto coordinate upper bounds in the covariance Mahalanobis metric.

    For at most four registered direct-defect bounds, enumerate all dual KKT
    active sets. This is deterministic and leaves unconstrained coordinates
    free to move according to covariance, unlike elementwise clipping.
    """
    anchor = np.asarray(anchor, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    indices = np.asarray(indices, dtype=int)
    upper = np.asarray(upper, dtype=float)
    if anchor.ndim != 1 or covariance.shape != (len(anchor), len(anchor)):
        raise ValueError("aligned anchor and covariance matrix required")
    if indices.ndim != 1 or upper.shape != indices.shape or len(indices) > 4:
        raise ValueError("at most four aligned direct-defect limits required")
    if len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= len(anchor)).any():
        raise ValueError("direct-defect indices must be unique and in range")
    if not np.isfinite(anchor).all() or not np.isfinite(covariance).all() or not np.isfinite(upper).all():
        raise ValueError("finite projection inputs required")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
        raise ValueError("symmetric covariance required")
    if np.linalg.eigvalsh(covariance).min() <= 0.0:
        raise ValueError("positive-definite covariance required")
    if len(indices) == 0:
        return anchor.copy(), np.empty(0)

    q = covariance[np.ix_(indices, indices)]
    delta = anchor[indices] - upper
    if (delta <= tolerance).all():
        return anchor.copy(), np.zeros(len(indices))
    possibilities = []
    for size in range(1, len(indices) + 1):
        for active_tuple in combinations(range(len(indices)), size):
            active = np.asarray(active_tuple, dtype=int)
            dual = np.zeros(len(indices))
            dual[active] = np.linalg.solve(q[np.ix_(active, active)], delta[active])
            residual = delta - q @ dual
            if (dual[active] >= -tolerance).all() and (residual <= tolerance).all() and (np.abs(residual[active]) <= tolerance).all():
                dual = np.maximum(dual, 0.0)
                projected = anchor - covariance[:, indices] @ dual
                score = float(dual @ delta - 0.5 * dual @ q @ dual)
                possibilities.append((score, projected, dual))
    if not possibilities:
        raise ValueError("no numerically feasible KKT active set")
    _, projected, dual = max(possibilities, key=lambda item: item[0])
    if (projected[indices] > upper + tolerance).any():
        raise ValueError("Mahalanobis projection violated a registered upper bound")
    return projected, dual


def migration_path(source, endpoint, lambdas, *, include_identity=True):
    source = np.asarray(source, dtype=float)
    endpoint = np.asarray(endpoint, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if source.ndim != 1 or endpoint.shape != source.shape or not np.isfinite(source).all() or not np.isfinite(endpoint).all():
        raise ValueError("finite aligned source and endpoint required")
    if lambdas.ndim != 1 or not len(lambdas) or not np.isfinite(lambdas).all() or (lambdas <= 0).any() or (lambdas > 1).any() or (np.diff(lambdas) <= 0).any():
        raise ValueError("strictly increasing nonzero lambdas in (0, 1] required")
    values = (1.0 - lambdas[:, None]) * source + lambdas[:, None] * endpoint
    return np.vstack([source[None, :], values]) if include_identity else values


class EBStyleGeometry:
    """All-data post-selection style geometry with fixed mean/covariance shrinkage."""

    def fit(self, x, styles, *, mean_kappa=15.0, covariance_kappa=30.0):
        values = np.asarray(x, dtype=float)
        labels = np.asarray(styles, dtype=str)
        if values.ndim != 2 or labels.shape != (len(values),) or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError("finite chemical matrix and aligned style labels required")
        if mean_kappa < 0 or covariance_kappa < 0:
            raise ValueError("nonnegative shrinkage required")
        self.global_mean = values.mean(axis=0)
        self.global_scale = np.std(values, axis=0)
        self.global_scale = np.where(self.global_scale > 1e-12, self.global_scale, 1.0)
        standardized = (values - self.global_mean) / self.global_scale
        global_covariance = _positive_definite(LedoitWolf().fit(standardized).covariance_)
        self.mean_kappa = float(mean_kappa)
        self.covariance_kappa = float(covariance_kappa)
        self.styles = {}
        for style in np.unique(labels):
            block = standardized[labels == style]
            if len(block) < 2:
                raise ValueError("each fitted style needs two chemical rows")
            mean_weight = len(block) / (len(block) + self.mean_kappa) if self.mean_kappa else 1.0
            covariance_weight = len(block) / (len(block) + self.covariance_kappa) if self.covariance_kappa else 1.0
            covariance = _positive_definite(covariance_weight * LedoitWolf().fit(block).covariance_ + (1.0 - covariance_weight) * global_covariance)
            self.styles[str(style)] = {
                "n": int(len(block)), "mean": mean_weight * block.mean(axis=0),
                "covariance": covariance, "mean_weight": float(mean_weight),
                "covariance_weight": float(covariance_weight),
            }
        return self

    def anchor(self, source, source_style, target_style):
        if source_style not in self.styles or target_style not in self.styles:
            raise ValueError("fitted source and target styles required")
        source = np.asarray(source, dtype=float)
        if source.shape != self.global_mean.shape or not np.isfinite(source).all():
            raise ValueError("finite source spectrum aligned to fitted features required")
        shift = self.styles[target_style]["mean"] - self.styles[source_style]["mean"]
        return source + shift * self.global_scale

    def covariance(self, style):
        if style not in self.styles:
            raise ValueError("unseen style")
        return self.styles[style]["covariance"].copy()

    def project(self, anchor, target_style, indices, upper_stored):
        anchor = np.asarray(anchor, dtype=float)
        indices = np.asarray(indices, dtype=int)
        upper_stored = np.asarray(upper_stored, dtype=float)
        standardized = (anchor - self.global_mean) / self.global_scale
        upper = (upper_stored - self.global_mean[indices]) / self.global_scale[indices]
        projected, dual = mahalanobis_upper_projection(standardized, self.covariance(target_style), indices, upper)
        return projected * self.global_scale + self.global_mean, dual
