"""Nonnegative OOF model weights with free target intercepts and fixed shrinkage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class NNLSStackingFit:
    weights: np.ndarray
    intercept: np.ndarray
    global_weights: np.ndarray
    scope: str
    shrinkage: float

    def predict(self, stack) -> np.ndarray:
        values = np.asarray(stack, dtype=float)
        if values.ndim != 3 or values.shape[1:] != self.weights.shape:
            raise ValueError("prediction tensor must have the fitted targets/models shape")
        if not np.isfinite(values).all():
            raise ValueError("prediction tensor must be finite")
        return np.einsum("ntm,tm->nt", values, self.weights) + self.intercept


def _solve(centered_stack, centered_y, scales) -> np.ndarray:
    design = (centered_stack / scales[None, :, None]).reshape(-1, centered_stack.shape[2])
    response = (centered_y / scales[None, :]).reshape(-1)
    if not np.any(design):
        return np.zeros(design.shape[1], dtype=float)
    weights, _ = nnls(design, response, maxiter=max(100, design.shape[1] * 30))
    return weights


def fit_nnls_stacking(stack, y, *, groups=None, scope="shared", shrinkage=.5) -> NNLSStackingFit:
    """Fit on outer-training inner-OOF predictions and their training labels only.

    Weights are nonnegative but need not sum to one. Target centering removes
    intercepts before NNLS, so an intercept is free to be negative afterward.
    Standardizing each training target gives shared/group fits equal target units.
    """
    predictions, truth = np.asarray(stack, dtype=float), np.asarray(y, dtype=float)
    if (predictions.ndim != 3 or min(predictions.shape) < 1
            or predictions.shape[0] < 3 or truth.shape != predictions.shape[:2]
            or not np.isfinite(predictions).all() or not np.isfinite(truth).all()):
        raise ValueError("finite samples-by-targets-by-models predictions must align with y")
    if scope not in {"shared", "group", "target"}:
        raise ValueError("scope must be shared, group or target")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    if scope == "group" and (groups is None or len(groups) != truth.shape[1]):
        raise ValueError("group labels must cover all registered targets")
    means = predictions.mean(axis=0)
    target_means = truth.mean(axis=0)
    centered, y_centered = predictions - means, truth - target_means
    scales = np.std(truth, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.)
    global_weights = _solve(centered, y_centered, scales)
    weights = np.tile(global_weights, (truth.shape[1], 1))
    if scope != "shared":
        labels = np.arange(truth.shape[1]) if scope == "target" else np.asarray(groups, dtype=str)
        for label in dict.fromkeys(labels.tolist()):
            positions = np.flatnonzero(labels == label)
            local = _solve(centered[:, positions], y_centered[:, positions], scales[positions])
            weights[positions] = (1. - shrinkage) * local + shrinkage * global_weights
    intercept = target_means - np.einsum("tm,tm->t", means, weights)
    return NNLSStackingFit(weights.copy(), intercept.copy(), global_weights.copy(),
                           str(scope), 1. if scope == "shared" else float(shrinkage))
