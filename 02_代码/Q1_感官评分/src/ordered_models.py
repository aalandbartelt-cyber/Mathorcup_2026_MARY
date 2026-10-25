"""Leakage-safe cumulative ordinal models for Q1 sensory targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class OrdinalPrediction:
    """Decoded ordinal predictions and their fold-local calibration state."""

    continuous: np.ndarray
    expected_level: np.ndarray
    survival_probability: np.ndarray
    anchors: np.ndarray


def _as_finite_target_matrix(y: np.ndarray) -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim != 2:
        raise ValueError("target matrix must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("target matrix must be finite")
    return array


def training_quantile_anchors(
    y_train: np.ndarray, n_levels: int = 7
) -> np.ndarray:
    """Return seven continuous anchors estimated from the training fold only."""
    targets = _as_finite_target_matrix(y_train)
    if n_levels < 2:
        raise ValueError("n_levels must be at least two")
    return np.quantile(
        targets, np.linspace(0.0, 1.0, n_levels), axis=0
    ).T


def _training_level_thresholds(
    y_train: np.ndarray, n_levels: int = 7
) -> np.ndarray:
    targets = _as_finite_target_matrix(y_train)
    return np.quantile(
        targets, np.linspace(0.0, 1.0, n_levels + 1)[1:-1], axis=0
    ).T


def project_survival_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip and project P(level > k) to a non-increasing sequence."""
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 3:
        raise ValueError("survival probabilities must be three-dimensional")
    clipped = np.clip(array, 0.0, 1.0)
    return np.minimum.accumulate(clipped, axis=-1)


def decode_expected_level(survival_probability: np.ndarray) -> np.ndarray:
    """Decode a 0--6 expected level from six cumulative probabilities."""
    projected = project_survival_probabilities(survival_probability)
    return projected.sum(axis=-1)


def expected_level_to_continuous(
    expected_level: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Map expected levels monotonically to the training-fold target scale."""
    levels = np.asarray(expected_level, dtype=float)
    anchor_array = np.asarray(anchors, dtype=float)
    if levels.ndim != 2 or anchor_array.ndim != 2:
        raise ValueError("levels and anchors must be two-dimensional")
    if levels.shape[1] != anchor_array.shape[0]:
        raise ValueError("target counts disagree between levels and anchors")
    grade_axis = np.arange(anchor_array.shape[1], dtype=float)
    decoded = np.empty_like(levels, dtype=float)
    for target_position in range(levels.shape[1]):
        decoded[:, target_position] = np.interp(
            np.clip(levels[:, target_position], grade_axis[0], grade_axis[-1]),
            grade_axis,
            anchor_array[target_position],
        )
    return decoded


def fit_predict_cumulative_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2_regularization: float,
    max_iter: int,
    random_state: int,
) -> OrdinalPrediction:
    """Fit six cumulative HGB classifiers per target and decode predictions."""
    targets = _as_finite_target_matrix(y_train)
    if len(x_train) != len(targets):
        raise ValueError("x_train and y_train have different lengths")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_features = imputer.fit_transform(np.asarray(x_train, dtype=float))
    validation_features = imputer.transform(np.asarray(x_validation, dtype=float))
    anchors = training_quantile_anchors(targets)
    thresholds = _training_level_thresholds(targets)
    raw_survival = np.empty(
        (len(validation_features), targets.shape[1], thresholds.shape[1]),
        dtype=float,
    )

    for target_position in range(targets.shape[1]):
        for threshold_position, threshold in enumerate(thresholds[target_position]):
            binary = (targets[:, target_position] > threshold).astype(np.int8)
            probability = float(binary.mean())
            if np.unique(binary).size == 1:
                raw_survival[:, target_position, threshold_position] = probability
                continue
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=(
                    random_state + target_position * thresholds.shape[1]
                    + threshold_position
                ),
            )
            model.fit(train_features, binary)
            raw_survival[:, target_position, threshold_position] = (
                model.predict_proba(validation_features)[:, 1]
            )

    survival = project_survival_probabilities(raw_survival)
    expected_level = decode_expected_level(survival)
    continuous = expected_level_to_continuous(expected_level, anchors)
    if not np.isfinite(continuous).all():
        raise RuntimeError("cumulative HGB produced non-finite predictions")
    return OrdinalPrediction(
        continuous=continuous,
        expected_level=expected_level,
        survival_probability=survival,
        anchors=anchors,
    )
