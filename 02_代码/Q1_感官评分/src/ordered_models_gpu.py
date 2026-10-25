"""CUDA XGBoost cumulative ordinal candidate for Q1."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from ordered_models import (
    OrdinalPrediction,
    _as_finite_target_matrix,
    _training_level_thresholds,
    decode_expected_level,
    expected_level_to_continuous,
    project_survival_probabilities,
    training_quantile_anchors,
)


def fit_predict_cumulative_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    max_depth: int,
    min_child_weight: float,
    learning_rate: float,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int,
) -> OrdinalPrediction:
    """Fit six CUDA XGBoost cumulative classifiers per target."""
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
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                random_state=(
                    random_state + target_position * thresholds.shape[1]
                    + threshold_position
                ),
                n_jobs=1,
                verbosity=0,
            )
            model.fit(train_features, binary)
            raw_survival[:, target_position, threshold_position] = (
                model.predict_proba(validation_features)[:, 1]
            )

    survival = project_survival_probabilities(raw_survival)
    expected_level = decode_expected_level(survival)
    continuous = expected_level_to_continuous(expected_level, anchors)
    if not np.isfinite(continuous).all():
        raise RuntimeError("CUDA cumulative XGBoost produced non-finite predictions")
    return OrdinalPrediction(
        continuous=continuous,
        expected_level=expected_level,
        survival_probability=survival,
        anchors=anchors,
    )
