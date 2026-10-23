"""Small, testable building blocks for the Q1 CPU baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import KFold


def fit_predict_global_mean(y_train: np.ndarray, n_validation: int) -> np.ndarray:
    """Predict each target with its training-fold mean."""
    mean = np.asarray(y_train, dtype=float).mean(axis=0, keepdims=True)
    return np.repeat(mean, repeats=n_validation, axis=0)


def fit_predict_shrunk_style(
    y_train: np.ndarray,
    train_style: pd.Series,
    validation_style: pd.Series,
    shrinkage: float,
) -> np.ndarray:
    """Predict style means shrunk toward the training-fold global mean."""
    y_train = np.asarray(y_train, dtype=float)
    train_style = pd.Series(train_style).reset_index(drop=True)
    validation_style = pd.Series(validation_style).reset_index(drop=True)
    if len(y_train) != len(train_style):
        raise ValueError("y_train and train_style have different lengths")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")

    global_mean = y_train.mean(axis=0)
    predictions = []
    for style in validation_style:
        mask = train_style.eq(style).to_numpy()
        if not mask.any():
            predictions.append(global_mean)
            continue
        n_style = int(mask.sum())
        style_mean = y_train[mask].mean(axis=0)
        prediction = (n_style * style_mean + shrinkage * global_mean) / (
            n_style + shrinkage
        )
        predictions.append(prediction)
    return np.asarray(predictions)


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a fold-local impute/scale/Ridge pipeline and predict all targets."""
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


def fit_predict_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit a fold-local impute/scale/PLS2 pipeline and predict all targets."""
    max_components = min(len(x_train) - 1, np.asarray(x_train).shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components={n_components} outside valid range 1..{max_components}"
        )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", PLSRegression(n_components=n_components, scale=True, max_iter=1000)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


def _quantile_preprocessor(n_train: int, random_state: int) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "quantile",
                            QuantileTransformer(
                                n_quantiles=min(100, n_train),
                                output_distribution="normal",
                                subsample=None,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "missing_indicator",
                MissingIndicator(features="missing-only", error_on_new=False),
            ),
        ]
    )


def fit_predict_ridge_quantile(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    alpha: float,
    random_state: int,
) -> np.ndarray:
    """Ridge after a training-fold rank-Gaussian transform."""
    pipeline = Pipeline(
        [
            ("preprocessor", _quantile_preprocessor(len(x_train), random_state)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


def fit_predict_pls_quantile(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """PLS2 after a training-fold rank-Gaussian transform."""
    max_components = min(len(x_train) - 1, np.asarray(x_train).shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components={n_components} outside valid range 1..{max_components}"
        )
    pipeline = Pipeline(
        [
            ("preprocessor", _quantile_preprocessor(len(x_train), random_state)),
            ("model", PLSRegression(n_components=n_components, scale=True, max_iter=1000)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


def fit_predict_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    max_depth: int | None,
    random_state: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Fit a fold-local median-imputed multi-output ExtraTrees model."""
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


def empirical_cdf_grade(
    reference_targets: np.ndarray, predicted_targets: np.ndarray
) -> np.ndarray:
    """Map predictions to [0, 6] using training-only empirical CDFs."""
    reference_targets = np.asarray(reference_targets, dtype=float)
    predicted_targets = np.asarray(predicted_targets, dtype=float)
    if reference_targets.ndim != 2 or predicted_targets.ndim != 2:
        raise ValueError("reference_targets and predicted_targets must be 2-D")
    if reference_targets.shape[1] != predicted_targets.shape[1]:
        raise ValueError("reference and prediction target counts disagree")
    grades = np.empty_like(predicted_targets, dtype=float)
    for target_position in range(reference_targets.shape[1]):
        ordered = np.sort(reference_targets[:, target_position])
        ranks = np.searchsorted(
            ordered, predicted_targets[:, target_position], side="right"
        )
        grades[:, target_position] = 6.0 * ranks / len(ordered)
    return grades


def select_parameter_by_mae(
    targets: np.ndarray,
    candidates: Sequence[Any],
    predict_fold: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    n_splits: int,
    random_state: int,
) -> tuple[Any, dict[Any, float]]:
    """Select one candidate using mean inner-fold multi-output MAE."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    targets = np.asarray(targets, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: dict[Any, float] = {}
    for candidate in candidates:
        fold_scores = []
        for train_index, validation_index in splitter.split(np.arange(len(targets))):
            prediction = np.asarray(
                predict_fold(train_index, validation_index, candidate), dtype=float
            )
            fold_scores.append(
                float(np.mean(np.abs(targets[validation_index] - prediction)))
            )
        scores[candidate] = float(np.mean(fold_scores))

    selected = min(candidates, key=lambda candidate: scores[candidate])
    return selected, scores


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return unclipped regression metrics for one target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")

    pred_sd = float(np.std(y_pred, ddof=0))
    if pred_sd == 0:
        spearman = np.nan
        calibration_intercept = float(np.mean(y_true))
        calibration_slope = 0.0
    else:
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        design = np.column_stack([np.ones(len(y_pred)), y_pred])
        calibration_intercept, calibration_slope = np.linalg.lstsq(
            design, y_true, rcond=None
        )[0]

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "spearman": spearman,
        "calibration_intercept": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
    }

