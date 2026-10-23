"""Small, testable building blocks for the Q1 CPU baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=n_components, scale=False, max_iter=1000)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_validation))


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

