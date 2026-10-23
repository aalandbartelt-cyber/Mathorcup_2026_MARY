"""Testable model primitives for Q2 ranking and graph-constrained selection."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPreprocessed:
    x_train: np.ndarray
    x_validation: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing_indicator_features: np.ndarray
    n_original_features: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def fit_fold_preprocessor(
    x_train: pd.DataFrame | np.ndarray,
    x_validation: pd.DataFrame | np.ndarray,
    *,
    winsor_quantiles: tuple[float, float] = (0.01, 0.99),
) -> FoldPreprocessed:
    """Fit fold-local winsorization, imputation, indicators, and scaling.

    The clipping bounds are estimated from the training rows only.  This guards
    the linear selectors against isolated transcription-scale values without
    deleting beers or leaking validation-fold distribution information.
    """
    train = np.asarray(x_train, dtype=float)
    validation = np.asarray(x_validation, dtype=float)
    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("X matrices must be two-dimensional")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("training and validation feature counts disagree")
    lower_q, upper_q = map(float, winsor_quantiles)
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    lower_bounds = np.nanquantile(train, lower_q, axis=0)
    upper_bounds = np.nanquantile(train, upper_q, axis=0)
    train = np.clip(train, lower_bounds, upper_bounds)
    validation = np.clip(validation, lower_bounds, upper_bounds)
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    train_imputed = imputer.fit_transform(train)
    validation_imputed = imputer.transform(validation)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    validation_scaled = scaler.transform(validation_imputed)
    indicator_features = (
        np.asarray(imputer.indicator_.features_, dtype=int)
        if imputer.indicator_ is not None
        else np.array([], dtype=int)
    )
    return FoldPreprocessed(
        x_train=np.asarray(train_scaled, dtype=float),
        x_validation=np.asarray(validation_scaled, dtype=float),
        medians=np.asarray(imputer.statistics_, dtype=float),
        means=np.asarray(scaler.mean_, dtype=float),
        scales=np.asarray(scaler.scale_, dtype=float),
        missing_indicator_features=indicator_features,
        n_original_features=train.shape[1],
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        upper_bounds=np.asarray(upper_bounds, dtype=float),
    )


def _fit_lasso_with_retry(
    augmented_x: np.ndarray,
    augmented_y: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> Lasso:
    """Fit Lasso and retry once at four times the iteration budget."""
    last_warning = False
    for iteration_budget in (int(max_iter), int(max_iter) * 4):
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=iteration_budget,
            tol=float(tol),
            selection="cyclic",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(augmented_x, augmented_y)
        last_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        iterations = np.atleast_1d(np.asarray(model.n_iter_, dtype=int))
        if not last_warning and np.all(iterations < iteration_budget):
            return model
    raise RuntimeError(
        "Lasso failed to converge after automatic retry; "
        f"max_iter={int(max_iter) * 4}"
    )


def fit_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    n_components: int,
) -> tuple[PLSRegression, np.ndarray]:
    """Fit PLS2 and return the model and validation predictions."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_validation = np.asarray(x_validation, dtype=float)
    upper = min(x_train.shape[0] - 1, x_train.shape[1])
    if not 1 <= int(n_components) <= upper:
        raise ValueError(f"n_components must be in 1..{upper}")
    model = PLSRegression(
        n_components=int(n_components), scale=False, max_iter=1000, tol=1e-7
    )
    model.fit(x_train, y_train)
    return model, np.asarray(model.predict(x_validation), dtype=float)


def target_vip_scores(model: PLSRegression) -> np.ndarray:
    """Calculate target-specific variable-importance-in-projection scores."""
    scores = np.asarray(model.x_scores_, dtype=float)
    weights = np.asarray(model.x_weights_, dtype=float)
    loadings = np.asarray(model.y_loadings_, dtype=float)
    if loadings.ndim == 1:
        loadings = loadings.reshape(1, -1)
    component_energy = np.sum(scores**2, axis=0)
    weight_norm = np.sum(weights**2, axis=0)
    weight_fraction = np.divide(
        weights**2,
        weight_norm[None, :],
        out=np.zeros_like(weights),
        where=weight_norm[None, :] > 0,
    )
    n_features = weights.shape[0]
    vip = np.zeros((n_features, loadings.shape[0]), dtype=float)
    for target in range(loadings.shape[0]):
        explained = component_energy * loadings[target] ** 2
        total = float(explained.sum())
        if total > 0:
            vip[:, target] = np.sqrt(
                n_features * (weight_fraction @ explained) / total
            )
    return vip


@dataclass
class GraphElasticNetModel:
    coef_: np.ndarray
    intercept_: float
    n_iter_: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.coef_ + self.intercept_


@dataclass
class GraphMultiTargetModel:
    coef_: np.ndarray
    intercept_: np.ndarray
    n_iter_: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.coef_.T + self.intercept_[None, :]


def build_quadratic_penalty_rows(
    laplacian: np.ndarray,
    *,
    n_samples: int,
    lambda_l2: float,
    lambda_graph: float,
) -> np.ndarray:
    """Precompute augmented rows whose cross-product is the quadratic penalty."""
    laplacian = np.asarray(laplacian, dtype=float)
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian must be square")
    if int(n_samples) < 1 or min(lambda_l2, lambda_graph) < 0:
        raise ValueError("sample count must be positive and penalties non-negative")
    if not np.allclose(laplacian, laplacian.T, atol=1e-10):
        raise ValueError("laplacian must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if eigenvalues.min() < -1e-8:
        raise ValueError("laplacian must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    n_features = laplacian.shape[0]
    blocks = []
    if lambda_graph > 0:
        root = np.sqrt(eigenvalues)[:, None] * eigenvectors.T
        blocks.append(np.sqrt(2.0 * n_samples * lambda_graph) * root)
    if lambda_l2 > 0:
        blocks.append(
            np.sqrt(2.0 * n_samples * lambda_l2) * np.eye(n_features)
        )
    if not blocks:
        return np.empty((0, n_features), dtype=float)
    return np.vstack(blocks)


def fit_lasso_with_penalty_rows(
    x: np.ndarray,
    y: np.ndarray,
    penalty_rows: np.ndarray,
    *,
    lambda_l1: float,
    max_iter: int = 50000,
    tol: float = 1e-6,
) -> GraphElasticNetModel:
    """Fit one response after a quadratic penalty has been precomputed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    penalty_rows = np.asarray(penalty_rows, dtype=float)
    if x.ndim != 2 or len(y) != len(x):
        raise ValueError("X and y shapes disagree")
    if penalty_rows.ndim != 2 or penalty_rows.shape[1] != x.shape[1]:
        raise ValueError("penalty row shape disagrees with X")
    if lambda_l1 <= 0:
        raise ValueError("lambda_l1 must be positive for the Lasso solver")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    augmented_x = np.vstack([x_centered, penalty_rows])
    augmented_y = np.concatenate([y_centered, np.zeros(len(penalty_rows))])
    n_samples = len(y)
    n_augmented = len(augmented_y)
    model = _fit_lasso_with_retry(
        augmented_x,
        augmented_y,
        alpha=float(lambda_l1) * n_samples / n_augmented,
        max_iter=max_iter,
        tol=tol,
    )
    coefficient = np.asarray(model.coef_, dtype=float)
    intercept = y_mean - float(x_mean @ coefficient)
    return GraphElasticNetModel(
        coef_=coefficient,
        intercept_=intercept,
        n_iter_=int(model.n_iter_),
    )


def fit_multi_target_with_penalty_rows(
    x: np.ndarray,
    y: np.ndarray,
    penalty_rows: np.ndarray,
    *,
    lambda_l1: float,
    max_iter: int = 50000,
    tol: float = 1e-6,
) -> GraphMultiTargetModel:
    """Fit independent Lasso responses together using shared augmented rows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    penalty_rows = np.asarray(penalty_rows, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(y) != len(x):
        raise ValueError("X and multi-target Y shapes disagree")
    if penalty_rows.ndim != 2 or penalty_rows.shape[1] != x.shape[1]:
        raise ValueError("penalty row shape disagrees with X")
    if lambda_l1 <= 0:
        raise ValueError("lambda_l1 must be positive for the Lasso solver")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    augmented_x = np.vstack([x_centered, penalty_rows])
    augmented_y = np.vstack(
        [y_centered, np.zeros((len(penalty_rows), y.shape[1]))]
    )
    n_samples = len(y)
    n_augmented = len(augmented_y)
    model = _fit_lasso_with_retry(
        augmented_x,
        augmented_y,
        alpha=float(lambda_l1) * n_samples / n_augmented,
        max_iter=max_iter,
        tol=tol,
    )
    coefficient = np.asarray(model.coef_, dtype=float)
    intercept = y_mean - x_mean @ coefficient.T
    n_iter = np.atleast_1d(np.asarray(model.n_iter_, dtype=int))
    return GraphMultiTargetModel(
        coef_=coefficient,
        intercept_=np.asarray(intercept, dtype=float),
        n_iter_=n_iter,
    )


def fit_graph_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    laplacian: np.ndarray,
    *,
    lambda_l1: float,
    lambda_l2: float,
    lambda_graph: float,
    max_iter: int = 50000,
    tol: float = 1e-6,
) -> GraphElasticNetModel:
    """Fit L1 + L2 + signed-graph quadratic regression by data augmentation.

    The optimized objective is
    ``||y-Xb||²/(2n) + lambda_l1*|b|_1 + lambda_l2*||b||²
    + lambda_graph*b'Lb``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    laplacian = np.asarray(laplacian, dtype=float)
    if x.ndim != 2 or len(y) != len(x):
        raise ValueError("X and y shapes disagree")
    if laplacian.shape != (x.shape[1], x.shape[1]):
        raise ValueError("laplacian shape disagrees with X")
    if min(lambda_l1, lambda_l2, lambda_graph) < 0:
        raise ValueError("penalties must be non-negative")
    if lambda_l1 == 0:
        raise ValueError("lambda_l1 must be positive for the Lasso solver")
    penalty_rows = build_quadratic_penalty_rows(
        laplacian,
        n_samples=len(x),
        lambda_l2=lambda_l2,
        lambda_graph=lambda_graph,
    )
    return fit_lasso_with_penalty_rows(
        x,
        y,
        penalty_rows,
        lambda_l1=lambda_l1,
        max_iter=max_iter,
        tol=tol,
    )


def spearman_target_scores(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Return feature-by-target pairwise-complete Spearman correlations."""
    if len(x) != len(y):
        raise ValueError("X and Y row counts disagree")
    if set(x.columns) & set(y.columns):
        raise ValueError("X and Y column names must be distinct")
    combined = pd.concat(
        [x.reset_index(drop=True), y.reset_index(drop=True)], axis=1
    )
    correlation = combined.corr(method="spearman")
    return correlation.loc[x.columns, y.columns]


def rank_score_matrix(
    scores: pd.DataFrame,
    method: str,
    eligible: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert a signed feature-by-target score matrix into deterministic ranks."""
    if eligible is None:
        eligible = pd.Series(True, index=scores.index)
    eligible = eligible.reindex(scores.index)
    if eligible.isna().any():
        raise ValueError("eligible mask does not cover every score row")
    rows = []
    for target in scores.columns:
        values = pd.DataFrame(
            {
                "feature_code": scores.index,
                "score": scores[target].to_numpy(float),
                "eligible": eligible.to_numpy(bool),
            }
        ).query("eligible").drop(columns="eligible")
        values["abs_score"] = values["score"].abs()
        values = values.sort_values(
            ["abs_score", "feature_code"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        values.insert(0, "rank", np.arange(1, len(values) + 1))
        values.insert(0, "target_code", target)
        values.insert(0, "method", method)
        values["direction"] = np.select(
            [values["score"] > 0, values["score"] < 0],
            ["positive", "negative"],
            default="zero",
        )
        rows.append(values)
    return pd.concat(rows, ignore_index=True)
