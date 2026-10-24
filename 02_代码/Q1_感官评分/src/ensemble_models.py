"""Continuous HGB and constrained ensemble helpers for Q1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from q1_metrics import matrix_metrics


@dataclass(frozen=True)
class BlendSelection:
    """Selected convex weights and their inner-OOF evidence."""

    weights: np.ndarray
    metrics: dict[str, float]
    feasible: bool


def simplex_weight_grid(n_models: int, step: float) -> np.ndarray:
    """Enumerate deterministic nonnegative weights summing to one."""
    units = int(round(1.0 / step))
    if n_models < 1 or step <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one and n_models must be positive")
    rows = [
        row
        for row in product(range(units + 1), repeat=n_models)
        if sum(row) == units
    ]
    return np.asarray(rows, dtype=float) / units


def apply_convex_blend(
    prediction_stack: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Blend a samples×targets×models prediction tensor."""
    stack = np.asarray(prediction_stack, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if stack.ndim != 3 or stack.shape[-1] != len(weight):
        raise ValueError("prediction stack and weights disagree")
    if not np.isfinite(stack).all() or not np.isfinite(weight).all():
        raise ValueError("prediction stack and weights must be finite")
    if (weight < 0).any() or not np.isclose(weight.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return np.tensordot(stack, weight, axes=([-1], [0]))


def select_feasible_convex_blend(
    y_true: np.ndarray,
    prediction_stack: np.ndarray,
    candidates: np.ndarray,
    r2_floor: float,
    mae_ceiling: float,
) -> BlendSelection:
    """Select the best ordered score among continuous-feasible blends."""
    scored: list[dict] = []
    for order, weight in enumerate(np.asarray(candidates, dtype=float)):
        prediction = apply_convex_blend(prediction_stack, weight)
        metrics = matrix_metrics(y_true, prediction)
        feasible = metrics["r2"] >= r2_floor and metrics["mae"] <= mae_ceiling
        violation = max(0.0, r2_floor - metrics["r2"]) + max(
            0.0, metrics["mae"] - mae_ceiling
        )
        rank = (
            metrics["spearman"],
            metrics["qwk_proxy"],
            metrics["acc_pm1_proxy"],
            metrics["r2"],
            -metrics["mae"],
        )
        scored.append(
            {
                "order": order,
                "weights": weight,
                "metrics": metrics,
                "feasible": feasible,
                "violation": violation,
                "rank": rank,
            }
        )
    if not scored:
        raise ValueError("candidates must not be empty")
    feasible_rows = [row for row in scored if row["feasible"]]
    if feasible_rows:
        chosen = max(
            feasible_rows,
            key=lambda row: (*row["rank"], -row["order"]),
        )
    else:
        chosen = min(
            scored,
            key=lambda row: (
                row["violation"],
                *(-value for value in row["rank"]),
                row["order"],
            ),
        )
    return BlendSelection(
        weights=np.asarray(chosen["weights"], dtype=float),
        metrics=dict(chosen["metrics"]),
        feasible=bool(chosen["feasible"]),
    )


def shrink_target_weights(
    target_weights: np.ndarray,
    global_weights: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Shrink target-specific weights toward one global vector."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    target = np.asarray(target_weights, dtype=float)
    global_weight = np.asarray(global_weights, dtype=float)
    if target.ndim != 2 or global_weight.shape != (target.shape[1],):
        raise ValueError("target and global weight shapes disagree")
    result = (1.0 - shrinkage) * target + shrinkage * global_weight
    if (result < 0).any() or not np.allclose(result.sum(axis=1), 1.0):
        raise ValueError("shrunk weights are not convex")
    return result


def fit_predict_hgb_multioutput(
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
) -> np.ndarray:
    """Fit one fold-local HistGradientBoostingRegressor per target."""
    targets = np.asarray(y_train, dtype=float)
    if targets.ndim != 2 or not np.isfinite(targets).all():
        raise ValueError("y_train must be a finite two-dimensional matrix")
    if len(x_train) != len(targets):
        raise ValueError("x_train and y_train have different lengths")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_features = imputer.fit_transform(np.asarray(x_train, dtype=float))
    validation_features = imputer.transform(np.asarray(x_validation, dtype=float))
    columns = []
    for target_position in range(targets.shape[1]):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=random_state + target_position,
        )
        model.fit(train_features, targets[:, target_position])
        columns.append(model.predict(validation_features))
    prediction = np.column_stack(columns)
    if not np.isfinite(prediction).all():
        raise RuntimeError("continuous HGB produced non-finite predictions")
    return prediction
