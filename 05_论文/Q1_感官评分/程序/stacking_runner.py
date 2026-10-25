from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from baseline_models import fit_predict_extra_trees, fit_predict_ridge_quantile, fit_predict_pls_quantile
from ensemble_models import fit_predict_hgb_multioutput
from q1_metrics import matrix_metrics
from nnls_stacking import fit_nnls_stacking
def make_inner_splits(n_samples,n_splits,random_state):
    return list(KFold(n_splits=n_splits,shuffle=True,random_state=random_state).split(np.arange(n_samples)))

def _candidates(config: dict, model: str) -> list[object]:
    if model == "extra_trees":
        return [
            {"max_features": float(max_features), "min_samples_leaf": int(leaf)}
            for max_features in config["extra_trees"]["max_features"]
            for leaf in config["extra_trees"]["min_samples_leaf"]
        ]
    if model == "hgb":
        return list(config["hgb_profiles"])
    if model == "ridge_quantile":
        return [float(value) for value in config["ridge_alphas"]]
    if model == "pls2_quantile":
        return [int(value) for value in config["pls_components"]]
    raise ValueError(f"unsupported base base model: {model}")

def _fit_base(
    model: str,
    parameter: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: dict,
    random_state: int,
) -> np.ndarray:
    if model == "extra_trees":
        values = dict(parameter)
        return fit_predict_extra_trees(
            x_train,
            y_train,
            x_validation,
            n_estimators=int(config["extra_trees"]["n_estimators"]),
            max_features=float(values["max_features"]),
            min_samples_leaf=int(values["min_samples_leaf"]),
            max_depth=config["extra_trees"]["max_depth"],
            random_state=random_state,
            n_jobs=1,
        )
    if model == "hgb":
        values = dict(parameter)
        return fit_predict_hgb_multioutput(
            x_train,
            y_train,
            x_validation,
            max_leaf_nodes=int(values["max_leaf_nodes"]),
            min_samples_leaf=int(values["min_samples_leaf"]),
            learning_rate=float(values["learning_rate"]),
            l2_regularization=float(values["l2_regularization"]),
            max_iter=int(values["max_iter"]),
            random_state=random_state,
        )
    if model == "ridge_quantile":
        return fit_predict_ridge_quantile(
            x_train,
            y_train,
            x_validation,
            alpha=float(parameter),
            random_state=random_state,
        )
    if model == "pls2_quantile":
        return fit_predict_pls_quantile(
            x_train,
            y_train,
            x_validation,
            n_components=int(parameter),
            random_state=random_state,
        )
    raise ValueError(f"unsupported base base model: {model}")

def _parameter_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["mae"],
        -metrics["r2"],
        -metrics["spearman"],
        -metrics["qwk_proxy"],
        -metrics["acc_pm1_proxy"],
    )

def fit_outer_nnls(x_train, y_train, x_validation, *, base_config, groups,
                   scopes, shrinkage, inner_splits, seed) -> dict:
    """Rebuild base-model OOF entirely inside outer training, then fit NNLS."""
    y = np.asarray(y_train, dtype=float)
    splits = make_inner_splits(len(y), int(inner_splits), int(seed))
    assignment = np.full(len(y), -1, dtype=int)
    for fold, (_, valid) in enumerate(splits):
        assignment[valid] = fold
    inner_predictions, outer_predictions, selections = [], [], {}
    for model_position, model in enumerate(base_config["base_models"]):
        candidate_rows = []
        for candidate_position, parameter in enumerate(_candidates(base_config, model)):
            prediction = np.full_like(y, np.nan)
            for fold, (train, valid) in enumerate(splits):
                prediction[valid] = _fit_base(
                    model, parameter, x_train[train], y[train], x_train[valid], base_config,
                    random_state=int(seed) + model_position * 100000 + candidate_position * 1000 + fold,
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError("stacking base inner OOF is incomplete")
            candidate_rows.append({"parameter": parameter, "prediction": prediction,
                                   "metrics": matrix_metrics(y, prediction)})
        selected = min(candidate_rows, key=lambda row: _parameter_key(row["metrics"]))
        inner_predictions.append(selected["prediction"])
        outer_predictions.append(_fit_base(
            model, selected["parameter"], x_train, y, x_validation, base_config,
            random_state=int(seed) + model_position * 100000,
        ))
        selections[model] = {
            "parameter": selected["parameter"],
            "candidate_metrics": [{"parameter": row["parameter"], "metrics": row["metrics"]}
                                  for row in candidate_rows],
        }
    inner_stack, outer_stack = np.stack(inner_predictions, axis=-1), np.stack(outer_predictions, axis=-1)
    predictions, parameters = {}, {}
    for scope in scopes:
        fitted = fit_nnls_stacking(inner_stack, y, groups=groups, scope=scope, shrinkage=shrinkage)
        predictions[str(scope)] = fitted.predict(outer_stack)
        parameters[str(scope)] = {"weights": fitted.weights.tolist(),
                                  "intercept": fitted.intercept.tolist(),
                                  "global_weights": fitted.global_weights.tolist()}
    return {"predictions": predictions, "parameters": parameters, "base_selection": selections,
            "inner_fold_assignment": assignment, "inner_prediction_stack": inner_stack,
            "outer_base_stack": outer_stack}
