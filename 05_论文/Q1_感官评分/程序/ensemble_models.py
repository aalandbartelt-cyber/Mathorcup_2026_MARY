from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

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
