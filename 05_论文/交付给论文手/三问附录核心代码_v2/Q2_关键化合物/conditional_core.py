# Source logical name: conditional_residual_model; SHA256: 17d4745dc0fdf0c58b750b5a1c3044f287394001a48e9799515cadca97293693
# Source logical name: final_nnls_hierarchical_attribution; SHA256: 9bab3fb361c66e1f721a1eb11b89bd975a44bd8ca0809c5efd86f1cb2879f8f1
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
try:
    from .frozen_nnls_core import raw_module_permutation
except ImportError:
    from frozen_nnls_core import raw_module_permutation


@dataclass
class ConditionalFeatureModel:
    pipeline: Pipeline
    n_conditioners: int

    def predict(self, conditioners, style) -> np.ndarray:
        numeric = np.asarray(conditioners, dtype=float)
        styles = np.asarray(style, dtype=str)
        if numeric.ndim != 2 or numeric.shape[1] != self.n_conditioners:
            raise ValueError("conditioner shape disagrees with fitted model")
        if styles.shape != (len(numeric),):
            raise ValueError("style vector must align with conditioners")
        frame = pd.DataFrame(
            numeric,
            columns=[f"x_{position}" for position in range(self.n_conditioners)],
        )
        frame["style"] = styles
        return np.asarray(self.pipeline.predict(frame), dtype=float)


def fit_conditional_feature_model(
    conditioners, feature, style
) -> ConditionalFeatureModel:
    """Fit compact HGB E[X_j | module peers, style] on training rows only."""
    numeric = np.asarray(conditioners, dtype=float)
    target = np.asarray(feature, dtype=float)
    styles = np.asarray(style, dtype=str)
    if numeric.ndim != 2 or numeric.shape[0] < 20 or numeric.shape[1] < 1:
        raise ValueError("conditioners must contain at least 20 rows and one column")
    if target.shape != (len(numeric),) or styles.shape != (len(numeric),):
        raise ValueError("feature and style must align with conditioners")
    if not np.isfinite(target).all():
        raise ValueError("conditional feature target must be finite")
    frame = pd.DataFrame(
        numeric,
        columns=[f"x_{position}" for position in range(numeric.shape[1])],
    )
    frame["style"] = styles
    numeric_columns = list(frame.columns[:-1])
    transformer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "style",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["style"],
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("transform", transformer),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_leaf_nodes=7,
                    min_samples_leaf=20,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    max_iter=100,
                    random_state=20261025,
                ),
            ),
        ]
    )
    pipeline.fit(frame, target)
    return ConditionalFeatureModel(pipeline, numeric.shape[1])


def residual_permutation(
    conditional_mean, residual, rng: np.random.Generator
) -> np.ndarray:
    """Keep the conditional mean fixed and permute only residual variation."""
    mean = np.asarray(conditional_mean, dtype=float)
    error = np.asarray(residual, dtype=float)
    if mean.ndim != 1 or error.shape != mean.shape or not len(mean):
        raise ValueError("conditional mean and residual must be aligned vectors")
    if not np.isfinite(mean).all() or not np.isfinite(error).all():
        raise ValueError("conditional mean and residual must be finite")
    return mean + error[rng.permutation(len(error))]


def _fold_field_rows(model,x_train,x_valid,y_valid,style_train,style_valid,modules,feature_codes,target_codes,repeat,fold,seed,repeats=10):
    rows=[];feature_position={v:i for i,v in enumerate(model.feature_names)}
    baseline=model.predict(x_valid);base_mae=np.mean(np.abs(y_valid-baseline),axis=0)
    strict_position={v:i for i,v in enumerate(feature_codes)}
    for module_id,members in enumerate(modules,1):
        selected=[feature_position[v] for v in members]
        # Module-level permutation is raw-input and keeps missing values attached.
        for permutation_repeat in range(repeats):
            changed_prediction,order=raw_module_permutation(model,x_valid,members,strict_feature_codes=feature_codes,rng=np.random.default_rng(seed+module_id*1000+permutation_repeat))
            delta=np.mean(np.abs(y_valid-changed_prediction),axis=0)-base_mae
            for target_position,target in enumerate(target_codes):rows.append({'repeat':repeat,'fold':fold,'module_id':f'M{module_id:03d}','module_size':len(members),'module_members':'|'.join(members),'target_code':target,'permutation_repeat':permutation_repeat,'delta_mae':delta[target_position],'module_source':'fold_local_all_modules_no_validation_gate'})
        for feature in members:
            pos=feature_position[feature];peers=[feature_position[v] for v in members if v!=feature]
            train_feature=x_train.iloc[:,pos].to_numpy(float);observed=np.isfinite(train_feature)
            valid_feature=x_valid.iloc[:,pos].to_numpy(float);valid_observed=np.isfinite(valid_feature)
            if len(peers) and observed.sum()>=20 and valid_observed.any():
                cond_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                train_cond=cond_imputer.fit_transform(x_train.iloc[:,peers].to_numpy(float))
                valid_cond=cond_imputer.transform(x_valid.iloc[:,peers].to_numpy(float))
                cond=fit_conditional_feature_model(train_cond[observed],train_feature[observed],style_train[observed])
                valid_pos=np.flatnonzero(valid_observed);conditional=cond.predict(valid_cond[valid_pos],style_valid[valid_pos])
                residual=valid_feature[valid_pos]-conditional
                for permutation_repeat in range(repeats):
                    changed=x_valid.copy();changed_values=valid_feature.copy();changed_values[valid_pos]=residual_permutation(conditional,residual,np.random.default_rng(seed+pos*1000+permutation_repeat));changed.iloc[:,pos]=changed_values
                    delta=np.mean(np.abs(y_valid-model.predict(changed)),axis=0)-base_mae
                    for target_position,target in enumerate(target_codes):rows.append({'repeat':repeat,'fold':fold,'module_id':f'M{module_id:03d}','module_size':len(members),'module_members':'|'.join(members),'feature_code':feature,'target_code':target,'permutation_repeat':permutation_repeat,'delta_mae':delta[target_position],'module_source':'fold_local_all_modules_no_validation_gate','conditional_scope':'observed_only_missing_mask_preserved'})
    return pd.DataFrame(rows)
