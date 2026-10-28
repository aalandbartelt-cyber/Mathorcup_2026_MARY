# Source logical name: frozen_sensory_nnls_interface; SHA256: facf4d043c999208c3a9d3bad56b6e18400c222a384ce5fd7e31e923d7b768b8
from __future__ import annotations
import numpy as np
import pandas as pd


class FrozenFoldNNLS:
    """A four-learner raw-input predictor with already-shrunk frozen weights."""

    def __init__(self,models,weights,intercept,feature_names):
        self.models=list(models);self.weights=np.asarray(weights,float).copy()
        self.intercept=np.asarray(intercept,float).copy();self.feature_names=tuple(feature_names)
        if (len(self.models)!=4 or self.weights.ndim!=2 or self.weights.shape[1]!=4
                or self.intercept.shape!=(self.weights.shape[0],) or (self.weights<0).any()
                or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all()):
            raise ValueError('invalid frozen weights or free target intercepts')
        self.weights.setflags(write=False);self.intercept.setflags(write=False)

    def predict(self,values):
        if isinstance(values,pd.DataFrame) and tuple(values.columns.astype(str))!=self.feature_names:
            raise ValueError('raw feature order differs from frozen Q1')
        x=np.asarray(values,float)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or np.isinf(x).any():
            raise ValueError('predict expects the complete ordered raw feature matrix')
        stack=np.stack([np.asarray(model.predict(x),float) for model in self.models],axis=-1)
        if stack.shape[1:]!=self.weights.shape or not np.isfinite(stack).all():
            raise ValueError('base predictions disagree with frozen target/model dimensions')
        return np.einsum('ntm,tm->nt',stack,self.weights)+self.intercept


def raw_module_permutation(model,values,module_codes,*,strict_feature_codes,rng):
    """Move raw module values and masks together, holding other fields fixed.

    The other 13 fields remain conditioning context. This is a predictive
    permutation diagnostic, not a chemically coherent causal intervention.
    """
    names=list(model.feature_names);members=list(module_codes)
    if not members or len(set(members))!=len(members) or not set(members)<=set(strict_feature_codes):
        raise ValueError('module members must be distinct strict compound fields')
    if isinstance(values,pd.DataFrame) and tuple(values.columns.astype(str))!=tuple(names):
        raise ValueError('raw feature order differs from frozen Q1')
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=len(names) or not len(x):raise ValueError('complete raw input required')
    positions=[names.index(name) for name in members]
    order=rng.permutation(len(x));changed=x.copy();changed[:,positions]=x[order][:,positions]
    return model.predict(changed),order
