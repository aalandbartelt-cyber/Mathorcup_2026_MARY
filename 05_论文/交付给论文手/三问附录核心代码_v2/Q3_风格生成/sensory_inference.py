"""Pure inference with caller-supplied fitted learners and frozen parameters.

The caller must validate model provenance before constructing the predictor.
This module neither loads model caches nor fits learners or stacking weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


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


def verify_archived_prediction(prediction, expected):
    """Match the archive's storage precision, rather than relax an error gate."""
    prediction=np.asarray(prediction,float);expected=np.asarray(expected)
    if prediction.shape!=expected.shape or not np.isfinite(prediction).all() or not np.isfinite(expected).all():
        raise ValueError('Finite aligned archived predictions required')
    error=float(np.max(np.abs(prediction-expected)))
    if expected.dtype==np.dtype('float32'):
        if not np.array_equal(prediction.astype(np.float32),expected):
            raise ValueError('Reconstructed prediction differs at archived float32 precision')
    elif error>1e-9:
        raise ValueError('Reconstructed float64 prediction differs from archive')
    return error


def migration_model_input(values, source_positions, reference_x):
    values=np.asarray(values,float);raw=np.asarray(reference_x,float)
    positions=np.asarray(source_positions,int)
    if (values.ndim!=2 or raw.ndim!=2 or values.shape[1]!=raw.shape[1]
        or positions.shape!=(len(values),) or np.any(positions<0) or np.any(positions>=len(raw))
        or np.isinf(values).any() or np.isinf(raw).any()):
        raise ValueError('Aligned raw chemical spectra and source positions required')
    output=values.copy()
    output[np.isnan(raw[positions])]=np.nan
    return output


def support_audit(x_train, query, *, quantile=.95, outside_fraction=.05):
    train=np.asarray(x_train,float);query=np.asarray(query,float)
    if train.ndim!=2 or query.ndim!=2 or train.shape[1]!=query.shape[1] or len(train)<2:
        raise ValueError('Training and query matrices must align')
    if np.isinf(train).any() or np.isinf(query).any() or not 0<quantile<1 or not 0<=outside_fraction<=1:
        raise ValueError('Invalid support geometry')
    median=np.nanmedian(train,axis=0)
    if not np.isfinite(median).all():raise ValueError('A support field has no observed training values')
    low,high=np.nanquantile(train,[.25,.75],axis=0)
    scale=np.where(high-low>1e-12,high-low,np.nanstd(train,axis=0))
    scale=np.where(scale>1e-12,scale,1.)
    a=(np.where(np.isnan(train),median,train)-median)/scale
    b=(np.where(np.isnan(query),median,query)-median)/scale
    self_distance=cdist(a,a)/np.sqrt(a.shape[1]);np.fill_diagonal(self_distance,np.inf)
    cutoff=float(np.quantile(self_distance.min(axis=1),quantile))
    distance=cdist(b,a).min(axis=1)/np.sqrt(a.shape[1])
    observed=np.isfinite(query)
    out=observed&((query<np.nanmin(train,axis=0))|(query>np.nanmax(train,axis=0)))
    fractions=out.sum(axis=1)/np.maximum(observed.sum(axis=1),1)
    return {'nearest_distance':distance,'distance_cutoff':cutoff,
            'outside_range_fraction':fractions,
            'supported':(distance<=cutoff+1e-12)&(fractions<=outside_fraction)}


def predict_single_target(model,values,target):
    """Same frozen NNLS computation, retaining just the requested output column."""
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=len(model.feature_names) or np.isinf(x).any():
        raise ValueError('Full raw feature matrix required')
    out=[]
    # Multi-output tree shares splits, so use its original prediction path.
    out.append(model.models[0].predict(x)[:,target])
    hgb=model.models[1]
    out.append(hgb.models[target].predict(hgb.imputer.transform(x)))
    ridge=model.models[2];transformed=ridge[:-1].transform(x)
    out.append(transformed @ ridge[-1].coef_[target]+ridge[-1].intercept_[target])
    pls=model.models[3];transformed=pls[:-1].transform(x);estimator=pls[-1]
    # PLS coefficients multiply its centered X input (sklearn predict convention).
    out.append((transformed-estimator._x_mean) @ estimator.coef_[target]+estimator.intercept_[target])
    values=np.column_stack(out) @ model.weights[target]+model.intercept[target]
    if not np.isfinite(values).all():raise ValueError('Non-finite frozen single-target prediction')
    return values
