"""Read-only final-predictor reuse and conservative sensory-proxy diagnostics."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
from scipy.spatial.distance import cdist


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_cache_files(cache_dir, repeat, fold, signature):
    root=Path(cache_dir)
    model=root/f'R{repeat:02d}F{fold}.joblib'
    meta=root/f'R{repeat:02d}F{fold}.json'
    if not model.is_file() or not meta.is_file():
        raise FileNotFoundError(f'Frozen predictor missing; automatic refit is forbidden: {model}')
    record=json.loads(meta.read_text())
    if (record.get('signature')!=signature or record.get('model_sha256')!=sha(model)
        or record.get('repeat')!=repeat or record.get('fold')!=fold
        or any(record.get(key) is not False for key in ('nnls_refit','hyperparameter_search','q1_outputs_written'))):
        raise ValueError('Frozen predictor cache identity/hash/policy mismatch')
    return model,record


def load_read_only_nnls(final, cache_dir, repeat, fold, x_validation):
    path,record=verify_cache_files(cache_dir,repeat,fold,final.signature)
    model=joblib.load(path)
    _,weights,intercept=final.fold_parameters(repeat,fold)
    if (tuple(model.feature_names)!=final.feature_names or
        not np.array_equal(model.weights,weights) or not np.array_equal(model.intercept,intercept)):
        raise ValueError('Cached predictor does not contain frozen feature order/weights')
    error=float(np.max(np.abs(model.predict(x_validation)-final.fold_prediction(repeat,fold))))
    if not np.isfinite(error) or error>1e-10:
        raise ValueError('Cached NNLS no longer reproduces frozen validation predictions')
    return model,{'model_sha256':record['model_sha256'],'reconstruction_max_abs_error':error}


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


def proxy_screen(primary_deltas, audit_deltas, primary_reliable, audit_reliable, *, tolerance=1e-8):
    p=np.asarray(primary_deltas,float);a=np.asarray(audit_deltas,float)
    if p.shape!=(10,) or a.shape!=(10,) or not np.isfinite([*p,*a,tolerance]).all() or tolerance<0:
        raise ValueError('Ten finite paired-repeat changes and a numerical tolerance required')
    pm,am=float(p.mean()),float(a.mean());pq,aq=float(np.quantile(p,.9)),float(np.quantile(a,.9))
    if not primary_reliable or not audit_reliable:status='proxy_unreliable'
    elif (pm>tolerance)!=(am>tolerance):status='proxy_disagreement'
    elif pm>tolerance or am>tolerance:status='predicted_worsening'
    elif pq>tolerance or aq>tolerance:status='proxy_variation_uncertain'
    else:status='predicted_nonworsening'
    return {'proxy_status':status,'primary_mean_delta':pm,'audit_mean_delta':am,
            'primary_repeat_q90_delta':pq,'audit_repeat_q90_delta':aq,
            'physical_safety_certified':False}


def candidate_status(*, available, n_failed, n_unknown, target_gain, proxy_status, support_fraction):
    if not available:return 'rejected_no_structural_candidate'
    if n_failed>0:return 'rejected_direct_constraint'
    if not target_gain:return 'no_target_gain'
    if n_unknown>0:return 'needs_direct_verification'
    if support_fraction<.9:return 'outside_predictor_support'
    if proxy_status!='predicted_nonworsening':return proxy_status
    return 'eligible_statistical_candidate'
