"""Registered path construction and finite-candidate selection rules."""
import numpy as np
from concentration_constraints import project_direct_constraints


def constrained_paths(source,anchors,lambdas,feature_indices,source_values,known,thresholds,scales):
    source=np.asarray(source,float);anchors=np.asarray(anchors,float);lambdas=np.asarray(lambdas,float)
    if (source.ndim!=1 or anchors.ndim!=2 or anchors.shape[1]!=len(source)
        or not np.isfinite(source).all() or not np.isfinite(anchors).all()
        or not np.isfinite(lambdas).all() or (lambdas<=0).any() or (lambdas>1).any()):
        raise ValueError('Finite anchors and strictly positive lambdas in (0,1] required')
    anchor_ids=np.repeat(np.arange(len(anchors)),len(lambdas));path_lambdas=np.tile(lambdas,len(anchors))
    values=(1-path_lambdas[:,None])*source+path_lambdas[:,None]*anchors[anchor_ids]
    for row in values:
        row[feature_indices],_=project_direct_constraints(row[feature_indices],source_values,known,thresholds,scales,'C1')
    return values,anchor_ids,path_lambdas


def select_candidates(gain,change,mean_delta,q90_delta,support_fraction,*,minimum_gain=.01,guard_margin=.01):
    gain,change,mean_delta,q90_delta,support_fraction=np.broadcast_arrays(
        *[np.asarray(x,float) for x in (gain,change,mean_delta,q90_delta,support_fraction)])
    if gain.ndim!=1 or not np.isfinite([*gain,*change,*mean_delta,*q90_delta,*support_fraction]).all():
        raise ValueError('Finite aligned one-dimensional primary diagnostics required')
    feasible=(gain>=minimum_gain)&(mean_delta<=-guard_margin)&(q90_delta<=-guard_margin)&(support_fraction>=.9)
    idx=np.flatnonzero(feasible)
    if not len(idx):return {k:None for k in ['minimum_change','maximum_target_gain','maximum_proxy_margin']},feasible
    # Stable input index breaks exact objective ties; the auditor is not an argument.
    rules={
        'minimum_change':(idx,-gain[idx],change[idx]),
        'maximum_target_gain':(idx,change[idx],-gain[idx]),
        'maximum_proxy_margin':(idx,change[idx],q90_delta[idx]),
    }
    return {name:int(idx[np.lexsort(keys)[0]]) for name,keys in rules.items()},feasible
