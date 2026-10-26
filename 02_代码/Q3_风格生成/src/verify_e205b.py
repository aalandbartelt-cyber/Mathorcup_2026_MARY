"""Recompute search choices and spectra from frozen inputs, without reselection."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from run_e205b_sensory_search import inputs,RESULT,CONFIG
from run_e204_constrained_migration import _direct_vectors
from constrained_migration import empirical_quantile_anchor
from sensory_search import select_candidates,constrained_paths
from sensory_audit import sha,candidate_status


def run():
    config,data,hashes,signature=inputs();final=data['final']
    lock_path=RESULT/'q3_candidate_lock_E205B.json';lock=json.loads(lock_path.read_text())
    manifest_path=RESULT/'q3_run_manifest_E205B.json';manifest=json.loads(manifest_path.read_text())
    if (lock['signature']!=signature or manifest['candidate_lock_sha256']!=sha(lock_path)
        or manifest['status']!='complete_once_only_audit' or manifest['no_auditor_based_reselection'] is not True):
        raise ValueError('Candidate lock or once-only audit identity changed')
    for collection in [lock,manifest]:
        for name,digest in collection['output_sha256'].items():
            if sha(RESULT/name)!=digest:raise ValueError(f'Frozen output changed: {name}')
    for name,digest in lock['checkpoint_metadata_sha256'].items():
        p=RESULT/'q3_search_checkpoints_E205B'/name
        if sha(p)!=digest:raise ValueError('Search checkpoint metadata changed')
        for name,digest in json.loads(p.read_text())['output_sha256'].items():
            if sha(p.parent/name)!=digest:raise ValueError('Search checkpoint output changed')
    features=list(final.feature_names);x=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(data['x'])
    style=final.arrays['style'].astype(str);beers=final.arrays['beer'];sources=np.flatnonzero(style=='Pils/Lager')
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv');defects=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    fields=pd.read_csv(RESULT/'q3_field_contract_E200.csv');pool=pd.read_csv(RESULT/'q3_target_pool_E204.csv.gz')
    diagnostics=pd.read_csv(RESULT/'q3_search_diagnostics_E205B.csv.gz')
    selections=pd.read_csv(RESULT/'q3_search_selection_E205B.csv')
    spectra=pd.read_csv(RESULT/'q3_locked_spectra_E205B.csv.gz').set_index('selection_id')
    outcome=pd.read_csv(RESULT/'q3_once_audit_E205B.csv')
    if (len(diagnostics)!=59367 or len(selections)!=99 or len(spectra)!=87 or len(outcome)!=99
        or selections.selection_id.duplicated().any() or outcome.selection_id.duplicated().any()):
        raise ValueError('Registered search/selection counts changed')
    verified=0;max_spectrum_error=0.
    for (beer,target),group in diagnostics.groupby(['beer','target_style'],sort=False):
        group=group.reset_index(drop=True)
        selected,feasible=select_candidates(group.relative_target_gain,group.source_change_rms,
            group.primary_mean_delta,group.primary_repeat_q90_delta,group.support_repeat_fraction)
        if int(feasible.sum())!=int(group.primary_feasible.sum()):raise ValueError('Primary feasibility changed')
        rows=selections[selections.beer.eq(beer)&selections.target_style.eq(target)]
        if len(rows)!=3:raise ValueError('Each source-target pair must retain three predeclared rules')
        pos=int(np.flatnonzero(beers==beer)[0]);source=x[pos]
        direct,df,names,scales,sv,known,thresholds=_direct_vectors(int(beer),source_table,defects,fields)
        indices=[features.index(c) for c in df]
        for rule,index in selected.items():
            row=rows[rows.selection_rule==rule].iloc[0]
            if bool(row.candidate_available)!=(index is not None):raise ValueError('A failed search became an available candidate')
            if index is None:continue
            expected=group.iloc[index]
            for col in ['anchor_pool_id','lambda','relative_target_gain','source_change_rms','primary_mean_delta','primary_repeat_q90_delta']:
                if not np.isclose(row[col],expected[col],atol=1e-12,rtol=0):raise ValueError('Locked rule differs from fixed primary-only search')
            if int(row.anchor_pool_id)==-1:
                anchor=empirical_quantile_anchor(source,x[sources[sources!=pos]],x[style==target])
            else:
                anchor=pool[pool['style'].eq(target)&pool.sample_id.eq(int(row.anchor_pool_id))][features].to_numpy(float)[0]
            candidate,_,_=constrained_paths(source,anchor[None],[float(row['lambda'])],indices,sv,known,thresholds,scales)
            error=float(np.abs(candidate[0]-spectra.loc[row.selection_id,features].to_numpy(float)).max())
            if error>1e-10:raise ValueError('Locked spectrum no longer matches the registered projected path')
            max_spectrum_error=max(max_spectrum_error,error);verified+=1
    joined=outcome.merge(selections,on='selection_id',validate='one_to_one',suffixes=('','_locked'))
    for col in ['beer','target_style','selection_rule','candidate_available']:
        if not joined[col].equals(joined[col+'_locked']):raise ValueError('Once-only audit changed candidate selection identities')
    if outcome.physical_safety_certified.any():raise ValueError('Statistical design was mislabeled physical safety')
    for row in outcome.itertuples():
        if not row.candidate_available:
            if row.audited_status!='rejected_no_primary_feasible_candidate' or row.eligible:raise ValueError('Search rejection lost')
            continue
        status=candidate_status(available=True,n_failed=row.n_failed,n_unknown=row.n_unknown,
            target_gain=row.relative_target_gain>=.01,proxy_status=row.proxy_status,support_fraction=row.audit_support_repeat_fraction)
        if status!=row.audited_status or row.eligible!=(status=='eligible_statistical_candidate'):
            raise ValueError('Audit outcome bypassed unchanged E205 gates')
    primary=outcome[outcome.selection_rule.eq('minimum_change')]
    report={'status':'complete','n_fixed_rule_choices_verified':99,'n_locked_spectra_reconstructed':verified,
        'maximum_spectrum_reconstruction_error':max_spectrum_error,'n_primary_pairs':len(primary),
        'n_primary_eligible':int(primary.eligible.sum()),'n_primary_eligible_source_beers':int(primary[primary.eligible].beer.nunique()),
        'frozen_input_hash_count':len(hashes),'frozen_input_hashes_unchanged':True,
        'candidate_lock_sha256':sha(lock_path),'input_manifest_sha256':sha(manifest_path),'source_sha256':sha(__file__),
        'all_three_selection_rules_unchanged':True,'no_unknown_marked_eligible':True,'physical_safety_certified':False}
    (RESULT/'q3_e205b_audit_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
