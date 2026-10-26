"""Evaluate once after committing the primary-only candidate lock."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from run_e205b_sensory_search import RESULT,ROOT,CONFIG,inputs
from run_e205_sensory_audit import Q2_RESULT,transform_elastic
from run_e204_constrained_migration import _direct_vectors,_candidate_direct
from selection_models import fit_fold_preprocessor,build_quadratic_penalty_rows,fit_multi_target_with_penalty_rows
from sensory_audit import (sha,load_read_only_nnls,migration_model_input,verify_archived_prediction,
    proxy_screen,candidate_status,support_audit)
from q3_constraints import audit_concentrations


def verify_committed_lock(root,lock_path):
    root=Path(root);lock_path=Path(lock_path)
    relative=str(lock_path.relative_to(root))
    committed=subprocess.check_output(['git','show','HEAD:'+relative],cwd=root,stderr=subprocess.PIPE)
    if committed!=lock_path.read_bytes():raise ValueError('Candidate lock must be committed unchanged before audit')
    lock=json.loads(committed)
    if lock.get('status')!='locked_before_model_family_audit' or lock.get('auditor_accessed') is not False:
        raise ValueError('Search must freeze candidates before evaluating the auditor')
    for name,digest in lock['output_sha256'].items():
        if sha(lock_path.parent/name)!=digest:raise ValueError('Locked selected output changed')
    return lock


def run():
    output_manifest=RESULT/'q3_run_manifest_E205B.json'
    if output_manifest.exists():
        existing=json.loads(output_manifest.read_text())
        for name,digest in existing['output_sha256'].items():
            if sha(RESULT/name)!=digest:raise ValueError('Completed audit output changed')
        print('E205B already audited; candidates were not reselected')
        return existing
    lock_path=RESULT/'q3_candidate_lock_E205B.json';lock=verify_committed_lock(ROOT,lock_path)
    config,data,hashes,signature=inputs()
    if signature!=lock['signature']:raise ValueError('Frozen search inputs changed before audit')
    final=data['final'];codes=list(final.target_codes);features=list(final.feature_names);j=codes.index('stale_hops')
    selected=pd.read_csv(RESULT/'q3_search_selection_E205B.csv')
    spectra=pd.read_csv(RESULT/'q3_locked_spectra_E205B.csv.gz')
    if len(selected)!=99 or selected.selection_id.duplicated().any() or spectra.selection_id.duplicated().any():
        raise ValueError('Candidate rules or identities are incomplete')
    if set(spectra.selection_id)!=set(selected.loc[selected.candidate_available,'selection_id']):
        raise ValueError('Selected spectrum identities do not match availability')
    metadata=spectra[['selection_id']].merge(selected,on='selection_id',validate='one_to_one')
    n=len(spectra);positions=np.array([list(final.arrays['beer']).index(int(v)) for v in metadata.beer])
    query=migration_model_input(spectra[features],positions,data['x'])
    primary=np.full((10,n,50),np.nan);auditor=np.full_like(primary,np.nan);supported=np.zeros((10,n),bool)
    audit_error=[]
    for repeat in range(10):
        for fold in range(3):
            mask=final.arrays['fold_assignment'][repeat]==fold;valid=np.flatnonzero(mask);train=np.flatnonzero(~mask)
            candidates=np.flatnonzero(mask[positions])
            if not len(candidates):continue
            model,info=load_read_only_nnls(final,Q2_RESULT/'q2_nnls_cache_E115',repeat,fold,data['x'].iloc[valid])
            primary[repeat,candidates]=model.predict(query[candidates])
            processed=fit_fold_preprocessor(data['x'].iloc[train],data['x'].iloc[valid])
            parameter=data['parameters'].query('repeat==@repeat and fold==@fold').iloc[0];p=processed.x_train.shape[1]
            penalty=build_quadratic_penalty_rows(np.zeros((p,p)),n_samples=len(train),lambda_l2=float(parameter.lambda_l2),lambda_graph=0.)
            estimator=fit_multi_target_with_penalty_rows(processed.x_train,data['y'][train],penalty,lambda_l1=float(parameter.lambda_l1))
            error=verify_archived_prediction(estimator.predict(processed.x_validation),data['auditor_oof'][repeat,valid])
            auditor[repeat,candidates]=estimator.predict(transform_elastic(processed,query[candidates]))
            supported[repeat,candidates]=support_audit(data['x'].iloc[train],query[candidates])['supported']
            audit_error.append({'repeat':repeat,'fold':fold,'primary_error':info['reconstruction_max_abs_error'],
                                'auditor_archive_quantization_error':error,'auditor_exact_at_float32':True})
    if not np.isfinite(primary).all() or not np.isfinite(auditor).all():raise ValueError('Locked candidate predictions incomplete')
    dp=primary-final.arrays['prediction'][:,positions];da=auditor-data['auditor_oof'][:,positions].astype(float)
    # Use rebuilt auditor source predictions: the float32 archive is a precision
    # reference only. Quantization must not create a false tiny source delta.
    for repeat in range(10):
        for fold in range(3):
            ci=np.flatnonzero(final.arrays['fold_assignment'][repeat,positions]==fold)
            if not len(ci):continue
            train=np.flatnonzero(final.arrays['fold_assignment'][repeat]!=fold);valid=np.flatnonzero(final.arrays['fold_assignment'][repeat]==fold)
            processed=fit_fold_preprocessor(data['x'].iloc[train],data['x'].iloc[valid])
            parameter=data['parameters'].query('repeat==@repeat and fold==@fold').iloc[0];p=processed.x_train.shape[1]
            penalty=build_quadratic_penalty_rows(np.zeros((p,p)),n_samples=len(train),lambda_l2=float(parameter.lambda_l2),lambda_graph=0.)
            estimator=fit_multi_target_with_penalty_rows(processed.x_train,data['y'][train],penalty,lambda_l1=float(parameter.lambda_l1))
            da[repeat,ci]=auditor[repeat,ci]-estimator.predict(transform_elastic(processed,data['x'].iloc[positions[ci]]))
    reliability=pd.read_csv(RESULT/'q3_proxy_model_reliability_E205.csv')
    reliable=reliability[reliability.target_code=='stale_hops'].set_index('model').proxy_reliable
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv');defect_table=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    field_table=pd.read_csv(RESULT/'q3_field_contract_E200.csv')
    predictions=[];checks=[]
    for i,row in metadata.iterrows():
        record={'selection_id':row.selection_id,**proxy_screen(dp[:,i,j],da[:,i,j],bool(reliable.final_NNLS),
                        bool(reliable.ElasticNet_auditor))}
        record['audit_support_repeat_fraction']=float(supported[:,i].mean())
        if (abs(record['primary_mean_delta']-row.primary_mean_delta)>1e-10
            or abs(record['primary_repeat_q90_delta']-row.primary_repeat_q90_delta)>1e-10):
            raise ValueError('Full final predictor changed primary-only search values')
        direct,df,defects,scales,sv,known,thresholds=_direct_vectors(int(row.beer),source_table,defect_table,field_table)
        concentration=_candidate_direct(spectra.loc[i,features].to_numpy(float),[features.index(f) for f in df],scales)
        for scenario in ['C1','C2','C3']:
            table=audit_concentrations(sv,concentration,known,scenario,threshold=thresholds)
            for k,check in enumerate(table.to_dict('records')):
                checks.append({'selection_id':row.selection_id,'beer':int(row.beer),'target_style':row.target_style,
                    'selection_rule':row.selection_rule,'defect':defects[k],**check})
            if scenario=='C1':
                record['n_failed']=int(table.passed.eq(False).fillna(False).sum());record['n_unknown']=int(table.passed.isna().sum())
        record['audited_status']=candidate_status(available=True,n_failed=record['n_failed'],n_unknown=record['n_unknown'],
            target_gain=bool(row.relative_target_gain>=.01),proxy_status=record['proxy_status'],
            support_fraction=record['audit_support_repeat_fraction'])
        predictions.append(record)
    outcome=selected.merge(pd.DataFrame(predictions).drop(columns='physical_safety_certified'),on='selection_id',
                           how='left',validate='one_to_one',suffixes=('_search',''))
    outcome.loc[~outcome.candidate_available,'audited_status']='rejected_no_primary_feasible_candidate'
    outcome['eligible']=outcome.audited_status.eq('eligible_statistical_candidate');outcome['physical_safety_certified']=False
    all_profiles=metadata.loc[metadata.index.repeat(50),['selection_id','beer','target_style','selection_rule']].reset_index(drop=True)
    all_profiles['target_code']=np.tile(codes,n)
    all_profiles['prediction_z']=primary.mean(axis=0).ravel();all_profiles['audit_prediction_z']=auditor.mean(axis=0).ravel()
    all_profiles['primary_delta']=dp.mean(axis=0).ravel();all_profiles['audit_delta']=da.mean(axis=0).ravel()
    outputs=[]
    for name,table in [('q3_once_audit_E205B.csv',outcome),('q3_direct_sensitivity_E205B.csv',pd.DataFrame(checks)),
        ('q3_selected_profiles_E205B.csv.gz',all_profiles),('q3_predictor_reconstruction_E205B.csv',pd.DataFrame(audit_error))]:
        path=RESULT/name;table.to_csv(path,index=False,compression='gzip' if name.endswith('.gz') else None);outputs.append(path)
    comparison=[]
    old=pd.read_csv(RESULT/'q3_migration_proxy_audit_E205.csv')
    for style in config['target_styles']:
        base=old[old.scenario.eq('C1')&old.target_style.eq(style)&old.strategy.eq('constrained_path')]
        for rule in config['selection_rules']:
            group=outcome[outcome.target_style.eq(style)&outcome.selection_rule.eq(rule)]
            comparison.append({'style':style,'selection_rule':rule,'n_source_target_pairs':11,
                'old_n_eligible_pairs':int(base[base.eligible].beer.nunique()),'new_n_eligible_pairs':int(group.eligible.sum()),
                'new_n_candidates':int(group.candidate_available.sum()),'new_n_needs_direct_verification':int(group.audited_status.eq('needs_direct_verification').sum()),
                'new_available_mean_target_gain':float(group.relative_target_gain.mean()),
                'new_available_mean_source_change':float(group.source_change_rms.mean())})
    path=RESULT/'q3_search_comparison_E205B.csv';pd.DataFrame(comparison).to_csv(path,index=False);outputs.append(path)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during once-only audit')
    primary_rows=outcome[outcome.selection_rule=='minimum_change']
    manifest={'experiment_id':'E205B','status':'complete_once_only_audit','scope':config['scope'],
        'candidate_lock_sha256':sha(lock_path),'candidate_lock_commit':subprocess.check_output(
            ['git','log','-1','--format=%H','--',str(lock_path.relative_to(ROOT))],cwd=ROOT,text=True).strip(),
        'audit_source_sha256':sha(__file__),'n_source_target_pairs':33,'n_selection_rules':3,
        'n_rule_rows':len(outcome),'n_locked_available':n,'n_primary_rule_eligible_pairs':int(primary_rows.eligible.sum()),
        'n_auxiliary_eligible_rule_rows':int(outcome.eligible.sum()-primary_rows.eligible.sum()),
        'status_counts_primary_rule':{str(k):int(v) for k,v in primary_rows.audited_status.value_counts().items()},
        'status_counts_all_rules':{str(k):int(v) for k,v in outcome.audited_status.value_counts().items()},
        'no_auditor_based_reselection':True,'q1_retrained':False,'nnls_refit':False,'gpu_used':False,
        'physical_safety_certified':False,'output_sha256':{p.name:sha(p) for p in outputs}}
    output_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2));return manifest


if __name__=='__main__':run()
