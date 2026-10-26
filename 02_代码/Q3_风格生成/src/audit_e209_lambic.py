"""Once-only model-family audit after committing the fixed Lambic search lock."""
import gc
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from run_e209_lambic_search import inputs,anchor_pool_ids,ROOT,RESULT
from audit_e205b_sensory_search import verify_committed_lock
from run_e205_sensory_audit import Q2_RESULT,transform_elastic
from run_e204_constrained_migration import TargetGeometry,_direct_vectors,_candidate_direct
from constrained_migration import empirical_quantile_anchor
from sensory_search import constrained_paths
from sensory_audit import (sha,load_read_only_nnls,migration_model_input,support_audit,
    verify_archived_prediction,proxy_screen,candidate_status)
from q3_constraints import audit_concentrations
from selection_models import fit_fold_preprocessor,build_quadratic_penalty_rows,fit_multi_target_with_penalty_rows


def stronger_status(screen,*,n_failed,n_unknown,gain,change,support,margin=.01):
    if not np.isfinite(margin) or margin<0:raise ValueError('Finite nonnegative proxy margin required')
    guarded=all(screen[k]<=-margin for k in ['primary_mean_delta','audit_mean_delta',
        'primary_repeat_q90_delta','audit_repeat_q90_delta'])
    proxy=screen['proxy_status']
    if proxy=='predicted_nonworsening' and not guarded:proxy='proxy_margin_insufficient'
    return candidate_status(available=True,n_failed=n_failed,n_unknown=n_unknown,
        target_gain=bool(gain>=.01 and change<=1.),proxy_status=proxy,support_fraction=support)


def validate_selection(selected,spectra,features,source_beers):
    expected={f'{int(b)}_Lambic_{rule}' for b in source_beers for rule in
        ['minimum_change','maximum_target_gain','maximum_proxy_margin']}
    if (len(selected)!=33 or selected.selection_id.duplicated().any()
        or set(selected.selection_id)!=expected or not selected.target_style.eq('Lambic').all()
        or not selected.scenario.eq('C1').all()):raise ValueError('Complete fixed 11-source/3-rule denominator required')
    if (spectra.selection_id.duplicated().any() or list(spectra.columns)!=['selection_id']+features
        or set(spectra.selection_id)!=set(selected.loc[selected.candidate_available,'selection_id'])):
        raise ValueError('Locked spectra do not match available rule identities')


def provenance(selected,spectra,data,config):
    final=data['final'];features=list(final.feature_names)
    x=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(data['x'])
    styles=final.arrays['style'].astype(str);beers=final.arrays['beer'];sources=np.flatnonzero(styles=='Pils/Lager')
    validate_selection(selected,spectra,features,beers[sources])
    pool=pd.read_csv(RESULT/'q3_lambic_proposals_E209.csv.gz')
    if (len(pool)!=6000 or pool.proposal_id.tolist()!=list(range(6000))
        or pool.groupby('proposal_family').size().to_dict()!=
        {'frozen_copula':2000,'beta_linear':2000,'beta_step':2000}):raise ValueError('Fixed proposal pool incomplete')
    scale=np.std(x,axis=0);scale=np.where(scale>1e-12,scale,1.)
    geometry=TargetGeometry(x[styles=='Lambic'],x.mean(axis=0),scale)
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv')
    defects=pd.read_csv(RESULT/'q3_defect_registry_E200.csv');fields=pd.read_csv(RESULT/'q3_field_contract_E200.csv')
    records=[]
    for i,row in spectra.merge(selected,on='selection_id',validate='one_to_one').iterrows():
        pos=int(np.flatnonzero(beers==int(row.beer))[0]);source=x[pos]
        ids=anchor_pool_ids(pool[features].to_numpy(float),source,scale,config,int(row.beer))
        rank=empirical_quantile_anchor(source,x[sources[sources!=pos]],x[styles=='Lambic'])
        anchor_id=int(row.anchor_id)
        if anchor_id<0 or anchor_id>len(ids):raise ValueError('Locked anchor id outside fixed proposal set')
        anchor=pool.iloc[ids[anchor_id]][features].to_numpy(float) if anchor_id<len(ids) else rank
        expected_pool_id=int(ids[anchor_id]) if anchor_id<len(ids) else -1
        if int(row.proposal_id)!=expected_pool_id or float(row['lambda']) not in config['path_lambdas']:
            raise ValueError('Locked candidate did not come from registered anchor/lambda')
        _,df,_,dscales,sv,known,thresholds=_direct_vectors(int(row.beer),source_table,defects,fields)
        di=[features.index(f) for f in df]
        expected=constrained_paths(source,anchor[None,:],[float(row['lambda'])],di,sv,known,thresholds,dscales)[0][0]
        actual=row[features].to_numpy(float);error=float(np.max(np.abs(expected-actual)))
        if error>1e-10:raise ValueError('Locked spectrum differs from registered projected path')
        change=float(np.sqrt(np.mean(((actual-source)/scale)**2)))
        sd=float(geometry.distance(source)[0]);gain=float((sd-geometry.distance(actual)[0])/max(sd,1e-12))
        if abs(change-row.source_change_rms)>1e-10 or abs(gain-row.relative_target_gain)>1e-10:
            raise ValueError('Locked geometric diagnostics changed')
        records.append({'selection_id':row.selection_id,'maximum_spectrum_error':error,
            'proposal_id':expected_pool_id,'lambda':row['lambda'],'source_change_rms':change,'relative_target_gain':gain})
    return pd.DataFrame(records,columns=['selection_id','maximum_spectrum_error','proposal_id','lambda',
        'source_change_rms','relative_target_gain'])


def run():
    manifest_path=RESULT/'q3_run_manifest_E209.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for n,h in previous['output_sha256'].items():
            if sha(RESULT/n)!=h:raise ValueError('Completed once-only audit output changed')
        print('E209 already audited; no reselection');return previous
    lock_path=RESULT/'q3_candidate_lock_E209.json';lock=verify_committed_lock(ROOT,lock_path)
    config,data,hashes,signature=inputs()
    if lock['signature']!=signature:raise ValueError('Frozen E209 search inputs changed')
    final=data['final'];features=list(final.feature_names);codes=list(final.target_codes);j=codes.index('stale_hops')
    selected=pd.read_csv(RESULT/'q3_search_selection_E209.csv');spectra=pd.read_csv(RESULT/'q3_locked_spectra_E209.csv.gz')
    trace=provenance(selected,spectra,data,config)
    meta=spectra[['selection_id']].merge(selected,on='selection_id',validate='one_to_one')
    positions=np.array([list(final.arrays['beer']).index(int(b)) for b in meta.beer],dtype=int)
    n=len(spectra);query=migration_model_input(spectra[features],positions,data['x'])
    primary=np.full((10,n,50),np.nan);auditor=np.full_like(primary,np.nan)
    dp=np.full_like(primary,np.nan);da=np.full_like(primary,np.nan);supported=np.zeros((10,n),bool);reconstruction=[]
    for repeat in range(10):
        for fold in range(3):
            valid=np.flatnonzero(final.arrays['fold_assignment'][repeat]==fold)
            train=np.flatnonzero(final.arrays['fold_assignment'][repeat]!=fold)
            ci=np.flatnonzero(final.arrays['fold_assignment'][repeat,positions]==fold)
            if not len(ci):continue
            model,info=load_read_only_nnls(final,Q2_RESULT/'q2_nnls_cache_E115',repeat,fold,data['x'].iloc[valid])
            primary[repeat,ci]=model.predict(query[ci]);dp[repeat,ci]=primary[repeat,ci]-final.arrays['prediction'][repeat,positions[ci]]
            del model;gc.collect()
            processed=fit_fold_preprocessor(data['x'].iloc[train],data['x'].iloc[valid])
            parameter=data['parameters'].query('repeat==@repeat and fold==@fold').iloc[0];p=processed.x_train.shape[1]
            penalty=build_quadratic_penalty_rows(np.zeros((p,p)),n_samples=len(train),lambda_l2=float(parameter.lambda_l2),lambda_graph=0.)
            estimator=fit_multi_target_with_penalty_rows(processed.x_train,data['y'][train],penalty,lambda_l1=float(parameter.lambda_l1))
            error=verify_archived_prediction(estimator.predict(processed.x_validation),data['auditor_oof'][repeat,valid])
            auditor[repeat,ci]=estimator.predict(transform_elastic(processed,query[ci]))
            # Both source and candidate use rebuilt float64 auditor; its float32 archive is an identity check only.
            da[repeat,ci]=auditor[repeat,ci]-estimator.predict(transform_elastic(processed,data['x'].iloc[positions[ci]]))
            supported[repeat,ci]=support_audit(data['x'].iloc[train],query[ci])['supported']
            reconstruction.append({'repeat':repeat,'fold':fold,'primary_error':info['reconstruction_max_abs_error'],
                'auditor_archive_quantization_error':error,'auditor_exact_at_float32':True})
    if not all(np.isfinite(a).all() for a in [primary,auditor,dp,da]):raise ValueError('Incomplete once-only predictions')
    reliability=pd.read_csv(RESULT/'q3_proxy_model_reliability_E205.csv')
    reliable=reliability[reliability.target_code=='stale_hops'].set_index('model').proxy_reliable
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv');defects=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    fields=pd.read_csv(RESULT/'q3_field_contract_E200.csv');records=[];direct_rows=[];repeats=[]
    for i,row in meta.iterrows():
        screen=proxy_screen(dp[:,i,j],da[:,i,j],bool(reliable.final_NNLS),bool(reliable.ElasticNet_auditor))
        if (abs(screen['primary_mean_delta']-row.primary_mean_delta)>1e-10
            or abs(screen['primary_repeat_q90_delta']-row.primary_repeat_q90_delta)>1e-10):
            raise ValueError('Full NNLS does not reproduce primary-only search')
        if max(screen['primary_mean_delta'],screen['primary_repeat_q90_delta'])>-.03+1e-10:
            raise ValueError('A selected candidate violates stronger primary guard')
        _,df,names,dscales,sv,known,thresholds=_direct_vectors(int(row.beer),source_table,defects,fields)
        concentration=_candidate_direct(spectra.loc[i,features].to_numpy(float),[features.index(f) for f in df],dscales)
        support=float(supported[:,i].mean());counts={}
        for scenario in ['C1','C2','C3']:
            checks=audit_concentrations(sv,concentration,known,scenario,threshold=thresholds)
            for k,check in enumerate(checks.to_dict('records')):
                direct_rows.append({'selection_id':row.selection_id,'beer':int(row.beer),'defect':names[k],**check})
            if scenario=='C1':counts={'n_failed':int(checks.passed.eq(False).fillna(False).sum()),'n_unknown':int(checks.passed.isna().sum())}
        statuses={}
        for margin in [0.,.01,.02]:
            status=stronger_status(screen,**counts,gain=row.relative_target_gain,change=row.source_change_rms,support=support,margin=margin)
            statuses[f'status_margin_{margin:.2f}']=status
        records.append({'selection_id':row.selection_id,**screen,**counts,'audit_support_repeat_fraction':support,
            **statuses,'audited_status':statuses['status_margin_0.01'],'eligible':statuses['status_margin_0.01']=='eligible_statistical_candidate'})
        for repeat in range(10):
            repeats.append({'selection_id':row.selection_id,'repeat':repeat,'primary_stale_change':dp[repeat,i,j],
                'auditor_stale_change':da[repeat,i,j],'supported':bool(supported[repeat,i])})
    rec=pd.DataFrame(records).drop(columns='physical_safety_certified',errors='ignore')
    if not len(rec):rec=pd.DataFrame(columns=['selection_id','audited_status','eligible'])
    outcome=selected.merge(rec,on='selection_id',how='left',validate='one_to_one',suffixes=('_search',''))
    outcome.loc[~outcome.candidate_available,'audited_status']='rejected_no_primary_feasible_candidate'
    outcome['eligible']=outcome.audited_status.eq('eligible_statistical_candidate');outcome['physical_safety_certified']=False
    profiles=meta.loc[meta.index.repeat(50),['selection_id','beer','target_style','selection_rule']].reset_index(drop=True)
    profiles['target_code']=np.tile(codes,n);profiles['prediction_z']=primary.mean(axis=0).ravel()
    profiles['audit_prediction_z']=auditor.mean(axis=0).ravel();profiles['primary_delta']=dp.mean(axis=0).ravel();profiles['audit_delta']=da.mean(axis=0).ravel()
    outputs=[]
    for filename,table in [('q3_once_audit_E209.csv',outcome),('q3_direct_sensitivity_E209.csv',pd.DataFrame(direct_rows)),
        ('q3_selected_profiles_E209.csv.gz',profiles),('q3_proxy_changes_by_repeat_E209.csv',pd.DataFrame(repeats)),
        ('q3_predictor_reconstruction_E209.csv',pd.DataFrame(reconstruction)),('q3_spectrum_provenance_E209.csv',trace)]:
        path=RESULT/filename;table.to_csv(path,index=False,compression='gzip' if filename.endswith('.gz') else None);outputs.append(path)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during once-only audit')
    main=outcome[outcome.selection_rule=='minimum_change']
    report={'experiment_id':'E209','status':'complete_once_only_audit','n_source_target_pairs':11,'n_rule_rows':len(outcome),
        'n_available':n,'n_primary_rule_eligible_pairs':int(main.eligible.sum()),
        'n_auxiliary_eligible_rule_rows':int(outcome.eligible.sum()-main.eligible.sum()),
        'status_counts_primary_rule':{str(k):int(v) for k,v in main.audited_status.value_counts().items()},
        'status_counts_all_rules':{str(k):int(v) for k,v in outcome.audited_status.value_counts().items()},
        'candidate_lock_sha256':sha(lock_path),'candidate_lock_commit':subprocess.check_output(
            ['git','log','-1','--format=%H','--',str(lock_path.relative_to(ROOT))],cwd=ROOT,text=True).strip(),
        'audit_source_sha256':sha(__file__),'final_proxy_margin':.01,'audit_only_after_lock_commit':True,
        'no_auditor_based_reselection':True,'q1_retrained':False,'nnls_refit':False,'gpu_used':False,
        'physical_safety_certified':False,'scope':'exploratory_postselection_not_new_independent_validation',
        'output_sha256':{p.name:sha(p) for p in outputs}}
    manifest_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
