"""Expanded Lambic proposals; primary-only search, lock before family audit."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml

from run_e205b_sensory_search import inputs as base_inputs,RESULT,ROOT
from run_e205_sensory_audit import Q2_RESULT
from run_e204_constrained_migration import TargetGeometry,_direct_vectors,_candidate_direct
from constrained_migration import empirical_quantile_anchor
from joint_generation import JointStyleGenerator
from sensory_search import constrained_paths,predict_single_target,select_candidates
from sensory_audit import sha,load_read_only_nnls,migration_model_input,support_audit
from q3_constraints import audit_concentrations

CONFIG=Path(__file__).resolve().parents[1]/'configs/E209_lambic_search.yaml'


def validate_config(config):
    required={'source_style':'Pils/Lager','target_style':'Lambic','scenario':'C1',
        'new_pool_per_beta_method':2000,'nearest_anchors':512,'random_other_anchors':512,
        'path_lambdas':[.025,.05,.1,.2,.35,.5,.65,.8,1.],
        'minimum_relative_target_gain':.01,'maximum_source_change_rms':1.,'primary_proxy_guard_margin':.03,
        'final_two_model_guard_margin':.01,'minimum_support_repeat_fraction':.9,
        'selection_rules':['minimum_change','maximum_target_gain','maximum_proxy_margin'],
        'audit_used_during_search':False,'physical_safety_certified':False,'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    for k,v in required.items():
        if config.get(k)!=v:raise ValueError(f'E209 registered policy changed: {k}')


def anchor_pool_ids(pool,source,scale,config,beer):
    dist=np.mean(((pool-source)/scale)**2,axis=1)
    order=np.lexsort((np.arange(len(pool)),dist))
    nearest=order[:config['nearest_anchors']];others=order[config['nearest_anchors']:]
    random=np.random.default_rng(config['anchor_seed']+beer).choice(others,config['random_other_anchors'],replace=False)
    return np.concatenate([nearest,random])


def inputs():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    unused,data,hashes,signature=base_inputs()
    path=RESULT/'q3_e208_audit_manifest.json';m=json.loads(path.read_text())
    if m.get('status')!='complete':raise ValueError('E208 formal audit must finish before E209')
    for n,h in m['output_sha256'].items():
        if sha(RESULT/n)!=h:raise ValueError('E208 audit outputs changed')
    files=[CONFIG,Path(__file__),Path(__file__).with_name('joint_generation.py'),path,RESULT/'q3_run_manifest_E208.json']
    hashes.update({str(p.resolve()):sha(p) for p in files})
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return config,data,hashes,signature


def run():
    lock_path=RESULT/'q3_candidate_lock_E209.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        for n,h in lock['output_sha256'].items():
            if sha(RESULT/n)!=h:raise ValueError('E209 locked output changed')
        print('E209 already locked; no reselection');return lock
    config,data,hashes,signature=inputs();final=data['final'];features=list(final.feature_names);j=final.target_codes.index('stale_hops')
    x=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(data['x'])
    styles=final.arrays['style'].astype(str);beers=final.arrays['beer'];target=config['target_style']
    scale=np.std(x,axis=0);scale=np.where(scale>1e-12,scale,1.)
    old=pd.read_csv(RESULT/'q3_target_pool_E204.csv.gz');old=old[old['style']==target].reset_index(drop=True)
    values=[old[features].to_numpy(float)];pool_methods=['frozen_copula']*len(old)
    model=JointStyleGenerator().fit(x,styles,scale=scale,tie_seed=config['pool_seed'],target_styles=[target])
    for number,method in enumerate(['beta_linear','beta_step']):
        sample=model.generate(target,config['new_pool_per_beta_method'],method,seed=config['pool_seed']+number)
        values.append(sample);pool_methods.extend([method]*len(sample))
    pool=np.vstack(values);pool_table=pd.DataFrame(pool,columns=features)
    pool_table.insert(0,'proposal_family',pool_methods);pool_table.insert(0,'proposal_id',np.arange(len(pool)))
    pool_path=RESULT/'q3_lambic_proposals_E209.csv.gz'
    pool_table.to_csv(pool_path,index=False,compression='gzip')
    geom=TargetGeometry(x[styles==target],x.mean(axis=0),scale)
    sources=np.flatnonzero(styles==config['source_style']);checkpoint=RESULT/'q3_search_checkpoints_E209';checkpoint.mkdir(exist_ok=True)
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv');defects=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    fields=pd.read_csv(RESULT/'q3_field_contract_E200.csv')
    all_selected=[];all_spectra=[];all_diagnostics=[];max_error=0.
    for number,pos in enumerate(sources,1):
        beer=int(beers[pos]);source=x[pos];slot=checkpoint/f'beer_{beer}';slot.mkdir(exist_ok=True);commit=slot/'manifest.json'
        if commit.exists():
            meta=json.loads(commit.read_text())
            if meta['signature']!=signature:raise ValueError('E209 source checkpoint signature changed')
            for n,h in meta['files'].items():
                if sha(slot/n)!=h:raise ValueError('E209 source checkpoint changed')
            all_selected.append(pd.read_csv(slot/'selected.csv'));all_spectra.append(pd.read_csv(slot/'spectra.csv.gz'))
            all_diagnostics.append(pd.read_csv(slot/'diagnostics.csv.gz'));max_error=max(max_error,meta['single_target_error'])
            print(f'E209 {number}/11 source={beer} cached=True',flush=True);continue
        pool_ids=anchor_pool_ids(pool,source,scale,config,beer)
        rank=empirical_quantile_anchor(source,x[sources[sources!=pos]],x[styles==target])
        anchors=np.vstack([pool[pool_ids],rank])
        direct,df,names,dscales,sv,known,thresholds=_direct_vectors(beer,source_table,defects,fields);di=[features.index(f) for f in df]
        candidate,anchor_ids,lam=constrained_paths(source,anchors,config['path_lambdas'],di,sv,known,thresholds,dscales)
        query=migration_model_input(candidate,np.repeat(pos,len(candidate)),data['x'])
        source_distance=float(geom.distance(source)[0]);distance=geom.distance(candidate)
        change=np.sqrt(np.mean(((candidate-source)/scale)**2,axis=1));gain=(source_distance-distance)/max(source_distance,1e-12)
        deltas=[];supports=[];max_local_error=0.
        for repeat in range(10):
            fold=int(final.arrays['fold_assignment'][repeat,pos]);valid=np.flatnonzero(final.arrays['fold_assignment'][repeat]==fold)
            train=np.flatnonzero(final.arrays['fold_assignment'][repeat]!=fold)
            predictor,info=load_read_only_nnls(final,Q2_RESULT/'q2_nnls_cache_E115',repeat,fold,data['x'].iloc[valid])
            error=float(np.max(np.abs(predict_single_target(predictor,data['x'].iloc[valid],j)-final.arrays['prediction'][repeat,valid,j])))
            if error>1e-10:raise ValueError('Final NNLS single-target identity changed')
            max_local_error=max(max_local_error,error)
            deltas.append(predict_single_target(predictor,query,j)-final.arrays['prediction'][repeat,pos,j])
            supports.append(support_audit(data['x'].iloc[train],query)['supported'])
            del predictor;gc.collect()
        deltas=np.stack(deltas);supports=np.stack(supports);dm=deltas.mean(axis=0);dq=np.quantile(deltas,.9,axis=0);support=supports.mean(axis=0)
        selection,feasible=select_candidates(gain,change,dm,dq,np.where(change<=1.,support,0.),guard_margin=.03)
        diagnostics=pd.DataFrame({'beer':beer,'anchor_id':anchor_ids,'proposal_id':[int(pool_ids[v]) if v<len(pool_ids) else -1 for v in anchor_ids],
            'lambda':lam,'relative_target_gain':gain,'source_change_rms':change,'primary_mean_delta':dm,'primary_repeat_q90_delta':dq,
            'support_repeat_fraction':support,'primary_feasible':feasible})
        selected=[];spectra=[]
        for rule,index in selection.items():
            selection_id=f'{beer}_Lambic_{rule}'
            record={'selection_id':selection_id,'beer':beer,'target_style':target,'scenario':'C1','selection_rule':rule,
                'candidate_available':index is not None,'n_primary_feasible':int(feasible.sum()),'physical_safety_certified':False,'auditor_accessed':False}
            if index is None:record['status']='rejected_no_primary_feasible_candidate'
            else:
                record.update(diagnostics.iloc[index].drop('primary_feasible').to_dict());record['status']='locked_primary_candidate'
                projection=audit_concentrations(sv,_candidate_direct(candidate[index],di,dscales),known,'C1',threshold=thresholds)
                record['n_failed']=int(projection.passed.eq(False).fillna(False).sum());record['n_unknown']=int(projection.passed.isna().sum())
                if record['n_failed']!=0:raise ValueError('Known direct constraint violated after projection')
                spectra.append({'selection_id':selection_id,**dict(zip(features,candidate[index]))})
            selected.append(record)
        selected=pd.DataFrame(selected);spectra=pd.DataFrame(spectra,columns=['selection_id']+features)
        outputs=[]
        for n,f in [('selected.csv',selected),('spectra.csv.gz',spectra),('diagnostics.csv.gz',diagnostics)]:
            p=slot/n;f.to_csv(p,index=False,compression='gzip' if n.endswith('.gz') else None);outputs.append(p)
        meta={'signature':signature,'beer':beer,'single_target_error':max_local_error,
              'files':{p.name:sha(p) for p in outputs}}
        commit.write_text(json.dumps(meta,indent=2)+'\n');all_selected.append(selected);all_spectra.append(spectra);all_diagnostics.append(diagnostics)
        max_error=max(max_error,max_local_error)
        print(f'E209 {number}/11 source={beer} feasible={int(feasible.sum())} locked={len(spectra)}/3',flush=True)
    outputs=[pool_path]
    for name,frame in [('q3_search_selection_E209.csv',pd.concat(all_selected,ignore_index=True)),
        ('q3_locked_spectra_E209.csv.gz',pd.concat(all_spectra,ignore_index=True)),
        ('q3_search_diagnostics_E209.csv.gz',pd.concat(all_diagnostics,ignore_index=True))]:
        p=RESULT/name;frame.to_csv(p,index=False,compression='gzip' if name.endswith('.gz') else None);outputs.append(p)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Frozen input changed during E209')
    lock={'experiment_id':'E209','status':'locked_before_model_family_audit','n_source_target_pairs':11,'n_rule_rows':33,
        'n_available':sum(len(f) for f in all_spectra),'n_search_points':sum(len(f) for f in all_diagnostics),
        'signature':signature,'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in outputs},
        'single_target_max_abs_error':max_error,'auditor_accessed':False,'q1_retrained':False,'nnls_refit':False,'gpu_used':False,
        'physical_safety_certified':False,'scope':'exploratory_postselection_lambic_proposals_not_generator_promotion'}
    lock_path.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in lock.items() if not k.endswith('sha256')},ensure_ascii=False),flush=True);return lock


if __name__=='__main__':run()
