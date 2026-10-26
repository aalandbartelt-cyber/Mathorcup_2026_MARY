"""Search with frozen NNLS only; lock selected spectra before auditor access."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from run_e205_sensory_audit import CANONICAL,Q2_RESULT,ROOT,load_inputs
from run_e204_constrained_migration import TargetGeometry,_direct_vectors,_candidate_direct
from constrained_migration import empirical_quantile_anchor
from sensory_audit import sha,load_read_only_nnls,migration_model_input,support_audit
from sensory_search import constrained_paths,select_candidates,predict_single_target
from q3_constraints import audit_concentrations

RESULT=ROOT/'03_结果/Q3_风格生成'
CONFIG=Path(__file__).resolve().parents[1]/'configs/E205B_sensory_search.yaml'
RULES=['minimum_change','maximum_target_gain','maximum_proxy_margin']


def inputs():
    config=yaml.safe_load(CONFIG.read_text())
    required={'source_style':'Pils/Lager','target_styles':['Tripel','Kriek','Lambic'],'scenario':'C1',
        'nearest_pool_anchors':256,'include_rank_anchor':True,'path_lambdas':[.05,.1,.2,.35,.5,.75,1.],
        'minimum_relative_target_gain':.01,'primary_proxy_guard_margin':.01,'primary_delta_quantile':.9,
        'minimum_support_repeat_fraction':.9,'selection_rules':RULES,
        'audit_used_during_search':False,'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    for key,value in required.items():
        if config.get(key)!=value:raise ValueError(f'Registered search config changed: {key}')
    e205_path=RESULT/'q3_run_manifest_E205.json';e205=json.loads(e205_path.read_text())
    audit_path=RESULT/'q3_e205_audit_manifest.json';audit=json.loads(audit_path.read_text())
    if e205.get('status')!='complete' or audit.get('status')!='complete':raise ValueError('Audited E205 required')
    for path,digest in e205['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('E205 frozen input changed')
    config205=yaml.safe_load((CONFIG.parent/'E205_sensory_audit.yaml').read_text())
    data=load_inputs(config205)
    files=[CONFIG,Path(__file__),Path(__file__).with_name('sensory_search.py'),
        Path(__file__).with_name('constrained_migration.py'),Path(__file__).with_name('q3_constraints.py'),
        Path(__file__).with_name('run_e204_constrained_migration.py'),e205_path,audit_path,
        RESULT/'q3_source_defects_E200.csv',RESULT/'q3_defect_registry_E200.csv',RESULT/'q3_field_contract_E200.csv']
    hashes=dict(data['input_hashes']);hashes.update({str(p.resolve()):sha(p) for p in files})
    signature=__import__('hashlib').sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return config,data,hashes,signature


def run():
    lock_path=RESULT/'q3_candidate_lock_E205B.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        for name,digest in lock['output_sha256'].items():
            if sha(RESULT/name)!=digest:raise ValueError('Locked candidate output changed')
        print('E205B search already locked; no reselection',flush=True)
        return lock
    config,data,hashes,signature=inputs();final=data['final'];features=list(final.feature_names)
    x=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(data['x'])
    styles=final.arrays['style'].astype(str);beer=final.arrays['beer'];j=final.target_codes.index('stale_hops')
    mean=x.mean(axis=0);scale=np.std(x,axis=0);scale=np.where(scale>1e-12,scale,1.)
    pool=pd.read_csv(RESULT/'q3_target_pool_E204.csv.gz')
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv')
    defect_table=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    field_table=pd.read_csv(RESULT/'q3_field_contract_E200.csv')
    source_positions=np.flatnonzero(styles==config['source_style'])
    geometries={s:TargetGeometry(x[styles==s],mean,scale) for s in config['target_styles']}
    checkpoint=RESULT/'q3_search_checkpoints_E205B';checkpoint.mkdir(exist_ok=True)
    summaries=[];spectra=[];diagnostics=[];max_error=0.;cache_hashes={}
    for source_number,pos in enumerate(source_positions):
        beer_id=int(beer[pos]);name=f'beer_{beer_id}';meta_path=checkpoint/(name+'.json')
        selected_path=checkpoint/(name+'_selected.csv');spectra_path=checkpoint/(name+'_spectra.csv.gz')
        diagnostics_path=checkpoint/(name+'_search.csv.gz')
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta['signature']!=signature or any(sha(checkpoint/n)!=h for n,h in meta['output_sha256'].items()):
                raise ValueError('Search checkpoint identity changed')
            summaries.append(pd.read_csv(selected_path));spectra.append(pd.read_csv(spectra_path))
            diagnostics.append(pd.read_csv(diagnostics_path));max_error=max(max_error,meta['single_target_max_abs_error'])
            cache_hashes.update(meta['cache_sha256']);print(f'E205B {source_number+1}/11 source={beer_id} cached=True',flush=True)
            continue
        source=x[pos];direct,df,defects,dscales,sv,known,thresholds=_direct_vectors(beer_id,source_table,defect_table,field_table)
        di=[features.index(v) for v in df]
        blocks=[];descriptions=[]
        for style in config['target_styles']:
            target_pool=pool[pool['style']==style];values=target_pool.loc[:,features].to_numpy(float)
            distances=np.mean(((values-source)/scale)**2,axis=1)
            chosen=np.lexsort((target_pool.sample_id.to_numpy(),distances))[:config['nearest_pool_anchors']]
            rank_anchor=empirical_quantile_anchor(source,x[source_positions[source_positions!=pos]],x[styles==style])
            anchors=np.vstack([values[chosen],rank_anchor])
            paths,ai,lam=constrained_paths(source,anchors,config['path_lambdas'],di,sv,known,thresholds,dscales)
            source_distance=float(geometries[style].distance(source)[0])
            target_distance=geometries[style].distance(paths)
            desc=pd.DataFrame({'beer':beer_id,'target_style':style,'anchor_index':ai,'lambda':lam,
                'candidate_distance':target_distance,'source_distance':source_distance,
                'relative_target_gain':(source_distance-target_distance)/max(source_distance,1e-12),
                'source_change_rms':np.sqrt(np.mean(((paths-source)/scale)**2,axis=1))})
            desc['anchor_pool_id']=[int(target_pool.sample_id.iloc[chosen[k]]) if k<len(chosen) else -1 for k in ai]
            blocks.append(paths);descriptions.append(desc)
        values=np.vstack(blocks);description=pd.concat(descriptions,ignore_index=True)
        query=migration_model_input(values,np.repeat(pos,len(values)),data['x'])
        deltas=[];supports=[];source_cache={};source_error=0.
        for repeat in range(10):
            fold=int(final.arrays['fold_assignment'][repeat,pos]);valid=np.flatnonzero(final.arrays['fold_assignment'][repeat]==fold)
            train=np.flatnonzero(final.arrays['fold_assignment'][repeat]!=fold)
            model,info=load_read_only_nnls(final,Q2_RESULT/'q2_nnls_cache_E115',repeat,fold,data['x'].iloc[valid])
            error=float(np.max(np.abs(predict_single_target(model,data['x'].iloc[valid],j)-final.arrays['prediction'][repeat,valid,j])))
            if error>1e-10:raise ValueError('Single-target optimization differs from frozen NNLS')
            source_error=max(source_error,error);source_cache[f'R{repeat:02d}F{fold}']=info['model_sha256']
            delta=predict_single_target(model,query,j)-float(final.arrays['prediction'][repeat,pos,j]);deltas.append(delta)
            support=support_audit(data['x'].iloc[train],query);supports.append(support['supported'])
            del model;gc.collect()
        deltas=np.stack(deltas);supports=np.stack(supports)
        description['primary_mean_delta']=deltas.mean(axis=0);description['primary_repeat_q90_delta']=np.quantile(deltas,.9,axis=0)
        description['support_repeat_fraction']=supports.mean(axis=0)
        chosen_rows=[];chosen_spectra=[];description['primary_feasible']=False
        for style,index in description.groupby('target_style',sort=False).groups.items():
            index=np.array(list(index));subset=description.loc[index]
            selections,feasible=select_candidates(subset.relative_target_gain,subset.source_change_rms,
                subset.primary_mean_delta,subset.primary_repeat_q90_delta,subset.support_repeat_fraction)
            description.loc[index,'primary_feasible']=feasible
            for rule,selected in selections.items():
                selection_id=f'{beer_id}_{style}_{rule}'
                base={'selection_id':selection_id,'beer':beer_id,'target_style':style,'scenario':'C1','selection_rule':rule,
                      'candidate_available':selected is not None,'physical_safety_certified':False,
                      'n_primary_feasible':int(feasible.sum()),'n_search_points':len(subset),'auditor_accessed':False}
                if selected is None:
                    base['search_status']='rejected_no_primary_feasible_candidate'
                else:
                    loc=int(index[selected]);base.update(description.loc[loc].drop(labels=['primary_feasible']).to_dict())
                    concentration=_candidate_direct(values[loc],di,dscales)
                    audit=audit_concentrations(sv,concentration,known,'C1',threshold=thresholds)
                    base['n_direct_failed']=int(audit.passed.eq(False).fillna(False).sum())
                    base['n_direct_unknown']=int(audit.passed.isna().sum())
                    base['search_status']='locked_primary_candidate'
                    if base['n_direct_failed']!=0:raise ValueError('Projected selected candidate violates a known direct constraint')
                    row={'selection_id':selection_id,**dict(zip(features,values[loc]))};chosen_spectra.append(row)
                chosen_rows.append(base)
        selected=pd.DataFrame(chosen_rows);selected_spectra=pd.DataFrame(chosen_spectra,columns=['selection_id']+features)
        selected.to_csv(selected_path,index=False);selected_spectra.to_csv(spectra_path,index=False,compression='gzip')
        description.to_csv(diagnostics_path,index=False,compression='gzip')
        meta={'signature':signature,'beer':beer_id,'single_target_max_abs_error':source_error,'cache_sha256':source_cache,
              'n_selected':len(selected_spectra),'n_primary_feasible':int(description.primary_feasible.sum()),
              'output_sha256':{p.name:sha(p) for p in [selected_path,spectra_path,diagnostics_path]}}
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
        summaries.append(selected);spectra.append(selected_spectra);diagnostics.append(description)
        max_error=max(max_error,source_error);cache_hashes.update(source_cache)
        print(f'E205B {source_number+1}/11 source={beer_id} locked={len(selected_spectra)}/9 feasible={meta["n_primary_feasible"]}',flush=True)
    summary=pd.concat(summaries,ignore_index=True);spectra=pd.concat(spectra,ignore_index=True)
    diagnostic=pd.concat(diagnostics,ignore_index=True)
    outputs=[]
    for name,frame in [('q3_search_selection_E205B.csv',summary),('q3_locked_spectra_E205B.csv.gz',spectra),
                       ('q3_search_diagnostics_E205B.csv.gz',diagnostic)]:
        path=RESULT/name;frame.to_csv(path,index=False,compression='gzip' if name.endswith('.gz') else None);outputs.append(path)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during search')
    lock={'experiment_id':'E205B','status':'locked_before_model_family_audit','scope':config['scope'],
          'n_source_target_pairs':33,'n_selection_rows':len(summary),'n_locked_spectra':len(spectra),
          'n_search_points':len(diagnostic),'n_sources':11,'selection_rules':RULES,
          'primary_single_target_max_abs_error':max_error,'auditor_accessed':False,'q1_retrained':False,
          'nnls_refit':False,'gpu_used':False,'physical_safety_certified':False,'signature':signature,
          'input_sha256':hashes,'model_cache_sha256':cache_hashes,
          'checkpoint_metadata_sha256':{p.name:sha(p) for p in checkpoint.glob('beer_*.json')},
          'output_sha256':{p.name:sha(p) for p in outputs}}
    lock_path.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in lock.items() if not k.endswith('sha256')},ensure_ascii=False),flush=True)
    return lock


if __name__=='__main__':
    run()
