"""Reuse frozen fold predictors for descriptive profiles and proxy diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[3]
CANONICAL=Path('/home/mary/mathorcup')
Q2=CANONICAL/'.worktrees/q2-evidence-opt'
Q2_SRC=Q2/'02_代码/Q2_关键化合物/src'
sys.path.insert(0,str(Q2_SRC))
sys.path.insert(0,str(CANONICAL/'02_代码/Q1_感官评分/src'))
from q1_final_adapter import FrozenQ1FinalModel
from selection_models import fit_fold_preprocessor, build_quadratic_penalty_rows, fit_multi_target_with_penalty_rows
from baseline_models import empirical_cdf_grade
from sensory_audit import sha, load_read_only_nnls, migration_model_input, support_audit,verify_archived_prediction

RESULT=ROOT/'03_结果/Q3_风格生成'
CONFIG=Path(__file__).resolve().parents[1]/'configs/E205_sensory_audit.yaml'
Q2_RESULT=Q2/'03_结果/Q2_关键化合物'
KEYS=['beer','target_style','scenario','strategy','lambda']


def validate_config(config):
    required={'q1_retrained':False,'nnls_refit':False,'hyperparameter_search':False,
              'gpu_used':False,'physical_safety_certified':False,'q2_top3_in_optimizer':False,
              'source_missing_mask':'preserve_for_migration_predictors',
              'proxy_target':'stale_hops','profile_samples_per_style':300,
              'proxy_positive_r2_repeats':7,'proxy_mean_r2_min':0.,
              'proxy_delta_repeat_quantile':.9,'proxy_numerical_tolerance':1e-8,
              'support_train_loo_quantile':.95,'support_max_outside_range_fraction':.05,
              'support_min_repeat_fraction':.9}
    for key,value in required.items():
        if config.get(key)!=value:raise ValueError(f'E205 registered policy changed: {key}')
    if config.get('target_styles')!=['Tripel','Kriek','Lambic']:
        raise ValueError('Target style contract changed')


def transform_elastic(processed, values):
    values=np.asarray(values,float)
    clipped=np.clip(values,processed.lower_bounds,processed.upper_bounds)
    filled=np.where(np.isnan(clipped),processed.medians,clipped)
    transformed=np.column_stack([filled,np.isnan(clipped[:,processed.missing_indicator_features]).astype(float)])
    return (transformed-processed.means)/processed.scales


def load_inputs(config):
    final=FrozenQ1FinalModel(CANONICAL)
    chemistry_path=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    chemistry=pd.read_csv(chemistry_path)
    x=chemistry.loc[:,final.feature_names];y=final.arrays['y_true']
    final.validate_data(x,pd.DataFrame(y,columns=final.target_codes),chemistry.beer,chemistry.tasting_category_fine)
    e204_path=RESULT/'q3_run_manifest_E204.json'
    e204=json.loads(e204_path.read_text());e204_audit=RESULT/'q3_e204_audit_manifest.json'
    if e204.get('status')!='complete' or json.loads(e204_audit.read_text()).get('status')!='complete':
        raise ValueError('Completed and audited E204 required')
    if e204['signature']['q1_final_oof_sha256']!=final.registry['oof_sha256']:
        raise ValueError('Q1 OOF differs from E204')
    for name,digest in e204['output_sha256'].items():
        if sha(RESULT/name)!=digest:raise ValueError(f'Frozen E204 hash changed: {name}')
    e115_path=Q2_RESULT/'q2_run_manifest_E115_final_nnls.json'
    e115=json.loads(e115_path.read_text())
    if (e115.get('status')!='complete' or e115.get('n_outer_folds')!=30
        or e115.get('adapter_signature')!=final.signature or e115.get('q1_retrained') is not False
        or e115.get('nnls_refit') is not False):
        raise ValueError('Completed identity-matching final NNLS attribution required')
    spectra_path=RESULT/'q3_migration_spectra_E204.csv.gz'
    spectra=pd.read_csv(spectra_path)
    summary=pd.read_csv(RESULT/'q3_migration_summary_E204.csv')
    if len(spectra)!=844 or len(summary)!=891 or spectra.duplicated(KEYS).any() or summary.duplicated(KEYS).any():
        raise ValueError('Frozen E204 candidate identities changed')
    beer_positions={int(v):i for i,v in enumerate(final.arrays['beer'])}
    positions=np.array([beer_positions[int(v)] for v in spectra.beer])
    migration_x=migration_model_input(spectra.loc[:,final.feature_names],positions,x)
    pool_path=RESULT/'q3_target_pool_E204.csv.gz';pool=pd.read_csv(pool_path)
    samples=[]
    for i,style in enumerate(config['target_styles']):
        block=pool[pool['style']==style]
        selected=np.random.default_rng(config['profile_seed']+i).choice(len(block),config['profile_samples_per_style'],replace=False)
        samples.append(block.iloc[selected].copy())
    generated=pd.concat(samples,ignore_index=True)
    parameters_path=Q2_RESULT/'q2_selected_parameters_E101_E104_E109_wide.csv'
    parameters=pd.read_csv(parameters_path)
    parameters=parameters[parameters.experiment_id=='E103'].copy()
    if len(parameters)!=30 or parameters.duplicated(['repeat','fold']).any():raise ValueError('ElasticNet fold parameters incomplete')
    auditor_oof_path=Q2_RESULT/'q2_oof_predictions_E101_E104_E109_wide.npz'
    with np.load(auditor_oof_path,allow_pickle=False) as a:
        if not np.array_equal(a['beer'],final.arrays['beer']) or tuple(a['target_codes'].astype(str))!=final.target_codes:
            raise ValueError('Auditor OOF identities differ from Q1')
        auditor_oof=a['E103_ElasticNet'].copy()
    files=[CONFIG,Path(__file__),Path(__file__).with_name('sensory_audit.py'),
           Path(__file__).with_name('summarize_e205.py'),chemistry_path,e204_path,e204_audit,e115_path,
           spectra_path,RESULT/'q3_migration_summary_E204.csv',pool_path,parameters_path,auditor_oof_path,
           final.registry_path,final.oof_path,final.parameter_path,final.config_path,
           Q2_SRC/'q1_final_adapter.py',Q2_SRC/'selection_models.py',
           CANONICAL/'02_代码/Q1_感官评分/src/baseline_models.py']
    for repeat in range(10):
        for fold in range(3):
            files.extend([Q2_RESULT/f'q2_nnls_cache_E115/R{repeat:02d}F{fold}.json',
                          RESULT/f'q3_checkpoints_E202/R{repeat:02d}F{fold}/generated.csv.gz'])
    input_hashes={str(p.resolve()):sha(p) for p in files}
    signature=hashlib.sha256(json.dumps(input_hashes,sort_keys=True).encode()).hexdigest()
    return dict(final=final,x=x,y=y,chemistry=chemistry,spectra=spectra,summary=summary,positions=positions,
                migration_x=migration_x,generated=generated,parameters=parameters,auditor_oof=auditor_oof,
                input_hashes=input_hashes,signature=signature)


def run(max_folds=None):
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    data=load_inputs(config);final=data['final'];x=data['x'];y=data['y'];checkpoint=RESULT/'q3_checkpoints_E205'
    checkpoint.mkdir(exist_ok=True);rows=[];completed=[]
    for repeat in range(10):
        for fold in range(3):
            if max_folds is not None and len(completed)>=max_folds:break
            started=time.perf_counter();name=f'R{repeat:02d}F{fold}'
            array_path=checkpoint/(name+'.npz');meta_path=checkpoint/(name+'.json')
            mask=final.arrays['fold_assignment'][repeat]==fold
            valid=np.flatnonzero(mask);train=np.flatnonzero(~mask)
            mi=np.flatnonzero(mask[data['positions']])
            outer_path=RESULT/f'q3_checkpoints_E202/{name}/generated.csv.gz'
            outergen=pd.read_csv(outer_path);outergen=outergen[outergen.method=='style_copula_shrunk'].reset_index(drop=True)
            reused=False
            if array_path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text())
                if (meta.get('signature')!=data['signature'] or meta.get('output_sha256')!=sha(array_path)
                    or meta.get('repeat')!=repeat or meta.get('fold')!=fold):
                    raise ValueError(f'Checkpoint signature/hash changed: {name}')
                reused=True
            else:
                model,cache_info=load_read_only_nnls(final,Q2_RESULT/'q2_nnls_cache_E115',repeat,fold,x.iloc[valid])
                processed=fit_fold_preprocessor(x.iloc[train],x.iloc[valid])
                parameter=data['parameters'].query('repeat==@repeat and fold==@fold').iloc[0]
                p=processed.x_train.shape[1]
                penalty=build_quadratic_penalty_rows(np.zeros((p,p)),n_samples=len(train),
                                                    lambda_l2=float(parameter.lambda_l2),lambda_graph=0.)
                auditor=fit_multi_target_with_penalty_rows(processed.x_train,y[train],penalty,
                                                         lambda_l1=float(parameter.lambda_l1))
                expected=data['auditor_oof'][repeat,valid]
                auditor_error=verify_archived_prediction(auditor.predict(processed.x_validation),expected)
                blocks=[x.iloc[valid].to_numpy(float),data['migration_x'][mi],
                        data['generated'].loc[:,final.feature_names].to_numpy(float),
                        outergen.loc[:,final.feature_names].to_numpy(float)]
                counts=np.cumsum([0]+[len(v) for v in blocks]);query=np.vstack(blocks)
                prediction=model.predict(query);grade=empirical_cdf_grade(y[train],prediction)
                audit_prediction=auditor.predict(transform_elastic(processed,query))
                support=support_audit(x.iloc[train],query,quantile=config['support_train_loo_quantile'],
                                      outside_fraction=config['support_max_outside_range_fraction'])
                arrays={'validation_index':valid,'migration_index':mi}
                for i,label in enumerate(['validation','migration','generated','outergen']):
                    region=slice(counts[i],counts[i+1])
                    arrays['primary_'+label]=prediction[region]
                    arrays['grade_'+label]=grade[region]
                    arrays['auditor_'+label]=audit_prediction[region]
                    arrays['supported_'+label]=support['supported'][region]
                    arrays['support_distance_'+label]=support['nearest_distance'][region]
                    arrays['outside_fraction_'+label]=support['outside_range_fraction'][region]
                tmp=array_path.with_suffix('.npz.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
                tmp.replace(array_path)
                meta={'signature':data['signature'],'repeat':repeat,'fold':fold,
                      'output_sha256':sha(array_path),'q1_cache':cache_info,
                      'auditor_reconstruction_max_abs_error':auditor_error,
                      'auditor_archive_precision':'float32_exact_after_cast',
                      'support_distance_cutoff':support['distance_cutoff'],
                      'n_validation':len(valid),'n_migration':len(mi),'n_descriptive_generated':len(blocks[2]),
                      'n_outer_training_generated':len(outergen),'q1_retrained':False,'nnls_refit':False,
                      'hyperparameter_search':False,'elapsed_seconds':time.perf_counter()-started}
                meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
            completed.append((repeat,fold));rows.append(meta)
            progress={'experiment_id':'E205','status':'running' if len(completed)<30 else 'checkpoints_complete',
                      'n_outer_folds':len(completed),'last_checkpoint':name,'signature':data['signature']}
            (RESULT/'q3_progress_E205.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(f'E205 {len(completed)}/30 {name} cached={reused} NNLS_error={meta["q1_cache"]["reconstruction_max_abs_error"]:.3g} auditor_error={meta["auditor_reconstruction_max_abs_error"]:.3g}',flush=True)
        if max_folds is not None and len(completed)>=max_folds:break
    if len(completed)<30:return progress
    from summarize_e205 import summarize
    outputs,decision=summarize(data,config,checkpoint)
    outputs.append(RESULT/'q3_progress_E205.json')
    for path,digest in data['input_hashes'].items():
        if sha(path)!=digest:raise ValueError(f'Frozen input changed during E205: {path}')
    manifest={'experiment_id':'E205','status':'complete','scope':config['scope'],'n_outer_folds':30,
              'n_repeats':10,'n_targets':50,'q1_retrained':False,'nnls_refit':False,'hyperparameter_search':False,
              'gpu_used':False,'physical_safety_certified':False,'signature':data['signature'],
              'input_sha256':data['input_hashes'],'q1_final_model':final.registry['final_model'],
              'q1_final_oof_sha256':final.registry['oof_sha256'],
              'checkpoint_sha256':{p.name:sha(p) for p in sorted(checkpoint.iterdir()) if p.suffix in ('.npz','.json')},
              'max_primary_reconstruction_error':max(r['q1_cache']['reconstruction_max_abs_error'] for r in rows),
              'max_auditor_reconstruction_error':max(r['auditor_reconstruction_max_abs_error'] for r in rows),
              'decision':decision,'output_sha256':{p.name:sha(p) for p in outputs}}
    (RESULT/'q3_run_manifest_E205.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','n_outer_folds':30,'decision':decision},ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-folds',type=int)
    args=parser.parse_args()
    if args.max_folds is not None and not 1<=args.max_folds<=30:parser.error('--max-folds must be 1..30')
    run(args.max_folds)
