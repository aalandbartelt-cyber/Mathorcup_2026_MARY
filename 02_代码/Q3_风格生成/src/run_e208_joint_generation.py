"""Nested joint-distribution candidate comparison with frozen baseline reuse."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml

from q1_interface import FrozenQ1V5OOF
from run_e200_contract import validate_cv
from audit_e202 import training_winsorize
from joint_generation import JointStyleGenerator,CANDIDATES,diversity_diagnostics,balanced_inner_splits

ROOT=Path(__file__).resolve().parents[3];CANONICAL=Path('/home/mary/mathorcup')
RESULT=ROOT/'03_结果/Q3_风格生成'
CONFIG=Path(__file__).resolve().parents[1]/'configs/E208_joint_generation.yaml'
METRICS=['mean_w1','energy_v','coverage','near_duplicate_fraction','spearman_pair_rmse',
         'pairwise_spread_ratio','log_spread_error','density_k1','collapsed']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if (config.get('target_styles')!=['Tripel','Kriek','Lambic'] or config.get('n_repeats')!=10
        or config.get('n_folds')!=3 or config.get('inner_folds')!=2
        or config.get('methods')!=['beta_linear','beta_step','mixture_selected','joint_selected']
        or config.get('selection_candidates')!=CANDIDATES or len(config.get('seeds',[]))!=5
        or len(config.get('inner_seeds',[]))!=3 or config.get('inner_spread_log_weight')!=.25
        or any(config.get(k) is not False for k in ['q1_retrained','nnls_refit','gpu_used'])):
        raise ValueError('E208 registered methods/folds/selection policy changed')


def metrics_for_scales(real,generated,train,local):
    scale=np.std(train,axis=0);scale=np.where(scale>1e-12,scale,1.)
    raw=diversity_diagnostics(real,generated,local,scale)
    clipped_train,(cr,cg),robust_scale,(lo,hi)=training_winsorize(train,real,generated)
    robust=diversity_diagnostics(cr,cg,np.clip(local,lo,hi),robust_scale)
    for value in (raw,robust):value.pop('w1_by_feature');value.pop('scope')
    return {'raw':raw,'robust':robust}


def select_inside_training(raw_train,styles,config):
    scores=[];splits=balanced_inner_splits(styles,seed=int(config['inner_split_seed']))
    for inner_fold,(train,valid) in enumerate(splits):
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        tx=imputer.fit_transform(raw_train[train]);vx=imputer.transform(raw_train[valid])
        scale=np.std(tx,axis=0);scale=np.where(scale>1e-12,scale,1.)
        generator=JointStyleGenerator().fit(tx,styles[train],scale=scale,tie_seed=config['tie_seed'],
                                           target_styles=config['target_styles'])
        for style_index,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            if min(len(real),len(local))<2:raise ValueError('Insufficient target style rows in balanced inner split')
            for candidate in CANDIDATES:
                losses=[];copies=[]
                for seed_index,base_seed in enumerate(config['inner_seeds']):
                    seed=int(base_seed)+inner_fold*100+style_index*10
                    sample=generator.generate(style,len(real),candidate,seed=seed)
                    metrics=metrics_for_scales(real,sample,tx,local)['robust']
                    # Raw copying is checked separately: robust clipping must not
                    # turn the copying test into a comparison of clipped rows.
                    raw_metrics=metrics_for_scales(real,sample,tx,local)['raw']
                    loss=metrics['mean_w1']+metrics['energy_v']+.25*metrics['log_spread_error']
                    if raw_metrics['near_duplicate_fraction']>0 or raw_metrics['collapsed']:loss=np.inf
                    losses.append(loss);copies.append(raw_metrics['near_duplicate_fraction'])
                scores.append({'inner_fold':inner_fold,'style':style,'candidate':candidate,
                               'selection_loss':float(np.mean(losses)),'copy_fraction':float(np.mean(copies)),
                               'n_inner_training_style':len(local),'n_inner_validation_style':len(real)})
    table=pd.DataFrame(scores)
    means=table.groupby(['style','candidate'],as_index=False).selection_loss.mean()
    means['complexity_order']=means.candidate.map({name:i for i,name in enumerate(CANDIDATES)})
    selected={}
    for style in config['target_styles']:
        subset=means[means['style']==style].sort_values(['selection_loss','complexity_order'],kind='mergesort')
        mixed=subset[subset.candidate.str.startswith('mix_')]
        if not np.isfinite(subset.iloc[0].selection_loss) or not np.isfinite(mixed.iloc[0].selection_loss):
            raise ValueError('No finite inner-selected model')
        selected[style]={'joint_selected':subset.iloc[0].candidate,'mixture_selected':mixed.iloc[0].candidate}
    table['selected_joint']=[candidate==selected[style]['joint_selected'] for style,candidate in zip(table['style'],table.candidate)]
    table['selected_mixture']=[candidate==selected[style]['mixture_selected'] for style,candidate in zip(table['style'],table.candidate)]
    return selected,table


def run(max_folds=None):
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    q1=FrozenQ1V5OOF(CANONICAL)
    attachment=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv'
    chem=pd.read_csv(attachment);splits=pd.read_csv(split_path);validate_cv(chem,splits,n_repeats=10,n_folds=3)
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']]
    if len(features)!=231 or len(chem)!=250:raise ValueError('Frozen Q3 feature/sample contract changed')
    raw=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    inputs=[CONFIG,Path(__file__),Path(__file__).with_name('joint_generation.py'),
        Path(__file__).with_name('gaussian_baselines.py'),Path(__file__).with_name('empirical_generation.py'),
        Path(__file__).with_name('q3_evaluation.py'),Path(__file__).with_name('audit_e202.py'),attachment,split_path,
        RESULT/'q3_run_manifest_E201.json',RESULT/'q3_run_manifest_E202.json']
    for experiment in ['E201','E202']:
        base=json.loads((RESULT/f'q3_run_manifest_{experiment}.json').read_text())
        if base.get('status')!='complete' or base.get('n_outer_folds')!=30:raise ValueError('Complete frozen baselines required')
        for repeat in range(10):
            for fold in range(3):
                slot=RESULT/f'q3_checkpoints_{experiment}/R{repeat:02d}F{fold}'
                manifest=slot/'manifest.json';record=json.loads(manifest.read_text())
                inputs.extend([manifest,slot/'generated.csv.gz'])
                if record['files']['generated.csv.gz']!=sha(slot/'generated.csv.gz'):
                    raise ValueError('Frozen baseline generated samples changed')
    hashes={str(p.resolve()):sha(p) for p in inputs}
    signature={'input_sha256':hashes,'q1_final_oof_sha256':q1.signature}
    checkpoint=RESULT/'q3_checkpoints_E208';checkpoint.mkdir(exist_ok=True)
    all_metrics=[];all_selections=[];completed=[];started=time.perf_counter()
    for number,((repeat,fold),group) in enumerate(splits.groupby(['repeat','fold'],sort=True),1):
        if max_folds is not None and number>max_folds:break
        repeat=int(repeat);fold=int(fold);name=f'R{repeat:02d}F{fold}';slot=checkpoint/name;slot.mkdir(exist_ok=True)
        commit=slot/'manifest.json';valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(raw)),valid)
        if commit.exists():
            meta=json.loads(commit.read_text())
            if meta['signature']!=signature:raise ValueError('E208 checkpoint source/input identity changed')
            for n,h in meta['files'].items():
                if sha(slot/n)!=h:raise ValueError('E208 checkpoint damaged')
            all_metrics.append(pd.read_csv(slot/'metrics.csv'));all_selections.append(pd.read_csv(slot/'inner_selection.csv'))
            completed.append(name);print(f'E208 {number}/30 {name} cached=True',flush=True);continue
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        selected,selection=select_inside_training(raw[train],styles[train],config)
        selection.insert(0,'fold',fold);selection.insert(0,'repeat',repeat)
        scale=np.std(tx,axis=0);scale=np.where(scale>1e-12,scale,1.)
        generator=JointStyleGenerator().fit(tx,styles[train],scale=scale,tie_seed=config['tie_seed'],target_styles=config['target_styles'])
        rows=[];generated=[]
        for style_index,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            for seed_index,base_seed in enumerate(config['seeds']):
                seed=int(base_seed)+repeat*1000+fold*100+style_index*10
                for method in config['methods']:
                    actual=selected[style][method] if method in ['joint_selected','mixture_selected'] else method
                    values=generator.generate(style,len(real),actual,seed=seed)
                    for scope,metrics in metrics_for_scales(real,values,tx,local).items():
                        rows.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'selected_candidate':actual,
                                     'seed_index':seed_index,'metric_scale':scope,'scope':'outer_validation_evaluation',**metrics})
                    frame=pd.DataFrame(values,columns=features)
                    for key,value in [('sample_position',np.arange(len(real))),('seed_index',seed_index),('style',style),
                                      ('method',method),('fold',fold),('repeat',repeat),('scope','outer_training_generated')]:frame.insert(0,key,value)
                    generated.append(frame)
        # Re-evaluate immutable baseline samples with the exact same metrics.
        for experiment in ['E201','E202']:
            base_path=RESULT/f'q3_checkpoints_{experiment}/{name}'
            record=json.loads((base_path/'manifest.json').read_text())
            if record['train_beer']!=chem.beer.iloc[train].astype(int).tolist() or record['validation_beer']!=chem.beer.iloc[valid].astype(int).tolist():
                raise ValueError('Baseline fold identities differ from E208')
            base=pd.read_csv(base_path/'generated.csv.gz')
            for (method,style,seed_index),sample in base.groupby(['method','style','seed_index'],sort=True):
                real=vx[styles[valid]==style];local=tx[styles[train]==style];values=sample[features].to_numpy(float)
                for scope,metrics in metrics_for_scales(real,values,tx,local).items():
                    rows.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'selected_candidate':method,
                        'seed_index':seed_index,'metric_scale':scope,'scope':'frozen_baseline_samples_re_evaluated',**metrics})
        metrics_path=slot/'metrics.csv';gen_path=slot/'generated.csv.gz';selection_path=slot/'inner_selection.csv'
        table=pd.DataFrame(rows);table.to_csv(metrics_path,index=False);selection.to_csv(selection_path,index=False)
        pd.concat(generated,ignore_index=True).to_csv(gen_path,index=False,compression='gzip')
        meta={'signature':signature,'train_beer':chem.beer.iloc[train].astype(int).tolist(),
            'validation_beer':chem.beer.iloc[valid].astype(int).tolist(),'selected_candidates':selected,
            'files':{p.name:sha(p) for p in [metrics_path,gen_path,selection_path]}}
        commit.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        all_metrics.append(table);all_selections.append(selection);completed.append(name)
        (RESULT/'q3_progress_E208.json').write_text(json.dumps({'status':'running','completed_folds':len(completed),
            'last_fold':name,'elapsed_seconds':time.perf_counter()-started},indent=2)+'\n')
        print(f'E208 {number}/30 {name} selections={selected}',flush=True)
    if len(completed)<30:return {'status':'checkpoint_only','completed_folds':len(completed)}
    combined=pd.concat(all_metrics,ignore_index=True);selections=pd.concat(all_selections,ignore_index=True)
    repeat_metrics=combined.groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    summary=repeat_metrics.groupby(['method','style','metric_scale'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    outputs=[]
    for n,f in [('q3_distribution_metrics_E208.csv',combined),('q3_metrics_by_repeat_E208.csv',repeat_metrics),
        ('q3_baseline_summary_E208.csv',summary),('q3_inner_selection_E208.csv',selections)]:
        p=RESULT/n;f.to_csv(p,index=False);outputs.append(p)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during E208')
    manifest={'experiment_id':'E208','status':'complete','n_outer_folds':30,'n_repeats':10,'n_seeds':5,
        'n_metric_rows':len(combined),'methods':config['methods'],'signature':signature,
        'scope':'nested_outer_validation_training_only_selection','q1_retrained':False,'nnls_refit':False,'gpu_used':False,
        'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in outputs},
        'checkpoint_manifest_sha256':{name:sha(checkpoint/name/'manifest.json') for name in completed},
        'elapsed_seconds':time.perf_counter()-started}
    (RESULT/'q3_run_manifest_E208.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (RESULT/'q3_progress_E208.json').write_text(json.dumps({'status':'complete','completed_folds':30,'last_fold':completed[-1]},indent=2)+'\n')
    print(json.dumps({'status':'complete','n_metric_rows':len(combined)},ensure_ascii=False),flush=True);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-folds',type=int);args=parser.parse_args()
    if args.max_folds is not None and not 1<=args.max_folds<=30:parser.error('--max-folds must be 1..30')
    run(args.max_folds)
