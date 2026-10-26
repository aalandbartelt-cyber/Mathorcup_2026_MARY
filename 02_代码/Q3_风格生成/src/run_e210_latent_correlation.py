"""Nested comparison of tuned rank correlation against the frozen Copula."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml

from run_e208_joint_generation import ROOT,CANONICAL,RESULT,METRICS,metrics_for_scales,sha
from joint_generation import balanced_inner_splits
from latent_correlation import LatentCorrelationGenerator
from gaussian_baselines import StyleGaussianGenerator
from run_e200_contract import validate_cv
from q1_interface import FrozenQ1V5OOF

CONFIG=Path(__file__).resolve().parents[1]/'configs/E210_latent_correlation.yaml'
RHOS=[1.,.5,.25,.1,.05,0.]
CANDIDATES=['copula']+[f'rho_{int(r*100):03d}' for r in RHOS]
RHO_BY_NAME=dict(zip(CANDIDATES[1:],RHOS))


def validate_config(config):
    expected={'experiment':'E210','target_styles':['Tripel','Kriek','Lambic'],'n_repeats':10,'n_folds':3,
        'inner_folds':2,'inner_spread_log_weight':.25,'correlation_shrinkages':[0.,.05,.1,.25,.5,1.],
        'methods':['latent_selected','latent_00','latent_50','latent_100'],
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    if any(config.get(k)!=v for k,v in expected.items()) or len(config.get('seeds',[]))!=5 or len(config.get('inner_seeds',[]))!=3:
        raise ValueError('E210 registered protocol changed')


def select_inside_training(raw,styles,config):
    rows=[]
    for inner,(train,valid) in enumerate(balanced_inner_splits(styles,seed=config['inner_split_seed'])):
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        latent=LatentCorrelationGenerator().fit(tx,styles[train],target_styles=config['target_styles'])
        for si,style in enumerate(config['target_styles']):
            local=tx[styles[train]==style];real=vx[styles[valid]==style]
            if min(len(local),len(real))<2:raise ValueError('Inner target style too small')
            copula=StyleGaussianGenerator().fit(local,np.repeat(style,len(local)))
            for candidate in CANDIDATES:
                losses=[]
                for base_seed in config['inner_seeds']:
                    seed=base_seed+inner*100+si*10
                    sample=(copula.generate(style,len(real),method='style_copula_shrunk',seed=seed)['values']
                        if candidate=='copula' else latent.generate(style,len(real),shrinkage=RHO_BY_NAME[candidate],seed=seed))
                    metrics=metrics_for_scales(real,sample,tx,local);m=metrics['robust'];r=metrics['raw']
                    loss=m['mean_w1']+m['energy_v']+.25*m['log_spread_error']
                    losses.append(np.inf if r['near_duplicate_fraction']>0 or r['collapsed'] else loss)
                rows.append({'inner_fold':inner,'style':style,'candidate':candidate,'selection_loss':float(np.mean(losses)),
                    'n_inner_training_style':len(local),'n_inner_validation_style':len(real)})
    table=pd.DataFrame(rows);scores=table.groupby(['style','candidate'],as_index=False).selection_loss.mean()
    scores['order']=scores.candidate.map({c:i for i,c in enumerate(CANDIDATES)})
    selected={}
    for style in config['target_styles']:
        best=scores[scores['style']==style].sort_values(['selection_loss','order'],kind='mergesort').iloc[0]
        if not np.isfinite(best.selection_loss):raise ValueError('No finite inner selected candidate')
        selected[style]=best.candidate
    table['selected']=[c==selected[s] for s,c in zip(table['style'],table.candidate)]
    return selected,table


def run(max_folds=None):
    config=yaml.safe_load(CONFIG.read_text());validate_config(config);q1=FrozenQ1V5OOF(CANONICAL)
    attachment=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv';chem=pd.read_csv(attachment);splits=pd.read_csv(split_path)
    validate_cv(chem,splits,n_repeats=10,n_folds=3)
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']]
    if len(features)!=231 or len(chem)!=250:raise ValueError('Frozen data contract changed')
    x=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    source=Path(__file__).parent
    files=[CONFIG,Path(__file__),source/'latent_correlation.py',source/'joint_generation.py',source/'gaussian_baselines.py',
        source/'run_e208_joint_generation.py',source/'q3_evaluation.py',source/'audit_e202.py',attachment,split_path,
        RESULT/'q3_e208_audit_manifest.json',RESULT/'q3_distribution_metrics_E208.csv']
    for repeat in range(10):
        for fold in range(3):
            slot=RESULT/f'q3_checkpoints_E202/R{repeat:02d}F{fold}'
            for n in ['manifest.json','generated.csv.gz']:files.append(slot/n)
    hashes={str(p.resolve()):sha(p) for p in files}
    signature={'input_sha256':hashes,'q1_final_oof_sha256':q1.signature}
    baseline=pd.read_csv(RESULT/'q3_distribution_metrics_E208.csv')
    baseline=baseline[baseline.method=='style_copula_shrunk'].copy()
    checkpoint=RESULT/'q3_checkpoints_E210';checkpoint.mkdir(exist_ok=True)
    all_metrics=[];all_selections=[];completed=[]
    for number,((repeat,fold),group) in enumerate(splits.groupby(['repeat','fold'],sort=True),1):
        if max_folds is not None and number>max_folds:break
        repeat=int(repeat);fold=int(fold);name=f'R{repeat:02d}F{fold}';slot=checkpoint/name;slot.mkdir(exist_ok=True)
        valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
        commit=slot/'manifest.json'
        if commit.exists():
            meta=json.loads(commit.read_text())
            if meta['signature']!=signature:raise ValueError('E210 checkpoint identity changed')
            for n,h in meta['files'].items():
                if sha(slot/n)!=h:raise ValueError('E210 checkpoint damaged')
            all_metrics.append(pd.read_csv(slot/'metrics.csv'));all_selections.append(pd.read_csv(slot/'inner_selection.csv'))
            completed.append(name);print(f'E210 {number}/30 {name} cached=True',flush=True);continue
        base_slot=RESULT/f'q3_checkpoints_E202/{name}';base_meta=json.loads((base_slot/'manifest.json').read_text())
        if (base_meta['files']['generated.csv.gz']!=sha(base_slot/'generated.csv.gz')
            or base_meta['train_beer']!=chem.beer.iloc[train].astype(int).tolist()
            or base_meta['validation_beer']!=chem.beer.iloc[valid].astype(int).tolist()):raise ValueError('Frozen Copula fold identity changed')
        frozen=pd.read_csv(base_slot/'generated.csv.gz');frozen=frozen[frozen.method=='style_copula_shrunk']
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(x[train]);vx=imputer.transform(x[valid])
        selected,selection=select_inside_training(x[train],styles[train],config)
        selection.insert(0,'fold',fold);selection.insert(0,'repeat',repeat)
        latent=LatentCorrelationGenerator().fit(tx,styles[train],target_styles=config['target_styles'])
        rows=[];generated=[]
        for si,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            for seed_index,base_seed in enumerate(config['seeds']):
                seed=base_seed+repeat*1000+fold*100+si*10
                for method in config['methods']:
                    actual=selected[style] if method=='latent_selected' else {'latent_00':'rho_000','latent_50':'rho_050','latent_100':'rho_100'}[method]
                    sample=(frozen[frozen['style'].eq(style)&frozen.seed_index.eq(seed_index)][features].to_numpy(float)
                        if actual=='copula' else latent.generate(style,len(real),shrinkage=RHO_BY_NAME[actual],seed=seed))
                    if len(sample)!=len(real):raise ValueError('Generated/validation style count mismatch')
                    for scale,m in metrics_for_scales(real,sample,tx,local).items():
                        rows.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'selected_candidate':actual,
                            'seed_index':seed_index,'metric_scale':scale,'scope':'outer_validation_evaluation',**m})
                    frame=pd.DataFrame(sample,columns=features)
                    for key,value in [('sample_position',np.arange(len(sample))),('seed_index',seed_index),('style',style),('method',method),
                        ('fold',fold),('repeat',repeat),('scope','outer_training_generated')]:frame.insert(0,key,value)
                    generated.append(frame)
        metrics=pd.concat([pd.DataFrame(rows),baseline[baseline.repeat.eq(repeat)&baseline.fold.eq(fold)]],ignore_index=True)
        paths=[]
        for filename,table in [('metrics.csv',metrics),('inner_selection.csv',selection),('generated.csv.gz',pd.concat(generated,ignore_index=True))]:
            path=slot/filename;table.to_csv(path,index=False,compression='gzip' if filename.endswith('.gz') else None);paths.append(path)
        meta={'signature':signature,'train_beer':chem.beer.iloc[train].astype(int).tolist(),
            'validation_beer':chem.beer.iloc[valid].astype(int).tolist(),'selected_candidates':selected,'files':{p.name:sha(p) for p in paths}}
        commit.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        completed.append(name);all_metrics.append(metrics);all_selections.append(selection)
        (RESULT/'q3_progress_E210.json').write_text(json.dumps({'status':'running','completed_folds':len(completed),'last_fold':name},indent=2)+'\n')
        print(f'E210 {number}/30 {name} selected={selected}',flush=True)
    if len(completed)!=30:return {'status':'checkpoint_only','completed_folds':len(completed)}
    table=pd.concat(all_metrics,ignore_index=True);selection=pd.concat(all_selections,ignore_index=True)
    by_repeat=table.groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    outputs=[]
    for filename,frame in [('q3_distribution_metrics_E210.csv',table),('q3_metrics_by_repeat_E210.csv',by_repeat),('q3_inner_selection_E210.csv',selection)]:
        path=RESULT/filename;frame.to_csv(path,index=False);outputs.append(path)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Frozen input changed during E210')
    manifest={'experiment_id':'E210','status':'complete','n_outer_folds':30,'n_repeats':10,'n_seeds':5,
        'n_metric_rows':len(table),'input_sha256':hashes,'signature':signature,'output_sha256':{p.name:sha(p) for p in outputs},
        'checkpoint_manifest_sha256':{name:sha(checkpoint/name/'manifest.json') for name in completed},
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False,'scope':'exploratory_nested_outer_validation_not_independent_confirmation'}
    (RESULT/'q3_run_manifest_E210.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (RESULT/'q3_progress_E210.json').write_text(json.dumps({'status':'complete','completed_folds':30},indent=2)+'\n')
    print('E210 status complete 30/30',flush=True);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-folds',type=int);args=parser.parse_args()
    if args.max_folds is not None and not 1<=args.max_folds<=30:parser.error('--max-folds must be 1..30')
    run(args.max_folds)
