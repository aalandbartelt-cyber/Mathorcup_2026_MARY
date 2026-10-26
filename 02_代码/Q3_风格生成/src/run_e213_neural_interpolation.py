"""Fixed nested neural residual/curvature comparison; no sensory-model refit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import torch
import yaml

from neural_interpolation import ResidualInterpolator
from joint_generation import balanced_inner_splits
from run_e208_joint_generation import ROOT,CANONICAL,RESULT,METRICS,sha,metrics_for_scales
from run_e200_contract import validate_cv
from q1_interface import FrozenQ1V5OOF

CONFIG=Path(__file__).resolve().parents[1]/'configs/E213_neural_interpolation.yaml'
METHODS=['pca_noise','neural_noise','neural_curve','neural_selected']
ALL_METHODS=METHODS+['style_copula_shrunk','raw_mixture']
EXPECTED_MODEL={'hidden_dim':32,'latent_dim':8,'train_steps':200,'learning_rate':.003,'weight_decay':.01,
    'denoising_std':.05,'gradient_clip':1.,'winsor_quantiles':[.01,.99],'covariance_pseudocount':20,
    'residual_noise_strength':.5,'curvature_strength':.5,'neighbors':3}


def validate_config(config):
    expected={'experiment':'E213','target_styles':['Tripel','Kriek','Lambic'],'n_repeats':10,'n_folds':3,
        'seeds':[20261025,20261026,20261027,20261028,20261029],'inner_folds':2,
        'inner_seeds':[20261026,20261027,20261028],'inner_split_seed':20261026,
        'inner_spread_log_weight':.25,'training_seed':20261026,'device':'cuda','methods':METHODS,
        'mixture_probabilities':[.25,.5,.75],'q1_retrained':False,'nnls_refit':False,'kaggle_used_for_training':False}
    if any(config.get(k)!=v for k,v in expected.items()) or config.get('model')!=EXPECTED_MODEL:
        raise ValueError('Registered E213 data/training/nesting protocol changed')


def choices_from_table(table,config):
    means=table.groupby(['style','family','alpha','beta'],as_index=False).selection_loss.mean();choices={}
    for style in config['target_styles']:
        choices[style]={}
        for method in METHODS:
            family='pca' if method=='pca_noise' else 'neural'
            block=means[means['style'].eq(style)&means.family.eq(family)]
            if method!='neural_selected':block=block[block.beta.eq(.5 if method=='neural_curve' else 0.)]
            best=block.sort_values(['selection_loss','beta','alpha'],kind='mergesort').iloc[0]
            if not np.isfinite(best.selection_loss):raise ValueError('No finite training-only choice')
            choices[style][method]={'alpha':float(best.alpha),'beta':float(best.beta),'family':family}
    return choices


def select_inside_training(raw,styles,config,*,train_seed):
    rows=[];histories=[]
    for inner,(train,valid) in enumerate(balanced_inner_splits(styles,seed=config['inner_split_seed'])):
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        model=ResidualInterpolator().fit(tx,styles[train],config['model'],target_styles=config['target_styles'],
            train_seed=train_seed+inner+1,device=config['device'])
        history=pd.DataFrame(model.history_);history['inner_fold']=inner;history['training_seed']=train_seed+inner+1
        histories.append(history)
        for si,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            if min(len(real),len(local))<2:raise ValueError('Insufficient balanced inner style rows')
            for family,beta in [('pca',0.),('neural',0.),('neural',.5)]:
                for alpha in config['mixture_probabilities']:
                    losses=[];clips=[]
                    for base_seed in config['inner_seeds']:
                        seed=int(base_seed)+inner*100+si*10
                        sample=model.generate(style,len(real),family,alpha=alpha,beta=beta,gamma=.5,seed=seed)
                        metrics=metrics_for_scales(real,sample['values'],tx,local);m=metrics['robust'];r=metrics['raw']
                        value=m['mean_w1']+m['energy_v']+config['inner_spread_log_weight']*m['log_spread_error']
                        losses.append(np.inf if r['near_duplicate_fraction']>0 or r['collapsed'] else value)
                        clips.append(float(sample['local_preclip_cell_fraction'].mean()))
                    rows.append({'inner_fold':inner,'style':style,'family':family,'alpha':alpha,'beta':beta,
                        'selection_loss':float(np.mean(losses)),'local_cell_clipping':float(np.mean(clips)),
                        'n_inner_training_style':len(local),'n_inner_validation_style':len(real)})
    table=pd.DataFrame(rows);chosen=choices_from_table(table,config)
    for method in METHODS:
        table['selected_'+method]=[family==chosen[s][method]['family'] and alpha==chosen[s][method]['alpha']
            and beta==chosen[s][method]['beta'] for s,family,alpha,beta in zip(table['style'],table.family,table.alpha,table.beta)]
    return chosen,table,pd.concat(histories,ignore_index=True)


def load_inputs():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config);q1=FrozenQ1V5OOF(CANONICAL)
    attachment=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv'
    chem=pd.read_csv(attachment);splits=pd.read_csv(split_path);validate_cv(chem,splits,n_repeats=10,n_folds=3)
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']]
    if len(chem)!=250 or len(features)!=231:raise ValueError('Frozen official chemistry contract changed')
    src=Path(__file__).parent
    files=[CONFIG,Path(__file__),src/'neural_interpolation.py',src/'gaussian_baselines.py',src/'empirical_generation.py',
        src/'joint_generation.py',src/'run_e208_joint_generation.py',src/'q3_evaluation.py',src/'audit_e202.py',
        src/'q1_interface.py',src/'run_e200_contract.py',attachment,split_path,
        RESULT/'q3_run_manifest_E211.json',RESULT/'q3_e211_audit_manifest.json',RESULT/'q3_distribution_metrics_E211.csv']
    for experiment in ['E202','E211']:
        for repeat in range(10):
            for fold in range(3):
                slot=RESULT/f'q3_checkpoints_{experiment}/R{repeat:02d}F{fold}'
                meta=json.loads((slot/'manifest.json').read_text())
                if meta['files']['generated.csv.gz']!=sha(slot/'generated.csv.gz'):raise ValueError('Frozen spectra changed')
                files.extend([slot/'manifest.json',slot/'generated.csv.gz'])
    hashes={str(p.resolve()):sha(p) for p in files}
    signature={'input_sha256':hashes,'q1_final_oof_sha256':q1.signature}
    return config,chem,splits,features,hashes,signature


def write_frame(slot,name,frame):
    path=slot/name;frame.to_csv(path,index=False,compression='gzip' if name.endswith('.gz') else None);return path


def run(max_folds=None):
    manifest_path=RESULT/'q3_run_manifest_E213.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous.get('status')!='complete':raise ValueError('Unexpected incomplete final manifest')
        for p,h in previous['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Completed E213 input changed')
        for n,h in previous['output_sha256'].items():
            if sha(RESULT/n)!=h:raise ValueError('Completed E213 output changed')
        print('E213 complete; frozen outputs verified without rewriting',flush=True);return previous
    config,chem,splits,features,hashes,signature=load_inputs()
    raw=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    baseline=pd.read_csv(RESULT/'q3_distribution_metrics_E211.csv')
    baseline=baseline[baseline.method.isin(['style_copula_shrunk','raw_mixture'])].copy()
    checkpoint=RESULT/'q3_checkpoints_E213';checkpoint.mkdir(exist_ok=True)
    tables={k:[] for k in ['metrics.csv','inner_selection.csv','training_history.csv','residual_diagnostics.csv','clipping.csv','unclipped_metrics.csv']}
    completed=[]
    for number,((repeat,fold),group) in enumerate(splits.groupby(['repeat','fold'],sort=True),1):
        if max_folds is not None and number>max_folds:break
        repeat=int(repeat);fold=int(fold);name=f'R{repeat:02d}F{fold}';slot=checkpoint/name;slot.mkdir(exist_ok=True)
        commit=slot/'manifest.json';valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(raw)),valid)
        if commit.exists():
            meta=json.loads(commit.read_text())
            if meta['signature']!=signature:raise ValueError('E213 checkpoint identity changed')
            for n,h in meta['files'].items():
                if sha(slot/n)!=h:raise ValueError('E213 checkpoint damaged')
            for n in tables:tables[n].append(pd.read_csv(slot/n))
            completed.append(name);print(f'E213 {number}/30 {name} cached=True',flush=True);continue
        for experiment in ['E202','E211']:
            old=json.loads((RESULT/f'q3_checkpoints_{experiment}/{name}/manifest.json').read_text())
            if old['train_beer']!=chem.beer.iloc[train].astype(int).tolist() or old['validation_beer']!=chem.beer.iloc[valid].astype(int).tolist():
                raise ValueError('Baseline outer identities differ')
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        training_seed=config['training_seed']+repeat*1000+fold*10
        chosen,selection,history=select_inside_training(raw[train],styles[train],config,train_seed=training_seed)
        model=ResidualInterpolator().fit(tx,styles[train],config['model'],target_styles=config['target_styles'],
            train_seed=training_seed,device=config['device'])
        outer_history=pd.DataFrame(model.history_);outer_history['inner_fold']=-1;outer_history['training_seed']=training_seed
        history=pd.concat([history,outer_history],ignore_index=True)
        diagnostics=pd.DataFrame(model.residual_diagnostics_)
        for frame in [selection,history,diagnostics]:frame.insert(0,'fold',fold);frame.insert(0,'repeat',repeat)
        rows=[];unclipped=[];samples=[];clipping=[];beers=chem.beer.iloc[train].to_numpy(int)
        for si,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            for seed_index,base_seed in enumerate(config['seeds']):
                seed=int(base_seed)+repeat*1000+fold*100+si*10
                for method in METHODS:
                    choice=chosen[style][method];alpha=choice['alpha'];beta=choice['beta'];family=choice['family']
                    sample=model.generate(style,len(real),family,alpha=alpha,beta=beta,gamma=.5,seed=seed)
                    for target,values in [(rows,sample['values']),(unclipped,sample['unclipped_values'])]:
                        for scope,m in metrics_for_scales(real,values,tx,local).items():
                            target.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'alpha':alpha,'beta':beta,
                                'seed_index':seed_index,'metric_scale':scope,'scope':'outer_validation_evaluation',**m})
                    clipping.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'seed_index':seed_index,
                        'local_cell_clipping':float(sample['local_preclip_cell_fraction'].mean()),
                        'local_row_clipping':float(sample['local_preclip_row'].mean()),
                        'generated_cell_clipping':float(np.mean(sample['local_preclip_cell_fraction']*sample['local_selected']))})
                    frame=pd.DataFrame(sample['values'],columns=features)
                    metadata={'repeat':repeat,'fold':fold,'method':method,'style':style,'seed_index':seed_index,
                        'sample_position':np.arange(len(real)),'alpha':alpha,'beta':beta,'family':family,
                        'local_selected':sample['local_selected'],'anchor_a_beer':beers[sample['anchor_a']],
                        'anchor_b_beer':beers[sample['anchor_b']],'interpolation_t':sample['interpolation_t'],
                        'local_preclip_cell_fraction':sample['local_preclip_cell_fraction'],'local_preclip_row':sample['local_preclip_row'],
                        **dict(zip(['selector_seed','copula_seed','local_seed','noise_seed'],sample['stream_seeds'])),
                        'scope':'outer_training_generated_unprojected'}
                    for k,v in reversed(list(metadata.items())):frame.insert(0,k,v)
                    samples.append(frame)
        metrics=pd.concat([pd.DataFrame(rows),baseline[baseline.repeat.eq(repeat)&baseline.fold.eq(fold)]],ignore_index=True)
        frames={'metrics.csv':metrics,'inner_selection.csv':selection,'training_history.csv':history,
            'residual_diagnostics.csv':diagnostics,'clipping.csv':pd.DataFrame(clipping),'unclipped_metrics.csv':pd.DataFrame(unclipped)}
        paths=[write_frame(slot,n,f) for n,f in frames.items()]
        paths.append(write_frame(slot,'generated.csv.gz',pd.concat(samples,ignore_index=True)))
        state=slot/'model_state.npz';np.savez_compressed(state,feature_names=np.array(features,dtype=str),
            imputer_statistics=imputer.statistics_,**model.state_arrays());paths.append(state)
        meta={'signature':signature,'train_beer':chem.beer.iloc[train].astype(int).tolist(),
            'validation_beer':chem.beer.iloc[valid].astype(int).tolist(),'selected':chosen,
            'training_seed':training_seed,'training_steps':config['model']['train_steps'],'n_parameters':model.n_parameters_,
            'outer_training_seconds':model.training_seconds_,'gpu_used':model.gpu_used_,
            'files':{p.name:sha(p) for p in paths}}
        commit.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');completed.append(name)
        for n,f in frames.items():tables[n].append(f)
        (RESULT/'q3_progress_E213.json').write_text(json.dumps({'status':'running','completed_folds':len(completed),'last_fold':name},indent=2)+'\n')
        print(f'E213 {number}/30 {name} selected={chosen} network_seconds={model.training_seconds_:.3f}',flush=True)
    if len(completed)!=30:return {'status':'checkpoint_only','completed_folds':len(completed)}
    combined={n:pd.concat(frames,ignore_index=True) for n,frames in tables.items()}
    repeat_metrics=combined['metrics.csv'].groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    names={'metrics.csv':'q3_distribution_metrics_E213.csv','inner_selection.csv':'q3_inner_selection_E213.csv',
        'training_history.csv':'q3_training_history_E213.csv','residual_diagnostics.csv':'q3_residual_diagnostics_E213.csv',
        'clipping.csv':'q3_clipping_E213.csv','unclipped_metrics.csv':'q3_unclipped_metrics_E213.csv'}
    outputs=[write_frame(RESULT,names[n],frame) for n,frame in combined.items()]
    outputs.append(write_frame(RESULT,'q3_metrics_by_repeat_E213.csv',repeat_metrics))
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Frozen input changed during E213')
    manifest={'experiment_id':'E213','status':'complete','n_outer_folds':30,'n_repeats':10,'n_seeds':5,
        'n_metric_rows':len(combined['metrics.csv']),'n_inner_selection_rows':len(combined['inner_selection.csv']),
        'methods':ALL_METHODS,'signature':signature,'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in outputs},
        'checkpoint_manifest_sha256':{n:sha(checkpoint/n/'manifest.json') for n in completed},
        'q1_retrained':False,'nnls_refit':False,'gpu_used':True,'kaggle_used_for_training':False,
        'torch_version':torch.__version__,'gpu_name':torch.cuda.get_device_name(0),
        'scope':'exploratory_nested_outer_validation_not_independent_confirmation',
        'random_stream_policy':'independent_four_way','generated_spectra_projected_or_filtered':False,
        'local_generator_training_style_range_clipping_recorded':True}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (RESULT/'q3_progress_E213.json').write_text(json.dumps({'status':'complete','completed_folds':30},indent=2)+'\n')
    print('E213 status complete 30/30',flush=True);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-folds',type=int);args=parser.parse_args()
    if args.max_folds is not None and not 1<=args.max_folds<=30:parser.error('--max-folds must be 1..30')
    run(args.max_folds)
