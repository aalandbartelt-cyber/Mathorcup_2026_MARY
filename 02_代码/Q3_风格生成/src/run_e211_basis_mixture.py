"""Frozen nested comparison of raw, PCA and chemical-basis local mixtures."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
import yaml

from basis_mixture import ChemicalMixtureGenerator
from joint_generation import balanced_inner_splits
from run_e208_joint_generation import ROOT,CANONICAL,RESULT,METRICS,sha,metrics_for_scales
from run_e200_contract import validate_cv
from q1_interface import FrozenQ1V5OOF

CONFIG=Path(__file__).resolve().parents[1]/'configs/E211_basis_mixture.yaml'
GEOMETRIES={'raw_mixture':'raw','pca_mixture':'pca','basis_mixture':'basis'}
ALL_METHODS=list(GEOMETRIES)+['style_copula_shrunk','legacy_raw_mixture']
EXPECTED_GEOMETRY={'n_strict_fields':218,'winsor_quantiles':[.01,.99],'coordinate_dim':4,'prototype_count':8,
    'prototype_seed':20261026,'bandwidth_multipliers':[.5,1.],'neighbors':3,'pca_support_neighbors':6,
    'basis_zero_tolerance':1e-8,'stability_bootstraps':5,'stability_dimensions':[3,5],'stability_minimum_median_jaccard':.5}


def validate_config(config):
    expected={'experiment':'E211','target_styles':['Tripel','Kriek','Lambic'],'n_repeats':10,'n_folds':3,'inner_folds':2,
        'methods':list(GEOMETRIES),'mixture_probabilities':[.25,.5,.75],'inner_spread_log_weight':.25,
        'random_stream_policy':'independent_three_way','legacy_mixture_read_only':True,
        'fallback_order':['basis_with_increment','raw_if_base_rules','frozen_copula'],
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    if (any(config.get(k)!=v for k,v in expected.items()) or config.get('geometry')!=EXPECTED_GEOMETRY
        or len(config.get('seeds',[]))!=5 or len(config.get('inner_seeds',[]))!=3):
        raise ValueError('Registered E211 protocol changed')


def select_inside_training(raw,styles,indices,config):
    rows=[]
    for inner,(train,valid) in enumerate(balanced_inner_splits(styles,seed=config['inner_split_seed'])):
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        generator=ChemicalMixtureGenerator().fit(tx,styles[train],indices,config['geometry'],target_styles=config['target_styles'])
        for si,style in enumerate(config['target_styles']):
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            if min(len(real),len(local))<2:raise ValueError('Balanced inner style has fewer than two observations')
            for method,geometry in GEOMETRIES.items():
                for alpha in config['mixture_probabilities']:
                    losses=[];copies=[]
                    for base_seed in config['inner_seeds']:
                        seed=int(base_seed)+inner*100+si*10
                        sample=generator.generate(style,len(real),geometry,alpha=alpha,seed=seed)['values']
                        metrics=metrics_for_scales(real,sample,tx,local);m=metrics['robust'];r=metrics['raw']
                        loss=m['mean_w1']+m['energy_v']+.25*m['log_spread_error']
                        losses.append(np.inf if r['near_duplicate_fraction']>0 or r['collapsed'] else loss)
                        copies.append(r['near_duplicate_fraction'])
                    rows.append({'inner_fold':inner,'style':style,'method':method,'alpha':alpha,
                        'selection_loss':float(np.mean(losses)),'copy_fraction':float(np.mean(copies)),
                        'n_inner_training_style':len(local),'n_inner_validation_style':len(real)})
    table=pd.DataFrame(rows)
    means=table.groupby(['style','method','alpha'],as_index=False).selection_loss.mean();selected={}
    for style in config['target_styles']:
        selected[style]={}
        for method in GEOMETRIES:
            subset=means[means['style'].eq(style)&means.method.eq(method)].sort_values(['selection_loss','alpha'],kind='mergesort')
            if not np.isfinite(subset.iloc[0].selection_loss):raise ValueError('No finite training-only mixture choice')
            selected[style][method]=float(subset.iloc[0].alpha)
    table['selected']=[a==selected[s][m] for s,m,a in zip(table['style'],table.method,table.alpha)]
    return selected,table


def load_inputs():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config);q1=FrozenQ1V5OOF(CANONICAL)
    attachment=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv';dictionary=ROOT/'03_结果/Q2_关键化合物/q2_candidate_dictionary.csv'
    chem=pd.read_csv(attachment);splits=pd.read_csv(split_path);validate_cv(chem,splits,n_repeats=10,n_folds=3)
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']]
    candidate=pd.read_csv(dictionary);strict=candidate.loc[candidate.eligible_strict,'field_name_R'].tolist()
    if len(chem)!=250 or len(features)!=231 or len(strict)!=218 or not set(strict)<=set(features):
        raise ValueError('Frozen sample/full field/strict geometry contract changed')
    indices=[features.index(f) for f in strict];src=Path(__file__).parent
    files=[CONFIG,Path(__file__),src/'basis_mixture.py',src/'gaussian_baselines.py',src/'joint_generation.py',
        src/'empirical_generation.py',src/'run_e208_joint_generation.py',src/'q3_evaluation.py',src/'audit_e202.py',src/'q1_interface.py',
        attachment,split_path,dictionary,RESULT/'q3_run_manifest_E208.json',RESULT/'q3_e208_audit_manifest.json',
        RESULT/'q3_distribution_metrics_E208.csv']
    for experiment in ['E202','E208']:
        for repeat in range(10):
            for fold in range(3):
                slot=RESULT/f'q3_checkpoints_{experiment}/R{repeat:02d}F{fold}'
                meta=json.loads((slot/'manifest.json').read_text())
                if meta['files']['generated.csv.gz']!=sha(slot/'generated.csv.gz'):raise ValueError('Frozen baseline generated samples changed')
                files.extend([slot/'manifest.json',slot/'generated.csv.gz'])
    hashes={str(p.resolve()):sha(p) for p in files}
    signature={'input_sha256':hashes,'q1_final_oof_sha256':q1.signature}
    return config,chem,splits,features,indices,hashes,signature


def write_frame(slot,name,frame):
    p=slot/name;frame.to_csv(p,index=False,compression='gzip' if name.endswith('.gz') else None);return p


def run(max_folds=None):
    manifest_path=RESULT/'q3_run_manifest_E211.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous.get('status')!='complete':raise ValueError('Unexpected incomplete final manifest')
        for p,h in previous['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Completed E211 input changed')
        for n,h in previous['output_sha256'].items():
            if sha(RESULT/n)!=h:raise ValueError('Completed E211 output changed')
        print('E211 complete; frozen outputs verified without rewriting',flush=True);return previous
    config,chem,splits,features,indices,hashes,signature=load_inputs()
    raw=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    baseline=pd.read_csv(RESULT/'q3_distribution_metrics_E208.csv')
    baseline=baseline[baseline.method.isin(['style_copula_shrunk','mixture_selected'])].copy()
    baseline['method']=baseline.method.replace({'mixture_selected':'legacy_raw_mixture'})
    checkpoint=RESULT/'q3_checkpoints_E211';checkpoint.mkdir(exist_ok=True)
    tables={k:[] for k in ['metrics.csv','inner_selection.csv','stability.csv','neighborhood_diagnostics.csv','feature_w1.csv']};completed=[]
    for number,((repeat,fold),group) in enumerate(splits.groupby(['repeat','fold'],sort=True),1):
        if max_folds is not None and number>max_folds:break
        repeat=int(repeat);fold=int(fold);name=f'R{repeat:02d}F{fold}';slot=checkpoint/name;slot.mkdir(exist_ok=True)
        commit=slot/'manifest.json';valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(raw)),valid)
        if commit.exists():
            meta=json.loads(commit.read_text())
            if meta['signature']!=signature:raise ValueError('E211 checkpoint identity changed')
            for n,h in meta['files'].items():
                if sha(slot/n)!=h:raise ValueError('E211 checkpoint damaged')
            for n in tables:tables[n].append(pd.read_csv(slot/n))
            completed.append(name);print(f'E211 {number}/30 {name} cached=True',flush=True);continue
        for experiment in ['E202','E208']:
            old=json.loads((RESULT/f'q3_checkpoints_{experiment}/{name}/manifest.json').read_text())
            if old['train_beer']!=chem.beer.iloc[train].astype(int).tolist() or old['validation_beer']!=chem.beer.iloc[valid].astype(int).tolist():
                raise ValueError('Frozen baseline fold identities differ')
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(raw[train]);vx=imputer.transform(raw[valid])
        selected,selection=select_inside_training(raw[train],styles[train],indices,config)
        for col,val in [('fold',fold),('repeat',repeat)]:selection.insert(0,col,val)
        model=ChemicalMixtureGenerator().fit(tx,styles[train],indices,config['geometry'],target_styles=config['target_styles'])
        stability=model.stability_table_.copy();stability.insert(0,'fold',fold);stability.insert(0,'repeat',repeat)
        diagnostics=[];neighbors=[];rows=[];samples=[];feature_rows=[]
        scale=tx.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
        for si,style in enumerate(config['target_styles']):
            ids=model.positions_[style];beers=chem.beer.iloc[train].to_numpy(int)
            d=model.diagnostics_[style].copy();d['anchor_beer']=beers[ids];d['style']=style;d['repeat']=repeat;d['fold']=fold
            diagnostics.append(d)
            for geometry,nn in model.neighbors_[style].items():
                for anchor,peers in zip(ids,nn):
                    for rank,neighbor in enumerate(peers,1):neighbors.append({'repeat':repeat,'fold':fold,'style':style,'geometry':geometry,
                        'anchor_beer':int(beers[anchor]),'neighbor_beer':int(beers[neighbor]),'rank_in_geometry':rank,'scope':'outer_training_only'})
            real=vx[styles[valid]==style];local=tx[styles[train]==style]
            w1s={m:[] for m in GEOMETRIES}
            for seed_index,base_seed in enumerate(config['seeds']):
                seed=int(base_seed)+repeat*1000+fold*100+si*10
                for method,geometry in GEOMETRIES.items():
                    alpha=selected[style][method];sample=model.generate(style,len(real),geometry,alpha=alpha,seed=seed)
                    values=sample['values']
                    for scope,m in metrics_for_scales(real,values,tx,local).items():
                        rows.append({'repeat':repeat,'fold':fold,'style':style,'method':method,'alpha':alpha,
                            'seed_index':seed_index,'metric_scale':scope,'scope':'outer_validation_evaluation',**m})
                    w1s[method].append([wasserstein_distance(real[:,j],values[:,j])/scale[j] for j in range(len(features))])
                    frame=pd.DataFrame(values,columns=features)
                    metadata={'repeat':repeat,'fold':fold,'method':method,'style':style,'seed_index':seed_index,
                        'sample_position':np.arange(len(real)),'alpha':alpha,'local_selected':sample['local_selected'],
                        'local_proposal_anchor_a_beer':beers[sample['anchor_a']],
                        'local_proposal_anchor_b_beer':beers[sample['anchor_b']],'interpolation_t':sample['interpolation_t'],
                        'selector_seed':sample['selector_seed'],'copula_seed':sample['copula_seed'],'local_seed':sample['local_seed'],
                        'scope':'outer_training_generated_unprojected'}
                    for k,v in reversed(list(metadata.items())):frame.insert(0,k,v)
                    samples.append(frame)
            for method,vectors in w1s.items():
                for f,value in zip(features,np.mean(vectors,axis=0)):feature_rows.append({'repeat':repeat,'fold':fold,
                    'style':style,'method':method,'feature_code':f,'mean_seed_w1':float(value),'metric_scale':'raw'})
        metrics=pd.concat([pd.DataFrame(rows),baseline[baseline.repeat.eq(repeat)&baseline.fold.eq(fold)]],ignore_index=True)
        frames={'metrics.csv':metrics,'inner_selection.csv':selection,'stability.csv':stability,
            'neighborhood_diagnostics.csv':pd.concat(diagnostics,ignore_index=True),'feature_w1.csv':pd.DataFrame(feature_rows)}
        paths=[write_frame(slot,n,f) for n,f in frames.items()]
        paths.extend([write_frame(slot,'neighbors.csv',pd.DataFrame(neighbors)),write_frame(slot,'generated.csv.gz',pd.concat(samples,ignore_index=True))])
        g=model.geometry_;state=slot/'geometry_state.npz'
        np.savez_compressed(state,feature_names=np.array(features,dtype=str),geometry_indices=np.asarray(indices),
            imputer_statistics=imputer.statistics_,lower=g.lower_,upper=g.upper_,center=g.center_,scale=g.scale_,
            pca_mean=g.pca_mean_,pca_components=g.components_,prototypes=g.prototypes_,bandwidths=g.bandwidths_)
        paths.append(state)
        meta={'signature':signature,'train_beer':chem.beer.iloc[train].astype(int).tolist(),
            'validation_beer':chem.beer.iloc[valid].astype(int).tolist(),'selected_alpha':selected,'n_strict_fields':len(indices),
            'basis_stability_fallback':model.unstable_,'files':{p.name:sha(p) for p in paths}}
        commit.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        completed.append(name)
        for n,f in frames.items():tables[n].append(f)
        (RESULT/'q3_progress_E211.json').write_text(json.dumps({'status':'running','completed_folds':len(completed),'last_fold':name},indent=2)+'\n')
        print(f'E211 {number}/30 {name} alpha={selected} fallback={model.unstable_}',flush=True)
    if len(completed)!=30:return {'status':'checkpoint_only','completed_folds':len(completed)}
    combined={n:pd.concat(frames,ignore_index=True) for n,frames in tables.items()}
    repeat_metrics=combined['metrics.csv'].groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    names={'metrics.csv':'q3_distribution_metrics_E211.csv','inner_selection.csv':'q3_inner_selection_E211.csv',
        'stability.csv':'q3_neighborhood_stability_E211.csv','neighborhood_diagnostics.csv':'q3_neighborhood_diagnostics_E211.csv',
        'feature_w1.csv':'q3_feature_w1_E211.csv'}
    outputs=[write_frame(RESULT,names[n],frame) for n,frame in combined.items()]
    outputs.append(write_frame(RESULT,'q3_metrics_by_repeat_E211.csv',repeat_metrics))
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Frozen input changed during E211')
    manifest={'experiment_id':'E211','status':'complete','n_outer_folds':30,'n_repeats':10,'n_seeds':5,
        'n_metric_rows':len(combined['metrics.csv']),'n_inner_selection_rows':len(combined['inner_selection.csv']),
        'methods':ALL_METHODS,'signature':signature,'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in outputs},
        'checkpoint_manifest_sha256':{n:sha(checkpoint/n/'manifest.json') for n in completed},
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False,'scope':'exploratory_nested_outer_validation_not_independent_confirmation',
        'random_stream_policy':config['random_stream_policy'],'generated_spectra_projected_or_filtered':False}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (RESULT/'q3_progress_E211.json').write_text(json.dumps({'status':'complete','completed_folds':30},indent=2)+'\n')
    print('E211 status complete 30/30',flush=True);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-folds',type=int);args=p.parse_args()
    if args.max_folds is not None and not 1<=args.max_folds<=30:p.error('--max-folds must be 1..30')
    run(args.max_folds)
