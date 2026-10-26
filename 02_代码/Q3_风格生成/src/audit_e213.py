"""Replay neural weights, fixed nested choices and clipping-aware adoption."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml

from run_e213_neural_interpolation import CONFIG,RESULT,CANONICAL,ALL_METHODS,METHODS,METRICS,sha,validate_config,choices_from_table,metrics_for_scales
from neural_interpolation import ResidualInterpolator
from audit_e208 import EXPECTED_RULES
from audit_e211 import base_decision

INCREMENT={'references':['pca_noise','raw_mixture'],'energy_min_decrease':.002,'energy_better_repeats':7,
    'w1_max_increase':.005,'coverage_max_decrease':.02,'spread_max_increase':.02,'each_style_w1_energy_max_increase':.02}
CLIPPING={'average_cell_max':.15,'each_style_cell_max':.25}


def assess(repeats,metrics,clipping,config):
    if (config['replacement_rules']!=dict(EXPECTED_RULES,candidate='neural_selected')
        or config['neural_increment']!=INCREMENT or config['clipping_rules']!=CLIPPING
        or config['fallback_order']!=['neural_with_increment','pca_noise_if_base_rules','frozen_copula']):
        raise ValueError('E213 registered adoption/increment/clipping changed')
    repeats=repeats.copy();repeats[METRICS]=repeats[METRICS].astype(float)
    decisions={};paired=[];styles=[];clip_info={}
    for method in METHODS:
        d,p,s=base_decision(method,repeats,metrics);d['gpu_used']=True;decisions[method]=d
        p['comparison']=method+'_versus_frozen_copula';paired.append(p);s.insert(0,'method',method);styles.append(s)
        block=clipping[clipping.method.eq(method)];by_style=block.groupby('style').local_cell_clipping.mean()
        clip_info[method]={'acceptable':bool(len(by_style)==3 and block.local_cell_clipping.mean()<=.15 and by_style.max()<=.25),
            'mean_local_cell_clipping':float(block.local_cell_clipping.mean()),'by_style':{str(k):float(v) for k,v in by_style.items()}}
    increments={}
    for reference in INCREMENT['references']+['neural_noise']:
        method='neural_curve' if reference=='neural_noise' else 'neural_selected'
        joined=repeats[repeats.method.eq(method)].merge(repeats[repeats.method.eq(reference)],
            on=['style','repeat','metric_scale'],validate='one_to_one',suffixes=('_new','_base'))
        if len(joined)!=60:raise ValueError('Incomplete paired comparison')
        for metric in METRICS:joined[metric+'_delta']=joined[metric+'_new']-joined[metric+'_base']
        joined['comparison']=method+'_versus_'+reference;paired.append(joined)
        robust=joined[joined.metric_scale.eq('robust')];rd=robust.groupby('repeat').energy_v_delta.mean()
        per_style=robust.groupby('style')[['mean_w1_delta','energy_v_delta']].mean()
        checks={'energy_gain':bool(robust.energy_v_delta.mean()<=-.002),'repeat_gain':bool((rd<0).sum()>=7),
            'w1_noninferior':bool(robust.mean_w1_delta.mean()<=.005),'coverage_noninferior':bool(robust.coverage_delta.mean()>=-.02),
            'spread_noninferior':bool(robust.log_spread_error_delta.mean()<=.02),
            'no_style_regression':bool(per_style.to_numpy().max()<=.02)}
        increments[reference]={'supported':all(checks.values()),'checks':checks,
            'robust_energy_delta':float(robust.energy_v_delta.mean()),'robust_w1_delta':float(robust.mean_w1_delta.mean()),
            'robust_coverage_delta':float(robust.coverage_delta.mean()),'robust_log_spread_error_delta':float(robust.log_spread_error_delta.mean()),
            'energy_better_repeats':int((rd<0).sum())}
    supported=bool(all(increments[reference]['supported'] for reference in INCREMENT['references']))
    if decisions['neural_selected']['replace_copula'] and supported and clip_info['neural_selected']['acceptable']:chosen='neural_selected'
    elif decisions['pca_noise']['replace_copula'] and clip_info['pca_noise']['acceptable']:chosen='pca_noise'
    else:chosen='style_copula_shrunk'
    report={'selected_distribution':chosen,'base_decisions':decisions,'neural_increment_supported':supported,
        'increments':increments,'clipping':clip_info,'selection_status':'candidate_pending_full_fit_and_migration_audit' if chosen!='style_copula_shrunk' else 'retain_frozen_distribution',
        'q1_retrained':False,'nnls_refit':False,'kaggle_used_for_training':False,'gpu_used':True,
        'physical_safety_certified':False,'existing_full_data_profiles_and_migrations_replaced':False,
        'notes':['内层仅用官方化学谱，网络无感官标签/外部数据/外折早停',
            '神经选择六候选，PCA三候选；外折不参与选择，预算差异保留',
            '残差噪声与曲率增量分开；同风格训练范围剪裁不等于缺陷安全认证',
            '重复交叉验证是探索性比较，不等于独立外部确认']}
    return report,pd.concat(paired,ignore_index=True),pd.concat(styles,ignore_index=True)


def audit_checkpoint(slot,meta,chem,features,config):
    train_beer=meta['train_beer'];valid_beer=meta['validation_beer']
    if len(set(train_beer))!=len(train_beer) or len(set(valid_beer))!=len(valid_beer) or set(train_beer)&set(valid_beer) or set(train_beer)|set(valid_beer)!=set(chem.beer.astype(int)):
        raise ValueError('Outer training/validation identities invalid')
    for name,h in meta['files'].items():
        if sha(slot/name)!=h:raise ValueError('E213 checkpoint output changed')
    train=chem.set_index('beer').loc[train_beer];valid=chem.set_index('beer').loc[valid_beer]
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(train[features]);vx=imputer.transform(valid[features])
    with np.load(slot/'model_state.npz',allow_pickle=False) as state:
        if not np.array_equal(state['feature_names'],np.array(features)):raise ValueError('Feature order changed')
        np.testing.assert_allclose(imputer.statistics_,state['imputer_statistics'],atol=1e-12,rtol=0.)
        model=ResidualInterpolator.from_arrays(tx,train.tasting_category_fine.to_numpy(str),config['model'],
            target_styles=config['target_styles'],arrays=dict(state))
    history=pd.read_csv(slot/'training_history.csv')
    for inner,group in history.groupby('inner_fold'):
        if len(group)!=200 or not np.array_equal(group.step.to_numpy(),np.arange(1,201)) or not np.isfinite(group[['training_huber','gradient_norm']]).all().all():
            raise ValueError('Incomplete/invalid fixed neural training history')
        expected=meta['training_seed']+(int(inner)+1 if int(inner)>=0 else 0)
        if not group.training_seed.eq(expected).all():raise ValueError('Training seed mismatch')
    if set(history.inner_fold)!=set([-1,0,1]) or not meta['gpu_used'] or meta['training_steps']!=200:
        raise ValueError('Three registered training fits not present')
    inner=pd.read_csv(slot/'inner_selection.csv');selected=choices_from_table(inner,config)
    if selected!=meta['selected']:raise ValueError('Recorded choices differ from training-only selections')
    samples=pd.read_csv(slot/'generated.csv.gz');metrics=pd.read_csv(slot/'metrics.csv');unclipped=pd.read_csv(slot/'unclipped_metrics.csv')
    if samples.duplicated(['method','style','seed_index','sample_position']).any() or not samples.scope.eq('outer_training_generated_unprojected').all():
        raise ValueError('Duplicate/filtered sample rows')
    max_error=0.;metric_errors=[]
    for (method,style,seed_index),group in samples.groupby(['method','style','seed_index']):
        group=group.sort_values('sample_position');repeat=int(group.repeat.iloc[0]);fold=int(group.fold.iloc[0]);seed_index=int(seed_index)
        if not np.array_equal(group.sample_position,np.arange(len(group))):raise ValueError('Sample positions incomplete')
        real=vx[valid.tasting_category_fine.to_numpy(str)==style];local=tx[train.tasting_category_fine.to_numpy(str)==style]
        if len(group)!=len(real):raise ValueError('Generated sample count differs from validation style count')
        seed=int(config['seeds'][seed_index])+repeat*1000+fold*100+config['target_styles'].index(style)*10
        choice=selected[style][method]
        sample=model.generate(style,len(group),choice['family'],alpha=choice['alpha'],beta=choice['beta'],gamma=.5,seed=seed)
        for name,value in choice.items():
            if not group[name].eq(value).all():raise ValueError('Stored selection differs from inner-only choice')
        for column,value in zip(['selector_seed','copula_seed','local_seed','noise_seed'],sample['stream_seeds']):
            if not group[column].eq(value).all():raise ValueError('Independent random stream trace changed')
        for column,key in [('local_selected','local_selected'),('interpolation_t','interpolation_t'),('local_preclip_cell_fraction','local_preclip_cell_fraction'),('local_preclip_row','local_preclip_row')]:
            np.testing.assert_allclose(group[column].to_numpy(float),sample[key].astype(float),atol=1e-12,rtol=0.)
        for column,key in [('anchor_a_beer','anchor_a'),('anchor_b_beer','anchor_b')]:
            if not np.array_equal(group[column],np.asarray(train_beer)[sample[key]]):raise ValueError('Wrong/outside training anchor')
        if np.any(sample['anchor_a']==sample['anchor_b']):raise ValueError('Self-interpolation anchor')
        error=float(np.max(np.abs(group[features].to_numpy(float)-sample['values'])));max_error=max(max_error,error)
        if error>1e-8:raise ValueError('Neural state spectrum replay differs')
        for table,values in [(metrics,sample['values']),(unclipped,sample['unclipped_values'])]:
            selected_metrics=table[table.method.eq(method)&table['style'].eq(style)&table.seed_index.eq(seed_index)]
            for scale,m in metrics_for_scales(real,values,tx,local).items():
                row=selected_metrics[selected_metrics.metric_scale.eq(scale)]
                if len(row)!=1:raise ValueError('Missing metric row')
                # Rank ties may move at floating-point precision; keep that diagnostic separate.
                for metric in [v for v in METRICS if v!='spearman_pair_rmse']:
                    a=float(row[metric].iloc[0]);b=float(m[metric])
                    if not np.isclose(a,b,atol=1e-8,rtol=0.,equal_nan=True):raise ValueError('Replay evaluation differs: '+metric)
                metric_errors.append(abs(float(row.energy_v.iloc[0])-m['energy_v']))
    if samples.groupby(['method','style','seed_index']).ngroups!=60:raise ValueError('Incomplete method/style/seed coverage')
    return {'max_spectrum_error':max_error,'max_energy_replay_error':float(max(metric_errors)),
        'saved_state_replay_without_training':True,'fold_preprocessing_and_independent_streams_valid':True}


def run():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    mp=RESULT/'q3_run_manifest_E213.json';manifest=json.loads(mp.read_text())
    if manifest.get('status')!='complete' or manifest.get('n_outer_folds')!=30 or manifest.get('q1_retrained') or manifest.get('nnls_refit'):
        raise ValueError('Complete unchanged-interface E213 required')
    for path,h in manifest['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Frozen E213 input changed: '+path)
    for name,h in manifest['output_sha256'].items():
        if sha(RESULT/name)!=h:raise ValueError('E213 output changed')
    chem=pd.read_csv(CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv')
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']];reconstruction=[]
    for name,h in manifest['checkpoint_manifest_sha256'].items():
        slot=RESULT/'q3_checkpoints_E213'/name;meta=json.loads((slot/'manifest.json').read_text())
        if sha(slot/'manifest.json')!=h or meta['signature']!=manifest['signature']:raise ValueError('E213 checkpoint identity changed')
        reconstruction.append({'fold_name':name,**audit_checkpoint(slot,meta,chem,features,config)})
    metrics=pd.read_csv(RESULT/'q3_distribution_metrics_E213.csv');repeats=pd.read_csv(RESULT/'q3_metrics_by_repeat_E213.csv')
    inner=pd.read_csv(RESULT/'q3_inner_selection_E213.csv');clips=pd.read_csv(RESULT/'q3_clipping_E213.csv')
    keys=['repeat','fold','style','method','seed_index','metric_scale']
    if (len(metrics)!=5400 or metrics.duplicated(keys).any() or set(metrics.method)!=set(ALL_METHODS)
        or not metrics.groupby(['method','metric_scale']).size().eq(450).all() or len(inner)!=1620 or len(clips)!=1800):
        raise ValueError('Incomplete E213 formal coverage')
    recomputed=metrics.groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    sort=['method','style','repeat','metric_scale']
    np.testing.assert_allclose(recomputed.sort_values(sort)[METRICS],repeats.sort_values(sort)[METRICS],atol=1e-12,rtol=0.,equal_nan=True)
    old=pd.read_csv(RESULT/'q3_distribution_metrics_E211.csv');checks=[]
    for method in ['style_copula_shrunk','raw_mixture']:
        joined=metrics[metrics.method.eq(method)].merge(old[old.method.eq(method)],on=keys,validate='one_to_one',suffixes=('_new','_old'))
        if len(joined)!=900:raise ValueError('Frozen baseline incomplete')
        for metric in METRICS:
            a=joined[metric+'_new'].to_numpy(float);b=joined[metric+'_old'].to_numpy(float)
            np.testing.assert_allclose(a,b,atol=1e-9,rtol=0.,equal_nan=True)
            checks.append({'method':method,'metric':metric,'max_absolute_difference':float(np.nanmax(np.abs(a-b)))})
    decision,paired,by_style=assess(repeats,metrics,clips,config);outputs=[]
    path=RESULT/'q3_replacement_decision_E213.json';path.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');outputs.append(path)
    confidence=[]
    for (comparison,scale),group in paired.groupby(['comparison','metric_scale']):
        for metric in ['mean_w1','energy_v','coverage','log_spread_error']:
            delta=group.groupby('repeat')[metric+'_delta'].mean().to_numpy();rng=np.random.default_rng(20261026)
            lo,hi=np.quantile(delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(axis=1),[.025,.975])
            confidence.append(dict(comparison=comparison,metric_scale=scale,metric=metric,mean_delta=float(delta.mean()),
                repeat_cluster_low=float(lo),repeat_cluster_high=float(hi),scope='descriptive_dependent_repeated_cv_not_independent_confirmation'))
    frames=[('q3_paired_comparisons_E213.csv',paired),('q3_comparisons_by_style_E213.csv',by_style),
        ('q3_frozen_baseline_reproduction_E213.csv',pd.DataFrame(checks)),('q3_neural_replay_audit_E213.csv',pd.DataFrame(reconstruction)),
        ('q3_repeat_cluster_sensitivity_E213.csv',pd.DataFrame(confidence)),
        ('q3_all_method_summary_E213.csv',repeats.groupby(['method','metric_scale'])[METRICS].mean().reset_index())]
    for name,frame in frames:path=RESULT/name;frame.to_csv(path,index=False);outputs.append(path)
    report={'experiment_id':'E213','status':'complete','n_outer_folds':30,'n_metric_rows':len(metrics),'n_inner_selection_rows':len(inner),
        'n_frozen_inputs_verified':len(manifest['input_sha256']),'decision':decision,'audit_source_sha256':sha(__file__),
        'input_manifest_sha256':sha(mp),'output_sha256':{p.name:sha(p) for p in outputs},'q1_retrained':False,'nnls_refit':False,
        'kaggle_used_for_training':False,'gpu_used':True}
    (RESULT/'q3_e213_audit_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
