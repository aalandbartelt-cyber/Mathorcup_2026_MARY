"""Compare all generators with unchanged, preregistered joint-selection gates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer

from run_e208_joint_generation import RESULT,ROOT,CONFIG,sha,METRICS

EXPECTED_RULES={'candidate':'joint_selected','raw_w1_max_increase':.01,'raw_energy_max_increase':.02,
    'robust_w1_max_increase':.01,'robust_energy_min_decrease':.005,'robust_energy_better_repeats':7,
    'robust_coverage_min_increase':.05,'robust_log_spread_error_max_increase':.05,
    'each_style_robust_w1_max_increase':.02,'each_style_robust_energy_max_increase':.02,
    'near_duplicate_fraction_max':0.}


def tolerant_rank_error(real,generated,scale):
    # Diagnostic sensitivity only: it is never used to select a generator or
    # relax a registered replacement rule. Normalized near-ties share a rank.
    real=np.round(np.asarray(real,float)/scale,12)
    generated=np.round(np.asarray(generated,float)/scale,12)
    valid=(np.ptp(real,axis=0)>1e-12)&(np.ptp(generated,axis=0)>1e-12)
    n=int(valid.sum())
    if n<2:return np.nan,0
    a=np.corrcoef(rankdata(real[:,valid],axis=0),rowvar=False)
    b=np.corrcoef(rankdata(generated[:,valid],axis=0),rowvar=False)
    upper=np.triu_indices(n,1)
    return float(np.sqrt(np.mean((a[upper]-b[upper])**2))),n*(n-1)//2


def rank_precision_sensitivity(metrics):
    from audit_e202 import training_winsorize
    chemistry=pd.read_csv(ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv')
    features=[c for c in chemistry if c not in ['beer','beer_id','tasting_category_fine']]
    x=chemistry[features].to_numpy(float);styles=chemistry.tasting_category_fine.to_numpy(str)
    splits=pd.read_csv(ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv');rows=[]
    for (repeat,fold),group in splits.groupby(['repeat','fold']):
        valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(x[train]);vx=imputer.transform(x[valid])
        scale=np.std(tx,axis=0);scale=np.where(scale>1e-12,scale,1.)
        for experiment in ['E201','E202','E208']:
            name=f'R{int(repeat):02d}F{int(fold)}'
            frame=pd.read_csv(RESULT/f'q3_checkpoints_{experiment}/{name}/generated.csv.gz')
            for (method,style,seed_index),sample in frame.groupby(['method','style','seed_index']):
                real=vx[styles[valid]==style];generated=sample[features].to_numpy(float)
                _,(cr,cg),robust_scale,_=training_winsorize(tx,real,generated)
                for scope,a,b,s in [('raw',real,generated,scale),('robust',cr,cg,robust_scale)]:
                    error,pairs=tolerant_rank_error(a,b,s)
                    rows.append({'repeat':int(repeat),'fold':int(fold),'method':method,'style':style,
                        'seed_index':int(seed_index),'metric_scale':scope,'rank_error_tolerance':error,
                        'n_rank_pairs_tolerance':pairs,'normalized_decimal_places':12,'constant_ptp_tolerance':1e-12})
    return pd.DataFrame(rows)


def replacement_decision(repeats,raw_metrics,config):
    rules=config['replacement_rules']
    if rules!=EXPECTED_RULES:raise ValueError('E208 replacement rules differ from preregistration')
    candidate=repeats[repeats.method=='joint_selected'];baseline=repeats[repeats.method=='style_copula_shrunk']
    paired=candidate.merge(baseline,on=['style','repeat','metric_scale'],suffixes=('_new','_base'),validate='one_to_one')
    if len(paired)!=60:raise ValueError('Expected 3 styles x10 repeats x2 scales')
    for metric in METRICS:paired[metric+'_delta']=paired[metric+'_new']-paired[metric+'_base']
    raw=paired[paired.metric_scale=='raw'];robust=paired[paired.metric_scale=='robust']
    repeat_delta=robust.groupby('repeat').energy_v_delta.mean()
    by_style=robust.groupby('style')[['mean_w1_delta','energy_v_delta','coverage_delta','log_spread_error_delta']].mean()
    checks={
        'no_near_training_copies':bool(raw_metrics[raw_metrics.method=='joint_selected'].near_duplicate_fraction.max()<=0.),
        'no_collapsed_samples':bool(raw_metrics[raw_metrics.method=='joint_selected'].collapsed.eq(False).all()),
        'raw_w1_noninferior':bool(raw.mean_w1_delta.mean()<=.01),
        'raw_energy_noninferior':bool(raw.energy_v_delta.mean()<=.02),
        'robust_w1_noninferior':bool(robust.mean_w1_delta.mean()<=.01),
        'robust_energy_improves':bool(robust.energy_v_delta.mean()<=-.005),
        'robust_energy_at_least_seven_repeats':bool((repeat_delta<0).sum()>=7),
        'robust_coverage_improves':bool(robust.coverage_delta.mean()>=.05),
        'spread_not_worse':bool(robust.log_spread_error_delta.mean()<=.05),
        'no_style_w1_regression':bool(by_style.mean_w1_delta.max()<=.02),
        'no_style_energy_regression':bool(by_style.energy_v_delta.max()<=.02),
    }
    decision={'replace_copula':all(checks.values()),'checks':checks,
        'raw_mean_w1_delta':float(raw.mean_w1_delta.mean()),'raw_mean_energy_delta':float(raw.energy_v_delta.mean()),
        'robust_mean_w1_delta':float(robust.mean_w1_delta.mean()),'robust_mean_energy_delta':float(robust.energy_v_delta.mean()),
        'robust_mean_coverage_delta':float(robust.coverage_delta.mean()),
        'robust_mean_log_spread_error_delta':float(robust.log_spread_error_delta.mean()),
        'n_repeats_robust_energy_better':int((repeat_delta<0).sum()),
        'primary_distribution':'joint_selected' if all(checks.values()) else 'style_copula_shrunk',
        'physical_safety_certified':False,'q1_retrained':False,'gpu_used':False,
        'notes':['替换仅依据训练内自适应选择方法；单一消融最佳值不另选为主线',
                 '原始尺度与分位敏感性同时报告，重复CV不等于独立外部确认试验']}
    return decision,paired,by_style.reset_index()


def run():
    config=yaml.safe_load(CONFIG.read_text());mp=RESULT/'q3_run_manifest_E208.json';manifest=json.loads(mp.read_text())
    if manifest.get('status')!='complete' or manifest.get('n_outer_folds')!=30:raise ValueError('Complete E208 required')
    for path,digest in manifest['input_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'E208 frozen input changed: {path}')
    for name,digest in manifest['output_sha256'].items():
        if sha(RESULT/name)!=digest:raise ValueError('E208 output changed')
    for name,digest in manifest['checkpoint_manifest_sha256'].items():
        slot=RESULT/'q3_checkpoints_E208'/name;p=slot/'manifest.json';meta=json.loads(p.read_text())
        if sha(p)!=digest or meta['signature']!=manifest['signature']:raise ValueError('E208 checkpoint signature changed')
        for n,h in meta['files'].items():
            if sha(slot/n)!=h:raise ValueError('E208 checkpoint output changed')
    metrics=pd.read_csv(RESULT/'q3_distribution_metrics_E208.csv')
    repeats=pd.read_csv(RESULT/'q3_metrics_by_repeat_E208.csv')
    selection=pd.read_csv(RESULT/'q3_inner_selection_E208.csv')
    keys=['repeat','fold','style','method','seed_index','metric_scale']
    if len(metrics)!=8100 or metrics.duplicated(keys).any() or len(selection)!=1260:
        raise ValueError('E208 method/style/fold/seed coverage incomplete')
    if metrics[keys].isna().any().any() or not metrics.groupby(['method','metric_scale']).size().eq(450).all():
        raise ValueError('E208 model-scale coverage changed')
    if not selection.groupby(['repeat','fold','style','inner_fold']).selected_joint.sum().eq(1).all():
        raise ValueError('Inner joint choice does not have exactly one winner per style/fold')
    baseline_errors=[]
    for experiment,filename in [('E201','q3_distribution_metrics_E201.csv'),('E202','q3_distribution_metrics_E202.csv')]:
        old=pd.read_csv(RESULT/filename)
        old=old[old['method']!='real_real_reference']
        new=metrics[metrics.metric_scale=='raw']
        mergekeys=['repeat','fold','style','method','seed_index']
        joined=old.merge(new,on=mergekeys,validate='one_to_one',suffixes=('_old','_new'))
        if len(joined)!=len(old):raise ValueError('Frozen baseline re-evaluation missing rows')
        for metric in ['mean_w1','energy_v','coverage','near_duplicate_fraction','spearman_pair_rmse']:
            if not np.array_equal(np.isfinite(joined[metric+'_old']),np.isfinite(joined[metric+'_new'])):
                raise ValueError('Baseline missing/finite metric pattern changed')
            delta=(joined[metric+'_old']-joined[metric+'_new']).abs()
            error=float(delta.max())
            # The original rank diagnostic uses ptp>0, which is sensitive to
            # round-trip arithmetic on constant fields. Keep and expose it;
            # four non-rank primary diagnostics must still reproduce exactly.
            if metric!='spearman_pair_rmse' and error>1e-9:
                raise ValueError(f'Frozen {experiment} metric changed on re-evaluation: {metric} {error}')
            baseline_errors.append({'baseline':experiment,'metric':metric,'maximum_absolute_difference':error,
                'csv_roundtrip_rank_sensitive':metric=='spearman_pair_rmse' and error>1e-9})
    decision,paired,style_delta=replacement_decision(repeats,metrics,config)
    dp=RESULT/'q3_replacement_decision_E208.json';dp.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    outputs=[dp]
    for name,frame in [('q3_joint_vs_copula_paired_E208.csv',paired),('q3_joint_vs_copula_by_style_E208.csv',style_delta),
        ('q3_frozen_baseline_reproduction_E208.csv',pd.DataFrame(baseline_errors))]:
        path=RESULT/name;frame.to_csv(path,index=False);outputs.append(path)
    # Family comparisons retain unfavorable metrics; no post-hoc chosen ensemble.
    rows=[]
    base=repeats[repeats.method=='style_copula_shrunk']
    for method in repeats.method.unique():
        cand=repeats[repeats.method==method]
        paired_family=cand.merge(base,on=['style','repeat','metric_scale'],suffixes=('_new','_base'),validate='one_to_one')
        for scale,group in paired_family.groupby('metric_scale'):
            for metric in METRICS:
                delta=(group[metric+'_new']-group[metric+'_base']).replace([np.inf,-np.inf],np.nan)
                rows.append({'method':method,'metric_scale':scale,'metric':metric,
                    'mean_candidate_value':float(group[metric+'_new'].mean()),'mean_baseline_value':float(group[metric+'_base'].mean()),
                    'mean_candidate_minus_baseline':float(delta.mean()) if delta.notna().any() else np.nan})
    path=RESULT/'q3_all_method_comparison_E208.csv';pd.DataFrame(rows).to_csv(path,index=False);outputs.append(path)
    rank_table=rank_precision_sensitivity(metrics)
    path=RESULT/'q3_rank_precision_sensitivity_E208.csv';rank_table.to_csv(path,index=False);outputs.append(path)
    report={'status':'complete','experiment_id':'E208','n_outer_folds':30,'n_metric_rows':len(metrics),
        'n_inner_selection_rows':len(selection),'n_frozen_inputs_verified':len(manifest['input_sha256']),
        'frozen_baseline_distance_and_copy_metrics_reproduced':True,
        'rank_metric_csv_roundtrip_sensitive':True,'decision':decision,'source_sha256':sha(__file__),
        'input_manifest_sha256':sha(mp),'output_sha256':{p.name:sha(p) for p in outputs},
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False,
        'notes':['历史秩相关诊断ptp>0对常量列的CSV往返舍入敏感；新增12位标准化规约诊断，不覆盖旧数字',
                 '生成器选择与预登记替换规则不使用此秩相关指标；W1/energy/覆盖/复制复现不变']}
    (RESULT/'q3_e208_audit_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
