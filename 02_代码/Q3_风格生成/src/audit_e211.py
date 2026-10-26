"""Integrity, geometry increments and preregistered sequential E211 decisions."""
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from run_e211_basis_mixture import CONFIG,ROOT,RESULT,CANONICAL,ALL_METHODS,GEOMETRIES,METRICS,sha,validate_config
from audit_e208 import EXPECTED_RULES,replacement_decision

INCREMENT_RULES={'comparison_methods':['raw_mixture','pca_mixture'],'robust_energy_min_decrease':.002,
    'energy_better_repeats':7,'robust_w1_max_increase':.005,'robust_coverage_max_decrease':.02,
    'robust_log_spread_error_max_increase':.02,'each_style_w1_energy_max_increase':.02,
    'each_style_minimum_active_basis_fraction':.5}


def base_decision(method,repeats,metrics):
    r=repeats.copy();m=metrics.copy();r['method']=r.method.replace({method:'joint_selected'});m['method']=m.method.replace({method:'joint_selected'})
    decision,paired,by_style=replacement_decision(r,m,{'replacement_rules':EXPECTED_RULES.copy()})
    decision['primary_distribution']=method if decision['replace_copula'] else 'style_copula_shrunk'
    paired['method_new']=paired.method_new.replace({'joint_selected':method})
    return decision,paired,by_style


def assess(repeats,metrics,diagnostics,config):
    if (config['replacement_rules']!=dict(EXPECTED_RULES,candidate='basis_mixture')
        or config['basis_increment_rules']!=INCREMENT_RULES
        or config['fallback_order']!=['basis_with_increment','raw_if_base_rules','frozen_copula']):
        raise ValueError('E211 registered adoption/increment/fallback changed')
    repeats=repeats.copy()
    repeats[METRICS]=repeats[METRICS].astype(float)
    decisions={};pairs=[];styles=[]
    for method in GEOMETRIES:
        d,p,s=base_decision(method,repeats,metrics);decisions[method]=d
        p['comparison']='versus_frozen_copula';pairs.append(p)
        s.insert(0,'method',method);styles.append(s)
    fold_active=diagnostics.groupby(['style','repeat','fold']).basis_effective.mean().ge(.5)
    active_by_style=fold_active.groupby('style').mean()
    active=bool(len(active_by_style)==3 and active_by_style.min()>=.5)
    increments={}
    for reference in ['raw_mixture','pca_mixture']:
        joined=repeats[repeats.method=='basis_mixture'].merge(repeats[repeats.method==reference],
            on=['style','repeat','metric_scale'],validate='one_to_one',suffixes=('_new','_base'))
        if len(joined)!=60:raise ValueError('Incomplete basis paired comparison')
        for metric in METRICS:joined[metric+'_delta']=joined[metric+'_new']-joined[metric+'_base']
        joined['comparison']='basis_versus_'+reference;pairs.append(joined)
        robust=joined[joined.metric_scale=='robust'];rd=robust.groupby('repeat').energy_v_delta.mean()
        per_style=robust.groupby('style')[['mean_w1_delta','energy_v_delta']].mean()
        checks={'energy_gain':bool(robust.energy_v_delta.mean()<=-.002),'repeat_gain':bool((rd<0).sum()>=7),
            'w1_noninferior':bool(robust.mean_w1_delta.mean()<=.005),'coverage_noninferior':bool(robust.coverage_delta.mean()>=-.02),
            'spread_noninferior':bool(robust.log_spread_error_delta.mean()<=.02),
            'no_style_regression':bool(per_style.to_numpy().max()<=.02)}
        increments[reference]={'supported':all(checks.values()),'checks':checks,
            'robust_energy_delta':float(robust.energy_v_delta.mean()),'robust_w1_delta':float(robust.mean_w1_delta.mean()),
            'robust_coverage_delta':float(robust.coverage_delta.mean()),
            'robust_log_spread_error_delta':float(robust.log_spread_error_delta.mean()),'energy_better_repeats':int((rd<0).sum())}
    supported=bool(active and all(v['supported'] for v in increments.values()))
    if decisions['basis_mixture']['replace_copula'] and supported:chosen='basis_mixture'
    elif decisions['raw_mixture']['replace_copula']:chosen='raw_mixture'
    else:chosen='style_copula_shrunk'
    report={'selected_distribution':chosen,'basis_increment_supported':supported,'base_decisions':decisions,
        'basis_increments':increments,'active_basis_fraction_by_style':{str(k):float(v) for k,v in active_by_style.items()},
        'selection_status':'candidate_pending_full_fit_and_migration_audit' if chosen!='style_copula_shrunk' else 'retain_frozen_distribution',
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False,'physical_safety_certified':False,
        'existing_full_data_profiles_and_migrations_replaced':False,
        'notes':['仅按运行前写明的顺序选择，低维方案只作消融',
            '新混合使用独立随机流；旧混合只读保留，不视作独立确认',
            '分布选择不自动认证迁移、真实感官或化学安全']}
    return report,pd.concat(pairs,ignore_index=True),pd.concat(styles,ignore_index=True)


def audit_checkpoint(slot,meta,chem,features,config):
    train_beer=meta['train_beer'];valid_beer=meta['validation_beer']
    if set(train_beer)&set(valid_beer) or set(train_beer)|set(valid_beer)!=set(chem.beer.astype(int)):
        raise ValueError('Training/validation beer overlap or miss samples')
    for name,h in meta['files'].items():
        if sha(slot/name)!=h:raise ValueError('E211 checkpoint output changed')
    samples=pd.read_csv(slot/'generated.csv.gz');inner=pd.read_csv(slot/'inner_selection.csv')
    keys=['method','style','seed_index','sample_position']
    if samples.duplicated(keys).any() or not samples.scope.eq('outer_training_generated_unprojected').all():
        raise ValueError('Duplicate or projected/filtered generated samples')
    train=chem.set_index('beer').loc[train_beer];valid=chem.set_index('beer').loc[valid_beer]
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);tx=imputer.fit_transform(train[features])
    lookup={int(b):i for i,b in enumerate(train_beer)}
    with np.load(slot/'geometry_state.npz',allow_pickle=False) as state:
        if len(state['geometry_indices'])!=218 or not np.array_equal(state['feature_names'],np.array(features,dtype=str)):
            raise ValueError('Geometry/full output field identity changed')
        np.testing.assert_allclose(state['imputer_statistics'],imputer.statistics_,atol=0.,rtol=0.)
    neighbors=pd.read_csv(slot/'neighbors.csv');diagnostic=pd.read_csv(slot/'neighborhood_diagnostics.csv')
    if (not neighbors.scope.eq('outer_training_only').all() or not set(neighbors.anchor_beer)<=set(train_beer)
        or not set(neighbors.neighbor_beer)<=set(train_beer) or neighbors.anchor_beer.eq(neighbors.neighbor_beer).any()):
        raise ValueError('Neighbor graph contains held-out or self samples')
    if len(inner)!=54 or not inner.groupby(['inner_fold','style','method']).selected.sum().eq(1).all():
        raise ValueError('Incomplete inner selection denominator')
    neighbor_set=set(zip(neighbors['style'],neighbors.geometry,neighbors.anchor_beer,neighbors.neighbor_beer))
    max_error=0.
    for (method,style,seed),group in samples.groupby(['method','style','seed_index']):
        n=int(valid.tasting_category_fine.eq(style).sum())
        if len(group)!=n or set(group.sample_position)!=set(range(n)):raise ValueError('Generated count does not match held-out style')
        if group.alpha.nunique()!=1 or float(group.alpha.iloc[0])!=meta['selected_alpha'][style][method]:
            raise ValueError('Generated mixture proportion differs from inner choice')
        for _,row in group.iterrows():
            a=int(row.local_proposal_anchor_a_beer);b=int(row.local_proposal_anchor_b_beer)
            if (style,GEOMETRIES[method],a,b) not in neighbor_set:raise ValueError('Local proposal not in declared chemical neighborhood')
            if train.loc[a,'tasting_category_fine']!=style or train.loc[b,'tasting_category_fine']!=style:
                raise ValueError('Cross-style interpolation used')
            if bool(row.local_selected):
                t=float(row.interpolation_t)
                if not .2<=t<=.8:raise ValueError('Local interpolation coefficient outside registered range')
                expected=(1-t)*tx[lookup[a]]+t*tx[lookup[b]]
                error=float(np.max(np.abs(expected-row[features].to_numpy(float))));max_error=max(max_error,error)
                if error>1e-10:raise ValueError('Generated local spectrum differs from recorded anchors')
    if set(samples.method)!=set(GEOMETRIES) or len(samples.groupby(['method','style','seed_index']))!=45:
        raise ValueError('Incomplete new-method/style/seed coverage')
    return max_error


def run():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    mp=RESULT/'q3_run_manifest_E211.json';manifest=json.loads(mp.read_text())
    if manifest.get('status')!='complete' or manifest.get('n_outer_folds')!=30:raise ValueError('Complete E211 required')
    for path,h in manifest['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Frozen E211 input changed')
    for name,h in manifest['output_sha256'].items():
        if sha(RESULT/name)!=h:raise ValueError('E211 output changed')
    chem=pd.read_csv(CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv')
    features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']];reconstruction=[]
    for name,h in manifest['checkpoint_manifest_sha256'].items():
        slot=RESULT/'q3_checkpoints_E211'/name;meta=json.loads((slot/'manifest.json').read_text())
        if sha(slot/'manifest.json')!=h or meta['signature']!=manifest['signature']:raise ValueError('E211 checkpoint identity changed')
        error=audit_checkpoint(slot,meta,chem,features,config)
        reconstruction.append({'fold_name':name,'max_local_spectrum_error':error,'fold_identity_and_neighbor_scope_valid':True})
    metrics=pd.read_csv(RESULT/'q3_distribution_metrics_E211.csv');repeats=pd.read_csv(RESULT/'q3_metrics_by_repeat_E211.csv')
    inner=pd.read_csv(RESULT/'q3_inner_selection_E211.csv');diag=pd.read_csv(RESULT/'q3_neighborhood_diagnostics_E211.csv')
    keys=['repeat','fold','style','method','seed_index','metric_scale']
    if (len(metrics)!=4500 or metrics.duplicated(keys).any() or set(metrics.method)!=set(ALL_METHODS)
        or not metrics.groupby(['method','metric_scale']).size().eq(450).all() or len(inner)!=1620):
        raise ValueError('Incomplete formal E211 evaluation coverage')
    recomputed=metrics.groupby(['method','style','repeat','metric_scale'],as_index=False)[METRICS].mean()
    sort=['method','style','repeat','metric_scale']
    np.testing.assert_allclose(recomputed.sort_values(sort)[METRICS],repeats.sort_values(sort)[METRICS],atol=1e-12,rtol=0.,equal_nan=True)
    old=pd.read_csv(RESULT/'q3_distribution_metrics_E208.csv');checks=[]
    for current,previous in [('style_copula_shrunk','style_copula_shrunk'),('legacy_raw_mixture','mixture_selected')]:
        joined=metrics[metrics.method==current].merge(old[old.method==previous],on=[k for k in keys if k!='method'],
            suffixes=('_new','_old'),validate='one_to_one')
        if len(joined)!=900:raise ValueError('Frozen baseline coverage changed')
        for metric in METRICS:
            a=joined[metric+'_new'].to_numpy(float);b=joined[metric+'_old'].to_numpy(float)
            np.testing.assert_allclose(a,b,atol=1e-9,rtol=0.,equal_nan=True)
            checks.append({'method':current,'metric':metric,'max_absolute_difference':float(np.nanmax(np.abs(a-b)))})
    decision,paired,by_style=assess(repeats,metrics,diag,config)
    outputs=[]
    path=RESULT/'q3_replacement_decision_E211.json';path.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');outputs.append(path)
    for name,frame in [('q3_paired_comparisons_E211.csv',paired),('q3_comparisons_by_style_E211.csv',by_style),
        ('q3_frozen_baseline_reproduction_E211.csv',pd.DataFrame(checks)),('q3_local_provenance_audit_E211.csv',pd.DataFrame(reconstruction))]:
        p=RESULT/name;frame.to_csv(p,index=False);outputs.append(p)
    summary=repeats.groupby(['method','metric_scale'])[METRICS].mean().reset_index()
    p=RESULT/'q3_all_method_summary_E211.csv';summary.to_csv(p,index=False);outputs.append(p)
    report={'experiment_id':'E211','status':'complete','n_outer_folds':30,'n_metric_rows':len(metrics),
        'n_inner_selection_rows':len(inner),'n_frozen_inputs_verified':len(manifest['input_sha256']),
        'decision':decision,'audit_source_sha256':sha(__file__),'input_manifest_sha256':sha(mp),
        'output_sha256':{p.name:sha(p) for p in outputs},'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    (RESULT/'q3_e211_audit_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
