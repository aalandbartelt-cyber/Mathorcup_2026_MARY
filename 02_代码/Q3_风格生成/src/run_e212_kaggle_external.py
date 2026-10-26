"""Adopt static external directional auditing; no Q1/Q2/Q3 fitting or RL."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kaggle_external import FLAVORS,normalized_flavors,select_mapped,cluster_profiles,direction_status
from sensory_audit import sha
from q1_interface import FrozenQ1V5OOF

ROOT=Path(__file__).resolve().parents[3];CANONICAL=Path('/home/mary/mathorcup');RESULT=ROOT/'03_结果/Q3_风格生成'
CONFIG=Path(__file__).resolve().parents[1]/'configs/E212_kaggle_external.yaml'
DIMENSIONS={'Bitter':'bitternes','Sweet':'sweetness','Sour':'acidity','Hoppy':'F_hops_all','Malty':'F_malt_all'}


def run():
    path=RESULT/'q3_run_manifest_E212.json'
    if path.exists():
        old=json.loads(path.read_text())
        for p,h in old['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Completed external audit input changed')
        for n,h in old['output_sha256'].items():
            if sha(RESULT/n)!=h:raise ValueError('Completed external audit output changed')
        print('E212 already complete; verified without rewriting');return old
    config=yaml.safe_load(CONFIG.read_text());q1=FrozenQ1V5OOF(CANONICAL)
    if (config.get('dimensions')!=DIMENSIONS or config.get('target_styles')!=['Tripel','Kriek','Lambic']
        or config.get('n_cluster_bootstrap')!=1000 or config.get('bootstrap_seed')!=20261026
        or any(config.get(k) is not False for k in ['training_or_selection_use','q1_retrained','q2_results_written',
            'q3_generator_replaced','rl_training','gpu_used'])):raise ValueError('E212 descriptive-only scope changed')
    external=CANONICAL/'data/2026初赛/公开数据/Kaggle_top_beer_information';raw_path=external/'beer_data_set.csv'
    mapping_path=external/'processed/kaggle_official_style_mapping_draft.csv';proxy_path=external/'processed/kaggle_sensory_proxy_draft.csv'
    real_path=CANONICAL/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/4.csv'
    chemical_path=real_path.with_name('1.csv');profile_path=RESULT/'q3_generated_profiles_E205.csv.gz'
    source_manifest=RESULT/'q3_run_manifest_E205.json';pool_path=RESULT/'q3_target_pool_E204.csv.gz'
    files=[CONFIG,Path(__file__),Path(__file__).with_name('kaggle_external.py'),raw_path,mapping_path,proxy_path,
        real_path,chemical_path,profile_path,source_manifest,pool_path]
    reports=CANONICAL/'docs/analysis/2026初赛'
    for n in ['Kaggle数据预处理与外部校验方案.md','Kaggle数据深度分析与映射审计.md','Kaggle多变量对齐与子维度审计.md','Kaggle_Q1模型外部校验.md']:
        files.append(reports/n)
    files.extend(sorted((reports/'Kaggle深度分析').glob('*')))
    hashes={str(p.resolve()):sha(p) for p in files if p.is_file()}
    previous=json.loads(source_manifest.read_text())
    if previous.get('status')!='complete' or previous['output_sha256'].get(profile_path.name)!=sha(profile_path):
        raise ValueError('Frozen final-NNLS generated profiles do not match manifest')
    raw=pd.read_csv(raw_path)
    if raw.shape!=(5558,21) or raw['key'].duplicated().any():raise ValueError('External static beer table contract changed')
    clean=normalized_flavors(raw);mapping=pd.read_csv(mapping_path).fillna({'subset':''});proxy=pd.read_csv(proxy_path)
    for dimension,target in DIMENSIONS.items():
        row=proxy[proxy.kaggle_dimension==dimension]
        if len(row)!=1 or row.iloc[0].official_target_primary!=target:raise ValueError('Preexisting sensory proxy changed')
    selected=select_mapped(clean,mapping,config['target_styles'])
    usable={s:g[g.flavor_usable].copy() for s,g in selected.items()}
    profiles=pd.read_csv(profile_path)
    if (len(profiles)!=45000 or profiles.duplicated(['style','sample_id','target_code']).any()
        or profiles.groupby(['style','sample_id']).size().ne(50).any()
        or profiles.groupby('style').sample_id.nunique().to_dict()!={'Kriek':300,'Lambic':300,'Tripel':300}):
        raise ValueError('Frozen 900-sample/50-target generated profile contract changed')
    real=pd.read_csv(real_path);chemical=pd.read_csv(chemical_path)
    columns=['prop_'+d for d in DIMENSIONS];bootstrap=cluster_profiles(usable,columns,1000,seed=20261026)
    rows=[]
    for a,b in itertools.combinations(config['target_styles'],2):
        for i,(dimension,target) in enumerate(DIMENSIONS.items()):
            external_delta=float(usable[a]['prop_'+dimension].mean()-usable[b]['prop_'+dimension].mean())
            boot=bootstrap[a][:,i]-bootstrap[b][:,i];finite=boot[np.isfinite(boot)]
            if len(finite)<950:raise ValueError('Insufficient valid brewery bootstrap replicates')
            low,high=np.quantile(finite,[.025,.975])
            means=profiles[profiles.target_code==target].groupby('style')[['prediction_z','audit_prediction_z']].mean()
            real_means=real.groupby('tasting_category_fine')[target].mean()
            primary=float(means.loc[a,'prediction_z']-means.loc[b,'prediction_z'])
            auditor=float(means.loc[a,'audit_prediction_z']-means.loc[b,'audit_prediction_z'])
            observed=float(real_means[a]-real_means[b])
            rows.append({'style_a':a,'style_b':b,'dimension':dimension,'official_target':target,
                'external_delta_proportion':external_delta,'external_cluster_ci_low':float(low),'external_cluster_ci_high':float(high),
                'n_valid_cluster_bootstrap':len(finite),'official_real_delta_z':observed,'generated_final_nnls_delta_z':primary,
                'generated_elasticnet_delta_z':auditor,'real_direction_status':direction_status(low,high,observed),
                'nnls_direction_status':direction_status(low,high,primary),'elasticnet_direction_status':direction_status(low,high,auditor),
                'n_external_a':len(usable[a]),'n_external_b':len(usable[b]),'n_generated_a':300,'n_generated_b':300,
                'scope':'posthoc_external_descriptive_not_training_or_independent_validation'})
    directions=pd.DataFrame(rows)
    pool=pd.read_csv(pool_path);sample_ids=profiles[['style','sample_id']].drop_duplicates()
    spectra=sample_ids.merge(pool,on=['style','sample_id'],how='left',validate='one_to_one')
    if len(spectra)!=900 or spectra['ethanol..v.v.'].isna().any():raise ValueError('Generated ABV chemical lineage incomplete')
    sample_summary=[]
    for style in config['target_styles']:
        source=selected[style];abv=source.loc[source.ABV.gt(0)&source.ABV.le(20),'ABV'].to_numpy(float)
        generated=10**spectra.loc[spectra['style']==style,'ethanol..v.v.'].to_numpy(float)
        original=10**chemical.loc[chemical.tasting_category_fine==style,'ethanol..v.v.'].to_numpy(float)
        q=np.quantile(abv,[.05,.5,.95])
        record={'style':style,'n_external_total':len(source),'n_external_flavor_usable':len(usable[style]),
            'n_breweries_flavor_usable':int(usable[style].Brewery.nunique()),'n_external_valid_abv':len(abv),
            'external_abv_q05_percent':q[0],'external_abv_median_percent':q[1],'external_abv_q95_percent':q[2],
            'official_abv_median_percent':float(np.median(original)),'generated_abv_median_percent':float(np.median(generated)),
            'generated_fraction_within_external_abv_q05_q95':float(((generated>=q[0])&(generated<=q[2])).mean())}
        record.update({f'external_mean_prop_{d}':float(usable[style]['prop_'+d].mean()) for d in DIMENSIONS})
        sample_summary.append(record)
    outputs=[]
    for name,table in [('q3_external_directions_E212.csv',directions),('q3_external_style_summary_E212.csv',pd.DataFrame(sample_summary)),
        ('q3_external_mapping_snapshot_E212.csv',mapping[mapping.official_style.isin(config['target_styles'])]),
        ('q3_external_proxy_snapshot_E212.csv',proxy[proxy.kaggle_dimension.isin(DIMENSIONS)])]:
        p=RESULT/name;table.to_csv(p,index=False);outputs.append(p)
    diagnostic={'raw_shape':list(raw.shape),'n_styles':int(raw.Style.nunique()),'n_breweries':int(raw.Brewery.nunique()),
        'n_all_zero_descriptor_rows':int((~clean.flavor_usable).sum()),'raw_columns':raw.columns.tolist(),
        'ibu_nunique_max_within_style':raw.groupby('Style')[['Min IBU','Max IBU']].nunique().max().to_dict(),
        'logged_action_transition_records_available':False,'chemical_concentrations_available':False,
        'verified_sample_level_official_join_available':False,'rl_data_supported':False,
        'adopted_use':'fixed_mapping_external_directional_audit',
        'conditional_future_uses':['heldout_brewery_ABV_marginal_prior','coarse_flavor_latent_prior_with_domain_alignment'],
        'not_implemented_future_uses':True,'n_directional_rows':len(directions),
        'nnls_direction_status_counts':{str(k):int(v) for k,v in directions.nnls_direction_status.value_counts().items()},
        'elasticnet_direction_status_counts':{str(k):int(v) for k,v in directions.elasticnet_direction_status.value_counts().items()},
        'notes':['词频比例与标准化感官评分不作绝对尺度误差比较','15个方向不是15项独立证据，区间未做家族多重性控制',
            '原先bootstrap脚本只重采样官方啤酒且Kaggle固定，其区间不是两数据源不确定性；本轮单独量化Kaggle酒厂抽样',
            '原报告Procrustes字段是1-disparity，不把它称普通相关系数','外部数据此前已被检查，本轮属于事后描述，不用于选择生成模型',
            '如果以后用于训练/奖励，必须另划酒厂留出并重新定义其验证角色，不能同时称未使用的外部验证']}
    dp=RESULT/'q3_external_data_eligibility_E212.json';dp.write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n');outputs.append(dp)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Read-only external source changed during audit')
    manifest={'experiment_id':'E212','status':'complete','scope':'posthoc_external_descriptive_audit',
        'q1_final_oof_sha256':q1.signature,'n_directions':len(directions),'n_cluster_bootstrap':1000,
        'n_inputs_verified':len(hashes),'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in outputs},
        'q1_retrained':False,'q2_results_written':False,'q3_generator_replaced':False,'rl_training':False,'gpu_used':False,
        'training_or_selection_use':False,'source_mapping_was_previously_inspected':True}
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if 'sha256' not in k},ensure_ascii=False,indent=2));return manifest


if __name__=='__main__':run()
