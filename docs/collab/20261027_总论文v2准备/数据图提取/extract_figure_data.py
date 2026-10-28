"""Read-only descriptive data extraction for draft data-overview figures.

No fitting, imputation, ranking change, experiment runner, or manuscript edits.
"""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('/home/mary/mathorcup')
OUT=Path(__file__).resolve().parent
ATT=ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件'
Q2=ROOT/'03_结果/Q2_关键化合物'
Q3=ROOT/'.worktrees/q3-foundation/03_结果/Q3_风格生成'
SOURCE=ROOT/'.worktrees/q3-foundation/02_代码/Q3_风格生成/src'
STYLES=['Tripel','Kriek','Lambic']
TARGETS={'alcohol':'酒精感','bitternes':'苦味','F_hops_all':'酒花味',
         'X4vg':'4VG','acidity':'酸度','lactic':'乳酸感'}
SOURCES=[]

def track(path,role):
    path=Path(path)
    SOURCES.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path.read_bytes()).hexdigest(),'role':role})
    return path

def csv(frame,name):
    frame.to_csv(OUT/name,index=False)

def main():
    chemistry=pd.read_csv(track(ATT/'1.csv','raw chemical fields, styles and beer keys'))
    sensory=pd.read_csv(track(ATT/'4.csv','raw standardized observed sensory targets'))
    official=pd.read_csv(track(ATT/'2.csv','full-data descriptive official correlation matrix only'))
    # Read both attachment-8 sheets to record their actual scopes, without editing values.
    p8=track(ATT/'8.xlsx','field units and calibrated measurement reference')
    calibrated=pd.read_excel(p8,sheet_name='Calibrated chemical dataset')
    annotations=pd.read_excel(p8,sheet_name='Sensory thresholds')
    candidates=pd.read_csv(track(Q2/'q2_candidate_dictionary.csv','frozen strict-candidate eligibility'))
    final=pd.read_csv(track(Q2/'q2_final_top3_E112.csv','frozen final 150 relationships, counted only'))
    feature_cols=[x for x in chemistry if x not in ['beer','beer_id','tasting_category_fine']]
    strict=candidates.loc[candidates.eligible_strict,'feature_code'].tolist()
    excluded=candidates.loc[~candidates.eligible_strict,['feature_code','compound_name','factor_type','exclusion_reason','unit']].copy()
    assert len(feature_cols)==231 and len(strict)==218 and len(excluded)==13
    assert len(final)==150 and final.groupby('target_code').size().eq(3).all()
    assert set(final.feature_code)<=set(strict)
    csv(pd.DataFrame([
        {'stage_order':1,'stage_cn':'全部测量字段','count':231,'count_unit':'字段','scope':'数据输入'},
        {'stage_order':2,'stage_cn':'非严格单体字段','count':13,'count_unit':'字段','scope':'预先排除'},
        {'stage_order':3,'stage_cn':'严格单体候选','count':218,'count_unit':'字段','scope':'候选资格'},
        {'stage_order':4,'stage_cn':'感官目标','count':50,'count_unit':'目标','scope':'输出任务'},
        {'stage_order':5,'stage_cn':'各目标前三关系','count':150,'count_unit':'关系','scope':'50目标×3；非字段数'},
    ]),'Q2_输入与候选规模.csv')
    category_cn={'physicochemical':'理化指标','composition':'混合或聚合组成',
                 'element':'元素','derived_aggregate':'派生或汇总指标'}
    excluded['category_cn']=excluded.factor_type.map(category_cn)
    csv(excluded,'Q2_排除13字段.csv')
    counts=excluded.groupby(['factor_type','category_cn'],sort=False).size().rename('count').reset_index()
    csv(counts,'Q2_排除类型数量.csv')
    x=chemistry[strict]
    corr=x.corr(method='spearman',min_periods=3)
    counts_pair=x.notna().to_numpy(dtype=int).T@x.notna().to_numpy(dtype=int)
    i,j=np.triu_indices(len(strict),1)
    rho=corr.to_numpy()[i,j]
    pairs=pd.DataFrame({'field_a':np.asarray(strict)[i],'field_b':np.asarray(strict)[j],
                        'spearman':rho,'abs_spearman':np.abs(rho),'n_pairwise_complete':counts_pair[i,j],
                        'scope':'full_data_descriptive_pairwise_complete_no_imputation'})
    assert len(pairs)==23653 and np.isfinite(rho).all()
    csv(pairs,'Q2_218候选成对相关.csv')
    bins=np.array([0,.2,.4,.6,.8,1.00000000001])
    labels=['0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1.0']
    bin_frame=pairs.assign(bin=pd.cut(pairs.abs_spearman,bins,labels=labels,include_lowest=True,right=False))
    binned=bin_frame.groupby('bin',observed=False).size().rename('pair_count').reset_index()
    binned['fraction']=binned.pair_count/len(pairs);binned['scope']='full_data_descriptive'
    csv(binned,'Q2_相关强度分箱.csv')
    # Full-data official matrix is a diagnostic reference, never an outer-fold graph.
    official_values=official.iloc[:,1:].to_numpy(float)
    computed=chemistry[feature_cols].corr(method='spearman').to_numpy()
    matrix_diff=float(np.nanmax(np.abs(official_values-computed)))

    profiles=pd.read_csv(track(Q3/'q3_generated_profiles_E205.csv.gz','existing 900 generated profiles in long target form'))
    pool=pd.read_csv(track(Q3/'q3_target_pool_E204.csv.gz','existing full-data postselection 2000-per-style pool'))
    track(Q3/'q3_run_manifest_E205.json','profile provenance and frozen predictor identities')
    track(Q3/'q3_run_manifest_E202.json','outer-fold distribution evaluation scope')
    track(SOURCE/'run_e205_sensory_audit.py','actual inputs and Q2 frozen model use; no Top-3 optimizer input')
    track(SOURCE/'run_e204_constrained_migration.py','actual direct chemical input and pool generation')
    track(ROOT/'02_代码/Q3_风格生成/src/run_e200_contract.py','actual direct attachment1/4/8 reads')
    csv(pd.DataFrame({'target_code':list(TARGETS),'target_cn':list(TARGETS.values()),
                      'scale':['附件4标准化连续值；非原始0–6分']*6}),'Q3_六维定义.csv')
    raw=sensory.loc[sensory.tasting_category_fine.isin(STYLES),['beer','tasting_category_fine']+list(TARGETS)].copy()
    long=raw.melt(id_vars=['beer','tasting_category_fine'],var_name='target_code',value_name='observed_value')
    long=long.rename(columns={'tasting_category_fine':'style'})
    long['target_cn']=long.target_code.map(TARGETS)
    long['scope']='observed_attachment4_standardized_continuous_not_0_6'
    assert len(long)==294 and np.isfinite(long.observed_value).all()
    csv(long,'Q3_三风格六维实测长表.csv')
    summary=long.groupby(['style','target_code','target_cn'],sort=False).observed_value.agg(n='size',mean='mean',median='median',minimum='min',maximum='max').reset_index()
    csv(summary,'Q3_三风格六维实测摘要.csv')
    checks=[];selected=[]
    for style in STYLES:
        group=profiles.loc[profiles['style']==style]
        ids=group.sample_id.drop_duplicates()
        pool_group=pool.loc[pool['style']==style]
        spectra=pool_group.set_index('sample_id').loc[ids,feature_cols]
        assert len(ids)==300 and len(group)==15000 and group.target_code.nunique()==50
        assert not group.duplicated(['sample_id','target_code']).any()
        assert group.groupby('sample_id').size().eq(50).all()
        zz=group.prediction_z.to_numpy(float);grades=group.grade_0_6.to_numpy(float);xx=spectra.to_numpy(float)
        checks.append({'style':style,'n_real_observed':int((sensory.tasting_category_fine==style).sum()),
            'n_pool':len(pool_group),'n_profile_spectra':len(ids),'n_targets':50,'n_profile_rows':len(group),
            'finite_prediction_z':int(np.isfinite(zz).sum()),'finite_grade_0_6':int(np.isfinite(grades).sum()),
            'finite_spectrum_cells':int(np.isfinite(xx).sum()),'expected_spectrum_cells':int(xx.size),
            'all_prediction_z_finite':bool(np.isfinite(zz).all()),'all_spectra_finite':bool(np.isfinite(xx).all()),
            'duplicate_spectrum_ids':int(ids.duplicated().sum()),
            'scope':'existing_full_data_postselection_profiles_not_new_observations'})
        selected.append(pd.DataFrame({'style':style,'sample_id':ids.to_numpy(),'source_pool_scope':'full_data_postselection_target_pool'}))
    csv(pd.DataFrame(checks),'Q3_实测与画像完整性.csv')
    csv(pd.concat(selected,ignore_index=True),'Q3_900谱来源身份.csv')
    profile_counts=pd.DataFrame(checks)
    raw_styles=chemistry.tasting_category_fine.value_counts().rename_axis('style').rename('n').reset_index()
    csv(raw_styles,'Q1_22风格样本数_可复用.csv')
    dimensions={'attachment1':list(chemistry.shape),'attachment2':list(official.shape),'attachment4':list(sensory.shape),
                'attachment8_calibrated':list(calibrated.shape),'attachment8_thresholds':list(annotations.shape)}
    checks_json={'dimensions':dimensions,'q2':{'strict_candidates':len(strict),'excluded_fields':len(excluded),
        'candidate_pairs':len(pairs),'pairwise_n_min':int(pairs.n_pairwise_complete.min()),'pairwise_n_max':int(pairs.n_pairwise_complete.max()),
        'min_abs_rho':float(pairs.abs_spearman.min()),'max_abs_rho':float(pairs.abs_spearman.max()),
        'pairs_abs_rho_at_least_0_6':int((pairs.abs_spearman>=.6).sum()),
        'official2_vs_pairwise_spearman_max_abs_diff':matrix_diff},
        'q3':{'raw_style_counts':{s:int((sensory.tasting_category_fine==s).sum()) for s in STYLES},
        'raw_6d_rows':len(long),'profile_rows':len(profiles),'profile_spectra':int(profile_counts.n_profile_spectra.sum()),
        'finite_prediction_z':int(profile_counts.finite_prediction_z.sum()),'finite_spectrum_cells':int(profile_counts.finite_spectrum_cells.sum()),
        'q2_top3_in_optimizer':False,'q2_interface_actual':'frozen final NNLS caches + archived ElasticNet parameters/OOF; not final Top-3 ranking'},
        'no_training':True,'no_imputation':True,'no_experiment_rerun':True}
    (OUT/'描述统计核对.json').write_text(json.dumps(checks_json,ensure_ascii=False,indent=2))
    (OUT/'source_manifest.json').write_text(json.dumps({'sources':SOURCES,'outputs':[p.name for p in OUT.glob('*.csv')]},ensure_ascii=False,indent=2))
    print(json.dumps(checks_json,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
