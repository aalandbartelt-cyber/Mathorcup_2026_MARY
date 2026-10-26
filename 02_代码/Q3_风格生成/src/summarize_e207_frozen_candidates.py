"""Sensitivity of locked candidates; no candidate search or refitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_e205b_sensory_search import RESULT
from run_e204_constrained_migration import _direct_vectors,_candidate_direct
from q3_constraints import audit_concentrations
from sensory_audit import sha


def run():
    lock_path=RESULT/'q3_candidate_lock_E205B.json';manifest_path=RESULT/'q3_run_manifest_E205B.json'
    verification_path=RESULT/'q3_e205b_audit_manifest.json'
    lock=json.loads(lock_path.read_text());manifest=json.loads(manifest_path.read_text())
    verification=json.loads(verification_path.read_text())
    if manifest.get('status')!='complete_once_only_audit' or verification.get('status')!='complete':
        raise ValueError('Verified locked search and audit required before sensitivity')
    for collection in [lock,manifest]:
        for name,digest in collection['output_sha256'].items():
            if sha(RESULT/name)!=digest:raise ValueError('Frozen candidate/audit output changed')
    spectra=pd.read_csv(RESULT/'q3_locked_spectra_E205B.csv.gz').set_index('selection_id')
    outcome=pd.read_csv(RESULT/'q3_once_audit_E205B.csv')
    primary=outcome[outcome.selection_rule.eq('minimum_change')].copy()
    if len(primary)!=33:raise ValueError('Primary rule source-target denominator changed')
    source_table=pd.read_csv(RESULT/'q3_source_defects_E200.csv');defects=pd.read_csv(RESULT/'q3_defect_registry_E200.csv')
    fields=pd.read_csv(RESULT/'q3_field_contract_E200.csv');features=fields.feature_code.tolist()
    if set(features)!=set(spectra.columns):raise ValueError('Full chemical feature contract changed')
    rows=[]
    for row in primary.itertuples():
        if not row.candidate_available:continue
        direct,df,names,scales,sv,known,unused=_direct_vectors(int(row.beer),source_table,defects,fields)
        c=_candidate_direct(spectra.loc[row.selection_id,features].to_numpy(float),[features.index(f) for f in df],scales)
        for statistic in ['median','mean']:
            thresholds=direct['threshold_'+statistic].to_numpy(float)
            for scenario in ['C1','C2','C3']:
                audit=audit_concentrations(sv,c,known,scenario,threshold=thresholds)
                for i,record in enumerate(audit.to_dict('records')):
                    rows.append({'selection_id':row.selection_id,'beer':int(row.beer),'target_style':row.target_style,
                        'threshold_statistic':statistic,'defect':names[i],**record})
    direct=pd.DataFrame(rows)
    proxy=[]
    for margin in [0.,.01,.02]:
        for row in primary.itertuples():
            primary_ok=bool(row.candidate_available and row.primary_mean_delta<=-margin+1e-8
                            and row.primary_repeat_q90_delta<=-margin+1e-8)
            auditor_ok=bool(row.candidate_available and row.audit_mean_delta<=-margin+1e-8
                            and row.audit_repeat_q90_delta<=-margin+1e-8)
            proxy.append({'selection_id':row.selection_id,'margin_z':margin,'primary_proxy_nonworsening':primary_ok,
                'auditor_proxy_nonworsening':auditor_ok,'both_proxy_nonworsening':primary_ok and auditor_ok,
                'was_main_candidate_eligible':bool(row.eligible)})
    proxy=pd.DataFrame(proxy)
    # A readable status for every source-target pair, including unavailable ones.
    public=primary[['selection_id','beer','target_style','candidate_available','relative_target_gain','source_change_rms',
        'primary_mean_delta','primary_repeat_q90_delta','audit_mean_delta','audit_repeat_q90_delta',
        'n_failed','n_unknown','audit_support_repeat_fraction','audited_status','eligible','physical_safety_certified']].copy()
    status_names={'eligible_statistical_candidate':'可供复测的统计候选','needs_direct_verification':'直接化学证据不足',
        'rejected_no_primary_feasible_candidate':'固定搜索未找到可行谱','proxy_variation_uncertain':'感官代理变化不稳定',
        'proxy_disagreement':'感官模型方向分歧','predicted_worsening':'预测陈旧酒花感升高',
        'outside_predictor_support':'超出预测支持范围','no_target_gain':'目标靠近不足'}
    public['status_cn']=public.audited_status.map(status_names)
    public['scope']='postselection_statistical_design_not_brewing_validation'
    summary=[]
    for (stat,scenario),group in direct.groupby(['threshold_statistic','scenario']):
        summary.append({'threshold_statistic':stat,'scenario':scenario,'n_candidates':group.selection_id.nunique(),
            'n_direct_checks':len(group),'n_passed':int(group.passed.eq(True).sum()),
            'n_unknown':int(group.passed.isna().sum()),'n_failed':int(group.passed.eq(False).sum())})
    outputs=[]
    for name,frame in [('q3_frozen_candidate_sensitivity_E207.csv',direct),('q3_proxy_margin_sensitivity_E207.csv',proxy),
        ('q3_direct_sensitivity_summary_E207.csv',pd.DataFrame(summary)),('q3_primary_candidates_E207.csv',public)]:
        path=RESULT/name;frame.to_csv(path,index=False);outputs.append(path)
    accepted=set(primary.loc[primary.eligible,'selection_id'])
    accepted_direct=direct[direct.selection_id.isin(accepted)]
    c1=accepted_direct[accepted_direct.scenario.eq('C1')]
    if c1.passed.isna().any() or c1.passed.eq(False).any():raise ValueError('Accepted C1 candidate failed locked relative constraints')
    if public.physical_safety_certified.any():raise ValueError('Physical safety is not identifiable from these files')
    report={'experiment_id':'E207','status':'frozen_candidate_sensitivity_complete',
        'scope':'sensitivity_only_same_locked_spectra_no_reselection',
        'n_primary_source_target_pairs':33,'n_primary_available':int(primary.candidate_available.sum()),
        'n_primary_eligible':int(primary.eligible.sum()),'n_source_beers_of_eligible':int(primary[primary.eligible].beer.nunique()),
        'accepted_c1_known_failures':0,'absolute_threshold_status':'trans_2_nonenal_threshold_unknown',
        'margin_sensitivity_for_accepted':proxy[proxy.selection_id.isin(accepted)].to_dict('records'),
        'q1_retrained':False,'nnls_refit':False,'candidate_reselection':False,'physical_safety_certified':False,
        'input_sha256':{p.name:sha(p) for p in [lock_path,manifest_path,verification_path]},
        'source_sha256':sha(__file__),'output_sha256':{p.name:sha(p) for p in outputs},
        'notes':['均值/中位阈值不改变C1相对约束；未知绝对阈值始终保留未知',
            '未知LOD和缺失源浓度没有可证实的有限区间，不编造蒙特卡洛安全率',
            '0.01/0.02 z裕量只检视已锁候选的预测脆弱性，不修改主规则或重新选点']}
    (RESULT/'q3_run_manifest_E207.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
