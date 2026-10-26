"""Aggregate sensory profiles without turning model agreement into safety."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sensory_audit import proxy_screen,candidate_status,verify_archived_prediction

KEYS=['beer','target_style','scenario','strategy','lambda']


def summarize(data,config,checkpoint):
    result=checkpoint.parent;final=data['final'];n=len(data['spectra']);ng=len(data['generated'])
    codes=list(final.target_codes);y=data['y'];styles=final.arrays['style'].astype(str)
    primary=np.full((10,n,50),np.nan);auditor=np.full_like(primary,np.nan);grades=np.full_like(primary,np.nan)
    supported=np.zeros((10,n),bool);audit_oof=np.full((10,250,50),np.nan)
    generated=[];generated_audit=[];generated_grade=[];generated_support=[];bias=[]
    for repeat in range(10):
        for fold in range(3):
            name=f'R{repeat:02d}F{fold}'
            with np.load(checkpoint/(name+'.npz'),allow_pickle=False) as a:
                mi=a['migration_index'];valid=a['validation_index']
                primary[repeat,mi]=a['primary_migration'];auditor[repeat,mi]=a['auditor_migration'];grades[repeat,mi]=a['grade_migration']
                supported[repeat,mi]=a['supported_migration'];audit_oof[repeat,valid]=a['auditor_validation']
                generated.append(a['primary_generated']);generated_audit.append(a['auditor_generated'])
                generated_grade.append(a['grade_generated']);generated_support.append(a['supported_generated'])
                outergen=pd.read_csv(result/f'q3_checkpoints_E202/{name}/generated.csv.gz')
                outergen=outergen[outergen.method=='style_copula_shrunk'].reset_index(drop=True)
                for (style,seed),index in outergen.groupby(['style','seed_index']).groups.items():
                    measured=y[valid][styles[valid]==style]
                    if not len(measured):raise ValueError('Generated style has no matching real validation measurements')
                    measured_mean=measured.mean(axis=0)
                    real_pred=a['primary_validation'][styles[valid]==style].mean(axis=0)
                    real_audit=a['auditor_validation'][styles[valid]==style].mean(axis=0)
                    gen_pred=a['primary_outergen'][list(index)].mean(axis=0)
                    gen_audit=a['auditor_outergen'][list(index)].mean(axis=0)
                    bias.append({'scope':'outer_fold_generated_profile_mean_diagnostic',
                        'repeat':repeat,'fold':fold,'style':style,'seed_index':int(seed),'n_real':len(measured),
                        'n_generated':len(index),'primary_generated_vs_measured_mean_mae':float(np.abs(gen_pred-measured_mean).mean()),
                        'primary_real_oof_vs_measured_mean_mae':float(np.abs(real_pred-measured_mean).mean()),
                        'primary_generated_vs_real_oof_mean_mae':float(np.abs(gen_pred-real_pred).mean()),
                        'auditor_generated_vs_measured_mean_mae':float(np.abs(gen_audit-measured_mean).mean()),
                        'auditor_real_oof_vs_measured_mean_mae':float(np.abs(real_audit-measured_mean).mean())})
    for array in (primary,auditor,grades,audit_oof):
        if not np.isfinite(array).all():raise ValueError('A repeat/sample/target prediction is incomplete')
    verify_archived_prediction(audit_oof,data['auditor_oof'])
    generated=np.stack(generated);generated_audit=np.stack(generated_audit);generated_grade=np.stack(generated_grade)
    generated_support=np.stack(generated_support)
    rows=[]
    for model,prediction in [('final_NNLS',final.arrays['prediction']),('ElasticNet_auditor',audit_oof)]:
        for repeat in range(10):
            scores=r2_score(y,prediction[repeat],multioutput='raw_values')
            errors=np.abs(prediction[repeat]-y).mean(axis=0)
            for j,code in enumerate(codes):rows.append({'model':model,'repeat':repeat,'target_code':code,
                'r2':float(scores[j]),'mae':float(errors[j])})
    reliability_by_repeat=pd.DataFrame(rows)
    reliability=reliability_by_repeat.groupby(['model','target_code'],as_index=False).agg(
        repeat_mean_r2=('r2','mean'),repeat_mean_mae=('mae','mean'),positive_r2_repeats=('r2',lambda v:int((v>0).sum())))
    reliability['proxy_reliable']=(reliability.repeat_mean_r2>config['proxy_mean_r2_min'])&(
        reliability.positive_r2_repeats>=config['proxy_positive_r2_repeats'])
    stale=codes.index(config['proxy_target'])
    stale_reliability=reliability[reliability.target_code==config['proxy_target']].set_index('model')
    primary_reliable=bool(stale_reliability.loc['final_NNLS','proxy_reliable'])
    audit_reliable=bool(stale_reliability.loc['ElasticNet_auditor','proxy_reliable'])
    source_primary=final.arrays['prediction'][:,data['positions']]
    source_auditor=audit_oof[:,data['positions']]
    delta_primary=primary-source_primary;delta_auditor=auditor-source_auditor
    unchanged=data['spectra'].strategy.eq('source_unchanged').to_numpy()
    zero_error=float(max(np.abs(delta_primary[:,unchanged]).max(),np.abs(delta_auditor[:,unchanged]).max()))
    if zero_error>1e-9:raise ValueError(f'Unchanged source produced an artificial sensory change: {zero_error}')

    profiles=data['generated'][['style','sample_id']].loc[data['generated'].index.repeat(50)].reset_index(drop=True)
    profiles['scope']='full_data_postselection_generated_profile'
    profiles['target_code']=np.tile(codes,ng)
    for name,values in [('prediction_z',generated.mean(axis=0)),('grade_0_6',generated_grade.mean(axis=0)),
                        ('model_fold_sd',generated.std(axis=0)),('audit_prediction_z',generated_audit.mean(axis=0))]:
        profiles[name]=values.ravel()
    profiles['predictor_support_fraction']=np.repeat(generated_support.mean(axis=0),50)
    profile_summary=profiles.groupby(['style','target_code'],as_index=False).agg(
        n_generated=('sample_id','size'),generated_prediction_mean=('prediction_z','mean'),
        generated_prediction_q10=('prediction_z',lambda v:float(v.quantile(.1))),
        generated_prediction_q90=('prediction_z',lambda v:float(v.quantile(.9))),
        generated_relative_grade_mean=('grade_0_6','mean'),
        model_fold_sd_mean=('model_fold_sd','mean'),audit_prediction_mean=('audit_prediction_z','mean'),
        predictor_support_fraction=('predictor_support_fraction','mean'))
    measured_rows=[]
    for style in config['target_styles']:
        actual=styles==style
        for j,code in enumerate(codes):measured_rows.append({'style':style,'target_code':code,
            'n_real_style':int(actual.sum()),'measured_style_mean_full_data':float(y[actual,j].mean()),
            'real_q1_mean_oof':float(final.arrays['mean_prediction'][actual,j].mean())})
    profile_summary=profile_summary.merge(pd.DataFrame(measured_rows),on=['style','target_code'],validate='one_to_one')
    profile_summary['scope']='full_data_postselection_generated_profile_not_oof'
    migration_profiles=data['spectra'][KEYS].loc[data['spectra'].index.repeat(50)].reset_index(drop=True)
    migration_profiles['target_code']=np.tile(codes,n)
    for name,values in [('prediction_z',primary.mean(axis=0)),('source_prediction_z',source_primary.mean(axis=0)),
                        ('grade_0_6',grades.mean(axis=0)),('prediction_delta',delta_primary.mean(axis=0)),
                        ('audit_prediction_z',auditor.mean(axis=0)),('audit_prediction_delta',delta_auditor.mean(axis=0))]:
        migration_profiles[name]=values.ravel()
    migration_profiles['scope']='postselection_design_source_excluded_predictor_profile'
    migration_profiles['source_missing_mask_preserved']=True
    proxy_rows=[]
    for i,row in data['spectra'][KEYS].iterrows():
        record=row.to_dict();record['spectrum_index']=i
        record.update(proxy_screen(delta_primary[:,i,stale],delta_auditor[:,i,stale],primary_reliable,audit_reliable,
                                   tolerance=config['proxy_numerical_tolerance']))
        record['support_repeat_fraction']=float(supported[:,i].mean())
        record['source_missing_fields']=int(np.isnan(data['x'].iloc[data['positions'][i]]).sum())
        record['primary_proxy_reliable']=primary_reliable;record['audit_proxy_reliable']=audit_reliable
        proxy_rows.append(record)
    proxy=data['summary'].rename(columns={'status':'direct_stage_status','eligible':'direct_stage_eligible'}).merge(
        pd.DataFrame(proxy_rows).drop(columns='physical_safety_certified'),on=KEYS,how='left',validate='one_to_one')
    proxy['status']=[candidate_status(available=bool(row.candidate_available),
        n_failed=float(row.n_failed) if row.candidate_available else 0,
        n_unknown=float(row.n_unknown) if row.candidate_available else 0,
        target_gain=bool(row.target_gain) if row.candidate_available else False,
        proxy_status=row.proxy_status,support_fraction=float(row.support_repeat_fraction) if row.candidate_available else 0.)
        for row in proxy.itertuples()]
    proxy['eligible']=proxy.status.eq('eligible_statistical_candidate');proxy['physical_safety_certified']=False
    proxy['proxy_scope']='two_frozen_model_predictions_not_measurement_or_confidence_guarantee'
    bias=pd.DataFrame(bias)
    metric_columns=[c for c in bias if c.endswith('_mae')]
    by_repeat=[]
    for (repeat,style),group in bias.groupby(['repeat','style']):
        row={'repeat':repeat,'style':style,'scope':'repeat_aggregated_outer_profile_diagnostic'}
        for column in metric_columns:row[column]=float(np.average(group[column],weights=group.n_real))
        by_repeat.append(row)
    aggregate=proxy.groupby(['scenario','strategy'],as_index=False).agg(n_candidates=('beer','size'),
        n_available=('candidate_available','sum'),n_eligible=('eligible','sum'),
        mean_primary_stale_delta=('primary_mean_delta','mean'),mean_audit_stale_delta=('audit_mean_delta','mean'),
        mean_support_fraction=('support_repeat_fraction','mean'))
    decision={'status':'sensory_profile_and_proxy_audit_complete',
        'distribution_primary':'style_copula_shrunk','n_outer_folds':30,
        'n_generated_profiles':ng,'n_profile_summary_rows':len(profile_summary),'n_migration_candidates':len(proxy),
        'n_migration_spectra':n,'n_eligible_statistical_candidates':int(proxy.eligible.sum()),
        'status_counts':{str(k):int(v) for k,v in proxy.status.value_counts().items()},
        'stale_hops_reliability':stale_reliability.reset_index().to_dict('records'),
        'max_unchanged_source_prediction_delta':zero_error,
        'q1_retrained':False,'nnls_refit':False,'gpu_used':False,'physical_safety_certified':False,
        'q2_top3_used_for_model_weights':False,
        'notes':['完整候选分母含无可行结构的47条拒绝记录；未知直接证据不计通过',
                 '预测不恶化只指冻结模型输出；不代表新啤酒实测或物理安全认证',
                 '重复分位与模型间波动不是置信覆盖保证；全数据设计与外折分布诊断分开报告']}
    tables={'q3_generated_profiles_E205.csv.gz':profiles,'q3_generated_profile_summary_E205.csv':profile_summary,
            'q3_migration_sensory_profiles_E205.csv.gz':migration_profiles,
            'q3_migration_proxy_audit_E205.csv':proxy,'q3_migration_proxy_aggregate_E205.csv':aggregate,
            'q3_proxy_model_reliability_E205.csv':reliability,
            'q3_proxy_reliability_by_repeat_E205.csv':reliability_by_repeat,
            'q3_outer_profile_bias_E205.csv':bias,'q3_outer_profile_bias_by_repeat_E205.csv':pd.DataFrame(by_repeat)}
    outputs=[]
    for name,frame in tables.items():
        path=result/name;frame.to_csv(path,index=False,compression='gzip' if name.endswith('.gz') else None);outputs.append(path)
    path=result/'q3_sensory_decision_E205.json';path.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');outputs.append(path)
    return outputs,decision
