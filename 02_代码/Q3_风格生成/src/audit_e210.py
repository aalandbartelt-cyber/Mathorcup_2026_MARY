"""Apply the unchanged distribution replacement gates to E210."""
import json
import numpy as np
import pandas as pd
import yaml

from run_e210_latent_correlation import CONFIG,RESULT,sha,validate_config
from audit_e208 import replacement_decision,EXPECTED_RULES


def assess(repeats,metrics,config):
    if config['replacement_rules']!=dict(EXPECTED_RULES,candidate='latent_selected'):
        raise ValueError('E210 replacement rules changed')
    # Reuse the exact decision arithmetic; aliases are internal only.
    alias={'latent_selected':'joint_selected'}
    r=repeats.copy();r['method']=r.method.replace(alias)
    m=metrics.copy();m['method']=m.method.replace(alias)
    decision,paired,by_style=replacement_decision(r,m,dict(config,replacement_rules=EXPECTED_RULES.copy()))
    if decision['primary_distribution']=='joint_selected':decision['primary_distribution']='latent_selected'
    paired['method_new']=paired.method_new.replace({'joint_selected':'latent_selected'})
    decision['notes']=['本轮正式候选为训练内选择的秩空间相关收缩；单一ρ消融不作事后选择',
        '重复使用已有外折属于探索性比较，不能称独立确认']
    return decision,paired,by_style


def run():
    config=yaml.safe_load(CONFIG.read_text());validate_config(config)
    path=RESULT/'q3_run_manifest_E210.json';manifest=json.loads(path.read_text())
    if manifest['status']!='complete' or manifest['n_outer_folds']!=30:raise ValueError('Complete 30-fold run required')
    for p,h in manifest['input_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen E210 input changed')
    for n,h in manifest['output_sha256'].items():
        if sha(RESULT/n)!=h:raise ValueError('E210 output changed')
    for name,digest in manifest['checkpoint_manifest_sha256'].items():
        slot=RESULT/'q3_checkpoints_E210'/name;meta=json.loads((slot/'manifest.json').read_text())
        if sha(slot/'manifest.json')!=digest or meta['signature']!=manifest['signature']:raise ValueError('Checkpoint identity changed')
        if set(meta['train_beer'])&set(meta['validation_beer']) or len(meta['train_beer'])+len(meta['validation_beer'])!=250:
            raise ValueError('Fold identities overlap or miss samples')
        for n,h in meta['files'].items():
            if sha(slot/n)!=h:raise ValueError('Checkpoint output changed')
    metrics=pd.read_csv(RESULT/'q3_distribution_metrics_E210.csv');repeats=pd.read_csv(RESULT/'q3_metrics_by_repeat_E210.csv')
    inner=pd.read_csv(RESULT/'q3_inner_selection_E210.csv');keys=['repeat','fold','style','method','seed_index','metric_scale']
    if len(metrics)!=4500 or metrics.duplicated(keys).any() or not metrics.groupby(['method','metric_scale']).size().eq(450).all():
        raise ValueError('Expected complete 30-fold/3-style/5-seed/2-scale method coverage')
    if len(inner)!=1260 or not inner.groupby(['repeat','fold','style','inner_fold']).selected.sum().eq(1).all():
        raise ValueError('Incomplete training-only selection records')
    decision,paired,by_style=assess(repeats,metrics,config)
    outputs=[]
    dp=RESULT/'q3_replacement_decision_E210.json';dp.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');outputs.append(dp)
    for n,table in [('q3_latent_vs_copula_paired_E210.csv',paired),('q3_latent_vs_copula_by_style_E210.csv',by_style)]:
        p=RESULT/n;table.to_csv(p,index=False);outputs.append(p)
    audit={'experiment_id':'E210','status':'complete','n_outer_folds':30,'n_metric_rows':len(metrics),
        'n_inner_selection_rows':len(inner),'n_frozen_inputs_verified':len(manifest['input_sha256']),
        'decision':decision,'source_sha256':sha(__file__),'input_manifest_sha256':sha(path),
        'output_sha256':{p.name:sha(p) for p in outputs},'q1_retrained':False,'nnls_refit':False,'gpu_used':False}
    (RESULT/'q3_e210_audit_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2));return audit


if __name__=='__main__':run()
