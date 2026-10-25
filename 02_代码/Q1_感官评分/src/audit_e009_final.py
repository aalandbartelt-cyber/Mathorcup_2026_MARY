"""Recompute the frozen E009 audit without fitting any predictive model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from audit_q1_second_round import (
    METRICS, compare_full_frames, bottom_quartile_target_summary,
    target_delta_table, style_mae_table, worst_style_summary, paired_beer_bootstrap,
)
from q1_metrics import matrix_metrics, METRIC_COLUMNS
from nnls_stacking import fit_nnls_stacking

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / '03_结果/Q1_感官评分'
HARD = ['dms','diacetyl','t2n','stale_hops','barnyard','overall',
        'A_esters_flower','F_esters_flower','A_hops_tropical','F_hops_tropical']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out = R / 'audit_E009'
    out.mkdir(exist_ok=True)
    manifest = json.loads((R/'q1_run_manifest_E009.json').read_text())
    assert manifest['status']=='complete' and manifest['outer_folds']==30
    assert manifest['repeats']==list(range(10)) and manifest['primary_scope']=='group'
    assert all(digest(p)==v for p,v in manifest['input_source_hashes'].items())
    b=np.load(R/'q1_oof_predictions_E003.npz',allow_pickle=True)
    a=np.load(R/'q1_oof_predictions_E009.npz',allow_pickle=False)
    for key in ['y_true','beer','style','target_code','repeat','fold_assignment']:
        np.testing.assert_array_equal(a[key],b[key])
    assert a['y_true'].shape==(250,50) and len(np.unique(a['style']))==22
    frame=pd.read_csv(R/'q1_extended_metrics_E009.csv')
    base_frame=pd.read_csv(R/'q1_extended_metrics_E003.csv')
    checks=[]
    for scope in ['shared','group','target']:
        prediction=a['pred_nnls_'+scope]
        assert prediction.shape==(10,250,50) and np.isfinite(prediction).all()
        for rep in list(range(10))+['mean_oof']:
            evaluation='mean_oof' if rep=='mean_oof' else f'repeat_{rep}'
            value=prediction.mean(0) if rep=='mean_oof' else prediction[rep]
            recomputed=matrix_metrics(a['y_true'],value)
            saved=frame.loc[frame.evaluation.eq(f'pred_nnls_{scope}:{evaluation}')]
            assert len(saved)==1
            for metric in METRIC_COLUMNS:
                error=abs(recomputed[metric]-saved.iloc[0][metric])
                assert error < 1e-12,(scope,rep,metric,error)
                checks.append(dict(scope=scope,evaluation=evaluation,metric=metric,error=error))
    pd.DataFrame(checks).to_csv(out/'metric_recomputation.csv',index=False)
    params=pd.read_csv(R/'q1_selected_parameters_E009.csv')
    assert len(params)==30 and not params.duplicated(['repeat','fold']).any()
    assert set(map(tuple,params[['repeat','fold']].to_numpy()))==set((i,j) for i in range(10) for j in range(3))
    checkpoint=np.load(R/'q1_checkpoint_E009.npz',allow_pickle=False)
    checkpoint_rows=json.loads(str(checkpoint['parameter_rows_json'].item()))
    assert len(checkpoint_rows)==30
    dictionary=pd.read_csv(R/'q1_target_dictionary.csv').set_index('target_code')
    groups=dictionary.loc[a['target_code'].astype(str),'target_group'].astype(str).tolist()
    cache_checks=[]
    for _,row in params.iterrows():
        rep,fold=int(row['repeat']),int(row['fold'])
        candidates=list((R/'q1_base_cache_E009').glob(f'*r{rep:02d}*f{fold}*'))
        if len(candidates)!=1:
            candidates=list((R/'q1_base_cache_E009').glob(f'*repeat_{rep}*fold_{fold}*'))
        assert len(candidates)==1,(rep,fold,candidates)
        cache=np.load(candidates[0],allow_pickle=False)
        train=cache['train_index']; valid=cache['validation_index']
        assert not set(train)&set(valid) and sorted(np.r_[train,valid].tolist())==list(range(250))
        np.testing.assert_array_equal(valid,np.flatnonzero(a['fold_assignment'][rep]==fold))
        assert set(cache['inner_fold_assignment'].tolist())=={0,1,2}
        assert str(cache['run_signature'].item())==manifest['run_signature']
        fit=fit_nnls_stacking(cache['inner_prediction_stack'],a['y_true'][train],groups=groups,scope='group',shrinkage=.5)
        error=float(np.max(np.abs(fit.predict(cache['outer_base_stack'])-a['pred_nnls_group'][rep,valid])))
        assert error<1e-12,(rep,fold,error)
        assert np.min(fit.weights)>=0 and np.isfinite(fit.intercept).all()
        cache_checks.append(dict(repeat=rep,fold=fold,max_prediction_error=error,cache_sha256=digest(candidates[0])))
    pd.DataFrame(cache_checks).to_csv(out/'fold_cache_recomputation.csv',index=False)
    summary,paired,decision=compare_full_frames(base_frame,frame,experiment_id='E009',model='nnls_group',candidate_prefix='pred_nnls_group:repeat_')
    paired.to_csv(out/'paired_repeat_metrics.csv',index=False)
    bt=pd.read_csv(R/'q1_extended_metrics_by_target_E003.csv');bt=bt.loc[bt.evaluation.str.startswith('repeat_')]
    ct=pd.read_csv(R/'q1_extended_metrics_by_target_E009.csv');ct=ct.loc[ct.evaluation.str.startswith('pred_nnls_group:repeat_')]
    targets=target_delta_table(bt,ct);targets.to_csv(out/'all_target_deltas.csv',index=False)
    hard=targets.set_index('target_code').loc[HARD].reset_index();hard.to_csv(out/'preregistered_hard_targets.csv',index=False)
    tail=bottom_quartile_target_summary(bt,ct)
    targets[targets.target_code.isin(tail['bottom_targets'])].to_csv(out/'fixed_tail_13.csv',index=False)
    styles=style_mae_table(a['y_true'],a['style'],b['pred_extra_trees'],a['pred_nnls_group'])
    styles.to_csv(out/'styles_by_repeat.csv',index=False)
    style_mean=styles.groupby('style',as_index=False).agg(n_beers=('n_beers','first'),baseline_mae=('baseline_mae','mean'),candidate_mae=('candidate_mae','mean'),delta_mae=('delta_mae','mean'))
    style_mean.to_csv(out/'styles_22.csv',index=False)
    bootstrap=paired_beer_bootstrap(a['y_true'],b['pred_extra_trees'].mean(0),a['pred_nnls_group'].mean(0))
    bootstrap.to_csv(out/'paired_beer_bootstrap.csv',index=False)
    robust={
        'tail_13':tail,'hard_10_mean_deltas':{m:float(hard['delta_'+m].mean()) for m in METRICS},
        'hard_10_r2_declines':hard.loc[hard.delta_r2<0,'target_code'].tolist(),
        'worst_style':worst_style_summary(styles),
        'styles_with_lower_mae':int((style_mean.delta_mae<0).sum()),
        'style_increases':style_mean.loc[style_mean.delta_mae>0].to_dict('records'),
        'mean_oof':frame.loc[frame.evaluation.eq('pred_nnls_group:mean_oof'),METRIC_COLUMNS].iloc[0].to_dict(),
        'paired_bootstrap':bootstrap.to_dict('records'),
    }
    sources=[R/f'q1_{name}_E009{suffix}' for name,suffix in [('oof_predictions','.npz'),('extended_metrics','.csv'),('extended_metrics_by_target','.csv'),('selected_parameters','.csv'),('checkpoint','.npz'),('run_manifest','.json')]]
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    payload={'status':'complete','training_performed':False,'frozen_protocol':decision,'robustness':robust,
             'integrity':{'metric_values_recomputed':len(checks),'fold_cache_recomputations':len(cache_checks),
                          'source_hashes_verified':len(manifest['input_source_hashes']),'prediction_shape':[10,250,50],
                          'checkpoint_parameter_rows':len(checkpoint_rows),'source_artifact_sha256':source_hashes},
             'selection_under_user_instruction':{'selected':'E009','basis':'五项指标与稳健性综合选择；不改写原冻结规则',
             'tail_and_worst_style_material_regression':False,
             'limitations':'部分难目标与两种风格小幅变差；连续—有序融合的MAE/排序代理指标更优；原ACC±1绝对要求未达到'},
             'bootstrap_scope':'固定mean-OOF预测的啤酒级配对重采样，非重训、非训练样本不确定性区间'}
    (out/'formal_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    outputs=sorted(p for p in out.iterdir() if p.name!='SHA256SUMS' and p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in outputs))
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
