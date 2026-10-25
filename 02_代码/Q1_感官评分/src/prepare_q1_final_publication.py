"""Export audited final-model evidence under academic publication names."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from baseline_models import empirical_cdf_grade

ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'03_结果/Q1_感官评分'
OUT=R/'publication_final'
SPECS=[('E003','extra_trees','ExtraTrees基线'),('E005','continuous_blend','连续凸融合'),
       ('E006','dual_route','连续—有序融合'),('E007','shrunk_gate','逐目标收缩融合'),
       ('E009','nnls_group','感官组收缩NNLS')]

def main():
    audit=json.loads((R/'audit_E009/formal_audit.json').read_text())
    assert audit['status']=='complete' and audit['selection_under_user_instruction']['selected']=='E009'
    for name,digest in audit['integrity']['source_artifact_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    OUT.mkdir(exist_ok=True)
    a=np.load(R/'q1_oof_predictions_E009.npz',allow_pickle=False)
    b=np.load(R/'q1_oof_predictions_E003.npz',allow_pickle=True)
    y,p=a['y_true'],a['pred_nnls_group'];mean=p.mean(0)
    codes=a['target_code'].astype(str);styles=a['style'].astype(str)
    dictionary=pd.read_csv(R/'q1_target_dictionary.csv')
    dictionary['target_name_en']=dictionary['target_name_en'].str.replace(r'\s+',' ',regex=True).str.strip()
    parts=[];means=[]
    for experiment,key,name in SPECS:
        f=pd.read_csv(R/f'q1_extended_metrics_{experiment}.csv')
        prefix='repeat_' if experiment=='E003' else f'pred_{key}:repeat_'
        sub=f[f.evaluation.str.startswith(prefix)].copy()
        sub['repeat']=sub.evaluation.str.rsplit('_',n=1).str[-1].astype(int)
        sub['display_name']=name
        sub=sub[['display_name','repeat','r2','mae','rmse','spearman','qwk_proxy','acc_pm1_proxy']]
        assert len(sub)==10
        parts.append(sub)
        row={'display_name':name}
        for metric in ['r2','mae','rmse','spearman','qwk_proxy','acc_pm1_proxy']:
            row[metric+'_mean']=float(sub[metric].mean());row[metric+'_sd']=float(sub[metric].std(ddof=1))
        means.append(row)
    comparison=pd.concat(parts,ignore_index=True)
    comparison.to_csv(OUT/'model_metrics_by_repeat.csv',index=False)
    pd.DataFrame(means).to_csv(OUT/'model_summary.csv',index=False)
    fig=comparison.rename(columns={m:'macro_'+m for m in ['r2','mae','rmse','spearman']})
    fig.to_csv(OUT/'model_comparison.csv',index=False)
    m=pd.read_csv(R/'q1_extended_metrics_by_target_E009.csv')
    t=m[m.evaluation.eq('pred_nnls_group:mean_oof')].drop(columns=['experiment_id','evaluation','model']).merge(dictionary,on='target_code',validate='one_to_one')
    t.to_csv(OUT/'target_metrics_final.csv',index=False)
    pm=m[m.evaluation.str.startswith('pred_nnls_group:repeat_')].copy()
    pm['repeat']=pm.evaluation.str.rsplit('_',n=1).str[-1].astype(int)
    pm.drop(columns=['experiment_id','evaluation','model']).to_csv(OUT/'target_metrics_by_repeat.csv',index=False)
    long=pd.DataFrame({'beer':np.repeat(a['beer'],50),'tasting_category_fine':np.repeat(styles,50),
                       'target_code':np.tile(codes,250),'y_true':y.ravel(),'y_pred_mean':mean.ravel(),
                       'prediction_sd':p.std(0,ddof=1).ravel(),'residual':(y-mean).ravel()})
    long.to_csv(OUT/'final_oof_predictions.csv.gz',index=False,compression='gzip')
    grades=np.empty_like(p)
    for rep in range(10):
        for fold in range(3):
            mask=a['fold_assignment'][rep]==fold
            grades[rep,mask]=empirical_cdf_grade(y[~mask],p[rep,mask])
    long['grade_0_6']=grades.mean(0).ravel()
    long['grade_sd']=grades.std(0,ddof=1).ravel()
    long.to_csv(OUT/'final_oof_with_relative_grades.csv.gz',index=False,compression='gzip')
    # Fixed examples retain the previously documented choices, including a weak defect.
    representative=long[long.target_code.isin(['acidity','X4vg','stale_hops'])].copy()
    representative['tier']=representative.target_code.map({'acidity':'高','X4vg':'中','stale_hops':'低'})
    representative.to_csv(OUT/'representative_oof.csv.gz',index=False,compression='gzip')
    style_rows=[]
    for style in sorted(set(styles)):
        mask=styles==style
        style_rows.append({'tasting_category_fine':style,'n_beers':int(mask.sum()),
                           'mae_all_targets':float(np.abs(y[mask]-mean[mask]).mean()),
                           'baseline_mae':float(np.abs(y[mask]-b['pred_extra_trees'].mean(0)[mask]).mean()),
                           'rmse_all_targets':float(np.sqrt(((y[mask]-mean[mask])**2).mean())),
                           'mean_residual':float((y[mask]-mean[mask]).mean())})
    pd.DataFrame(style_rows).to_csv(OUT/'style_metrics_final.csv',index=False)
    # One fixed outer training fold defines the illustrative CDF curves.
    train=a['fold_assignment'][0]!=0
    mapping=[]
    for target in ['A_malt_all','F_esters_flower','acidity','alcohol']:
        pos=np.flatnonzero(codes==target)[0]
        grid=np.linspace(min(y[train,pos].min(),mean[:,pos].min()),max(y[train,pos].max(),mean[:,pos].max()),301)
        for x,value in zip(grid,empirical_cdf_grade(y[train,pos,None],grid[:,None]).ravel()):
            mapping.append({'target_code':target,'z_score':x,'grade_0_6':value})
    pd.DataFrame(mapping).to_csv(OUT/'scale_mapping_example.csv',index=False)
    params=pd.read_csv(R/'q1_selected_parameters_E009.csv')
    weights=[]
    base_names=['ExtraTrees','HGB','Ridge（秩高斯）','PLS2（秩高斯）']
    for _,row in params.iterrows():
        par=json.loads(row.nnls_parameters_json)['group']
        for j,target in enumerate(codes):
            for k,name in enumerate(base_names):
                weights.append({'repeat':int(row['repeat']),'fold':int(row['fold']),'target_code':target,
                                'target_group':dictionary.set_index('target_code').loc[target,'target_group'],
                                'base_model':name,'weight':par['weights'][j][k],
                                'intercept':par['intercept'][j]})
    w=pd.DataFrame(weights);w.to_csv(OUT/'stacking_weights_by_fold.csv',index=False)
    w.groupby(['target_group','base_model'],sort=False).agg(weight_mean=('weight','mean'),weight_sd=('weight','std')).reset_index().to_csv(OUT/'stacking_weights_summary.csv',index=False)
    ablations=[]
    full=pd.read_csv(R/'q1_extended_metrics_E009.csv')
    for scope,label in [('shared','共享权重NNLS'),('group','感官组收缩NNLS'),('target','逐目标收缩NNLS')]:
        z=full[full.evaluation.str.startswith(f'pred_nnls_{scope}:repeat_')]
        ablations.append({'model':label,**{c:float(z[c].mean()) for c in ['r2','mae','spearman','qwk_proxy','acc_pm1_proxy']}})
    pd.DataFrame(ablations).to_csv(OUT/'weight_structure_ablation.csv',index=False)
    for name in ['all_target_deltas','fixed_tail_13','preregistered_hard_targets','styles_22','paired_beer_bootstrap']:
        pd.read_csv(R/f'audit_E009/{name}.csv').to_csv(OUT/f'{name}.csv',index=False)
    clean={'model':'感官组收缩NNLS','base_learners':base_names,'shrinkage':.5,
           'weights_sum_to_one':False,'free_target_intercept':True,'outer_repeats':10,'outer_folds':3,'inner_folds':3,
           'repeat_mean':means[-1], 'mean_oof':audit['robustness']['mean_oof'],
           'relative_grades':'每个验证折仅以对应训练折目标经验分布映射，非原始评委均分',
           'ordered_metrics':'观测与预测分别七分位分档的事后排序代理，不是真实0—6准确率',
           'source_prediction_sha256':hashlib.sha256((R/'q1_oof_predictions_E009.npz').read_bytes()).hexdigest()}
    (OUT/'final_model_summary.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n')
    # A single pickle-free interface for downstream evaluation; no fitted decoder is claimed.
    np.savez_compressed(OUT/'final_model_oof.npz',y_true=y,prediction=p,mean_prediction=mean,
                        beer=a['beer'],style=a['style'],target_code=codes,fold_assignment=a['fold_assignment'],
                        relative_grades=grades)
    print('Publication evidence exported:',OUT)

if __name__=='__main__':
    main()
