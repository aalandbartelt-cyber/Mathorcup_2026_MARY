"""Robust, fold-local sensitivity audit for the completed E202 baseline."""

from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from q3_evaluation import distribution_diagnostics
from run_e200_contract import validate_cv


ROOT=Path(__file__).resolve().parents[3]
RESULT=ROOT/'03_结果/Q3_风格生成'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_winsorize(training,*arrays,lower=.01,upper=.99):
    training=np.asarray(training,float)
    values=[np.asarray(array,float) for array in arrays]
    if training.ndim!=2 or len(training)<2 or not np.isfinite(training).all():
        raise ValueError('finite two-dimensional training matrix required')
    if not 0.<=lower<upper<=1.:
        raise ValueError('invalid winsor quantiles')
    if any(array.ndim!=2 or array.shape[1]!=training.shape[1] or not np.isfinite(array).all() for array in values):
        raise ValueError('finite aligned matrices required')
    low=np.quantile(training,lower,axis=0);high=np.quantile(training,upper,axis=0)
    clipped_training=np.clip(training,low,high)
    clipped=[np.clip(array,low,high) for array in values]
    scale=np.std(clipped_training,axis=0);scale=np.where(scale>1e-12,scale,1.)
    return clipped_training,tuple(clipped),scale,(low,high)


def run():
    attachment=ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv'
    manifest_path=RESULT/'q3_run_manifest_E202.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete' or manifest.get('n_outer_folds')!=30:
        raise ValueError('complete E202 manifest required')
    chemistry=pd.read_csv(attachment);splits=pd.read_csv(split_path)
    validate_cv(chemistry,splits,n_repeats=10,n_folds=3)
    features=[column for column in chemistry if column not in ['beer','beer_id','tasting_category_fine']]
    x=chemistry[features].to_numpy(float);styles=chemistry.tasting_category_fine.to_numpy(str)
    rows=[];checkpoint_hashes={}
    for (repeat,fold),group in splits.groupby(['repeat','fold'],sort=True):
        repeat=int(repeat);fold=int(fold);split_id=f'R{repeat:02d}F{fold}'
        slot=RESULT/'q3_checkpoints_E202'/split_id
        checkpoint_hashes[split_id]=sha(slot/'manifest.json')
        generated=pd.read_csv(slot/'generated.csv.gz')
        valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        train_x=imputer.fit_transform(x[train]);valid_x=imputer.transform(x[valid])
        for (method,style,seed_index),sample in generated.groupby(['method','style','seed_index'],sort=True):
            real=valid_x[styles[valid]==style]
            local=train_x[styles[train]==style]
            generated_x=sample[features].to_numpy(float)
            clipped_training,(clipped_real,clipped_generated),scale,_=training_winsorize(
                train_x,real,generated_x,lower=.01,upper=.99,
            )
            clipped_local=np.clip(local,np.quantile(train_x,.01,axis=0),np.quantile(train_x,.99,axis=0))
            metrics=distribution_diagnostics(clipped_real,clipped_generated,clipped_local,scale)
            metrics.pop('w1_by_feature');diagnostic_scope=metrics.pop('scope')
            rows.append({
                'evaluation_scope':'outer_validation_fold_train_q01_q99_sensitivity',
                'diagnostic_scope':diagnostic_scope,'repeat':repeat,'fold':fold,'style':style,
                'method':method,'seed_index':int(seed_index),'status':'evaluated',**metrics,
            })
    table=pd.DataFrame(rows)
    if len(table)!=900:
        raise ValueError('E202 robust audit row count changed')
    table_path=RESULT/'q3_e202_robust_sensitivity.csv';table.to_csv(table_path,index=False)
    names=['mean_w1','energy_v','coverage','near_duplicate_fraction','spearman_pair_rmse']
    repeat_table=table.groupby(['method','style','repeat'],as_index=False)[names].mean()
    repeat_path=RESULT/'q3_e202_robust_by_repeat.csv';repeat_table.to_csv(repeat_path,index=False)
    summary=repeat_table.groupby(['method','style'])[names].agg(['mean','std'])
    summary.columns=['_'.join(column) for column in summary.columns]
    summary_path=RESULT/'q3_e202_robust_summary.csv';summary.reset_index().to_csv(summary_path,index=False)
    audit={
        'status':'complete','scope':'fold_training_q01_q99_sensitivity_not_primary_result',
        'n_rows':len(table),'n_outer_folds':30,'input_manifest_sha256':sha(manifest_path),
        'source_sha256':sha(Path(__file__)),'checkpoint_manifest_sha256':checkpoint_hashes,
        'output_sha256':{path.name:sha(path) for path in [table_path,repeat_path,summary_path]},
        'notes':['分位边界只由当前外层训练折估计','原始口径仍为主结果，本表仅用于异常值敏感性','未重新拟合或挑选生成模型'],
    }
    audit_path=RESULT/'q3_e202_audit_manifest.json';audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2));return audit


if __name__=='__main__':
    run()
