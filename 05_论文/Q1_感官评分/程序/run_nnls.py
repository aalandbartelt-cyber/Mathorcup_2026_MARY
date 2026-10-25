"""完整嵌套交叉验证与新样本推断；默认只显示帮助。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from stacking_runner import fit_outer_nnls
from q1_metrics import matrix_metrics
from baseline_models import empirical_cdf_grade

def load_tables(folder):
    x=pd.read_csv(folder/'1.csv');y=pd.read_csv(folder/'4.csv')
    for frame in [x,y]:
        if frame.beer.isna().any() or not frame.beer.is_unique:
            raise ValueError('beer必须完整且唯一')
    if set(x.beer)!=set(y.beer):raise ValueError('样本主键不匹配')
    y=y.set_index('beer').loc[x.beer].reset_index()
    meta=['beer','beer_id','tasting_category_fine']
    if not x.tasting_category_fine.equals(y.tasting_category_fine):raise ValueError('风格不匹配')
    return x,x.drop(columns=meta).to_numpy(float),y.drop(columns=meta)

def main():
    parser=argparse.ArgumentParser(description='感官组收缩NNLS：复算指标、完整嵌套验证或新样本推断')
    parser.add_argument('--attachments',type=Path)
    parser.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent.parent/'数据与结果')
    parser.add_argument('--output',type=Path,default=Path('运行结果'))
    parser.add_argument('--execute',action='store_true',help='执行完整10次重复3折嵌套验证')
    parser.add_argument('--predict-chemistry',type=Path,help='对新化学表预测；仅以现有250款训练')
    parser.add_argument('--recompute-oof',type=Path,help='只由已有归档复算，不训练')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.recompute_oof:
        a=np.load(args.recompute_oof,allow_pickle=False)
        rows=[{'evaluation':f'repeat_{r}',**matrix_metrics(a['y_true'],a['prediction'][r])} for r in range(10)]
        rows.append({'evaluation':'mean_oof',**matrix_metrics(a['y_true'],a['prediction'].mean(0))})
        pd.DataFrame(rows).to_csv(args.output/'预测指标.csv',index=False)
        return
    if not args.execute and not args.predict_chemistry:parser.print_help();return
    if args.attachments is None:parser.error('需要--attachments指向官方附件目录')
    chem,x,y_frame=load_tables(args.attachments);y=y_frame.to_numpy(float)
    dictionary=pd.read_csv(args.assets/'target_dictionary.csv').set_index('target_code')
    groups=dictionary.loc[y_frame.columns,'target_group'].astype(str).tolist()
    config=json.loads((Path(__file__).parent/'model_config.json').read_text())
    if args.predict_chemistry:
        new=pd.read_csv(args.predict_chemistry)
        feature_names=chem.drop(columns=['beer','beer_id','tasting_category_fine']).columns
        result=fit_outer_nnls(x,y,new.loc[:,feature_names].to_numpy(float),base_config=config,groups=groups,scopes=['group'],shrinkage=.5,inner_splits=3,seed=20261023)
        prediction=result['predictions']['group']
        frame=pd.DataFrame(prediction,columns=y_frame.columns);frame.insert(0,'beer',new.beer)
        frame.to_csv(args.output/'连续感官预测.csv',index=False)
        grade=pd.DataFrame(empirical_cdf_grade(y,prediction),columns=y_frame.columns);grade.insert(0,'beer',new.beer)
        grade.to_csv(args.output/'相对强度预测.csv',index=False)
        return
    splits=pd.read_csv(args.assets/'cross_validation_splits.csv')
    pred=np.full((10,len(x),len(y_frame.columns)),np.nan);folds=np.full((10,len(x)),-1,int);rows=[]
    for repeat in range(10):
        for fold in range(3):
            selected=splits[(splits.repeat==repeat)&(splits.fold==fold)]
            valid=np.sort(selected.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
            if not len(valid):raise ValueError('缺少外层划分')
            expected=chem.iloc[selected.row_position].beer.to_numpy()
            np.testing.assert_array_equal(expected,selected.beer.to_numpy())
            result=fit_outer_nnls(x[train],y[train],x[valid],base_config=config,groups=groups,scopes=['group'],shrinkage=.5,inner_splits=3,seed=20261023+repeat*10+fold)
            pred[repeat,valid]=result['predictions']['group'];folds[repeat,valid]=fold
            rows.append({'repeat':repeat,'fold':fold,'weights':result['parameters']['group']})
            print(f'完成重复{repeat+1}、折{fold+1}',flush=True)
    if not np.isfinite(pred).all():raise ValueError('折外预测未完整覆盖')
    np.savez_compressed(args.output/'模型折外预测.npz',y_true=y,prediction=pred,mean_prediction=pred.mean(0),beer=chem.beer.to_numpy(),style=chem.tasting_category_fine.to_numpy(str),target_code=y_frame.columns.to_numpy(str),fold_assignment=folds)
    (args.output/'模型权重.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
