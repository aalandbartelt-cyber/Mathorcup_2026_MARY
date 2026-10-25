"""Produce a standalone, academically named reproducibility package."""
from pathlib import Path
import ast
import json
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[3]
SRC=Path(__file__).parent
OUT=ROOT/'05_论文/Q1_感官评分/程序'

def functions(path,names):
    text=path.read_text();tree=ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names)

RUNNER='''"""完整嵌套交叉验证与新样本推断；默认只显示帮助。"""
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
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['baseline_models.py','nnls_stacking.py','q1_metrics.py']:
        text=(SRC/name).read_text().replace('BZD-compatible','seven-bin')
        (OUT/name).write_text(text.rstrip()+'\n')
    imports='''from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
'''
    (OUT/'ensemble_models.py').write_text(imports+'\n'+functions(SRC/'ensemble_models.py',{'fit_predict_hgb_multioutput'})+'\n')
    imports='''from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from baseline_models import fit_predict_extra_trees, fit_predict_ridge_quantile, fit_predict_pls_quantile
from ensemble_models import fit_predict_hgb_multioutput
from q1_metrics import matrix_metrics
from nnls_stacking import fit_nnls_stacking
def make_inner_splits(n_samples,n_splits,random_state):
    return list(KFold(n_splits=n_splits,shuffle=True,random_state=random_state).split(np.arange(n_samples)))
'''
    helper=functions(SRC/'run_e005_continuous_ensemble.py',{'_candidates','_fit_base','_parameter_key'})
    fitting=functions(SRC/'run_e009_nnls_stacking.py',{'fit_outer_nnls'})
    (OUT/'stacking_runner.py').write_text((imports+'\n'+helper+'\n\n'+fitting+'\n').replace('E005','base').replace('E009','stacking'))
    (OUT/'run_nnls.py').write_text(RUNNER)
    core=OUT.parent/'附录核心代码';core.mkdir(exist_ok=True)
    for name in ['nnls_stacking.py','stacking_runner.py']:
        shutil.copy2(OUT/name,core/name)
    (core/'附录代码说明.md').write_text('# 附录核心代码\n\n由论文手编排附录，不提供程序附录Word/PDF。nnls_stacking.py给出非负权重、自由截距、组间收缩与预测；stacking_runner.py给出折内基学习器选择及堆叠。依赖及完整运行入口在相邻程序目录。\n')
    cfg=yaml.safe_load((SRC.parent/'configs/E005_continuous_ensemble.yaml').read_text())
    keep=['base_models','extra_trees','hgb_profiles','ridge_alphas','pls_components']
    (OUT/'model_config.json').write_text(json.dumps({k:cfg[k] for k in keep},ensure_ascii=False,indent=2)+'\n')
    plot=(SRC/'plot_q1_final.R').read_text().replace('source("scripts/plot_style.R")','source("plot_style.R")').replace('"03_结果/Q1_感官评分/publication_final"','"../数据与结果"').replace('"04_图表/Q1_感官评分/final_v4"','"../图片"')
    (OUT/'plot_final.R').write_text(plot)
    shutil.copy2(ROOT/'scripts/plot_style.R',OUT/'plot_style.R')
    assets=OUT.parent/'数据与结果'
    assets.mkdir(exist_ok=True)
    for p in (ROOT/'03_结果/Q1_感官评分/publication_final').iterdir():
        if p.is_file():shutil.copy2(p,assets/p.name)
    for source,dest in [('q1_cv_splits.csv','cross_validation_splits.csv'),('q1_target_dictionary.csv','target_dictionary.csv')]:
        shutil.copy2(ROOT/'03_结果/Q1_感官评分'/source,assets/dest)
    import pandas as pd
    dictionary=pd.read_csv(assets/'target_dictionary.csv')
    dictionary['target_name_en']=dictionary['target_name_en'].str.replace(r'\s+',' ',regex=True).str.strip()
    dictionary.to_csv(assets/'target_dictionary.csv',index=False)
    (OUT/'README.md').write_text('''# 第一问复现程序

Python 3.10；numpy、pandas、scipy、scikit-learn；作图使用R/ggplot2、ragg、dplyr、tidyr、readr、patchwork、svglite。

程序使用附带的固定划分和候选参数。所有填补、变换、基学习器选择及非负权重估计限制在相应训练数据中。

只复算已有折外指标：
```
python run_nnls.py --recompute-oof ../数据与结果/final_model_oof.npz --output 复算结果
```

完整嵌套验证（需要官方附件1.csv、4.csv）：
```
python run_nnls.py --attachments 官方附件目录 --execute --output 运行结果
```

新样本推断（先在全部训练样本内部重建折外基预测与堆叠，再对新样本预测）：
```
python run_nnls.py --attachments 官方附件目录 --predict-chemistry 新化学表.csv --output 新样本结果
```

绘图时在本目录执行`Rscript plot_final.R`；图片写入相邻的图片目录。0—6为训练经验分布的相对强度，不是原始评分。
''')
    print('Standalone programs and evidence:',OUT)

if __name__=='__main__':main()
