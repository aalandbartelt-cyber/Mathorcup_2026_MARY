"""Minimal E202 pilot checkpoint; formal Gaussian/Copula run is not started."""
from pathlib import Path
import hashlib,json,yaml
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from gaussian_baselines import StyleGaussianGenerator
from q3_evaluation import distribution_diagnostics
from q1_interface import FrozenQ1V5OOF

ROOT=Path(__file__).resolve().parents[3];CANONICAL=Path('/home/mary/mathorcup');RESULT=ROOT/'03_结果/Q3_风格生成';CONFIG=Path(__file__).resolve().parents[1]/'configs/E202_gaussian_copula_baselines.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gate():
    e201=json.loads((RESULT/'q3_run_manifest_E201.json').read_text())
    e113=json.loads((CANONICAL/'.worktrees/q2-evidence-opt/03_结果/Q2_关键化合物/q2_progress_E113.json').read_text())
    if e201.get('status')!='complete' or e113.get('status')!='complete':raise ValueError('E201 and E113 complete gates required')
    return {'e201':sha(RESULT/'q3_run_manifest_E201.json'),'e113':e113.get('last_split_id'),'q1_final_oof':FrozenQ1V5OOF(CANONICAL).signature}
def run_pilot():
    config=yaml.safe_load(CONFIG.read_text());signature={'config':sha(CONFIG),'gate':gate()}
    chem=pd.read_csv(ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv');splits=pd.read_csv(ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv');features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']];x=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    row=splits[(splits.repeat==0)&(splits.fold==0)];valid=np.sort(row.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid);imp=SimpleImputer(strategy='median',keep_empty_features=True);tx=imp.fit_transform(x[train]);vx=imp.transform(x[valid]);model=StyleGaussianGenerator().fit(tx,styles[train],mean_shrinkage=float(config['mean_shrinkage']))
    rows=[]
    for style in config['target_styles']:
        mask=styles[valid]==style
        if mask.sum()<2 or (styles[train]==style).sum()<2:rows.append({'style':style,'status':'NA_insufficient_rows'});continue
        for method in config['methods']:
            sample=model.generate(style,int(mask.sum()),method=method,seed=20261025)
            metrics=distribution_diagnostics(vx[mask],sample['values'],tx[styles[train]==style],np.where(np.std(tx,axis=0)>1e-12,np.std(tx,axis=0),1.))
            metrics.pop('w1_by_feature');rows.append({'scope':'pilot_outer_train_to_validation','split_id':'R00F0','style':style,'method':method,'status':'pilot_only',**metrics})
    out=RESULT/'q3_e202_pilot_metrics.csv';pd.DataFrame(rows).to_csv(out,index=False);decision={'status':'pilot_only_no_formal_run','formal_run_started':False,'q1_retrained':False,'gpu_used':False,'methods':config['methods'],'signature':signature}
    (RESULT/'q3_e202_pilot_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':run_pilot()
