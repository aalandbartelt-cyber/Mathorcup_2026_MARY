"""Prepare publication-safe plotting tables from already audited Q1 evidence."""
from pathlib import Path
from hashlib import sha256
import json,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'05_论文/Q1_感官评分/数据与结果'
VAL=DATA/'validation_v1';ABL=DATA/'validation_ablation_v1'
source=ROOT/'02_代码/Q1_感官评分/src/plot_q1_final.R'
text=source.read_text()
def named_map(name):
    block=text.split(name+' <- c(',1)[1].split('\n)',1)[0]
    return dict(re.findall(r'"([^"]+)"\s*=\s*"([^"]+)"',block))
targets=named_map('target_labels_zh');styles=named_map('style_labels_zh')
rep=pd.read_csv(DATA/'model_metrics_by_repeat.csv')
rep=rep[rep.display_name.isin(['ExtraTrees基线','感官组收缩NNLS'])].copy()
rep['model']=rep.display_name.map({'ExtraTrees基线':'基线','感官组收缩NNLS':'最终模型'})
rep[['repeat','model','r2','mae']].to_csv(OUT/'model_repeats.csv',index=False,encoding='utf-8-sig')
cal=pd.read_csv(VAL/'q1_calibration_bins.csv')
cal['group_plot']=cal.group_plot.replace({'味觉':'味觉/口感','口感':'味觉/口感'})
cal[['target_code','bin','n','mean_pred','mean_truth','abs_gap','group_plot']].to_csv(OUT/'calibration_bins.csv',index=False,encoding='utf-8-sig')
cov=pd.read_csv(VAL/'q1_interval_coverage.csv')
rows=[]
for (m,n),g in cov.groupby(['method','nominal']):
    rows.append(dict(method={'repeat_sd':'重复标准差','cross_repeat_residual':'残差校准'}[m],nominal=n,
                     mean_cov=g.coverage.mean(),q10=g.coverage.quantile(.1),q90=g.coverage.quantile(.9)))
pd.DataFrame(rows).to_csv(OUT/'interval_summary.csv',index=False,encoding='utf-8-sig')
rel=pd.read_csv(VAL/'q1_target_reliability.csv')
rel['target_zh']=rel.target_code.map(targets)
assert rel.target_zh.notna().all() and rel.target_zh.is_unique
rel[['target_code','target_zh','r2','r2_ci_low','r2_ci_high','r2_boot_sd','mae','reliability_tier']].to_csv(OUT/'target_reliability.csv',index=False,encoding='utf-8-sig')
abl=pd.read_csv(ABL/'q1_ablation_summary.csv')
abl['setting']=abl.setting.replace({'完整模型（重复0，3折）':'完整模型','移除4个汇总特征':'去汇总特征','标签置换负对照':'标签置换'})
abl.to_csv(OUT/'ablation_summary.csv',index=False,encoding='utf-8-sig')
leave=pd.read_csv(ABL/'q1_leave_style_out.csv',dtype=str)
leave['style_zh']=leave['style'].map(styles)
assert leave.style_zh.notna().all()
leave.to_csv(OUT/'leave_style_out.csv',index=False,encoding='utf-8-sig')
paths=[DATA/'model_metrics_by_repeat.csv',VAL/'q1_calibration_bins.csv',VAL/'q1_interval_coverage.csv',VAL/'q1_target_reliability.csv',ABL/'q1_ablation_summary.csv',ABL/'q1_leave_style_out.csv',source]
manifest={'scope':'publication_plot_inputs_only','source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},'generated_csv_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}}
(OUT/'source_data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('prepared six plotting tables')
