"""Verify and document only the newly generated Q1 validation drafts."""
from pathlib import Path
from hashlib import sha256
import json,re
import numpy as np
import pandas as pd
import pymupdf

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'04_图表/Q1_感官评分/待审_20261027'
DATA=ROOT/'05_论文/Q1_感官评分/数据与结果'
checks={}
rel=pd.read_csv(OUT/'target_reliability.csv').sort_values('target_code')
src=pd.read_csv(DATA/'validation_v1/q1_target_reliability.csv').sort_values('target_code')
cols=['r2','r2_ci_low','r2_ci_high','mae','r2_boot_sd']
checks['reliability_numeric_unchanged']=bool(np.array_equal(rel[cols].to_numpy(),src[cols].to_numpy()))
checks['targets_unique_50']=bool(len(rel)==50 and rel.target_code.is_unique and rel.target_zh.is_unique and rel.target_zh.notna().all())
leave=pd.read_csv(OUT/'leave_style_out.csv').sort_values('style')
src=pd.read_csv(DATA/'validation_ablation_v1/q1_leave_style_out.csv').sort_values('style')
cols=['n','r2','mae','spearman']
checks['leave_style_numeric_unchanged']=bool(np.array_equal(leave[cols].to_numpy(),src[cols].to_numpy()))
abl=pd.read_csv(OUT/'ablation_summary.csv')
src=pd.read_csv(DATA/'validation_ablation_v1/q1_ablation_summary.csv')
cols=['r2','mae','rmse','spearman','qwk_proxy','acc_pm1_proxy']
checks['ablation_numeric_unchanged']=bool(np.array_equal(abl[cols].to_numpy(),src[cols].to_numpy()))
pattern=re.compile(r'(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|BZD|agent(?:2|3)?|学长版|参考组|对照组|晋级|门槛',re.I)
hits=[]
for p in OUT.rglob('*'):
    if not p.is_file():continue
    if pattern.search(str(p.relative_to(OUT))):hits.append(str(p)+' [filename]')
    if p.suffix.lower() in {'.py','.r','.csv','.json','.md'}:
        if pattern.search(p.read_text(encoding='utf-8-sig')):hits.append(str(p))
checks['formal_source_text_forbidden_hits']=len(hits)
checks['forbidden_locations']=hits
qa=json.loads((OUT/'figure_qa.json').read_text())
for record in qa['records']:
    doc=pymupdf.open(OUT/(record['stem']+'.pdf'))
    record['forbidden_text_hits']=pattern.findall('\n'.join(page.get_text() for page in doc))
qa['manual_review']={
  'reviewed':['color originals of all three figures','70% grayscale of all three figures'],
  'result':'No visible clipping, label overlap or data-covering legends in the reviewed drafts; all 50 target labels are Chinese or proper chemical names; shared reliability legend contains all three shapes.',
  'specific_fixes':['S2(b) is a point plot','S2 sample-size groups use circle vs hollow triangle with a shared legend','S3 legend merged after explicitly retaining all shape levels','S2 value annotations changed to dark text and the negative value moved clear of zero reference line'],
  'status':'draft_ready_for_root_and_user_review'
}
(OUT/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((OUT/'source_data_manifest.json').read_text())
manifest['figure_outputs_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix.lower() in {'.png','.pdf'}}
manifest['plotting_script_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix.lower() in {'.r','.py'}}
(OUT/'source_data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
paper=json.loads((HERE/'paper_addition.json').read_text())
rows=[]
for entry in paper['figures']+paper['appendix_figures']:
    rows.append({'path':entry['path'],'caption':entry['caption'],'note':entry['note']})
pd.DataFrame(rows).to_csv(OUT/'Q1_新增图清单_待审.csv',index=False,encoding='utf-8-sig')
(HERE/'figure_rebuild_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
assert checks['reliability_numeric_unchanged'] and checks['targets_unique_50']
assert checks['leave_style_numeric_unchanged'] and checks['ablation_numeric_unchanged']
assert not hits
