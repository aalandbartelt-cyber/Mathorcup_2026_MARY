"""Prepare neutral Q2 draft figure inputs without changing source artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

ROOT=Path('/home/mary/mathorcup')
RES=ROOT/'.worktrees/q2-evidence-opt/03_结果/Q2_关键化合物'
OLD=ROOT/'.worktrees/q2-evidence-opt/04_图表/Q2_关键化合物/v4'
OUT=ROOT/'04_图表/Q2_关键化合物/待审_20261027'
DATA=OUT/'source_data'
DATA.mkdir(parents=True,exist_ok=True)

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
retained=[]
for i in range(1,5):
    src=next(OLD.glob(f'图{i}_*_v4.png'))
    dst=OUT/src.name
    shutil.copy2(src,dst)
    assert sha(src)==sha(dst)
    retained.append({'figure':i,'source':str(src.relative_to(ROOT)),
                     'copy':str(dst.relative_to(ROOT)),'sha256':sha(dst),'byte_identical':True})
for p in (OLD/'source_data').glob('*.csv'):
    shutil.copy2(p,DATA/p.name)

shadow=pd.read_csv(RES/'q2_shadow_feature_audit_E117.csv')
pd.DataFrame([
    {'字段类型':'真实字段','总入选槽数':int(shadow.real_selected.sum()),'记录数':len(shadow),
     '每记录平均入选槽数':float(shadow.real_selected.mean())},
    {'字段类型':'影子字段','总入选槽数':int(shadow.shadow_selected.sum()),'记录数':len(shadow),
     '每记录平均入选槽数':float(shadow.shadow_selected.mean())},
]).to_csv(DATA/'图5a_影子字段入选槽.csv',index=False,encoding='utf-8-sig')

anchors=pd.read_csv(RES/'q2_anchor_recovery_E107.csv')
final=pd.read_csv(RES/'q2_final_top3_E112.csv')
labels={'X4vg':'4VG—4VG','alcohol':'酒精感—乙醇','body':'酒体—乙醇','co2':'碳酸感—CO₂',
        'dms':'DMS—DMS','t2n':'反-2-壬烯醛','diacetyl':'双乙酰—双乙酰'}
rows=[]
for target,label in labels.items():
    a=anchors.loc[anchors.target_code.eq(target)].iloc[0]
    f=final.loc[final.target_code.eq(target)&final.feature_code.eq(a.direct_anchor)]
    rows.append({'直接锚点':label,'bootstrap名次':int(a.bootstrap_rank),
                 '最终前三保留':'保留' if len(f) else '未保留',
                 '最终融合名次':int(f.iloc[0].consensus_rank) if len(f) else None})
pd.DataFrame(rows).to_csv(DATA/'图5b_直接锚点.csv',index=False,encoding='utf-8-sig')

real=pd.read_csv(RES/'q2_cpss_selection_E117.csv')
null=pd.read_csv(RES/'q2_null_target_permutation_E117.csv.gz')
pd.DataFrame([{'入选频率阈值':v,'真实关系通过数':int(real.selection_probability.ge(v).sum()),
               '目标置换关系通过数':int(null.null_selection_probability.ge(v).sum())}
              for v in [.6,.7,.8,.9]]).to_csv(DATA/'图5_目标置换阈值核验.csv',index=False,encoding='utf-8-sig')

sources=[RES/'q2_shadow_feature_audit_E117.csv',RES/'q2_anchor_recovery_E107.csv',
         RES/'q2_final_top3_E112.csv',RES/'q2_cpss_selection_E117.csv',
         RES/'q2_null_target_permutation_E117.csv.gz']
internal={'retained_figures':retained,'raw_input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
(Path(__file__).parent/'图稿来源哈希.json').write_text(json.dumps(internal,ensure_ascii=False,indent=2)+'\n')
public={'figures':[{'figure':x['figure'],'output':Path(x['copy']).name,'approved_bytes_retained':True,
                    'sha256':x['sha256']} for x in retained],
        'supplemental_figure':{'figure':5,'source_data':['source_data/图5a_影子字段入选槽.csv',
                                'source_data/图5b_直接锚点.csv','source_data/图5_目标置换阈值核验.csv'],
                              'script':'plot_q2_validation.R'},
        'source_data_sha256':{p.name:sha(p) for p in DATA.glob('*.csv')}}
(OUT/'source_data_manifest.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
print('copied approved figures:',len(retained),'neutral source CSV:',len(list(DATA.glob('*.csv'))))
