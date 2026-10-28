"""Read-only provenance and source-data extraction; does not run any experiment."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd

ROOT=Path('/home/mary/mathorcup')
OUT=Path(__file__).resolve().parent
DATA=OUT/'data'
ATT=ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件'
Q2=ROOT/'.worktrees/q2-evidence-opt'
Q3=ROOT/'.worktrees/q3-foundation'

def sha(p):return sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv(frame,name):frame.to_csv(DATA/name,index=False,encoding='utf-8-sig')

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    frames={};inventory=[];columns=[];hashes={}
    units=pd.read_excel(ATT/'8.xlsx',sheet_name='Sensory thresholds').set_index('field_name_R')['unit']
    for p in sorted(ATT.iterdir()):
        if p.suffix not in ['.csv','.xlsx']:continue
        hashes[p.name]=sha(p)
        sheets={'CSV':pd.read_csv(p)} if p.suffix=='.csv' else pd.read_excel(p,sheet_name=None)
        for sheet,frame in sheets.items():
            frames[(p.name,sheet)]=frame
            keys=[c for c in ['beer','Beer','field_name_R','R_name','Unnamed: 0'] if c in frame]
            key=keys[0] if keys else '无可用样本主键（表单或空表）'
            unique=bool(frame[key].notna().all() and frame[key].is_unique) if key in frame else None
            inv={'attachment':p.name,'sheet':sheet,'rows':len(frame),'columns':len(frame.columns),'missing_cells':int(frame.isna().sum().sum()),
                 'key_candidate':key,'key_complete_unique':unique,'sha256':hashes[p.name],
                 'parse_policy':'pandas header=0; formatted form dimensions are not observed sample counts' if p.name=='3.xlsx' else 'pandas header=0'}
            inventory.append(inv)
            for pos,col in enumerate(frame):
                value=units.loc[col] if p.name in ['1.csv','8.xlsx'] and isinstance(col,str) and col in units.index else None
                if p.name in ['2.csv','5.csv','6.csv'] and pos>0:value='Spearman相关系数，无量纲'
                elif p.name=='4.csv' and pos>2:value='附件标准化连续感官值，无量纲；非原始0—6'
                elif p.name=='7.csv' and pos>2:value='论坛描述符连续值（含负值）；本研究不作原计数或0—6解释'
                columns.append({'attachment':p.name,'sheet':sheet,'column_position_1based':pos+1,'column_name':str(col),'missing_cells':int(frame[col].isna().sum()),'declared_unit_or_scope':'' if value is None or pd.isna(value) else str(value)})
    csv(pd.DataFrame(inventory),'attachment_sheet_inventory.csv');csv(pd.DataFrame(columns),'attachment_columns.csv')
    scopes=[
      {'attachment':'1.csv','dimensions':'250×234（231测量+3元数据）','primary_key':'beer（唯一）；beer_id禁作连接','units_scope':'按字段混合单位与混合存储尺度；70字段可核验为37 log10/33原尺度，其余保留附件尺度'},
      {'attachment':'2.csv','dimensions':'231×232（行名+231相关列）','primary_key':'首列化学名称；行列一一对应','units_scope':'Spearman相关系数[-1,1]，无量纲；全数据描述/核验'},
      {'attachment':'3.xlsx','dimensions':'13 sheets；beer_1—12各100×33；summary 12×52','primary_key':'问卷槽位/表单，不与250款啤酒按Beer数值直接连接','units_scope':'0—6品评分级模板，summary为模板零值；非新增观测'},
      {'attachment':'4.csv','dimensions':'250×53（50感官+3元数据）','primary_key':'beer（唯一）；beer_id存在精度损失','units_scope':'给定标准化连续感官值，无量纲；不能唯一反演原0—6'},
      {'attachment':'5.csv','dimensions':'50×51（行名+50相关列）','primary_key':'首列感官英文名称；与附件4尾部需次序映射','units_scope':'Spearman相关系数[-1,1]，无量纲；表头另作目标名称'},
      {'attachment':'6.csv','dimensions':'44×51（化学名称+50感官相关列）','primary_key':'首列44个化学名称；非完整231维','units_scope':'Spearman相关系数[-1,1]，无量纲；仅核验不作监督筛选'},
      {'attachment':'7.csv','dimensions':'250×189（186论坛描述符+3元数据）','primary_key':'beer（唯一）','units_scope':'提供的论坛描述符连续值（含负值），非原始评论计数；本研究未使用其数值'},
      {'attachment':'8.xlsx','dimensions':'7 sheets；校准化学250×73，阈值261×11等详见逐sheet表','primary_key':'校准表beer；阈值表field_name_R；其他sheet各有标识','units_scope':'单位/阈值按Sensory thresholds逐字段记录；校准70字段；不使用附带GBM重要性/SHAP作为本队证据'}]
    pd.DataFrame(scopes).to_csv(OUT/'附件口径汇总.csv',index=False,encoding='utf-8-sig')
    chem=frames[('1.csv','CSV')];sens=frames[('4.csv','CSV')];feature=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']]
    target=[c for c in sens if c not in ['beer','beer_id','tasting_category_fine']]
    x=chem[feature];missing=x.isna()
    missing_by_field=pd.DataFrame({'feature_code':feature,'missing_cells':missing.sum().to_numpy(),'missing_rate':missing.mean().to_numpy()})
    name_map=pd.read_csv(ROOT/'01_数据/compound_name_map.csv')
    missing_by_field=missing_by_field.merge(name_map[['feature_code','compound_name','unit']],on='feature_code',how='left',validate='one_to_one')
    chinese_missing={'X2.3.butanedione':'双乙酰','X3.methylbutanal':'3-甲基丁醛','ethyl.isobutyrate':'异丁酸乙酯','methylpyrazine':'甲基吡嗪','furfural':'糠醛','amyl.acetate':'乙酸戊酯','phenylacetaldehyde':'苯乙醛','ethyl.nicotinate':'烟酸乙酯','aroma_hops_sum':'酒花香气汇总量'}
    missing_by_field['feature_zh']=missing_by_field.feature_code.map(chinese_missing).fillna('')
    csv(missing_by_field,'q1_missing_by_feature.csv')
    csv(missing_by_field.loc[missing_by_field.missing_cells>0].sort_values('missing_cells',ascending=False),'q1_missing_nine_fields_zh.csv')
    coords=[{'beer':int(chem.beer.iloc[i]),'row_position_1based':int(i+1),'feature_code':feature[j],'style':chem.tasting_category_fine.iloc[i]} for i,j in zip(*np.where(missing.to_numpy()))]
    csv(pd.DataFrame(coords),'q1_missing_cells.csv')
    csv(pd.DataFrame({'beer':chem.beer,'style':chem.tasting_category_fine,'n_missing':missing.sum(axis=1)}),'q1_missing_by_beer.csv')
    styles=chem.tasting_category_fine.value_counts().rename_axis('style').reset_index(name='n_beers')
    csv(styles,'q1_style_counts.csv')
    td=pd.read_csv(ROOT/'03_结果/Q1_感官评分/q1_target_dictionary.csv')
    labels={'Aroma':'香气','Flavour-core':'风味','Taste-other':'其他味觉','Palate':'口感','Overall':'总体喜好'}
    groups=td.groupby('target_group',sort=False).size().rename('n_targets').reset_index();groups['group_zh']=groups.target_group.map(labels)
    csv(groups,'q1_target_families.csv');csv(td,'q1_target_dictionary_source.csv')
    scale=pd.read_csv(ROOT/'03_结果/Q1_感官评分/q1_feature_scale_audit.csv')
    csv(scale,'q1_feature_scale_audit_source.csv')
    counts=scale.scale_in_1_csv.value_counts().rename_axis('scale').reset_index(name='n_fields');counts=pd.concat([counts,pd.DataFrame([{'scale':'unverified','n_fields':len(feature)-len(scale)}])],ignore_index=True)
    csv(counts,'q1_feature_scale_counts.csv')
    candidate=pd.read_csv(Q2/'03_结果/Q2_关键化合物/q2_candidate_dictionary.csv')
    strict=candidate.loc[candidate.eligible_strict.astype(bool),'feature_code'].tolist()
    csv(candidate.loc[~candidate.eligible_strict.astype(bool),['feature_code','compound_name','factor_type','exclusion_reason']],'q2_excluded_fields.csv')
    corr=x[strict].corr(method='spearman');a,b=np.triu_indices(len(strict),1)
    pairs=pd.DataFrame({'source':np.array(strict)[a],'target':np.array(strict)[b],'rho':corr.to_numpy()[a,b]});pairs['abs_rho']=pairs.rho.abs()
    pairs.to_csv(DATA/'q2_strict_pairwise_spearman.csv.gz',index=False,compression='gzip')
    bins=pd.cut(pairs.abs_rho,bins=np.linspace(0,1,11),include_lowest=True,right=True)
    binframe=bins.value_counts(sort=False).rename('n_pairs').reset_index(name='n_pairs') if False else pd.DataFrame({'bin':[str(t) for t in bins.cat.categories],'n_pairs':bins.value_counts(sort=False).to_numpy()})
    binframe['lower']=np.arange(10)/10;binframe['upper']=np.arange(1,11)/10;binframe['fraction']=binframe.n_pairs/len(pairs);csv(binframe,'q2_pairwise_correlation_bins.csv')
    csv(pd.DataFrame({'abs_rho_threshold':[.2,.3,.4,.5,.6,.7,.8,.9], 'n_pairs':[int((pairs.abs_rho>=v).sum()) for v in [.2,.3,.4,.5,.6,.7,.8,.9]],'total_pairs':len(pairs)}),'q2_pairwise_threshold_counts.csv')
    formal=Q2/'03_结果/Q2_关键化合物/q2_final_top3_E112.csv';ft=pd.read_csv(formal)
    csv(pd.DataFrame({'stage':['全部测量字段','严格单体候选','最终Top-3关系'],'count':[len(feature),len(strict),len(ft)],'unit':['字段','字段','关系（50×3）']}),'q2_candidate_scope.csv')
    profiles_path=Q3/'03_结果/Q3_风格生成/q3_generated_profiles_E205.csv.gz';profiles=pd.read_csv(profiles_path)
    rows=[]
    for style,block in profiles.groupby('style',sort=False):
        rows.append({'style':style,'n_observed_beers':int((chem.tasting_category_fine==style).sum()),'n_generated_spectra':int(block.sample_id.nunique()),'n_sensory_targets':int(block.target_code.nunique()),'n_long_records':len(block),'finite_predictions':int(np.isfinite(block.prediction_z).sum()),'finite_relative_grades':int(np.isfinite(block.grade_0_6).sum()),'scope':'full_data_postselection_generated_profile; not independent samples'})
    csv(pd.DataFrame(rows),'q3_observed_generated_completeness.csv')
    # Raw input data only: observed chemistry, no generated spectra mixed into this panel.
    selected_fields=[('ethanol..v.v.','乙醇'),('ethyl_acetate','乙酸乙酯'),('X4VG','4-乙烯基愈创木酚'),('lactic_acid.mg.L.','乳酸'),('acetic_acid.g.L.','乙酸')]
    available=[]
    for code,zh in selected_fields:
        if code in chem:
            for row in chem.loc[chem.tasting_category_fine.isin(['Tripel','Kriek','Lambic']),['beer','tasting_category_fine',code]].itertuples(index=False,name=None):
                available.append({'beer':int(row[0]),'style':row[1],'feature_code':code,'feature_zh':zh,'value_in_attachment1':float(row[2]),'unit':str(units.get(code,'')),'stored_scale':scale.set_index('feature_code').scale_in_1_csv.get(code,'unverified')})
    csv(pd.DataFrame(available),'q3_observed_input_chemical_distributions.csv')
    units8=frames[('8.xlsx','Sensory thresholds')][['field_name_R','unit','calibrated','threshold_median']].copy();csv(units8,'attachment8_unit_threshold_metadata.csv')
    # Matrix classification is by data-content role, not merely a file hash operation.
    states={
    '1.csv':[('直接输入','训练化学矩阵及风格分层；231字段'),('直接输入','严格单体筛选、各折统计和预测归因原始X'),('直接输入','化学生成器/迁移直接读取；不是仅从前问OOF间接取得')],
    '2.csv':[('仅作核验/描述','英文名-代码映射与相关定义核验；不参与Q1筛选'),('仅作核验/描述','全量共表达描述/数值交叉核验；监督图由附件1折内重算'),('未使用','虽核对compound_name_map的feature_code列序，但该列来自附件1；未消费附件2英文名、相关值、边或模块')],
    '3.xlsx':[('未使用','数值管线无read_*调用，仅完整性SHA登记；问卷定义可作背景参照'),('未使用','无数据内容读入证据'),('未使用','无数据内容读入证据')],
    '4.csv':[('直接输入','50维标准化感官监督Y'),('直接输入','同一50维监督Y与归因误差目标'),('派生/接口输入','正式画像与审计的y_true/目标代码经Q1最终OOF导入；原4.csv仅契约直接核验，非Copula监督输入；外部后验方向核验另有直接read4')],
    '5.csv':[('仅作核验/描述','读表头生成目标英文名/尾部次序；全矩阵只核验Spearman'),('派生/接口输入','经Q1目标字典提供50目标显示名；未读其全量相关数值训练'),('未使用','Q3目标代码来自最终OOF；无Q1英文目标字典或附件5的内容读入')],
    '6.csv':[('仅作核验/描述','相关定义与44字段次序核验，不作预测筛选'),('未使用','上游Q1边界审计不视为本问数值输入；Q2自行折内计算'),('未使用','无数值读入证据')],
    '7.csv':[('未使用','只SHA完整性登记，不用论坛描述符训练'),('未使用','无数据内容读入证据'),('未使用','无数据内容读入证据')],
    '8.xlsx':[('仅作核验/描述','70字段尺度核验及231字段名/单位映射；不替换X'),('直接输入','阈值与化学注释/校准浓度用于资格及OAV独立核验'),('直接输入','契约直接读取校准浓度/单位/阈值并派生缺陷状态；生成仍以附件1为训练X')]}
    rows=[{'attachment':att,'question':q,'status':state,'role':role} for att,triples in states.items() for q,(state,role) in zip(['Q1','Q2','Q3'],triples)]
    pd.DataFrame(rows).to_csv(OUT/'附件使用矩阵_长表.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'attachment':a,**{q:triples[i][0] for i,q in enumerate(['Q1','Q2','Q3'])}} for a,triples in states.items()]).to_csv(OUT/'附件使用矩阵.csv',index=False,encoding='utf-8-sig')
    # Evidence lines are kept with exact source hashes; no pipeline is imported/executed.
    code_paths=[ROOT/'02_代码/Q1_感官评分/src/data_contract.py',Q2/'02_代码/Q2_关键化合物/src/q2_contract.py',Q2/'02_代码/Q2_关键化合物/src/run_e100_contract.py',Q2/'02_代码/Q2_关键化合物/src/run_e112_chemical_evidence.py',Q3/'02_代码/Q3_风格生成/src/run_e200_contract.py',Q3/'02_代码/Q3_风格生成/src/run_e202_gaussian_baselines.py',Q3/'02_代码/Q3_风格生成/src/run_e205_sensory_audit.py']
    evidence=[]
    for path in code_paths:
        lines=path.read_text().splitlines();selected=[{'line':i+1,'text':s} for i,s in enumerate(lines) if any(t in s for t in ['read_csv','read_excel','attachment_','official_graph_role','validation_graph_role','target_dictionary','input_sha256','Calibrated chemical dataset','Sensory thresholds','compound_dictionary','names.feature_code',"['y_true']",'q2_top3_in_optimizer'])]
        evidence.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),'lines':selected})
    manifests=[ROOT/'03_结果/Q1_感官评分/q1_data_contract.json',ROOT/'03_结果/Q1_感官评分/q1_run_manifest_E009.json',Q2/'03_结果/Q2_关键化合物/q2_data_contract.json',Q3/'03_结果/Q3_风格生成/q3_run_manifest_E200.json',Q3/'03_结果/Q3_风格生成/q3_run_manifest_E202.json']
    records=[]
    for path in manifests:
        d=json.loads(path.read_text());checks={}
        if path.name=='q1_data_contract.json':checks={n:hashes[n]==v for n,v in d['official_attachment_sha256'].items()}
        elif path.name=='q1_run_manifest_E009.json':checks={Path(n).name:hashes[Path(n).name]==v for n,v in d['input_source_hashes'].items() if '/附件/' in n}
        elif path.name=='q2_data_contract.json':checks={n:hashes[n]==v for n,v in d['input_sha256'].items()}
        elif path.name=='q3_run_manifest_E200.json':checks={n:hashes[n]==v for key,v in d['input_sha256'].items() if key.startswith('attachment_') for n in [key.removeprefix('attachment_')]}
        elif path.name=='q3_run_manifest_E202.json':checks={'1.csv':hashes['1.csv']==d['signature']['input_sha256']}
        records.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),'input_hash_checks':checks,'all_matched':all(checks.values()),'recorded_input_hashes':d.get('official_attachment_sha256',d.get('input_sha256',d.get('signature',{}).get('input_sha256',d.get('input_source_hashes',{}))))})
    refs={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'03_结果/Q1_感官评分/q1_target_dictionary.csv',ROOT/'01_数据/compound_name_map.csv',ROOT/'03_结果/Q1_感官评分/q1_feature_scale_audit.csv',formal,profiles_path]}
    dump(OUT/'代码读入证据.json',{'code':evidence,'manifests':records,'derived_artifact_sha256':refs})
    summary={'attachment_sha256':hashes,'counts':{'n_beers':len(chem),'n_features':len(feature),'n_targets':len(target),'n_styles':len(styles),'n_missing_cells':int(missing.to_numpy().sum()),'n_missing_rows':int(missing.any(axis=1).sum()),'n_missing_features':int(missing.any(axis=0).sum()),'n_strict_candidates':len(strict),'n_exclusions':len(candidate)-len(strict),'n_final_pairs':len(ft),'n_profiles_long_records':len(profiles),'n_generated_spectra':int(profiles[['style','sample_id']].drop_duplicates().shape[0]),'n_finite_predictions':int(np.isfinite(profiles.prediction_z).sum())},'no_training':True,'input_values_changed':False,'official_hash_checks_all_match':all(t['all_matched'] for t in records),'outputs':{str(p.relative_to(OUT)):sha(p) for p in sorted(DATA.iterdir()) if p.is_file()}}
    dump(OUT/'描述统计与哈希.json',summary)
    report=['# 附件使用矩阵与描述统计审计','','本记录只读原附件、当前代码与已落盘结果，未训练/生成新模型或改变结果。矩阵按正式主线及其检验中实际消费的数据列分类，不沿文件血缘自动传播全部上游附件；单纯SHA登记不算使用数据。Q3后备外部数据研究的直接读入另行注明，不混入Copula训练输入。','','| 附件 | Q1 | Q2 | Q3 |','|---|---|---|---|']
    for name,triples in states.items():report.append('| '+name+' | '+' | '.join(v[0] for v in triples)+' |')
    report+=['','## 实际代码与初步口径的差异','','1. Q1实际读取附件5表头生成目标名，且附件2/5/6全量矩阵仅作相关定义核验；它们不参与Q1模型筛选。','2. Q2附件2在合同中明确为“全量验证与最终描述图”，有监督构图从附件1的训练折重算，不将全量相关阵当独立验证证据。','3. Q3直接读取附件1训练生成器，附件8校准信息用于直接缺陷约束；契约也直接读附件4作核验，但正式感官真值/目标代码经Q1最终OOF导入，主状态为派生/接口输入。不能概括为三种附件均只经接口，或均直接监督生成器。','4. 附件3/7没有当前数值管线的内容读入调用，Q1只登记它们的SHA。附件3是问卷模板：12张100×33排版表单和12×52零值summary；不能当成12款已观测感官样本。','5. 附件8实际用到Sensory thresholds和Calibrated chemical dataset；没有证据使用GBM importance MDI/GBM SHAP Values作为本队标签、预测或筛选，也没有读取Beer Prices训练。','6. 附件5经Q1目标字典为Q2提供英文显示名，属于派生元数据，不代表使用全量目标相关阵；Q3目标代码来自最终OOF，无使用此英文目标字典的证据。','7. 45,000是3风格×300谱×50目标的长表记录/感官数值数；生成谱900条，真实目标风格样本28/10/11款，不是新增45,000个独立观测。','8. Q3的q2_top3_in_optimizer被显式固定为False；Q2提供同折预测器缓存和独立审计器参数等接口，而非将正式Top-3塞入迁移优化。','9. Q3契约读取compound_name_map文件后只检查feature_code列序；该列源自附件1，不消费附件2产生的英文名、相关值、边或模块，所以附件2×Q3为未使用，不能沿文件血缘自动传递。','','## 文件规模和口径','','| 附件/Sheet | 行×列 | 主键/标识 | 缺失单元 |','|---|---:|---|---:|']
    for row in inventory:report.append(f'| {row["attachment"]}/{row["sheet"]} | {row["rows"]}×{row["columns"]} | {row["key_candidate"]} | {row["missing_cells"]} |')
    report+=['','注：CSV行数不含表头；xlsx使用pandas header=0。附件3为排版表单，空白/合并单元格不能解释成训练缺失。附件2/5/6相关系数均无量纲，第一列是标签而不是相关字段。附件1为混合存储尺度，单位由附件8逐字段提供；附件4为标准化感官连续值，不是可直接逆变换的0—6原评分。附件7是论坛描述符连续数据，本队未使用，不能把缺失0填作真实评论。','', '## 追溯入口','','- 代码读入证据.json：精确路径/行号/源码SHA与既有manifest的输入SHA对账。','- 描述统计与哈希.json：八附件SHA、派生表SHA及关键计数。','- data/attachment_sheet_inventory.csv：逐表实际维度与主键候选。','- data/attachment_columns.csv：全部列、缺失和已知单位。','- 详细24格用途见附件使用矩阵_长表.csv。','','## 图稿使用建议','','Q1：缺失字段/行分布、22风格样本量、五家族目标数均来自1/4及既有目标字典。Q2：231→218是字段数，150是关系数，画漏斗时必须标不同单位；候选成对相关性为218单体的23,653对，由全部250行仅为描述重算，不回写验证。Q3：真实三风格输入边缘与真实样本数是数据概览；900谱/45,000有限值仅作已落盘接口完整性，不当生成质量结论。']
    (OUT/'附件使用审计.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary['counts'],ensure_ascii=False));print(OUT)

if __name__=='__main__':main()
