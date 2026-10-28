"""Q2 draft-only QA sheets and exact replacement plan for the parent builder."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from PIL import Image,ImageOps,ImageDraw,ImageFont
import hashlib,json,re

ROOT=Path('/home/mary/mathorcup')
OUT=ROOT/'04_图表/Q2_关键化合物/待审_20261027'
AUDIT=Path(__file__).resolve().parent
QA=OUT/'qa'
QA.mkdir(exist_ok=True)
font=ImageFont.truetype('/home/mary/.local/share/fonts/simsun.ttc',26) if Path('/home/mary/.local/share/fonts/simsun.ttc').exists() else ImageFont.load_default()
figures=[next(OUT.glob(f'图{i}_*.png')) for i in range(1,6)]
for p in figures:
    im=Image.open(p).convert('RGB')
    small=im.resize((round(im.width*.7),round(im.height*.7)),Image.Resampling.LANCZOS)
    small.save(QA/f'{p.stem}_70pct.png',dpi=(300,300))
    gray=ImageOps.grayscale(im).convert('RGB')
    gray.save(QA/f'{p.stem}_灰度.png',dpi=(300,300))

for mode in ['彩色','灰度']:
    sheet=Image.new('RGB',(2040,1725),'white')
    draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(figures):
        im=Image.open(p).convert('RGB')
        if mode=='灰度':im=ImageOps.grayscale(im).convert('RGB')
        im.thumbnail((985,500),Image.Resampling.LANCZOS)
        col,row=i%2,i//2;x=col*1020+(1020-im.width)//2;y=row*575+45
        sheet.paste(im,(x,y))
        draw.text((col*1020+25,row*575+12),p.stem,font=font,fill='black')
    sheet.save(QA/f'Q2_五图_{mode}_contact.png',dpi=(150,150))

source=ROOT/'.worktrees/q2-evidence-opt/05_论文/Q2_关键化合物/第二问_论文正文_v4.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(source) as z:
    root=etree.fromstring(z.read('word/document.xml'))
    paragraphs=[''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in root.xpath('./w:body/w:p',namespaces=ns)]

new_paragraphs={
2:paragraphs[2].replace('自助入选频率中位数为0.428。','分层bootstrap频率选出集的入选频率中位数为0.428，四路融合最终Top-3为0.395。'),
38:paragraphs[38].replace('图ElasticNet指有符号图正则回归','图正则EN指有符号图正则回归'),
40:'500次分层自助中，按bootstrap频率选出的150条候选入选频率范围为0.182—0.992，中位数为0.428；达到0.60、0.70和0.80的候选分别有43、31和23条。四路融合最终Top-3的入选频率中位数为0.395，40/150条不低于0.60。逐目标Top-3两两Jaccard均值为0.245，说明共线字段替代和样本扰动不可忽略。稳定共聚类得到170个模块，其中155个为单字段模块，最大模块含15个字段。',
41:'相同参数下，未校正与风格校正Top-3的平均Jaccard为0.264，只有2/50个目标完全一致。逐一删除22种风格后，1,100个“风格—目标”组合的Top-3平均Jaccard为0.523。分层bootstrap频率选出集的留一风格平均保留率为0.6582；四路融合最终Top-3为0.6076，中位数为0.6591。总关联共识参考集的平均保留率为0.6406，该集合与bootstrap频率选出集仅重合113/150条，三个集合分别计算。图3按化学证据等级展示最终Top-3在样本扰动与风格扰动下的稳定性分布。',
}
replacements=[{'paragraph_index_zero_based':i,'old':paragraphs[i],'new':t} for i,t in new_paragraphs.items()]

supplement={
 'section_title':'补充检验：负对照、直接锚点与遮蔽敏感性',
 'insert_before_exact_paragraph':'合理性与局限',
 'paragraphs':[
  '为检验选择结果是否依赖真实信号，表5汇总目标置换、影子字段、直接锚点与字段遮蔽检验。目标置换在0.60、0.70、0.80和0.90四个入选频率阈值下均无关系通过；但影子字段在2,000条目标—重复—半样本记录中占1,524个入选槽，真实字段占4,476个槽。每条记录共有3个槽，影子字段占全部6,000个槽的25.4%。图5(a)比较两类字段每条记录的平均入选槽数，分别为2.238和0.762；该检验支持真实字段整体占优，同时表明局部前三仍存在非零噪声入选。',
  '图5(b)按bootstrap名次展示直接锚点。最终融合前三保留4VG、酒精感—乙醇、酒体—乙醇和碳酸感—二氧化碳4对关系。CO₂的bootstrap名次为第4，而最终融合名次为第3；DMS、双乙酰和反-2-壬烯醛分别为第29、92和61名，未进入最终融合前三。已知锚点的部分恢复支持方法能够识别若干已知关联，但不能将弱缺陷目标的前三字段解释为已验证的缺陷来源。',
  '字段遮蔽检验固定最终感官模型，将各外层训练折中条件效应排名前三的字段在验证集里替换为训练折中位数。50个目标的平均ΔMAE为0.0005806，目标均值的中位数为−0.00009657，22/50个目标误差上升；酒精感、酒花总体和苦度分别增加0.02244、0.01143和0.00501，其中酒精感相对增加6.99%。被遮蔽字段来自同折条件效应排名，与正式融合Top-3对应不同集合；本项只衡量冻结预测器对这些字段遮蔽的敏感程度，不包含重训不确定性，也不证明三字段具有充分性。',
 ],
 'table_title':'表5 补充检验的数值结果与解释范围',
 'table_columns':['检验项','结果','单位/口径','支持与限制'],
 'table_rows':[
  ['bootstrap候选稳定性','中位0.428；43/150≥0.60；留一风格均值0.6582','按bootstrap频率选出的150对关系','集合对应后支持候选稳定性描述'],
  ['最终融合稳定性','中位0.395；40/150≥0.60；留一风格均值0.6076、中位0.6591','正式150对关系','尾部仍有替代与风格敏感性'],
  ['目标置换','0/0/0/0条','入选频率阈值0.60/0.70/0.80/0.90','仅针对所列阈值'],
  ['影子字段','真实4,476槽；影子1,524槽（25.4%）','2,000条记录，共6,000个前三槽','真实字段占优，噪声入选非零'],
  ['直接锚点','最终保留4对；CO₂ bootstrap第4、最终第3','7对单体锚点中4对保留','DMS/双乙酰/反-2-壬烯醛为29/92/61'],
  ['折内前三字段遮蔽','平均ΔMAE +0.0005806；中位−0.00009657；22/50退化','固定最终感官模型，附件感官标准化尺度','对象为同折条件效应前三，不是充分性'],
  ['遮蔽后较大退化','酒精感+0.02244（+6.99%）；酒花总体+0.01143；苦度+0.00501','30折分别计算，再按目标平均','不包含重训或重新选择字段的变化'],
 ],
 'table_note':'注：总关联共识参考集、bootstrap频率选出集与四路融合最终Top-3是不同集合。目标置换的通过关系数与影子记录的入选槽数分母不同。遮蔽敏感性使用每个训练折内条件效应前三字段和训练中位数，不改动正式答案排序。',
 'figure_caption':(OUT/'figure_contract.md').read_text().split('## 图注草稿\n\n',1)[1].split('\n\n## QA交付',1)[0],
}

plan={'status':'待主Agent采用的更正稿','source_docx':str(source.relative_to(ROOT)),
      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'paragraph_index_scope':'word/document.xml下w:body直属w:p，零起始，不含表格内段落',
      'replacements':replacements,'supplement':supplement,
      'table4_cell_replacement':{
       'table_index_zero_based':10,'row_index_zero_based':4,'column_index_zero_based':2,
       'old':'500次分层自助入选频率中位数0.428；互补半样本与风格内置换用于补充检验',
       'new':'bootstrap频率选出集中位0.428；最终融合Top-3中位0.395；互补半样本与风格内置换用于补充检验'},
      'notes':['表5整表保持同页；图5与图题绑定。','三套稳定性集合的数值口径已按用户继续指令更正。','不生成DOCX/交付包，不修改原正文。']}
(AUDIT/'Q2_正文替换与补充表5.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')

root_addition={
 'heading':supplement['section_title'],
 'paragraphs':supplement['paragraphs'],
 'tables':[{'caption':supplement['table_title'],
            'rows':[['检验项','结果','口径与解释']]+
                   [[row[0],row[1],row[2]+'；'+row[3]] for row in supplement['table_rows']],
            'widths':[2100,3600,2940],
            'note':supplement['table_note']}],
 'figures':[{'path':'04_图表/Q2_关键化合物/待审_20261027/图5_负对照与直接锚点.png',
             'caption':'图5 负对照、影子字段与直接锚点恢复',
             'note':supplement['figure_caption'].split('恢复。',1)[1]}],
 'replacements':[{'old':x['old'],'new':x['new']} for x in replacements]+
                [{'old':plan['table4_cell_replacement']['old'],'new':plan['table4_cell_replacement']['new']}],
}
(AUDIT/'paper_addition.json').write_text(json.dumps(root_addition,ensure_ascii=False,indent=2)+'\n')

checks=[]
for p in OUT.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.md','.csv','.json','.r','.svg'}:
        content=p.read_text(encoding='utf-8-sig')
        hits=re.findall(r'(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|参考组|对照组|晋级|门槛|promotion|\bpass\b|BZD|agent(?:2|3)?|学长版',content,re.I)
        checks.append({'file':str(p.relative_to(OUT)),'hits':hits})
assert not any(x['hits'] for x in checks),checks
meta=[]
for p in figures:
    im=Image.open(p)
    meta.append({'file':p.name,'width':im.width,'height':im.height,'dpi':im.info.get('dpi')})
qa={'text_source_forbidden_hits':sum(len(x['hits']) for x in checks),'scanned_files':checks,
    'figures':meta,'preserved_figures':'图1—图4逐字节复制已审版本',
    'figure5_grayscale_redundancy':'实心圆/空心三角+类别标签/名次直接标值',
    'figure5_intended_units':'左每记录平均入选槽；右bootstrap名次；目标置换通过关系数单列图注',
    'visual_review':'由主Agent与用户继续审计；不以代码检查替代目检'}
(AUDIT/'图稿QA记录.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
print('QA and text plan written; source forbidden hits:',qa['text_source_forbidden_hits'])
