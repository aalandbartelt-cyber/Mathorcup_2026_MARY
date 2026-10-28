"""Embed approved figures and theory/response clarifications in a v1-derived copy.

Reads frozen outputs only. No estimator is fitted, and v1 is never overwritten.
"""
from copy import deepcopy as cp
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
import re
import shutil

from lxml import etree as E
from PIL import Image
import numpy as np
import pandas as pd
import native_math_total as nm

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'05_论文/总论文'
SOURCE=BASE/'v2_准备稿/总论文_v2_图前工作稿.docx'
TARGET=BASE/'2026_MathorCup_大数据竞赛_总论文_v2.docx'
AUDIT=ROOT/'docs/collab/20261027_总论文v2定稿'
FIG=BASE/'图稿_待审计/数据概览'
DATA=BASE/'数据与结果_v2'
NS={'w':nm.W,'m':nm.M,'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
W='{'+nm.W+'}';M='{'+nm.M+'}'
AI='本参赛队在竞赛过程中使用了AI工具，用于辅助赛题理解与任务拆解、术语规范与语言润色、公式排版、图表标题与参考文献格式检查、模型优点与局限的梳理、图表结果表述的量化修改，以及附录与支撑材料结构整理。模型假设、算法选择、参数设定、程序运行、结果判断与最终文字均由参赛队逐项复核，详细使用情况见支撑材料《AI工具使用详情.pdf》。'


def readzip(path):
    with ZipFile(path) as z:return {n:z.read(n) for n in z.namelist()}


def txt(x):return ''.join(x.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))


def prop(p,local,**attrs):
    x=p.find(W+local)
    if x is None:x=E.SubElement(p,W+local)
    for k,v in attrs.items():x.set(W+k,str(v))
    return x


def pp(p):
    pr=p.find(W+'pPr')
    if pr is None:pr=E.Element(W+'pPr');p.insert(0,pr)
    return pr


def main():
    AUDIT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    shutil.copy2(SOURCE,TARGET);files=readzip(TARGET)
    root=E.fromstring(files['word/document.xml']);body=root.find(W+'body')
    rels=E.fromstring(files['word/_rels/document.xml.rels'])
    para=cp(next(p for p in body.findall(W+'p') if p.find('w:pPr/w:pStyle',NS) is not None and p.find('w:pPr/w:pStyle',NS).get(W+'val')=='11' and len(txt(p))>80))
    cap=cp(next(p for p in body.findall(W+'p') if re.match(r'图1\s',txt(p))))
    picture=cp(next(p for p in body.findall(W+'p') if p.find('.//w:drawing',NS) is not None))
    S=nm.symbols();S['rank_a']=nm.sub('r',nm.seq('ℓ',nm.r(',',True),'a','j'))
    S['rank_b']=nm.sub('r',nm.seq('ℓ',nm.r(',',True),'b','j'))
    S['H_a']=nm.sub('H',nm.seq('a','j'));S['H_b']=nm.sub('H',nm.seq('b','j'))
    S['loss_g']=nm.sub(nm.r('L',script='script'),'g');S['candidate_a']=nm.sup('w',nm.r('a',True));S['candidate_b']=nm.sup('w',nm.r('b',True))
    S['grade_discrete']=nm.ss('h','ij',nm.r('disc',True))
    S['grade_round']=nm.seq(S['grade_discrete'],nm.r('=',True),nm.delim(nm.seq(S['ĥ_ij'],nm.r('+0.5',True)),'⌊','⌋'))
    S['c_final']=nm.seq(nm.sym('c','k','*'),nm.delim('t'));S['c_source']=nm.seq(S['c_k'],nm.delim(S['x_0']))
    S['c_path']=nm.seq(S['c_k'],nm.delim(S['x(t)']))
    S['lossineq']=nm.seq(S['loss_g'],nm.delim(S['wtilde_g']),nm.r('≤',True),nm.delim(nm.seq(nm.r('1−',True),'η')),S['loss_g'],nm.delim(S['w_g*']),nm.r('+',True),'η',S['loss_g'],nm.delim(S['w_all*']),nm.r('≤',True),S['loss_g'],nm.delim(S['w_all*']))
    S['rho_threshold']=nm.seq(nm.delim('ρ','|','|'),nm.r('=0.60',True))
    S['OAV_gt']=nm.seq(nm.r('OAV',True),nm.r('>1',True))
    def paragraph(value,template=None,bold=False):
        p=cp(template if template is not None else para);pr=pp(p)
        rp=p.find('w:r/w:rPr',NS);rp=cp(rp) if rp is not None else None
        for child in list(p):
            if child is not pr:p.remove(child)
        for tag in ['sectPr','pageBreakBefore','numPr','keepNext']:
            old=pr.find(W+tag)
            if old is not None:pr.remove(old)
        chunks=re.split(r'(\[\[[^\]]+\]\])',value)
        for chunk in chunks:
            if not chunk:continue
            if chunk.startswith('[['):p.append(nm.math(S[chunk[2:-2]]));continue
            r=E.SubElement(p,W+'r')
            if rp is not None:r.append(cp(rp))
            if bold:prop(r.find(W+'rPr') if r.find(W+'rPr') is not None else E.SubElement(r,W+'rPr'),'b',val=1)
            t=E.SubElement(r,W+'t');t.text=chunk;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        return p
    def find(prefix):return next(p for p in body.findall(W+'p') if txt(p).startswith(prefix))
    def proposition(value):
        p=paragraph(value)
        first=p.find(W+'r');t=first.find(W+'t')
        title,rest=t.text.split('。',1);t.text=title+'。'
        rp=first.find(W+'rPr')
        if rp is None:rp=E.Element(W+'rPr');first.insert(0,rp)
        prop(rp,'b',val=1)
        if rest:
            r=E.Element(W+'r');pr=cp(rp);prop(pr,'b',val=0);r.append(pr);tt=E.SubElement(r,W+'t');tt.text=rest;first.addnext(r)
        return p
    def insert_after(anchor,nodes):
        for node in nodes:anchor.addnext(node);anchor=node
        return anchor
    def image(path,number,caption_value):
        data=path.read_bytes();p=cp(picture);rid=f'rIdDataOverview{number}';name=f'data_overview_{number}.png'
        files['word/media/'+name]=data
        E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',Id=rid,Type=NS['r']+'/image',Target='media/'+name)
        p.find('.//a:blip',NS).set('{'+NS['r']+'}embed',rid)
        w,h=Image.open(BytesIO(data)).size;cx=round(15.25/2.54*914400);cy=round(cx*h/w)
        for x in p.xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=NS):x.set('cx',str(cx));x.set('cy',str(cy))
        for x in p.xpath('.//wp:docPr | .//pic:cNvPr',namespaces=NS):x.set('id',str(200+number));x.set('name',f'图{number}');x.set('descr',caption_value)
        prop(pp(p),'keepNext');prop(pp(p),'keepLines')
        return [p,paragraph(f'图{number} {caption_value}',cap)]
    # Remap references first, before inserting newly numbered figure references.
    figmap={i:i+(1 if i<=5 else 2 if i<=10 else 3) for i in range(1,15)}
    refmap={i:i+2 for i in range(1,12)}|{12:1,13:2}
    for t in root.xpath('//w:t',namespaces=NS):
        if t.text:
            t.text=re.sub(r'(?<!附)图(\d+)(?!\d)',lambda m:'图'+str(figmap.get(int(m[1]),int(m[1]))),t.text)
            t.text=re.sub(r'\[(\d+)\]',lambda m:'['+str(refmap.get(int(m[1]),int(m[1])))+']',t.text)
    # Move the AI entries ahead of scientific references, keeping all entries.
    references=[p for p in body.findall(W+'p') if re.match(r'^\[\d+\]\s',txt(p))]
    assert len(references)==13
    ref_heading=next(p for p in body.findall(W+'p') if txt(p)=='参考文献' and
                     p.find('w:pPr/w:pStyle',NS) is not None and p.find('w:pPr/w:pStyle',NS).get(W+'val')=='2')
    ordered=sorted(references,key=lambda p:int(re.match(r'^\[(\d+)\]',txt(p))[1]))
    for p in references:body.remove(p)
    insert_after(ref_heading,ordered)
    assert body.index(ordered[0])>body.index(next(x for x in body.findall(W+'bookmarkStart') if x.get(W+'name')=='MainText'))
    # The requested paragraph is retained verbatim; disclose the substantive
    # assistance evidenced by the session in a separate factual supplement.
    ai_heading=find('AI 工具使用声明');old_ai=ai_heading.getnext()
    body.replace(old_ai,paragraph(AI))
    ai_para=ai_heading.getnext()
    ai_supp=paragraph('AI辅助还涉及模型候选比较、代码实现及模型检验分析；具体采纳内容、人工修改和复核记录应在所附使用详情中如实对应。')
    ai_para.addnext(ai_supp)
    # Clearly state the actual schema resolution rather than treating standardized
    # labels as recoverable original scores.
    p=find('问题一要求利用化学')
    value='问题一要求依据附件1，给出附件3所定义Aroma、Flavour等维度的0—6分级评分。问题二要求利用附件2构建共表达图，结合附件1，为每个维度给出Top-3化合物及稳定性和化学论证。问题三要求以tasting_category_fine为条件，生成Tripel、Kriek、Lambic等风格的化合物谱与感官画像，并在跨风格迁移时检查DMS、双乙酰、4VG、反-2-壬烯醛及陈旧酒花味等缺陷。下文依次给出评分接口、50维完整前三项结果及有约束的统计候选。'
    body.replace(p,paragraph(value,p))
    intro=find('本研究以附件1和附件4为共同数据入口')
    schema=paragraph('题面概述中的样本与维度数量同实际附件存在差异，本文以提供数据对账作答：附件1/4均为250款，附件3的summary实际含50个感官目标；经明确别名对齐后与附件4一一对应。附件3只在此核对定义，不作为额外观测或参与拟合。附件4缺少原品评尺度参数，评分输出采用式(6)的训练经验分布相对强度。')
    intro.addnext(schema)
    q1_overview_intro=paragraph('图1展示评分数据的结构：缺失集中在9字段，50目标由五个家族组成，22风格样本数明显不均衡。缺失数以单元计、风格样本数以实测啤酒计；附件8的70个尺度可核验字段不等于70字段逐值全部一致。')
    schema.addnext(q1_overview_intro)
    insert_after(q1_overview_intro,image(FIG/'Q1_数据概览_待审.png',1,'感官预测数据的缺失结构、目标构成与风格样本量'))
    q2intro=find('本问在第5章数据契约与预测接口')
    q2text=paragraph('图7给出问题二的实际候选口径。231个字段中先排除13个不满足严格单体资格的字段，再对218个候选的23653对相关作描述，成对有效样本为211—250；最终50维各选3项得到150条关系，不能将其误作150个输入字段。附件2虽标称Pearson，其数值与附件1成对完整Spearman核验一致，用于本文共表达描述图；监督检验的相关与模块仍在训练折重算。')
    q2intro.addnext(q2text);insert_after(q2text,image(FIG/'Q2_数据概览_待审.png',7,'单体候选资格及其实际共线性分布'))
    q3intro=find('本问条件标签为啤酒风格')
    q3text=paragraph('图13分开显示真实输入与既有接口完整性。Tripel、Kriek、Lambic的实测样本为28/10/11；中间面板来自附件4真实观测，点与横线为中位数及四分位距。每风格300谱对应15000个有限预测数值，总量45000表示900份画像的50维数值，不是新增45000个独立样本。')
    q3intro.addnext(q3text);insert_after(q3text,image(FIG/'Q3_数据概览_待审.png',13,'目标风格的真实样本、实测感官分布与画像接口完整性'))
    # Three short analytic facts, matching the proof-bearing structure of 0043.
    # None is used as a proof of out-of-sample superiority or physical feasibility.
    eq3=next(t for t in body.findall(W+'tbl') if txt(t.findall('w:tr/w:tc',NS)[-1])=='(3)')
    theorem=proposition('命题5-1　家族收缩的训练损失界。固定式(2)的训练预测及尺度，将其非负二次目标记为[[loss_g]]。式(3)的最终家族权重仍非负，且其家族训练损失不超过共享权重的家族损失。')
    proof=paragraph('证明。式(2)为仿射残差平方和，因而是权重的凸函数；家族最小化解的损失不大于任意可行共享解。式(3)是两个非负向量的凸组合，由凸性得[[lossineq]]。该结论只控制固定训练目标的损失，不保证外层预测指标提高。')
    insert_after(eq3.getnext(),[theorem,proof])
    eq13=next(t for t in body.findall(W+'tbl') if txt(t.findall('w:tr/w:tc',NS)[-1])=='(13)')
    theorem=proposition('命题6-1　同支持数下的秩支配。对同一感官目标，若两个候选的Top-10、Top-3支持数相同，且候选甲在四路的名次均不劣于候选乙，并至少一路严格更优，则式(13)的融合分数对候选甲更高。')
    proof=paragraph('证明。各路名次为正数，平滑常数60固定，倒数函数在正数域严格递减；名次逐路不劣使每项倒数不小，至少一路严格不等使总和严格增大。支持数相同时才比较该分数，故该候选不会在融合分数层被名次受支配者反超；支持数不同不适用。')
    insert_after(eq13.getnext(),[theorem,proof])
    eq21=next(t for t in body.findall(W+'tbl') if txt(t.findall('w:tr/w:tc',NS)[-1])=='(21)')
    theorem=proposition('命题7-1　已知直接浓度的截断保序性。对身份、单位、尺度均可核验的缺陷字段，式(21)给出的候选浓度不高于源浓度；若源浓度不超过同单位感知阈值，则截断后亦不超过该阈值。')
    proof=paragraph('证明。两数的最小值不大于其中任一数，故[[c_final]]不大于[[c_source]]。若源浓度不超过阈值，传递性即得结论。源浓度已超阈时，该操作只保证不升，不能保证降至阈下；缺失或尺度未知字段、以及陈旧酒花味代理均不属于本命题。')
    insert_after(eq21.getnext(),[theorem,proof])
    # Preserve continuous results, adding only a named, nearest-integer display
    # of frozen relative grades to meet the seven-category output wording.
    eq6=next(t for t in body.findall(W+'tbl') if txt(t.findall('w:tr/w:tc',NS)[-1])=='(6)')
    grade_explanation=paragraph('为对应题面的七档分级展示，对已经冻结的相对强度再定义[[grade_round]]，即就近取0—6整数、半档向上。该整数仅是相对分位的展示码，连续预测及此前所有评价数字保持原口径；不能把排序代理指标称为原始评分准确率。完整250×50相对强度与七档展示随评分结果表提供。')
    insert_after(eq6.getnext(),[grade_explanation])
    q2summary=find('按bootstrap频率选出的150对关系')
    q2summary.addprevious(paragraph('本问的正式答案是50个感官目标各3项、共150条化合物—感官关系。完整顺序及各对入选频率、留一风格保留率和化学等级列于“问题二_50维Top3与证据.csv”；表11和图11仅展示其中8条A级关系，不代替完整答案。'))
    q3summary=find('固定最小改动规则用于11个源')
    q3summary.addprevious(paragraph('按题面缺陷名单，DMS、双乙酰、4VG及反-2-壬烯醛在直接浓度可比时逐项核对源值、候选值和阈值；陈旧酒花味无单一可比化学列，另用固定感官代理。已知源值不升与超阈状态分开报告，未知项不计为通过，完整风格画像和迁移审计表随文提供。'))
    # Fix appendix reading cues and captions/legends using font-only regenerated
    # supplementary images; the fourteen existing main images remain untouched.
    replacements={
      '本图补充正文相对量表映射与误差形态的解释。':'附图S1补充正文相对量表映射与误差形态的解释。',
      '本图补充正文结构消融和压力检验。':'附图S2补充正文结构消融和压力检验。',
      '本图补充正文50个目标的相对可靠性，':'附图S3补充正文50个目标的相对可靠性，',
    }
    supplement_names=['附图S1_残差诊断与量表映射.png','附图S2_消融与负对照.png','附图S3_逐目标预测可靠性.png']
    for t in root.xpath('//w:t',namespaces=NS):
        if t.text:
            for before,after in replacements.items():t.text=t.text.replace(before,after)
            t.text=t.text.replace('先排除13个聚合/理化指标','先排除13个不满足严格单体资格的字段').replace('排除13个聚合/理化指标','排除13个非严格单体字段')
    relmap={r.get('Id'):r.get('Target') for r in rels}
    for p in body.findall(W+'p'):
        if re.match(r'附图S[123]\s',txt(p)):
            value=txt(p);pr=pp(p);fresh=paragraph(value,cap);p.getparent().replace(p,fresh)
            idx=int(value[3])-1
            drawing=fresh.getprevious();rid=drawing.find('.//a:blip',NS).get('{'+NS['r']+'}embed')
            files['word/'+relmap[rid]]=(BASE/'附录图_v2'/supplement_names[idx]).read_bytes()
    # Corresponding frozen outputs, with only the integer display newly derived.
    q1path=ROOT/'05_论文/Q1_感官评分/数据与结果/final_oof_with_relative_grades.csv.gz'
    q1=pd.read_csv(q1path);q1['relative_grade_integer']=np.floor(q1['grade_0_6']+.5).astype(int)
    assert q1.relative_grade_integer.between(0,6).all() and len(q1)==12500
    q1.to_csv(DATA/'问题一_250啤酒50维相对评分.csv.gz',index=False,compression='gzip')
    q2path=ROOT/'05_论文/Q2_关键化合物/数据与结果/关键化合物Top3_当前版.csv';q2=pd.read_csv(q2path)
    q2cols=['target_code','target_name_en','consensus_rank','feature_code','compound_name','bootstrap_selection_frequency','leave_style_top3_frequency','chemical_evidence_grade','oav_median','claim_scope']
    q2[q2cols].to_csv(DATA/'问题二_50维Top3与证据.csv',index=False,encoding='utf-8-sig')
    q3root=ROOT/'05_论文/交付给论文手/待审_20261027/Q3_风格生成/数据与结果'
    shutil.copy2(q3root/'完整风格画像.csv',DATA/'问题三_完整风格画像.csv')
    for p in q3root.glob('*.csv'):
        if '迁移' in p.name:shutil.copy2(p,DATA/p.name)
    source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [SOURCE,q1path,q2path,q3root/'完整风格画像.csv']}
    # Keep all mathematical objects native. Paragraph/run property order follows
    # the retained mother schema; styles/settings/numbering are untouched.
    orders={'pPr':'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange','rPr':'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath rPrChange'}
    for local,order in orders.items():
        rank={W+n:i for i,n in enumerate(order.split())}
        for pr in root.findall('.//'+W+local):pr[:]=sorted(pr,key=lambda x:rank.get(x.tag,999))
    files['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    files['word/_rels/document.xml.rels']=E.tostring(rels,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
        for n,data in files.items():z.writestr(n,data)
    figure_list=[]
    for p in body.findall(W+'p'):
        if re.match(r'图\d+\s',txt(p)):
            caption=txt(p);n=int(re.match(r'图(\d+)',caption)[1]);prev=p.getprevious();rid=prev.find('.//a:blip',NS).get('{'+NS['r']+'}embed')
            entry={'number':n,'caption':caption,'embedded_image':relmap.get(rid,'media/data_overview_'+str(n)+'.png'),'image_sha256':sha256(files['word/'+dict((x.get('Id'),x.get('Target')) for x in rels)[rid]]).hexdigest()}
            figure_list.append(entry)
    pd.DataFrame(figure_list).to_csv(BASE/'总论文_v2_图清单.csv',index=False,encoding='utf-8-sig')
    record={'output':str(TARGET),'sources':source_hashes,'figure_map':figmap,'reference_map':refmap,
      'new_main_figures':[1,7,13],'main_figures':17,'supplement_images':3,'propositions':['5-1','6-1','7-1'],
      'ai_paragraph_exact':AI,'ai_scope_supplement':txt(ai_supp),'ai_version_and_human_review_basis':'user supplied; not independently verified',
      'no_retraining':True,'new_grade_display':'floor(frozen grade_0_6+0.5); continuous values unchanged'}
    (AUDIT/'assembly_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(TARGET);print('17 main figures, 3 supplementary images, 3 analytic propositions.')


if __name__=='__main__':main()
