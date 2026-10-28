"""Replace the contents of an exact 0043 mother copy, retaining its style parts.

No model is fitted and no per-question deliverable is modified.
"""
from copy import deepcopy as cp
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
import shutil

from lxml import etree as E
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261027_总论文整合'
OUT = ROOT / '05_论文/总论文'
MOTHER = Path('/mnt/d/上海大学__/26年数学建模/26数模国赛/0043优化成果/定稿.docx')
TARGET = OUT / '2026_MathorCup_大数据竞赛_总论文_v1.docx'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
PROTECTED = ['word/styles.xml','word/numbering.xml','word/theme/theme1.xml',
             'word/fontTable.xml','word/settings.xml']


def q(name):
    prefix, local = name.split(':')
    return '{'+NS[prefix]+'}'+local


def el(tagname, **attrs):
    x=E.Element(q(tagname))
    for k,v in attrs.items(): x.set(q('w:'+k),str(v))
    return x


def text(x): return ''.join(x.xpath('.//w:t/text()', namespaces=NS))


def setprop(parent, name, **attrs):
    x=parent.find(q(name))
    if x is None: x=E.SubElement(parent,q(name))
    for k,v in attrs.items(): x.set(q('w:'+k),str(v))
    return x


def props(p):
    x=p.find(q('w:pPr'))
    if x is None: x=el('w:pPr');p.insert(0,x)
    return x


def run(p, value, rp=None):
    r=E.SubElement(p,q('w:r'))
    if rp is not None:r.append(cp(rp))
    t=E.SubElement(r,q('w:t'));t.text=value
    t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    return r


def zipread(path):
    with ZipFile(path) as z:return {n:z.read(n) for n in z.namelist()}


class Source:
    def __init__(self,path):
        self.path=path;self.files=zipread(path)
        self.root=E.fromstring(self.files['word/document.xml'])
        self.body=self.root.find(q('w:body'))
        self.rels={r.get('Id'):r.get('Target') for r in E.fromstring(self.files['word/_rels/document.xml.rels'])}
        self.tables={};self.formulas={};self.figures={};self.notes={}
        en=0
        for node in self.body:
            if node.tag==q('w:tbl'):
                maths=node.xpath('.//m:oMath',namespaces=NS)
                if maths:
                    en+=1;self.formulas[en]=maths[0]
                else:
                    cap=node.getprevious();match=re.match(r'表(\d+)\s*(.*)',text(cap))
                    if not match:raise ValueError(text(cap))
                    self.tables[int(match[1])]=(match[2],node)
            elif node.tag==q('w:p') and node.find('.//w:drawing',NS) is not None:
                caption=text(node.getnext());match=re.match(r'图(\d+)\s*(.*)',caption)
                if not match:continue  # Q1 appendix is explicitly excluded this round.
                rid=node.find('.//a:blip',NS).get(q('r:embed'))
                target=self.rels[rid].lstrip('/')
                if not target.startswith('word/'):target='word/'+target
                self.figures[int(match[1])]=(match[2],self.files[target])
                note=node.getnext().getnext()
                self.notes[int(match[1])]=text(note) if note is not None and text(note).startswith('注') else ''


REFS={
'Geurts2006':'Geurts P, Ernst D, Wehenkel L. Extremely randomized trees[J]. Machine Learning, 2006, 63: 3–42. DOI:10.1007/s10994-006-6226-1.',
'Breiman1996':'Breiman L. Stacked regressions[J]. Machine Learning, 1996, 24: 49–64. DOI:10.1023/A:1018046112532.',
'Zhang2005':'Zhang B, Horvath S. A general framework for weighted gene co-expression network analysis[J]. Statistical Applications in Genetics and Molecular Biology, 2005, 4: Article 17. DOI:10.2202/1544-6115.1128.',
'Friedman2008':'Friedman J, Hastie T, Tibshirani R. Sparse inverse covariance estimation with the graphical lasso[J]. Biostatistics, 2008, 9(3): 432–441. DOI:10.1093/biostatistics/kxm045.',
'Strobl2008':'Strobl C, Boulesteix A L, Kneib T, et al. Conditional variable importance for random forests[J]. BMC Bioinformatics, 2008, 9: 307. DOI:10.1186/1471-2105-9-307.',
'Meinshausen2010':'Meinshausen N, Bühlmann P. Stability selection[J]. Journal of the Royal Statistical Society: Series B, 2010, 72(4): 417–473. DOI:10.1111/j.1467-9868.2010.00740.x.',
'Shah2013':'Shah R D, Samworth R J. Variable selection with error control: another look at stability selection[J]. Journal of the Royal Statistical Society: Series B, 2013, 75(1): 55–80. DOI:10.1111/j.1467-9868.2011.01034.x.',
'Lundberg2017':'Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems 30. 2017: 4765–4774.',
'Sklar1959':"Sklar A. Fonctions de répartition à n dimensions et leurs marges[J]. Publications de l'Institut de Statistique de l'Université de Paris, 1959, 8: 229–231.",
'Ledoit2004':'Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices[J]. Journal of Multivariate Analysis, 2004, 88(2): 365–411.',
'Mahalanobis1936':'Mahalanobis P C. On the generalized distance in statistics[J]. Proceedings of the National Institute of Sciences of India, 1936, 2: 49–55.'}

NOTATION=[['符号','定义','口径或范围'],
['i；k；j','啤酒、化学字段、感官目标索引','250款；231字段；50目标'],
['s；g；m','啤酒风格、感官家族、基学习器索引','22风格；5家族；4学习器'],
['T；xᵢ；yᵢ','当前训练集；化学输入；标准化感官向量','均为当前训练尺度'],
['zᵢⱼₘ；sⱼ','内层折外基预测；目标训练标准差','不使用外层验证统计量'],
['w；bⱼ；η','非负组合权重；自由截距；组收缩系数','权重不必和为1；η=0.5'],
['Fⱼ,tr；gᵢⱼ','训练目标经验分布；相对强度','g取0—6，不是原始评分'],
['ρ；Rᵢₖ；Rᵢⱼ','Spearman系数；输入秩；目标秩','无量纲秩统计量'],
['β；L；λ₁；λG','回归系数；有符号图拉普拉斯；惩罚参数','第二问训练折内估计'],
['Hₖⱼ；rₗₖⱼ','倒数秩融合分数；第l路名次','四路统计排序'],
['πₖⱼ；B；I(·)','bootstrap入选频率；次数；示性函数','B=500；频率取0—1'],
['cₖ；τₖ；OAVₖ','校准浓度；感知阈值；气味活性值','前两者须同单位；OAV无量纲'],
['uᵢₖ；Qₛₖ','Copula正态潜在变量；风格经验分位函数','第三问训练风格内估计'],
['Σₛ；Dₛ；Rₛ；αₛ','潜在协方差；对角阵；相关阵；收缩强度','不与秩变量Rᵢₖ混用'],
['x₀；a；t','源谱；目标端点；迁移路径系数','t∈[0,1]，区别于η'],
['dₛ；RMS','目标风格距离；化学设计坐标均方根改动','训练标准化设计坐标']]


def main():
    OUT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
    # The output begins as an actual byte-for-byte mother copy.
    shutil.copy2(MOTHER,TARGET)
    copy_hash=sha256(TARGET.read_bytes()).hexdigest()
    files=zipread(TARGET);root=E.fromstring(files['word/document.xml']);body=root.find(q('w:body'))
    old=list(body);eqtemplate=cp(body.findall(q('w:tbl'))[1])
    pictemplate=cp(next(p for p in body.findall(q('w:p')) if p.find('.//w:drawing',NS) is not None))
    section=cp(body.find(q('w:sectPr')))
    paragraph_templates={'p':cp(old[25]),'h1':cp(old[16]),'h2':cp(old[17]),'title':cp(old[0]),'abstract':cp(old[1]),'placeholder':cp(old[2]),'keywords':cp(old[8])}
    captemplate=cp(next(p for p in body.findall(q('w:p')) if re.match(r'图\d+\s',text(p))))
    for node in list(body):body.remove(node)
    sources={qname:Source(next((ROOT/'05_论文').glob(qname+'*/待审_20261027/'+name))) for qname,name in
             [('Q1','第一问_论文正文_v6_待审.docx'),('Q2','第二问_论文正文_v5_待审.docx'),('Q3','第三问_论文正文_v3_待审.docx')]}
    blocks=json.loads((AUDIT/'总论文内容源.json').read_text())['blocks']
    maps={'T':{},'F':{},'E':{},'C':{}}
    for block in blocks:
        key={'table':'T','figure':'F','eq':'E'}.get(block['type'])
        if key:maps[key][block['source']]=len(maps[key])+1
        for cite in re.findall(r'\{C:([^}]+)\}',block.get('text','')):
            if cite not in maps['C']:maps['C'][cite]=len(maps['C'])+1
    def substitute(value):
        value=value.replace('Lₛᵢgₙ','L').replace('mean-OOF','平均折外预测')
        return re.sub(r'\{([TFEC]):([^}]+)\}',lambda m:({'T':'表','F':'图','E':'式','C':''}[m[1]]+
                      (f'({maps[m[1]][m[2]]})' if m[1]=='E' else f'[{maps[m[1]][m[2]]}]' if m[1]=='C' else str(maps[m[1]][m[2]]))),value)
    def paragraph(value,kind='p',caption=False):
        p=cp(captemplate if caption else paragraph_templates[kind]);pp=props(p)
        rp=p.find('w:r/w:rPr',NS);rp=cp(rp) if rp is not None else None
        for n in list(p):
            if n is not pp:p.remove(n)
        for name in ['w:sectPr','w:pageBreakBefore','w:keepNext','w:numPr']:
            x=pp.find(q(name))
            if x is not None:pp.remove(x)
        if kind=='h1':
            num=E.SubElement(pp,q('w:numPr'));num.append(el('w:ilvl',val=0));num.append(el('w:numId',val=1))
        if kind=='h2':
            num=E.SubElement(pp,q('w:numPr'));num.append(el('w:numId',val=0))
        if kind=='p':
            setprop(pp,'w:ind',firstLine=480,firstLineChars=200)
            # The mother's sole Body Text paragraph starts with a bold question
            # label; its remaining prose is regular. Do not propagate that label.
            if rp is not None:
                setprop(rp,'w:b',val=0);setprop(rp,'w:bCs',val=0)
        if caption or kind in ['h1','h2']:
            setprop(pp,'w:keepLines')
        if kind in ['h1','h2'] or (caption and value.startswith('表')):setprop(pp,'w:keepNext')
        run(p,substitute(value),rp)
        return p
    rels=E.fromstring(files['word/_rels/document.xml.rels'])
    for r in list(rels):
        if r.get('Type').split('/')[-1] in ['image','customXml']:rels.remove(r)
    for name in list(files):
        if name.startswith(('word/media/','customXml/','docProps/thumbnail')):del files[name]
    ct=E.fromstring(files['[Content_Types].xml'])
    for c in list(ct):
        if c.get('PartName','').startswith('/customXml/'):ct.remove(c)
    if not any(c.get('Extension')=='png' for c in ct):
        E.SubElement(ct,'{'+ct.nsmap[None]+'}Default',Extension='png',ContentType='image/png')
    def data_table(rows,widths=None):
        columns=len(rows[0]);widths=widths or [8730//columns]*columns;widths[-1]=8730-sum(widths[:-1])
        table=el('w:tbl');tp=E.SubElement(table,q('w:tblPr'))
        tp.append(el('w:tblStyle',val='43'));setprop(tp,'w:tblW',w=8730,type='dxa');setprop(tp,'w:jc',val='center')
        setprop(tp,'w:tblLayout',type='fixed')
        margins=E.SubElement(tp,q('w:tblCellMar'))
        for edge,v in [('top',35),('bottom',35),('left',65),('right',65)]:setprop(margins,'w:'+edge,w=v,type='dxa')
        grid=E.SubElement(table,q('w:tblGrid'))
        for w in widths:grid.append(el('w:gridCol',w=w))
        for i,row in enumerate(rows):
            tr=E.SubElement(table,q('w:tr'));trp=E.SubElement(tr,q('w:trPr'));setprop(trp,'w:cantSplit')
            if i==0:setprop(trp,'w:tblHeader')
            for j,value in enumerate(row):
                tc=E.SubElement(tr,q('w:tc'));tcp=E.SubElement(tc,q('w:tcPr'));setprop(tcp,'w:tcW',w=widths[j],type='dxa');setprop(tcp,'w:vAlign',val='center')
                p=paragraph(value);pp=props(p)
                setprop(pp,'w:ind',firstLine=0,firstLineChars=0);setprop(pp,'w:jc',val='center' if j else 'left')
                setprop(pp,'w:spacing',before=0,after=0,line=240,lineRule='auto');setprop(pp,'w:snapToGrid',val=0)
                setprop(pp,'w:keepLines')
                if i<len(rows)-1:setprop(pp,'w:keepNext')
                rp=E.SubElement(p.find(q('w:r')),q('w:rPr')) if p.find('w:r/w:rPr',NS) is None else p.find('w:r/w:rPr',NS)
                setprop(rp,'w:rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体')
                setprop(rp,'w:sz',val=20);setprop(rp,'w:szCs',val=20)
                if i==0:setprop(rp,'w:b')
                # rPr must precede text, including when it was just created.
                r=p.find(q('w:r'));r.remove(rp);r.insert(0,rp)
                tc.append(p)
        return table
    def eq(source):
        qn,oldnum=source.split(':');math=cp(sources[qn].formulas[int(oldnum)])
        for t in math.xpath('.//m:t',namespaces=NS):
            if qn=='Q1' and t.text:t.text=t.text.replace('λ','η')
            if qn=='Q2' and oldnum=='5' and t.text=='S':t.text='H'
            if qn=='Q3' and oldnum in ['1','4'] and t.text:t.text=t.text.replace('z','u')
            if qn=='Q3' and oldnum in ['5','6'] and t.text:t.text=t.text.replace('λ','t')
        for rp in math.xpath('.//w:rPr',namespaces=NS):
            setprop(rp,'w:rFonts',ascii='Cambria Math',hAnsi='Cambria Math',eastAsia='Cambria Math')
        if source=='Q1:2':
            # Three nested sums must not stretch to the height of the fraction.
            for pr in math.xpath('.//m:naryPr/m:grow',namespaces=NS):pr.set(q('m:val'),'0')
        if source=='Q2:3':
            # The original subscript s meant signed, not the beer-style index.
            for sub in math.xpath('.//m:sSub[m:e/m:r/m:t="L"]',namespaces=NS):
                parent=sub.getparent();index=parent.index(sub)
                for child in reversed(list(sub.find(q('m:e')))):parent.insert(index,cp(child))
                parent.remove(sub)
        table=cp(eqtemplate);cells=table.findall('w:tr/w:tc',NS)
        for ci,c in enumerate(cells):
            tcp=c.find(q('w:tcPr'))
            for child in list(c):
                if child is not tcp:c.remove(child)
            p=paragraph('');pp=props(p);setprop(pp,'w:ind',firstLine=0,firstLineChars=0);setprop(pp,'w:jc',val='right' if ci==2 else 'center')
            setprop(pp,'w:spacing',before=20,after=20);setprop(pp,'w:keepLines')
            for r in p.findall(q('w:r')):p.remove(r)
            if ci==1:p.append(math)
            elif ci==2:run(p,f'({maps["E"][source]})')
            c.append(p)
        setprop(table.find('w:tr/w:trPr',NS) if table.find('w:tr/w:trPr',NS) is not None else E.SubElement(table.find(q('w:tr')),q('w:trPr')),'w:cantSplit')
        return table
    def figure(source):
        qn,oldnum=source.split(':');caption,data=sources[qn].figures[int(oldnum)]
        n=maps['F'][source];name=f'figure{n:02d}.png';rid=f'rIdFigure{n}'
        files['word/media/'+name]=data
        E.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=NS['r']+'/image',Target='media/'+name)
        p=cp(pictemplate)
        for t in p.xpath('.//w:t',namespaces=NS):t.getparent().remove(t)
        p.find('.//a:blip',NS).set(q('r:embed'),rid)
        width,height=Image.open(BytesIO(data)).size;cx=round(15.25/2.54*914400);cy=round(cx*height/width)
        for x in p.xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=NS):x.set('cx',str(cx));x.set('cy',str(cy))
        for x in p.xpath('.//wp:docPr | .//pic:cNvPr',namespaces=NS):x.set('id',str(n));x.set('name',f'图{n}');x.set('descr',caption)
        pp=props(p);setprop(pp,'w:keepNext');setprop(pp,'w:keepLines');setprop(pp,'w:spacing',before=20,after=0)
        setprop(pp,'w:ind',firstLine=0,firstLineChars=0)
        return p,paragraph(f'图{n} {caption}',caption=True)
    # Front page: exact mother paragraph formats with fully replaced text.
    body.append(paragraph('【题目待定】','title'))
    body.append(paragraph('摘 要','abstract'))
    body.append(paragraph('【摘要下一轮撰写】','placeholder'))
    body.append(paragraph('【关键词下一轮确定】','keywords'))
    toc_title=paragraph('目 录','abstract');setprop(props(toc_title),'w:pageBreakBefore');body.append(toc_title)
    toc=paragraph('');pp=props(toc);setprop(pp,'w:ind',firstLine=0,firstLineChars=0)
    for r in toc.findall(q('w:r')):toc.remove(r)
    r=E.SubElement(toc,q('w:r'));f=E.SubElement(r,q('w:fldChar'));f.set(q('w:fldCharType'),'begin')
    r=E.SubElement(toc,q('w:r'));instr=E.SubElement(r,q('w:instrText'));instr.text=' TOC \\o "1-3" \\h \\z \\b MainText '
    r=E.SubElement(toc,q('w:r'));f=E.SubElement(r,q('w:fldChar'));f.set(q('w:fldCharType'),'separate')
    run(toc,'目录正在由Word更新')
    r=E.SubElement(toc,q('w:r'));f=E.SubElement(r,q('w:fldChar'));f.set(q('w:fldCharType'),'end')
    body.append(toc)
    front=cp(section)
    for x in list(front):
        if x.tag in [q('w:headerReference'),q('w:footerReference'),q('w:pgNumType'),q('w:titlePg')]:front.remove(x)
    for kind in ['default','first','even']:
        x=el('w:footerReference',type=kind);x.set(q('r:id'),'rId9');front.insert(0,x)
    setprop(front,'w:type',val='nextPage')
    sectionp=paragraph('');props(sectionp).append(front);body.append(sectionp)
    bookmark=el('w:bookmarkStart',id=9000,name='MainText');body.append(bookmark)
    for block in blocks:
        typ=block['type']
        if typ in ['p','h1','h2']:body.append(paragraph(block['text'],typ))
        elif typ=='eq':body.append(eq(block['source']))
        elif typ=='figure':
            for x in figure(block['source']):body.append(x)
        elif typ=='table':
            source=block['source']
            if source=='notation':caption='全文主要符号及其含义';rows=NOTATION;widths=[2170,3440,3120]
            else:
                qn,oldnum=source.split(':');caption,t=sources[qn].tables[int(oldnum)]
                rows=[[text(c) for c in tr.findall(q('w:tc'))] for tr in t.findall(q('w:tr'))]
                # Correct the documented NNLS group-count prose, never any number in result cells.
                rows=[[v.replace('三个感官组','五个感官组').replace('mean-OOF','平均折外预测').replace('R1/R2/R3','强稳健/中等稳健/探索性') for v in row] for row in rows]
                cols=len(rows[0]);widths={3:[1950,3100,3680],4:[1780,2400,2320,2230],5:[2350,1595,1595,1595,1595],6:[2250,1296,1296,1296,1296,1296]}[cols]
            body.append(paragraph(f'表{maps["T"][source]} {caption}',caption=True));body.append(data_table(rows,widths))
    ref=paragraph('参考文献','h1');setprop(props(ref).find(q('w:numPr')),'w:numId',val=0);body.append(ref)
    for key,n in maps['C'].items():
        p=paragraph(f'[{n}] '+REFS[key]);pp=props(p);setprop(pp,'w:ind',firstLine=0,firstLineChars=0)
        setprop(pp,'w:keepLines');body.append(p)
    body.append(el('w:bookmarkEnd',id=9000))
    appendix=paragraph('附录（下一轮组装）','abstract');setprop(props(appendix),'w:pageBreakBefore');setprop(props(appendix),'w:outlineLvl',val=9);body.append(appendix)
    body.append(paragraph('待并入：固定目标残差与训练折相对量表映射；结构消融、标签置换与留一风格压力测试；50个目标的中文逐目标可靠性图；核心程序与完整复现材料。本轮仅保留位置，不组装附录。'))
    for x in list(section):
        if x.tag in [q('w:headerReference'),q('w:footerReference'),q('w:titlePg')]:section.remove(x)
    for kind in ['default','first','even']:
        x=el('w:footerReference',type=kind);x.set(q('r:id'),'rId8');section.insert(0,x)
    setprop(section,'w:pgNumType',fmt='decimal',start=1);body.append(section)
    # Word otherwise coalesces adjacent formula tables into a multi-row table.
    for node in list(body):
        previous=node.getprevious()
        if node.tag==q('w:tbl') and previous is not None and previous.tag==q('w:tbl'):
            spacer=paragraph('');pp=props(spacer)
            setprop(pp,'w:spacing',before=0,after=0,line=20,lineRule='exact')
            setprop(pp,'w:snapToGrid',val=0);setprop(pp,'w:keepNext')
            node.addprevious(spacer)
    # Order property children according to the schema; source mother order is retained otherwise.
    orders={
    'pPr':'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange',
    'rPr':'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath rPrChange',
    'tblPr':'tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription tblPrChange',
    'trPr':'cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden ins del trPrChange',
    'sectPr':'headerReference footerReference footnotePr endnotePr type pgSz pgMar paperSrc pgBorders lnNumType pgNumType cols formProt vAlign noEndnote titlePg textDirection bidi rtlGutter docGrid printerSettings sectPrChange'}
    for local,seq in orders.items():
        rank={q('w:'+s):i for i,s in enumerate(seq.split())}
        for node in root.xpath('.//w:'+local,namespaces=NS):node[:]=sorted(node,key=lambda x:rank.get(x.tag,999))
    for name in ['word/footnotes.xml','word/endnotes.xml']:
        if name in files:
            rr=E.fromstring(files[name])
            for child in list(rr):
                if child.get(q('w:type')) not in ['separator','continuationSeparator','continuationNotice']:rr.remove(child)
            files[name]=E.tostring(rr,xml_declaration=True,encoding='UTF-8',standalone=True)
    core='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
    files['docProps/core.xml']=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="{core}"/>'.encode()
    if 'docProps/custom.xml' in files:
        custom=E.fromstring(files['docProps/custom.xml'])
        for child in list(custom):custom.remove(child)
        files['docProps/custom.xml']=E.tostring(custom,xml_declaration=True,encoding='UTF-8',standalone=True)
    files['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    files['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    files['[Content_Types].xml']=E.tostring(ct,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    record={'mother':str(MOTHER),'mother_sha256':sha256(MOTHER.read_bytes()).hexdigest(),'initial_copy_sha256':copy_hash,
            'output':str(TARGET),'protected_parts':{p:sha256(files[p]).hexdigest() for p in PROTECTED},
            'sources':{qn:{'path':str(s.path),'sha256':sha256(s.path.read_bytes()).hexdigest()} for qn,s in sources.items()},
            'renumbering':maps,'counts':{k:len(v) for k,v in maps.items()},
            'symbol_changes':{'Q1:3':'λ→η（组收缩）','Q2:3':'L_s→L（s原为signed，避免与风格s混用）','Q2:5':'S→H（融合分数）','Q3:1,4':'z→u（Copula潜在坐标）','Q3:5,6':'λ→t（迁移路径）'}}
    (AUDIT/'assembly_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record['counts']));print(TARGET)


if __name__=='__main__':main()
