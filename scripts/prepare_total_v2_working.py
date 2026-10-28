"""Prepare a versioned, pre-figure Word review draft from v1; never fit models.

The three data-overview plots stay outside this document until user approval.
"""
from copy import deepcopy as cp
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
import re
import shutil
from lxml import etree as E
from PIL import Image
from io import BytesIO

import native_math_total as nm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_论文/总论文/v2_准备稿'
AUDIT=ROOT/'docs/collab/20261027_总论文v2准备'
SOURCE=ROOT/'05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v1.docx'
TARGET=OUT/'总论文_v2_图前工作稿.docx'
NS={'w':nm.W,'m':nm.M,'a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
W='{'+nm.W+'}';M='{'+nm.M+'}'
AI='本参赛队在竞赛过程中使用了 AI 工具，用于辅助赛题理解与任务拆解、术语规范与语言润色、公式排版、图表标题与参考文献格式检查、模型优点与局限的梳理、图表结果表述的量化修改，以及附录与支撑材料结构整理。模型假设、算法选择、参数设定、程序运行、结果判断与最终文字均由参赛队逐项复核，详细使用情况见支撑材料《AI 工具使用详情.pdf》。'


def text(x):return ''.join(x.xpath('.//w:t/text()',namespaces=NS))


def prop(x,name,**attrs):
    y=x.find(W+name)
    if y is None:y=E.SubElement(x,W+name)
    for k,v in attrs.items():y.set(W+k,str(v))
    return y


def pp(p):
    x=p.find(W+'pPr')
    if x is None:x=E.Element(W+'pPr');p.insert(0,x)
    return x


def readzip(p):
    with ZipFile(p) as z:return {n:z.read(n) for n in z.namelist()}


def equations(S):
    a=nm.seq;r=nm.r;su=nm.sub;sp=nm.sup;de=nm.delim;fr=nm.frac;sm=nm.summation
    op=lambda t:r(t,True)
    ar=lambda variable: a(nm.e('m:limLow'))
    def argmin(var):
        n=nm.e('m:limLow');n.append(nm.slot('e',r('arg min',True)));n.append(nm.slot('lim',var));return n
    yt=su('y','ij');yhat=S['ŷ_ij'];zc=S['z_ijm']
    centered=a(yt,op('−'),S['ȳ_jT'],op('−'),sm(a('m',op('=1')),op('4'),a(su('w','m'),de(a(zc,op('−'),S['zbar_jmT'])))))
    rankx=a(S['Rik_X'],op('−'),S['Rbar_kX']);ranky=a(S['Rij_Y'],op('−'),S['Rbar_jY'])
    valid=a('i',op('∈'),S['A_kj'])
    beta=S['β'];residual=a(S['y_jTc'],op('−'),S['X_Tc'],beta)
    delta=a(S['v(x)'],op('−'),S['μ_design'])
    quad=a(sp(de(delta),op('T')),S['Ω_s'],de(delta))
    d={
      1:nm.math(S['y_i'],op('='),'F',de(S['x_i']),op('+'),S['ε_i']),
      2:nm.math(S['w_g*'],op('='),argmin(a('w',op('≥0'))),sm(a('j',op('∈'),S['G_g']),[],sm(a('i',op('∈'),'T'),[],sp(de(fr(centered,S['σ_jT'])),op('2'))))),
      3:nm.math(S['wtilde_g'],op('='),de(a(op('1−'),'η')),S['w_g*'],op('+'),'η',S['w_all*'],op(','),'η',op('=0.5')),
      4:nm.math(S['b_j'],op('='),S['ȳ_jT'],op('−'),sm(a('m',op('=1')),op('4'),a(S['w_gjm'],S['zbar_jmT']))),
      5:nm.math(yhat,op('='),S['b_j'],op('+'),sm(a('m',op('=1')),op('4'),a(S['w_gjm'],S['f_mjT'],de(S['x_i'])))),
      6:nm.math(S['ĥ_ij'],op('=6·'),S['F_cdf'],de(yhat),op(','),op('0≤'),S['ĥ_ij'],op('≤6')),
      7:nm.math(S['R2_macro'],op('='),fr(op('1'),'J'),sm(a('j',op('=1')),'J',de(a(op('1−'),fr(sm(a('i',op('=1')),'N',sp(de(a(yt,op('−'),yhat)),op('2'))),sm(a('i',op('=1')),'N',sp(de(a(yt,op('−'),S['ȳ_jeval'])),op('2'))))),'[',']'))),
      8:nm.math(S['MAE_macro'],op('='),fr(op('1'),a('N','J')),sm(a('j',op('=1')),'J',sm(a('i',op('=1')),'N',de(a(yt,op('−'),yhat),'|','|')))),
      9:nm.math(yt,op('='),S['F_j'],de(S['x_i']),op('+'),S['ε_ij']),
      10:nm.math(S['ρ_kj'],op('='),fr(sm(valid,[],a(de(rankx),de(ranky))),nm.sqrt(a(sm(valid,[],sp(de(rankx),op('2'))),sm(valid,[],sp(de(ranky),op('2'))))))),
      11:nm.math(S['βhat_j'],op('='),argmin(beta),de(a(fr(nm.norm(residual,power=2),a(op('2'),S['n_T'])),op('+'),S['λ_1'],nm.norm(beta,order=1),op('+'),S['λ_2'],nm.norm(beta,power=2),op('+'),S['λ_G'],sp(beta,op('T')),'L',beta),'{','}')),
      12:nm.math(S['Δ_cond'],op('='),S['MAE'],de(a(S['y_Vj'],op(','),S['F_j'],de(a(S['X_V-k'],op(','),S['xtilde_k'])))),op('−'),S['MAE'],de(a(S['y_Vj'],op(','),S['F_j'],de(S['X_V'])))),
      13:nm.math(S['H_kj'],op('='),sm(a('ℓ',op('=1')),op('4'),fr(op('1'),a(op('60+'),S['r_ℓkj'])))),
      14:nm.math(S['π_kj'],op('='),fr(op('1'),'B'),sm(a('b',op('=1')),'B',a(r('I',True),de(a('k',op('∈'),S['S_boot']))))),
      15:nm.math(S['OAV_k'],op('='),fr(S['c_k'],S['τ_k']),op(','),S['τ_k'],op('>0')),
      16:nm.math(S['u_ik'],op('='),S['Φinv'],de(fr(a(S['r_ik_s'],op('−0.5')),S['n_s']))),
      17:nm.math(S['Σhat_s'],op('='),de(a(op('1−'),S['α_s'])),S['Σ_s'],op('+'),S['α_s'],fr(a(r('tr',True),de(S['Σ_s'])),'p'),S['I_p']),
      18:nm.math(S['R_s'],op('='),sp(S['D_s'],fr(op('−1'),op('2'))),S['Σhat_s'],sp(S['D_s'],fr(op('−1'),op('2')))),
      19:nm.math(S['x_k*'],op('='),S['Q_sk'],de(a(S['Φ'],de(S['u_k*'])))),
      20:nm.math(S['x(t)'],op('='),de(a(op('1−'),'t')),S['x_0'],op('+'),'t','a',op(','),op('0≤'),'t',op('≤1')),
      21:nm.math(S['cstar_k(t)'],op('='),r('min',True),de(a(S['c_k'],de(S['x(t)']),op(','),S['c_k'],de(S['x_0'])))),
      22:nm.math(S['d_s(x)'],op('='),nm.sqrt(fr(quad,'p'))),
      23:nm.math(S['Δ_RMS'],op('='),nm.sqrt(fr(sm(a('k',op('=1')),'p',sp(de(a(S['v_k(x)'],op('−'),S['v_k(x0)'])),op('2'))),'p'))),
    }
    # Sums with a set lower limit have no upper limit, expressed structurally.
    for math in d.values():
        for n in math.xpath('.//m:nary',namespaces=NS):
            if len(n.find(M+'sup'))==0:n.find(M+'naryPr').append(nm.e('m:supHide',val=1))
    return d


def main():
    OUT.mkdir(exist_ok=True);shutil.copy2(SOURCE,TARGET);files=readzip(TARGET)
    root=E.fromstring(files['word/document.xml']);body=root.find(W+'body');S=nm.symbols()
    S['y_Vj']=nm.sub('y',nm.seq('V',nm.r(',',True),'j'))
    S['D_invhalf']=nm.sup(S['D_s'],nm.frac(nm.r('−1',True),nm.r('2',True)))
    S['v_definition']=nm.seq(S['v_k(x)'],nm.r('=',True),nm.frac(nm.seq(S['x_k'],nm.r('−',True),S['μ_xk']),S['σ_xk']))
    S['n_T_size']=nm.seq(S['n_T'],nm.r('=',True),nm.delim('T','|','|'))
    S['t_range']=nm.seq('t',nm.r('∈',True),nm.delim(nm.seq(nm.r('0',True),nm.r(',',True),nm.r('1',True)),'[',']'))
    S['rho_threshold']=nm.seq(nm.delim('ρ','|','|'),nm.r('=0.60',True))
    S['OAV_ge1']=nm.seq(nm.r('OAV',True),nm.r('≥1',True))
    # Templates come only from the v1 mother copy, not per-question styles.
    prose=cp(next(p for p in body.findall(W+'p') if p.find('w:pPr/w:pStyle',NS) is not None and p.find('w:pPr/w:pStyle',NS).get(W+'val')=='11' and len(text(p))>80))
    caption=cp(next(p for p in body.findall(W+'p') if re.match(r'图1\s',text(p))))
    h1=cp(next(p for p in body.findall(W+'p') if text(p)=='问题重述'))
    h2=cp(next(p for p in body.findall(W+'p') if text(p).startswith('1.1 ')))
    image_template=cp(next(p for p in body.findall(W+'p') if p.find('.//w:drawing',NS) is not None))
    def paragraph(value,template=None):
        p=cp(template if template is not None else prose);pr=pp(p)
        for child in list(p):
            if child is not pr:p.remove(child)
        for tag in ['pageBreakBefore','numPr','sectPr','keepNext']:
            x=pr.find(W+tag)
            if x is not None:pr.remove(x)
        rr=E.SubElement(p,W+'r');rp=E.SubElement(rr,W+'rPr');prop(rp,'b',val=0);prop(rp,'bCs',val=0)
        t=E.SubElement(rr,W+'t');t.text=value;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        return p
    def heading(value,level=2,pagebreak=False):
        p=paragraph(value,h1 if level==1 else h2);pr=pp(p)
        prop(pr,'pStyle',val={'1':'2','2':'3','3':'4'}[str(level)])
        num=prop(pr,'numPr');prop(num,'numId',val=0);prop(pr,'keepNext');prop(pr,'keepLines')
        if pagebreak:prop(pr,'pageBreakBefore')
        p.find('w:r/w:rPr/w:b',NS).set(W+'val','1')
        return p
    # Resolve inline notation conflicts in old prose before converting to OMML.
    replacements={
      'zᵢⱼₘ':'[[z_ijm]]','xᵢ':'[[x_i]]','yᵢ':'[[y_i]]','sⱼ':'[[σ_jT]]',
      'Fⱼ,tr':'[[F_cdf]]','Rᵢₖ':'[[Rik_X]]','Rᵢⱼ':'[[Rij_Y]]',
      'λ₁':'[[λ_1]]','λG':'[[λ_G]]','Hₖⱼ':'[[H_kj]]','πₖⱼ':'[[π_kj]]',
      'cₖ':'[[c_k]]','τₖ':'[[τ_k]]','rᵢₖ':'[[r_ik_s]]','nₛ':'[[n_s]]',
      'uᵢₖ':'[[u_ik]]','Σₛ':'[[Σ_s]]','Dₛ':'[[D_s]]','Rₛ':'[[R_s]]','Qₛₖ':'[[Q_sk]]','x₀':'[[x_0]]',
      'Δcond':'[[Δ_cond]]','ΔR²':'[[DELTA_R2]]','ΔMAE':'[[DELTA_MAE]]',
      '0.01 z':'0.01个标准化感官单位','0.02 z':'0.02个标准化感官单位',
      '0.30 z':'0.30个标准化感官单位','零标准差按1处理':'标准差不超过10⁻¹²时按1处理',
      '0/0.01/0.02z':'0/0.01/0.02个标准化感官单位',
      't∈[0,1]':'[[t_range]]','|ρ|=0.60':'[[rho_threshold]]','OAV≥1':'[[OAV_ge1]]',
      '完整逐目标可靠性另列入附录待组装清单':'逐目标可靠性见附图S3；固定目标残差与相对量表映射见附图S1',
      '也不据此排除全部泄漏风险。':'也不据此排除全部泄漏风险。结构消融与留一风格压力检验见附图S2。',
      '少数风格的R²':'少数风格的[[R2]]',
    }
    S['DELTA_R2']=nm.seq('Δ',S['R2']);S['DELTA_MAE']=nm.seq('Δ',nm.r('MAE',True))
    for p in body.findall('.//'+W+'p'):
        if p.find('.//w:drawing',NS) is not None or p.find('.//m:oMath',NS) is not None:continue
        value=text(p)
        for before,after in replacements.items():value=value.replace(before,after)
        if value!=text(p):
            replacement=paragraph(value,p);p.getparent().replace(p,replacement)
    # Keep original metadata and ordering, standardize only the visible citation
    # separators to the requested author/title/type/journal/year/volume/pages form.
    for p in list(body.findall(W+'p')):
        value=text(p)
        if not re.match(r'^\[(?:[1-9]|10|11)\]\s',value):continue
        normalized=re.sub(r'(?<!\s)\[([JC])\]',r' [\1]',value)
        normalized=re.sub(r'(, \d{4}, \d+)\((\d+)\)',r'\1 (\2)',normalized)
        normalized=re.sub(r'(?<=\d)–(?=\d)', '-',normalized)
        p.getparent().replace(p,paragraph(normalized,p))
    # Definitions already present before formula blocks are shortened rather than
    # repeated. All first mathematical use gets the adjacent definition below.
    for p in list(body.findall(W+'p')):
        value=text(p)
        if value.startswith('全文索引'):
            new='表1汇总全文核心符号，其余符号在相应公式首次出现处说明。训练统计量均仅由对应训练子样本估计。'
        elif value.startswith('堆叠通过训练数据内部'):
            new='堆叠通过训练数据内部的折外基预测估计非负组合[2]。对训练目标和内层基预测分别中心化，再按训练目标标准差归一化，求解式(2)，避免高波动目标独占权重。五个家族预先定义为香气15维、风味15维、其他味觉15维、口感4维和总体喜好1维。'
        elif value.startswith('目标接近程度采用相关结构'):
            new='目标接近程度采用设计坐标中的Mahalanobis距离[11]，见式(22)；候选相对源谱的均方根改动见式(23)。两者均在固定设计参考尺度下计算，不等于酿造成本、工艺可实现性或实测迁移误差。'
        elif value.startswith('第一类证据采用Spearman'):
            new='第一类证据采用成对完整观测的Spearman边际关联，按式(10)计算秩协变；共表达网络用于描述共变与模块[3]。第二类采用弹性网（ElasticNet）与有符号图二次惩罚，式(11)同时包含L1稀疏、L2收缩和图平滑项。图及全部预处理在训练折估计，条件依赖估计参照graphical lasso思想[4]，全量描述图不参与监督验证。'
        else:continue
        p.getparent().replace(p,paragraph(new,p))
    # Replace all 23 display equations while keeping their existing 1x3 tables.
    eqs=equations(S)
    eq_tables=[t for t in body.findall(W+'tbl') if t.xpath('.//m:oMath',namespaces=NS)]
    assert len(eq_tables)==23
    notes=json.loads((AUDIT/'符号审计/逐式首次解释建议.json').read_text())['equations']
    note_strings=[]
    for number,table in enumerate(eq_tables,1):
        cells=table.findall('w:tr/w:tc',NS);assert len(cells)==3
        central=cells[1];p=central.find(W+'p')
        for child in list(p):
            if child is not p.find(W+'pPr'):p.remove(child)
        p.append(eqs[number]);prop(pp(p),'jc',val='center');prop(pp(p),'keepLines')
        # Fixed mother grid; no vertical or visible borders.
        grid=table.find(W+'tblGrid')
        for col,width in zip(grid,[396,7381,953]):col.set(W+'w',str(width))
        for c,width in zip(cells,[396,7381,953]):prop(c.find(W+'tcPr'),'tcW',w=width,type='dxa')
        prop(pp(cells[2].find(W+'p')),'jc',val='right')
        value=notes[number-1]['suggested_after']
        if number==11:
            value='式中，[[X_Tc]]和[[y_jTc]]分别是按训练列均值中心化后的设计矩阵与目标列，[[β]]为待估系数向量，[[βhat_j]]为其解。矩阵有[[n_T]]行、[[q]]列，[[n_T_size]]，[[q]]包括231个测量字段及本折缺失指示，不等于218个单体资格。[[L]]为拓展至这些列的有符号归一化图拉普拉斯，缺失指示不连接化学边；[[λ_1]]、[[λ_2]]、[[λ_G]]依次控制L1、L2及图惩罚，截距在拟合后恢复。'
        elif number==12:
            value='式中，[[Δ_cond]]为固定预测器的条件置换MAE增量，[[X_V]]与[[y_Vj]]为当前验证集的原测量输入及目标列；[[V]]表示外层验证样本集。[[X_V-k]]保留其他输入，[[xtilde_k]]按训练折条件方案替换字段且保留缺失掩码。正增量表示验证误差上升，描述预测依赖而不是干预因果。'
        elif number==18:
            value='式中，[[D_s]]为[[Σhat_s]]对角元素构成的对角矩阵，[[D_invhalf]]逐对角取倒平方根，[[R_s]]为正态Copula的相关矩阵。近零方差作下界保护并保持矩阵对称半正定；该相关矩阵与迁移设计空间的精度矩阵分开估计。'
        elif number==17:
            value='式中，[[Σ_s]]为当前风格潜在坐标的经验协方差，[[Σhat_s]]为其Ledoit–Wolf收缩估计，[[α_s]]为训练数据确定的收缩强度；[[tr]]为矩阵迹，[[I_p]]为[[p]]阶单位矩阵。球形收缩目标的平均方差由协方差矩阵迹除以字段维数得到。'
        elif number==21:
            value='式中，[[c_of_x]]表示按已核验的字段尺度将化学谱还原为直接浓度，[[cstar_k(t)]]为施加源浓度不升约束后的候选浓度。该式仅对源值、单位及尺度已知的直接缺陷字段使用；未知项保留未知，不由填补值构造安全上界。'
        elif number==22:
            value='式中，[[v(x)]]为化学谱在固定设计参考尺度下的坐标，第[[k]]分量按[[v_definition]]计算；[[μ_xk]]及[[σ_xk]]分别为字段的设计参考均值及标准差。[[μ_design]]为目标风格的设计坐标均值，[[Ω_s]]为相同坐标下Ledoit–Wolf协方差的精度矩阵。[[d_s(x)]]是按[[p]]维归一化的Mahalanobis距离，包含平方根与维数分母；该矩阵不等同于Copula潜在相关矩阵[[R_s]]。'
        elif number==8:
            value='式中，[[MAE_macro]]为50个目标绝对误差的等权平均，分母为全部评价单元的数量，误差以附件标准化感官单位计。'
        # Code-style advice strings become explicit native-math markers.
        tokens={
          'x_i':'x_i','y_i':'y_i','epsilon_i':'ε_i','G_g':'G_g','z_ijm':'z_ijm',
          'bar y_j,T':'ȳ_jT','bar z_jm,T':'zbar_jmT','sigma_j,T':'σ_jT',
          'w_g^*':'w_g*','w_all^*':'w_all*','tilde w_g(j),m':'w_gjm','tilde w_g':'wtilde_g',
          'eta':'η','b_j':'b_j','g(j)':'g(j)','hat y_ij':'ŷ_ij','f_mj^(T)':'f_mjT',
          'mathcal F_j,T':'F_cdf','hat h_ij':'ĥ_ij','R_macro^2':'R2_macro','bar y_j,eval':'ȳ_jeval',
          'MAE_macro':'MAE_macro','F_j':'F_j','epsilon_ij':'ε_ij','rho_kj':'ρ_kj',
          'R_ik^X':'Rik_X','R_ij^Y':'Rij_Y','bar R_k^X':'Rbar_kX','bar R_j^Y':'Rbar_jY',
          'mathcal A_kj':'A_kj','n_kj':'n_kj','n_T':'n_T','Delta_kj^cond':'Δ_cond',
          'X_-k':'X_V-k','tilde x_k':'xtilde_k','r_ell,kj':'r_ℓkj','H_kj':'H_kj','ell':'ℓ',
          'mathcal S_b^(j)':'S_boot','pi_kj':'π_kj','OAV_k':'OAV_k','c_k':'c_k','tau_k':'τ_k',
          'n_s':'n_s','r_ik^(s)':'r_ik_s','Phi^(-1)':'Φinv','Phi':'Φ','u_ik':'u_ik',
          'hat Sigma_s':'Σhat_s','Sigma_s':'Σ_s','alpha_s':'α_s','I_p':'I_p','tr':'tr',
          'D_s':'D_s','R_s':'R_s','Q_sk':'Q_sk','u_k^*':'u_k*','x_k^*':'x_k*',
          'x_0':'x_0','x(t)':'x(t)','c_k^*(t)':'cstar_k(t)','c_k(x)':'c_of_x',
          'v(x)':'v(x)','mu_x,k':'μ_xk','sigma_x,k':'σ_xk','mu_s^v':'μ_design',
          'mu_x':'mu_x','sigma_x':'sigma_x','Omega_s':'Ω_s','d_s':'distance_fn',
          'Delta(x,x_0)':'Δ_RMS','v_k(x_0)':'v_k(x0)','v_k(x)':'v_k(x)',
        }
        # Do not transform already marked fragments (notably new definitions11/12).
        if '[[' not in value:
            pattern=re.compile('|'.join(re.escape(t) for t in sorted(tokens,key=len,reverse=True)))
            value=pattern.sub(lambda m:'[['+tokens[m.group()]+']]',value)
        value=value.replace('10^-12','[[EPS12]]').replace('mu_s^{design}','[[μ_design]]')
        note=paragraph(value);table.addnext(note);note_strings.append(value)
    # Add grouped native symbol table, retaining the mother three-line table style.
    table=body.findall(W+'tbl')[0];oldrows=table.findall(W+'tr');header=cp(oldrows[0]);rowtemplate=cp(oldrows[1])
    for row in oldrows:table.remove(row)
    table.append(header)
    # User's latest instruction: halve the summary table and keep only core
    # notation. Full local definitions remain beside the respective equations.
    grouped=[
      (['x_i','y_i'],'化学输入与标准化感官向量','231字段；50目标'),
      (['N','J','p'],'啤酒数、感官目标数、化学字段数','250；50；231'),
      (['T','V'],'当前外层训练集与验证集','互不重叠'),
      (['wtilde_g','b_j','η'],'家族融合权重、目标截距、收缩系数','权重非负；收缩0.5'),
      (['ŷ_ij','ĥ_ij'],'连续预测与相对感官等级','等级0—6，非原评分'),
      (['H_kj','π_kj'],'四路融合分数与自助入选频率','频率0—1'),
      (['c_k','τ_k','OAV_k'],'校准浓度、阈值与气味活性值','前两者同单位'),
      (['R_s','Q_sk'],'Copula相关矩阵与经验分位函数','由训练风格估计'),
      (['x_0','a','t'],'源谱、目标端点与迁移比例','比例0—1'),
      (['d_s(x)','Δ_RMS'],'目标距离与相对源谱的RMS改动','设计坐标，无量纲'),
    ]
    widths=[3800,2900,2030]
    grid=table.find(W+'tblGrid')
    for c,width in zip(grid,widths):c.set(W+'w',str(width))
    for c,width in zip(header.findall(W+'tc'),widths):prop(c.find(W+'tcPr'),'tcW',w=width,type='dxa')
    for symbols,meaning,scope in grouped:
        row=cp(rowtemplate);cells=row.findall(W+'tc')
        for c,val,width in zip(cells,['',meaning,scope],widths):
            pr=c.find(W+'tcPr')
            prop(pr,'tcW',w=width,type='dxa')
            for child in list(c):
                if child is not pr:c.remove(child)
            p=paragraph(val);prop(pp(p),'ind',firstLine=0,firstLineChars=0);prop(pp(p),'jc',val='left');prop(pp(p),'spacing',before=0,after=0,line=210,lineRule='auto')
            for rr in p.findall(W+'r'):
                rp=rr.find(W+'rPr');prop(rp,'sz',val=20);prop(rp,'szCs',val=20)
            c.append(p)
        p=cells[0].find(W+'p')
        for rr in p.findall(W+'r'):p.remove(rr)
        parts=[]
        for idx,name in enumerate(symbols):
            if idx:parts.append(nm.r('；',True,size=20))
            parts.extend(nm.seq(S[name]))
        p.append(nm.math(parts,size=20));table.append(row)
    for idx,row in enumerate(table.findall(W+'tr')):
        trp=row.find(W+'trPr')
        if trp is None:trp=E.Element(W+'trPr');row.insert(0,trp)
        prop(trp,'cantSplit')
        for p in row.findall('.//'+W+'p'):
            prop(pp(p),'keepLines')
            if idx<len(grouped):prop(pp(p),'keepNext')
    # Keep the notation chapter introduction with its complete one-page table.
    table_caption=table.getprevious();intro=table_caption.getprevious()
    if intro is not None and intro.tag==W+'p':prop(pp(intro),'keepNext')
    # References and literal user-supplied disclosure, scoped out of the TOC.
    end_bookmark=next(x for x in body.findall(W+'bookmarkEnd') if x.get(W+'id')=='9000')
    for entry in ['[12] ChatGPT，GPT-5.6 Sol，OpenAI，2026-10-27','[13] DeepSeek，DeepSeek-V4.1-Flash，深度求索（DeepSeek），2026-10-27']:
        p=paragraph(entry);prop(pp(p),'ind',firstLine=0,firstLineChars=0);end_bookmark.addprevious(p)
    title=paragraph('AI 工具使用声明');prop(pp(title),'jc',val='center');prop(pp(title),'ind',firstLine=0,firstLineChars=0);prop(pp(title),'keepNext');title.find('w:r/w:rPr/w:b',NS).set(W+'val','1')
    end_bookmark.addnext(title)
    # Keep the user's candidate verbatim in the audit record, not as a verified
    # final statement: actual records include substantive model/code assistance.
    title.addnext(paragraph('【待按真实使用记录核定声明。用户提供的候选原文及需核实事项见图前工作稿说明；本稿尚不可作为提交稿。】'))
    # Clear the former placeholder; add only appendix A--E, no old code/list.
    appendix=next(p for p in body.findall(W+'p') if text(p)=='附录（下一轮组装）')
    while appendix.getnext() is not None and appendix.getnext().tag!=W+'sectPr':body.remove(appendix.getnext())
    replacement=heading('附录',1,True);body.replace(appendix,replacement);anchor=replacement
    def add(node):
        nonlocal anchor
        anchor.addnext(node);anchor=node
    add(heading('附录A 支撑材料文件列表'));add(paragraph('待支撑材料整理后补充'))
    add(heading('附录B 补充表格与图片'));add(heading('B.1 问题一补充图表',3))
    # Existing approved supplementary figures are copied byte-for-byte.
    src=next((ROOT/'05_论文').glob('Q1*/待审_20261027/第一问_论文正文_v6_待审.docx'));ff=readzip(src);rr=E.fromstring(ff['word/document.xml'])
    relsource={r.get('Id'):r.get('Target') for r in E.fromstring(ff['word/_rels/document.xml.rels'])};rels=E.fromstring(files['word/_rels/document.xml.rels'])
    captions=['附图S1 固定目标残差与相对量表映射','附图S2 结构消融、标签置换与留一风格检验','附图S3 逐目标折外预测可靠性']
    readings=[
      '本图补充正文相对量表映射与误差形态的解释。左侧检查固定代表目标的残差，右侧展示训练经验分布将连续预测转为0—6相对强度；它不恢复原始品评分，也不提供单个新样本预测区间。',
      '本图补充正文结构消融和压力检验。MAE面板采用点图；目标置换为全局一次啤酒目标行置换后固定交叉验证，留一风格仅以Spearman和MAE描述。重训未保存逐样本预测，因此不能据汇总值分解重新训练不确定性。',
      '本图补充正文50个目标的相对可靠性，中文名称逐项对应真实目标代码。两栏按固定顺序展示解释度和排序结果，三层是相对分组而非固定可用性标准；小风格与弱目标不据此获得独立泛化保证。',
    ]
    supplements=[]
    for p in rr.findall('w:body/w:p',NS):
        if p.find('.//w:drawing',NS) is None:continue
        caption_value=text(p.getnext())
        if not caption_value.startswith('附图S'):continue
        rid=p.find('.//a:blip',NS).get('{'+NS['r']+'}embed');data=ff['word/'+relsource[rid]];supplements.append(data)
    assert len(supplements)==3
    for idx,data in enumerate(supplements,1):
        add(paragraph(readings[idx-1]));pic=cp(image_template);blip=pic.find('.//a:blip',NS);rid='rIdSupplement'+str(idx);blip.set('{'+NS['r']+'}embed',rid)
        name=f'supplement_s{idx}.png';files['word/media/'+name]=data
        E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',Id=rid,Type=NS['r']+'/image',Target='media/'+name)
        w,h=Image.open(BytesIO(data)).size;cx=round(15.25/2.54*914400);cy=round(cx*h/w)
        for ex in pic.xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=NS):ex.set('cx',str(cx));ex.set('cy',str(cy))
        for dp in pic.findall('.//wp:docPr',NS):dp.set('id',str(100+idx));dp.set('name',captions[idx-1]);dp.set('descr',captions[idx-1])
        prop(pp(pic),'keepNext');add(pic);add(paragraph(captions[idx-1],caption))
    add(heading('B.2 问题二补充图表',3));add(paragraph('本问补充检验已并入正文表13、图10。表13区分两套稳定性集合、影子字段和字段遮蔽，图10比较噪声占槽与直接锚点恢复；影子字段非零、CO₂的自助名次为第4，遮蔽对象不是正式前三项，因此不能作为零误选或充分性的证据。'))
    add(heading('B.3 问题三补充图表',3));add(paragraph('联合一致性、实际结构消融和保持边缘的置换诊断已并入正文表19、表20及图14。比较时须固定尺度、样本数和重复分母；边缘W1由构造保持不变，联合距离仅有有限且不均匀的支持。小风格相关结构弱，两条迁移候选同源且对裕量敏感，不将生成完整性解释为真实酿造验证。'))
    for label in ['附录C 问题一模型核心程序','附录D 问题二模型核心程序','附录E 问题三模型核心程序']:add(heading(label))
    files['word/_rels/document.xml.rels']=E.tostring(rels,encoding='UTF-8',xml_declaration=True,standalone=True)
    # Convert every inline notation use and Unicode sub/superscript into native
    # mathematics. Bibliography and the exact disclosure are deliberately literal.
    S['c_of_x']=nm.seq(S['c_k'],nm.delim('x'));S['mu_x']=nm.sub('μ','x');S['sigma_x']=nm.sub('σ','x');S['distance_fn']=nm.sub('d','s')
    S['EPS6']=nm.sup(nm.r('10',True),nm.r('−6',True));S['EPS12']=nm.sup(nm.r('10',True),nm.r('−12',True));S['EPS15']=nm.sup(nm.r('10',True),nm.r('−15',True));S['EPS16']=nm.sup(nm.r('10',True),nm.r('−16',True))
    aliases={'R²':'R2','η':'η','ρ':'ρ','β':'β','λ₁':'λ_1','λ₂':'λ_2','λG':'λ_G',
             '10⁻⁶':'EPS6','10⁻¹²':'EPS12','10⁻¹⁵':'EPS15','10⁻¹⁶':'EPS16','√p':'SQRT_P','CO₂':'CO2'}
    S['SQRT_P']=nm.sqrt('p');S['CO2']=nm.sub(nm.r('CO',True),nm.r('2',True))
    # Standalone code tokens in the expert-authored explanation text.
    standalone={'i':'i','j':'j','k':'k','m':'m','s':'s','g':'g','G':'G_g','T':'T','N':'N','J':'J','p':'p','q':'q','B':'B','b':'b','I':'I_event','F':'F','a':'a','t':'t','z':'z_ijm'}
    literal_pattern=re.compile('|'.join(re.escape(k) for k in sorted(aliases,key=len,reverse=True)))
    single_pattern=re.compile(r'(?<![A-Za-z0-9_\[/])('+ '|'.join(standalone)+r')(?![A-Za-z0-9_\]/])')
    inline_count=0
    for p in root.findall('.//'+W+'p'):
        if p.find('.//m:oMath',NS) is not None or p.find('.//w:drawing',NS) is not None:continue
        value=text(p)
        if not value or value==AI or re.match(r'^\[\d+\]',value):continue
        # Preserve cached TOC field paragraphs; they are re-generated by Word.
        if p.find('.//w:fldChar',NS) is not None or p.find('.//w:hyperlink',NS) is not None:continue
        parts=re.split(r'(\[\[[^\]]+\]\])',value);chunks=[]
        for part in parts:
            if part.startswith('[['):chunks.append(('math',part[2:-2]));continue
            # Avoid altering literal math markers inserted by the first pass.
            prepared=literal_pattern.sub(lambda m:'[['+aliases[m.group()]+']]',part)
            split=re.split(r'(\[\[[^\]]+\]\])',prepared)
            for subpart in split:
                if subpart.startswith('[['):chunks.append(('math',subpart[2:-2]));continue
                pos=0
                matches=single_pattern.finditer(subpart) if (value.startswith('式中，') or value.startswith('全文索引')) else []
                for match in matches:
                    if match.start()>pos:chunks.append(('text',subpart[pos:match.start()]))
                    chunks.append(('math',standalone[match[1]]));pos=match.end()
                if pos<len(subpart):chunks.append(('text',subpart[pos:]))
        if not any(k=='math' for k,_ in chunks):continue
        pr=pp(p);rp=p.find('w:r/w:rPr',NS);rp=cp(rp) if rp is not None else None
        size=20 if p.getparent().tag==W+'tc' else 24
        for child in list(p):
            if child is not pr:p.remove(child)
        for kind,val in chunks:
            if kind=='math':
                assert val in S,(val,value);p.append(nm.math(S[val],size=size));inline_count+=1
            elif val:
                rr=E.SubElement(p,W+'r')
                if rp is not None:rr.append(cp(rp))
                tt=E.SubElement(rr,W+'t');tt.text=val;tt.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    # Validate XML ordering and remove stale physical-page metadata.
    orders={'pPr':'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange','rPr':'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath rPrChange'}
    for tag,order in orders.items():
        rank={W+x:i for i,x in enumerate(order.split())}
        for pr in root.findall('.//'+W+tag):pr[:]=sorted(pr,key=lambda x:rank.get(x.tag,999))
    files['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
        for n,data in files.items():z.writestr(n,data)
    allmath=root.xpath('//m:oMath',namespaces=NS);empty=root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]',namespaces=NS)
    record={'status':'pre_figure_working_draft_not_final_v2','source_sha256':sha256(SOURCE.read_bytes()).hexdigest(),
            'working_docx':str(TARGET),'display_formulas':23,'added_explanation_paragraphs':23,
            'symbol_concepts_audited':81,'notation_rows_replaced':15,'notation_rows_v2':len(grouped),
            'notation_column_omml':len(grouped),'inline_math_added':inline_count,'total_omml':len(allmath),'empty_math_runs':len(empty),
            'supplement_images':3,'data_overviews_inserted':0,'ai_statement_literal':AI,'ai_entries_user_supplied_version_not_independently_verified':True,
            'disclosure_pending':'使用范围与实际记录须核对；不能把本工作稿简述当作已充分披露或独立核验事实。',
            'formula_text_corrections':['Spearman使用成对完整观测集合','ElasticNet补实际已有L2并统一系数向量','设计距离改为sqrt(设计精度二次型/p)','RMS显式使用设计坐标'],
            'new_symbol_ids_covered':sorted(set(s for item in notes for s in item['new_symbol_ids'])|{'V_validation'})}
    (AUDIT/'working_docx_checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    assert not empty
    print(json.dumps({k:record[k] for k in ['display_formulas','added_explanation_paragraphs','notation_column_omml','inline_math_added','total_omml','empty_math_runs']},ensure_ascii=False));print(TARGET)


if __name__=='__main__':main()
