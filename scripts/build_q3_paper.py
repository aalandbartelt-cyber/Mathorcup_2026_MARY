from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import json
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/reference/agent3/分小问题文档/Q3.docx"
if not TEMPLATE.exists():
    TEMPLATE = ROOT.parents[1] / "docs/reference/agent3/分小问题文档/Q3.docx"
QDIR = ROOT / "05_论文/Q3_风格生成"
FIGDIR = ROOT / "04_图表/Q3_风格生成/draft_v2"
OUT_MAIN = QDIR / "第三问_论文正文_v1.docx"
OUT_ASSUMP = QDIR / "模型假设_Q3_v1.docx"
OUT_ANALYSIS = QDIR / "问题分析_Q3_v1.docx"

NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "{" + NS_M + "}"
W = "{" + NS_W + "}"


def set_cell_text(cell, text, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "宋体"
    r.font.size = Pt(9.5)
    return p


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_cell_width(cell, width):
    cell.width = Inches(width)
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = tcPr.find(qn("w:tcW"))
    if tcW is None:
        tcW = OxmlElement("w:tcW")
        tcPr.append(tcW)
    tcW.set(qn("w:w"), str(int(width * 1440)))
    tcW.set(qn("w:type"), "dxa")


def no_table_borders(table):
    # The template already supplies a valid table property order; leave the
    # border element absent when a borderless formula table is requested.
    return


def set_table_layout(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            set_cell_width(cell, widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table_title(doc, title):
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.name = "宋体"
    r.font.size = Pt(10)
    return p


def add_table(doc, title, headers, rows, widths=None):
    add_table_title(doc, title)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    try:
        table.style = "Table Grid"
    except KeyError:
        no_table_borders(table)
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_text(hdr[i], h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(hdr[i], "D9E7F2")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], val, align=WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT)
    if widths:
        set_table_layout(table, widths)
    return table


def mr(text, italic=False, bold=False):
    r = OxmlElement("m:r")
    rp = OxmlElement("m:rPr")
    sty = OxmlElement("m:sty")
    sty.set(M + "val", "i" if italic else "p")
    rp.append(sty)
    r.append(rp)
    t = OxmlElement("m:t")
    t.text = str(text)
    r.append(t)
    return r


def msub(base, sub):
    x = OxmlElement("m:sSub")
    e = OxmlElement("m:e"); e.append(base); x.append(e)
    s = OxmlElement("m:sub"); s.append(sub); x.append(s)
    return x


def msup(base, sup):
    x = OxmlElement("m:sSup")
    e = OxmlElement("m:e"); e.append(base); x.append(e)
    s = OxmlElement("m:sup"); s.append(sup); x.append(s)
    return x


def msubsup(base, sub, sup):
    x = OxmlElement("m:sSubSup")
    e = OxmlElement("m:e"); e.append(base); x.append(e)
    s = OxmlElement("m:sub"); s.append(sub); x.append(s)
    u = OxmlElement("m:sup"); u.append(sup); x.append(u)
    return x


def mfrac(num, den):
    x = OxmlElement("m:f")
    fp = OxmlElement("m:fPr"); typ = OxmlElement("m:type"); typ.set(M + "val", "bar"); fp.append(typ); x.append(fp)
    n = OxmlElement("m:num"); n.append(num); x.append(n)
    d = OxmlElement("m:den"); d.append(den); x.append(d)
    return x


def mradicand(body):
    x = OxmlElement("m:rad")
    rp = OxmlElement("m:radPr")
    deg_hide = OxmlElement("m:degHide"); deg_hide.set(M + "val", "1"); rp.append(deg_hide)
    x.append(rp)
    deg = OxmlElement("m:deg"); x.append(deg)
    e = OxmlElement("m:e")
    for part in body:
        if isinstance(part, str):
            e.append(mr(part))
        else:
            e.append(part)
    x.append(e)
    return x


def omath(parts):
    o = OxmlElement("m:oMath")
    for part in parts:
        if isinstance(part, str):
            o.append(mr(part))
        else:
            o.append(part)
    return o


def add_formula(doc, parts, number):
    tbl = doc.add_table(rows=1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    no_table_borders(tbl)
    widths = [0.35, 5.9, 0.55]
    for i, c in enumerate(tbl.rows[0].cells):
        set_cell_width(c, widths[i]); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = tbl.cell(0, 1).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(omath(parts))
    pn = tbl.cell(0, 2).paragraphs[0]
    pn.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = pn.add_run(f"({number})"); rr.font.name = "Times New Roman"; rr.font.size = Pt(10)
    return tbl


def clear_body(doc):
    body = doc._body._element
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def add_para(doc, text, style="Body Text", bold_prefix=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        p.add_run(bold_prefix).bold = True
        p.add_run(text[len(bold_prefix):])
    else:
        p.add_run(text)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.bold = True; r.font.name = "宋体"; r.font.size = Pt(9.5)
    return p


def add_figure(doc, path, caption, width=6.55):
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)
    return p


def build_main():
    doc = Document(str(TEMPLATE)); clear_body(doc)
    doc.add_paragraph("问题三模型建立与求解", style="Heading 1")
    doc.add_paragraph("问题三分析", style="Heading 2")
    add_para(doc, "问题三要求以啤酒风格为条件生成合理的化合物谱，给出生成样本的感官画像，并将源啤酒迁移到目标风格，同时控制缺陷化合物不恶化。该问同时涉及条件联合分布估计、感官映射和受约束搜索三个环节：生成模型负责保持风格内的边缘分布与相关结构，第一问的感官模型负责将生成谱映射为相对感官画像，迁移模型则在目标接近、谱改动和缺陷限制之间寻找可核验候选。")
    add_para(doc, "数据包含250款啤酒、231个化学及理化字段和22个风格标签；Tripel、Kriek、Lambic分别只有28、10、11个样本。高维字段数接近样本量，直接按风格估计完整协方差容易不稳定；部分字段还存在检测地板、缺失或参考值冲突。因此本文保留字段原尺度和观测状态，采用风格内经验边缘与收缩相关结构生成联合谱，并把未知源浓度与未知阈值从缺陷通过分母中分离。")

    doc.add_paragraph("问题三模型", style="Heading 2")
    doc.add_paragraph("数据口径与预处理", style="Heading 3")
    add_para(doc, "以beer作为唯一连接键，beer_id不参与建模。保留250个样本和原始缺失；对正值成对比对识别37个log10字段和33个原尺度字段，其余字段保持附件的存储尺度。进一步记录80个正值不一致单元和3个负参考值，将缺失、校准零、参考冲突与可比观测分开保存。生成与迁移均在训练折内完成分位、尺度、相关阵和边界计算，避免把全数据描述量带入外层评价。")
    add_table(doc, "表1  问题三数据口径与处理规则", ["对象", "处理规则", "保留的边界"], [
        ("主键", "按beer对齐化学谱、风格和感官数据", "beer_id不连接、不建模"),
        ("化学字段", "按字段级核验结果保留log10或原尺度", "不对所有字段统一指数化"),
        ("缺失/地板", "保留缺失掩码和地板状态", "未知浓度不作为安全上限"),
        ("感官画像", "使用第一问最终感官组收缩NNLS", "0–6仅作训练折经验映射"),
        ("迁移约束", "已知直接浓度逐项比较，未知项返回未知", "未知不计入通过分母"),
    ], widths=[1.1, 3.0, 2.7])

    doc.add_paragraph("风格条件收缩Copula", style="Heading 3")
    add_para(doc, "设风格为s，第k个字段的风格内经验分位函数为Qsk。对风格内观测按字段取平均秩，并把秩映射到标准正态空间：")
    add_formula(doc, [msubsup(mr("z", True), mr("ik", True), mr("", False)) if False else msub(mr("z", True), mr("ik", True)), " = ", msup(mr("Φ"), mr("−1")), "(", mfrac(msub(mr("r", True), mr("ik", True)), msub(mr("n", True), mr("s", True))), ")"], 1)
    add_para(doc, "在潜在空间中，对风格内协方差向总体对角结构收缩；Ds为协方差的对角阵，p为字段数。对称化和非负特征值修正只用于保证相关阵可采样。")
    add_formula(doc, [mr("Σ", True), "~", msub(mr("s", True), mr("s", True)), " = (1−", msub(mr("α", True), mr("s", True)), ")", mr("Σ", True), "^", " + ", msub(mr("α", True), mr("s", True)), "tr(", mr("Σ", True), ")/p I"], 2)
    add_formula(doc, [msub(mr("R", True), mr("s", True)), " = ", msup(mr("D", True), mr("−1/2")), msub(mr("Σ", True), mr("s", True)), msup(mr("D", True), mr("−1/2"))], 3)
    add_para(doc, "从标准正态相关分布采样后，按训练风格的经验边缘分位函数反变换，保留每个字段的非正态边缘并共享稳定的依赖结构。")
    add_formula(doc, [msup(mr("x", True), mr("*")), " = ", msub(mr("Q", True), mr("sk", True)), "(", mr("Φ"), "(", msub(msup(mr("z", True), mr("*")), mr("k", True)), "))"], 4)
    add_para(doc, "该模型不把三个目标风格的均值简单平移到同一空间，也不将生成尾部解释为实测浓度。评价同时使用边缘W1、联合energy、留出覆盖率、对数离散误差、秩相关误差和近训练复制率。")

    doc.add_paragraph("感官画像与相对等级", style="Heading 3")
    add_para(doc, "每个生成谱通过第一问最终感官组收缩NNLS得到50维标准化感官预测。画像汇总每种风格300份生成谱的均值和样本间第10–90百分位范围；该区间描述生成样本散布，不是均值置信区间。为便于读者比较，逐目标预测再按对应训练折的经验分布做单调0–6映射，完整50维结果保留在机器可读表中，正文只展示预先指定的六个维度。")
    add_table(doc, "表2  三种风格生成样本的六维相对画像均值", ["维度", "Tripel", "Kriek", "Lambic"], [
        ("酒精感", "4.4", "0.9", "1.1"),
        ("苦味", "3.9", "0.6", "1.1"),
        ("酒花味", "3.7", "0.8", "1.6"),
        ("4VG", "4.6", "0.8", "1.3"),
        ("酸度", "3.0", "5.6", "5.6"),
        ("乳酸感", "2.9", "5.6", "5.7"),
    ], widths=[2.2, 1.5, 1.5, 1.5])

    doc.add_paragraph("受约束迁移", style="Heading 3")
    add_para(doc, "设源谱为x0，目标风格候选端点为a，沿线性路径搜索：")
    add_formula(doc, [mr("x", True), "(", mr("λ", True), ") = (1−", mr("λ", True), ")", mr("x", True), "₀ + ", mr("λ", True), mr("a", True), ", 0≤", mr("λ", True), "≤1"], 5)
    add_para(doc, "对DMS、双乙酰、4VG和trans-2-nonenal，仅在源浓度可核验时施加不升约束；投影只下调超过上限的候选值，不用插补值替代未知源浓度。")
    add_formula(doc, [msub(mr("c", True), mr("k", True)), "* = min{", msub(mr("c", True), mr("k", True)), "(", mr("x", True), "(", mr("λ", True), ")), ", msub(mr("c", True), mr("k", True)), "(", mr("x", True), "₀)}"], 6)
    add_para(doc, "目标接近用目标风格相关结构下的Mahalanobis距离衡量，谱改动用全字段标准差归一化后的RMS衡量；两者是化学设计空间中的距离，不等同于酿造成本或成对迁移真值。感官搜索要求Stale hops的预测均值和重复90%分位均不升，并要求至少1%的目标接近。")
    add_formula(doc, [mr("d", True), "(", mr("x", True), ") = ", mradicand([mr("(", False), mr("x", True), "−", mr("μ", True), ")ᵀ", mr("R", True), "⁻¹(", mr("x", True), "−", mr("μ", True), ")"])], 7)
    add_formula(doc, [mr("Δ", True), "(", mr("x", True), ",", mr("x", True), "₀) = ", mradicand([mfrac(mr("1", False), mr("p", True)), "Σ", "(", mr("x", True), "−", mr("x", True), "₀", ")²"])], 8)

    doc.add_paragraph("问题三求解", style="Heading 2")
    doc.add_paragraph("求解流程", style="Heading 3")
    for text in [
        "第一步，按外层训练折完成字段口径、缺失掩码、经验边缘、协方差收缩和目标风格采样；验证折只用于分布指标计算。",
        "第二步，在相同的十次重复、三折外层协议下比较经验生成、收缩高斯、局部混合和共享低秩候选，联合报告边缘距离、联合距离、覆盖率与离散误差。",
        "第三步，用全数据拟合的最终生成器为Tripel、Kriek和Lambic各产生300份谱，经过感官预测器得到50维画像。画像用于描述风格差异，不回写生成模型。",
        "第四步，在11个源啤酒与3个目标风格构成的33组任务中，从固定端点池和路径网格中筛选最小改动候选；先锁定候选，再进行一次ElasticNet模型族复核。",
        "第五步，对未知源浓度、检测地板、绝对阈值缺失和感官代理不稳定分别标记，不把待核验记录计为缺陷约束通过。",
    ]:
        add_para(doc, text)

    doc.add_paragraph("生成质量与模型选择", style="Heading 3")
    add_para(doc, "收缩Copula在稳健尺度下的W1为0.4762、联合energy为0.5813、留出覆盖率为0.5580、对数离散误差为0.2475。整行局部混合的覆盖率提高到0.6599，但离散误差增至0.2925；PCA邻域混合和基函数邻域混合也存在覆盖与离散度的交换。基函数方案在Lambic上的稳健W1和energy分别增加0.0079和0.0115，未形成跨风格一致改进。因此主生成器保留为风格条件收缩Copula。")
    add_table(doc, "表3  生成方法的稳健尺度比较", ["方法", "W1↓", "energy↓", "覆盖率↑", "离散误差↓"], [
        ("风格收缩Copula", "0.4762", "0.5813", "0.5580", "0.2475"),
        ("整行局部混合", "0.4740", "0.5772", "0.6599", "0.2925"),
        ("PCA邻域混合", "0.4768", "0.5805", "0.6660", "0.3026"),
        ("基函数邻域混合", "0.4786", "0.5834", "0.6580", "0.3014"),
    ], widths=[2.3, 1.1, 1.2, 1.2, 1.2])
    add_figure(doc, FIGDIR / "图1_生成质量与多样性.png", "图1  风格条件生成方法在外层重复验证中的分布保真与多样性")

    doc.add_paragraph("三风格感官画像", style="Heading 3")
    add_para(doc, "画像显示出清晰的风格差异：Tripel在酒精感、苦味、酒花味和4VG上较高，Kriek与Lambic在酸度和乳酸感上较高。每种风格的300份谱均由同一全数据生成器产生，图中误差棒表示合成样本第10–90百分位。4VG的第一问外折解释度约为0.255，故该维度只作较弱的描述证据；画像不能替代真实品评。")
    add_figure(doc, FIGDIR / "图2_三风格感官画像.png", "图2  三目标风格生成样本的相对感官画像")

    doc.add_paragraph("跨风格迁移与缺陷约束", style="Heading 3")
    add_para(doc, "33组源—目标任务中，固定最小改动规则得到29条有谱候选：2条进入待实测统计候选，26条因直接化学证据不足，1条因感官代理不稳定，另4组未找到搜索可行谱。两条待实测统计候选均来自源啤酒230，分别指向Tripel和Kriek；目标距离分别下降38.36%和37.36%，谱RMS改动分别为0.4862和0.5822。两条候选均未取得物理安全认证。")
    add_table(doc, "表4  两条待实测统计候选的迁移结果", ["目标风格", "目标距离下降", "谱RMS改动", "代理模型状态", "化学状态"], [
        ("Tripel", "38.36%", "0.4862", "两模型预测不升", "四项中仍有待核验状态"),
        ("Kriek", "37.36%", "0.5822", "两模型预测不升", "四项中仍有待核验状态"),
    ], widths=[1.35, 1.45, 1.25, 1.7, 1.7])
    add_figure(doc, FIGDIR / "图3_迁移收益与审计状态.png", "图3  跨风格迁移候选的目标接近程度与证据状态")

    doc.add_paragraph("模型检验与结果讨论", style="Heading 2")
    doc.add_paragraph("稳健性与边界", style="Heading 3")
    add_para(doc, "生成模型的比较采用同一外层重复和折内计算，近训练复制率为0，完整谱重构误差处于10⁻¹²量级。局部混合提高覆盖率的同时增加离散误差，说明覆盖不能单独作为生成质量结论。地板感知边缘、共享低秩、神经残差和局部双锚点迁移均未提供足以替换收缩Copula的综合证据。")
    add_para(doc, "迁移结果的“预测不升”只针对冻结的感官预测器；Stale hops的外层解释度约0.03、平均绝对误差约0.30 z，预测变化不能直接等同于真实感官改善。已有数据还缺少完整的配方、发酵工艺和新实测，因此两条候选应作为待补测的统计设计，不应写成可直接执行的酿造处方或物理安全结论。")
    add_para(doc, "本问的模型适合在化学谱空间中生成风格条件样本、比较风格差异和提出约束迁移候选；若要给出生产层面的结论，需要补充直接浓度、检测限、工艺状态和感官评价，并重新进行端到端验证。")
    return doc


def build_short(kind):
    doc = Document(str(TEMPLATE)); clear_body(doc)
    if kind == "assump":
        doc.add_paragraph("问题三模型假设", style="Heading 1")
        add_para(doc, "为简化问题，本文做出以下假设：")
        items = [
            ("假设1", "经字段级核验后，同一化学测量在样本之间具有可比尺度。数据依据是附件表格可按beer逐项对照；若字段单位或映射不一致，则停止该字段的绝对阈值计算，保留已知存储尺度。"),
            ("假设2", "风格内主要依赖可由经验边缘与正态Copula近似。数据依据是每种风格样本少于字段数；若留出联合距离或覆盖不足，则保留经验基线并降低该风格生成结论。"),
            ("假设3", "小样本风格的潜在协方差需要向对角结构收缩。数据依据是风格样本量为10–28而字段数为231；若收缩后相关结构检验变差，则报告比较结果，不用验证集反调参数。"),
            ("假设4", "第一问感官预测器可在训练支持域附近提供带误差的画像。数据依据是逐目标外层OOF；若画像落在支持域外或模型族方向不稳，则将其标记为弱证据或拒绝。"),
            ("假设5", "满足已知直接浓度约束的化学谱可作为统计迁移候选。题目未提供完整工艺状态转移；若缺失浓度、检测地板或绝对阈值不可识别，则返回未知，不能认证真实安全。"),
        ]
        for title, text in items:
            add_para(doc, title + "：" + text, bold_prefix=title)
    else:
        doc.add_paragraph("问题三分析", style="Heading 1")
        doc.add_paragraph("任务本质", style="Heading 2")
        add_para(doc, "本问需要同时完成风格条件化合物谱生成、感官画像和跨风格迁移。生成模型要保持边缘分布与化合物间依赖，画像要沿用第一问的感官预测器，迁移则要在目标风格接近与缺陷不恶化之间取得可核验的折中。")
        doc.add_paragraph("主要难点", style="Heading 2")
        add_para(doc, "第一，化学字段维数为231，而Tripel、Kriek和Lambic样本数分别只有28、10、11，直接估计每种风格的完整协方差容易产生不稳定相关结构。第二，缺失、检测地板和参考冲突使部分缺陷浓度不可识别，不能把插补值或存储零当成安全上限。第三，感官画像和迁移收益依赖预测器，若候选离开训练支持域，均值下降可能掩盖模型不确定性。")
        doc.add_paragraph("对策", style="Heading 2")
        add_para(doc, "采用风格内经验边缘与收缩相关结构生成联合谱，以同一外层重复协议评价边缘距离、联合距离、覆盖和离散误差；用第一问最终感官模型生成50维画像，并将0–6仅作为训练折经验映射；在固定候选池中沿受约束路径搜索，先检查可核验化学状态，再用感官代理和支持域审计，未知项单列而不计入通过分母。")
    return doc


def main():
    QDIR.mkdir(parents=True, exist_ok=True)
    build_main().save(str(OUT_MAIN))
    build_short("assump").save(str(OUT_ASSUMP))
    build_short("analysis").save(str(OUT_ANALYSIS))
    # Remove a template-only settings child rejected by strict OOXML schema
    # checks; this does not alter styles, page layout, or formula objects.
    from zipfile import ZipFile, ZIP_DEFLATED
    from xml.etree import ElementTree as ET
    for path in (OUT_MAIN, OUT_ASSUMP, OUT_ANALYSIS):
        tmp = path.with_suffix('.tmp.docx')
        with ZipFile(path) as zin, ZipFile(tmp, 'w', ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == 'word/settings.xml':
                    root = ET.fromstring(data)
                    for node in list(root):
                        if node.tag == '{http://schemas.openxmlformats.org/officeDocument/2006/math}mathPr':
                            root.remove(node)
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                zout.writestr(item, data)
        tmp.replace(path)
    print("built", OUT_MAIN, OUT_ASSUMP, OUT_ANALYSIS)


if __name__ == "__main__":
    main()
