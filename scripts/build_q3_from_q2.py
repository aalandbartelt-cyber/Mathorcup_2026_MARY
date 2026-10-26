"""Rebuild the Q3 paper by cloning the accepted Q2 v5 Word structure.

The source document is deliberately the Q2 v5 DOCX itself.  Paragraphs,
tables, equation containers, captions, section settings, and page geometry
are cloned from that document; only Q3 text, equations, tables, and figures
are replaced.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil
import tempfile
import importlib.util

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
Q2_TEMPLATE = ROOT.parents[1] / ".worktrees/q2-evidence-opt/05_论文/Q2_关键化合物/第二问_论文正文_v5.docx"
QDIR = ROOT / "05_论文/Q3_风格生成"
FIGDIR = ROOT / "04_图表/Q3_风格生成/draft_v2"
OUT_MAIN = QDIR / "第三问_论文正文_v2.docx"
OUT_ASSUMP = QDIR / "模型假设_Q3_v2.docx"
OUT_ANALYSIS = QDIR / "问题分析_Q3_v2.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS, "m": M_NS, "r": R_NS, "a": A_NS}


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def load_omml():
    path = Path(__file__).with_name("q3_omml.py")
    spec = importlib.util.spec_from_file_location("q3_omml", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OM = load_omml()


def idx(symbol: str, sub: str):
    return OM.subscript([OM.var(symbol)], [OM.var(sub)])


def hat(symbol: str, sub: str | None = None):
    base = OM.accent([OM.var(symbol)])
    return OM.subscript([base], [OM.var(sub)]) if sub else base


def radical(parts):
    node = OM.OxmlElement("m:rad")
    prop = OM.OxmlElement("m:radPr")
    hide = OM.OxmlElement("m:degHide")
    hide.set(OM.qn("m:val"), "1")
    prop.append(hide)
    node.append(prop)
    degree = OM.OxmlElement("m:deg")
    node.append(degree)
    body = OM.OxmlElement("m:e")
    for part in parts:
        body.append(part)
    node.append(body)
    return node


def formula_z():
    return [
        idx("z", "ik"), OM.roman(" = "),
        OM.superscript([OM.var("Φ")], [OM.roman("−1")]),
        OM.delimiter([
            OM.fraction([
                idx("r", "ik"), OM.roman(" − 0.5")
            ], [idx("n", "s")])
        ])
    ]


def formula_shrink_cov():
    sigma_hat = hat("Σ", "s")
    sigma = idx("Σ", "s")
    alpha = idx("α", "s")
    trace = OM.function("tr", [sigma])
    return [
        sigma_hat, OM.roman(" = (1 − "), alpha, OM.roman(")"), sigma,
        OM.roman(" + "), alpha,
        OM.fraction([trace], [OM.var("p")]), OM.roman("I")
    ]


def formula_corr():
    return [
        idx("R", "s"), OM.roman(" = "),
        OM.superscript([idx("D", "s")], [OM.roman("−1/2")]),
        hat("Σ", "s"),
        OM.superscript([idx("D", "s")], [OM.roman("−1/2")]),
    ]


def formula_inverse_margin():
    return [
        OM.subsup([OM.var("x")], [OM.var("k")], [OM.roman("*")]),
        OM.roman(" = "), idx("Q", "sk"), OM.delimiter([
            OM.var("Φ"), OM.delimiter([
                OM.subsup([OM.var("z")], [OM.var("k")], [OM.roman("*")])
            ])
        ])
    ]


def formula_path():
    x0 = OM.subscript([OM.var("x")], [OM.roman("0")])
    return [
        OM.delimiter([OM.var("x"), OM.var("λ")]),
        OM.roman(" = (1 − "), OM.var("λ"), OM.roman(")"),
        x0,
        OM.roman(" + "), OM.var("λ"), OM.var("a"),
        OM.roman(",   0 ≤ "), OM.var("λ"), OM.roman(" ≤ 1"),
    ]


def formula_defect():
    ck_lambda = [idx("c", "k"), OM.delimiter([OM.var("x"), OM.var("λ")])]
    ck_zero = [idx("c", "k"), OM.delimiter([OM.var("x"), OM.roman("0")])]
    return [
        OM.subsup([OM.var("c")], [OM.var("k")], [OM.roman("*")]),
        OM.roman("(λ) = "), OM.function("min", ck_lambda + [OM.roman(", ")] + ck_zero)
    ]


def formula_distance():
    left = OM.delimiter([OM.var("x"), OM.roman(" − "), idx("μ", "s")])
    return [
        idx("d", "s"), OM.delimiter([OM.var("x")]), OM.roman(" = "),
        OM.superscript([left], [OM.roman("T")]),
        OM.superscript([idx("R", "s")], [OM.roman("−1")]),
        OM.delimiter([OM.var("x"), OM.roman(" − "), idx("μ", "s")]),
    ]


def formula_rms():
    sum_term = OM.nary([
        OM.superscript([
            OM.delimiter([idx("x", "k"), OM.roman(" − "), OM.subscript([OM.var("x")], [OM.var("0k")])])
        ], [OM.roman("2")])
    ], [OM.var("k"), OM.roman(" = 1")], [OM.var("p")])
    return [
        OM.var("Δ"), OM.delimiter([
            OM.var("x"), OM.roman(", "), OM.subscript([OM.var("x")], [OM.roman("0")])
        ]), OM.roman(" = "), radical([
            OM.fraction([sum_term], [OM.var("p")])
        ])
    ]


FORMULAS = [
    formula_z,
    formula_shrink_cov,
    formula_corr,
    formula_inverse_margin,
    formula_path,
    formula_defect,
    formula_distance,
    formula_rms,
]


def text_run_from_template(template_p, text: str):
    run = etree.Element(qn(W_NS, "r"))
    template_run = template_p.find("./w:r", NS)
    if template_run is not None:
        rpr = template_run.find("./w:rPr", NS)
        if rpr is not None:
            run.append(deepcopy(rpr))
    t = etree.SubElement(run, qn(W_NS, "t"))
    if text[:1].isspace() or text[-1:].isspace():
        t.set(qn(XML_NS, "space"), "preserve")
    t.text = text
    return run


def para_from(template_p, text: str):
    p = deepcopy(template_p)
    ppr = p.find("./w:pPr", NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    if text:
        p.append(text_run_from_template(template_p, text))
    return p


def clear_cell(tc):
    tcpr = tc.find("./w:tcPr", NS)
    for child in list(tc):
        if child is not tcpr:
            tc.remove(child)


def set_cell_text(tc, text: str):
    template_p = tc.find("./w:p", NS)
    if template_p is None:
        template_p = etree.Element(qn(W_NS, "p"))
    p = para_from(template_p, str(text))
    clear_cell(tc)
    tc.append(p)


def set_cell_width(tc, width: int):
    tcpr = tc.find("./w:tcPr", NS)
    tcw = tcpr.find("./w:tcW", NS)
    if tcw is None:
        tcw = etree.SubElement(tcpr, qn(W_NS, "tcW"))
    tcw.set(qn(W_NS, "w"), str(width))
    tcw.set(qn(W_NS, "type"), "dxa")


def resize_grid(tbl, widths):
    grid = tbl.find("./w:tblGrid", NS)
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = etree.SubElement(grid, qn(W_NS, "gridCol"))
        col.set(qn(W_NS, "w"), str(width))


def clone_data_table(base_tbl, headers, rows, widths):
    tbl = deepcopy(base_tbl)
    resize_grid(tbl, widths)
    for child in list(tbl):
        if child.tag != qn(W_NS, "tblPr") and child.tag != qn(W_NS, "tblGrid"):
            tbl.remove(child)
    header_template = base_tbl.find("./w:tr", NS)
    body_template = base_tbl.find("./w:tr[2]", NS)
    if body_template is None:
        body_template = header_template
    for row_index, values in enumerate([headers] + list(rows)):
        template = header_template if row_index == 0 else body_template
        row = deepcopy(template)
        trpr = row.find("./w:trPr", NS)
        for child in list(row):
            if child is not trpr:
                row.remove(child)
        templates = template.findall("./w:tc", NS)
        for i, value in enumerate(values):
            cell = deepcopy(templates[min(i, len(templates) - 1)])
            set_cell_width(cell, widths[i])
            set_cell_text(cell, value)
            row.append(cell)
        tbl.append(row)
    return tbl


def clone_formula_table(base_tbl, parts, number):
    tbl = deepcopy(base_tbl)
    row = tbl.find("./w:tr", NS)
    cells = row.findall("./w:tc", NS)
    middle = cells[1]
    p = middle.find("./w:p", NS)
    ppr = p.find("./w:pPr", NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    equation = etree.Element(qn(M_NS, "oMath"), nsmap={"m": M_NS})
    for node in parts:
        equation.append(deepcopy(node))
    p.append(equation)
    right = cells[2]
    set_cell_text(right, f"({number})")
    return tbl


def paragraph_map(body):
    paragraphs = body.xpath("./w:p", namespaces=NS)
    def by_style(style):
        for p in paragraphs:
            node = p.find("./w:pPr/w:pStyle", NS)
            if node is not None and node.get(qn(W_NS, "val")) == style:
                return p
        raise RuntimeError(f"missing paragraph style {style}")
    def by_prefix(prefix):
        for p in paragraphs:
            text = "".join(p.itertext()).strip()
            if text.startswith(prefix):
                return p
        raise RuntimeError(f"missing paragraph prefix {prefix}")
    return {
        "h1": by_style("Heading1"),
        "h2": by_style("Heading2"),
        "h3": by_style("Heading3"),
        "body": by_style("BodyText"),
        "table_caption": by_prefix("表1"),
        "figure": next(p for p in paragraphs if p.xpath(".//w:drawing", namespaces=NS)),
        "figure_caption": by_prefix("图1"),
        "note": by_prefix("注："),
    }


def tables_map(body):
    tables = body.xpath("./w:tbl", namespaces=NS)
    def dims(tbl):
        rows = tbl.findall("./w:tr", NS)
        return len(rows), len(rows[0].findall("./w:tc", NS)) if rows else 0
    return {
        "data": next(t for t in tables if dims(t) == (7, 4)),
        "formula": next(t for t in tables if dims(t) == (1, 3)),
        "metrics": next(t for t in tables if dims(t) == (8, 5)),
        "literature": next(t for t in tables if dims(t) == (6, 4)),
    }


def append_text(body, pmap, style, text):
    body.append(para_from(pmap[style], text))


def append_caption(body, pmap, text):
    body.append(para_from(pmap["table_caption"], text))


def append_figure(body, pmap, text, embed_id):
    image = deepcopy(pmap["figure"])
    blip = image.find(".//a:blip", NS)
    blip.set(qn(R_NS, "embed"), embed_id)
    body.append(image)
    body.append(para_from(pmap["figure_caption"], text))


def append_note(body, pmap, text):
    body.append(para_from(pmap["note"], text))


def replace_body(doc_xml: bytes) -> bytes:
    root = etree.fromstring(doc_xml)
    body = root.find("./w:body", NS)
    pmap = paragraph_map(body)
    tmap = tables_map(body)
    sect = body.find("./w:sectPr", NS)
    body.remove(sect)
    for child in list(body):
        body.remove(child)

    append_text(body, pmap, "h1", "问题三模型建立与求解")
    append_text(body, pmap, "h2", "摘要式结论")
    append_text(body, pmap, "body", "本文针对风格条件化合物谱生成、感官画像和跨风格迁移三个目标，使用风格内经验边缘与收缩相关结构构建联合生成模型，并将生成谱输入第一问最终感官组收缩NNLS。每种目标风格生成300份谱，稳健尺度下W1、联合energy、留出覆盖率和对数离散误差分别为0.4762、0.5813、0.5580和0.2475；Tripel、Kriek、Lambic的画像呈现酒精/苦味/酒花/4VG与酸度/乳酸感的清晰差异。33组迁移任务中形成2条待实测统计候选，但未知浓度、检测地板和感官代理不稳定仍被单独标记，候选不等同于实测结果或物理安全认证。")

    append_text(body, pmap, "h2", "问题三分析")
    append_text(body, pmap, "h3", "研究背景与任务界定")
    append_text(body, pmap, "body", "问题三要求以tasting_category_fine为条件生成Tripel、Kriek和Lambic等风格的化合物谱，给出生成样本的感官画像，并把源啤酒迁移到目标风格，同时控制DMS、双乙酰、4VG、trans-2-nonenal和Stale hops等缺陷或代理指标不恶化。问题本质是高维联合分布生成、跨模型感官映射和带约束的化学空间搜索。")
    append_text(body, pmap, "h3", "难点与证据链设计")
    append_text(body, pmap, "body", "本题的难点有三：231个化学及理化字段对应的风格样本数只有28、10、11；检测地板、缺失和参考值冲突使部分浓度不可直接比较；生成谱的感官画像与迁移收益都依赖第一问预测器，离开训练支持域后容易把模型外推误写成真实改善。本文先以折内边缘、相关结构和边界处理建立生成证据，再用留出分布指标、感官画像和化学可比性分层审计。")

    append_text(body, pmap, "h2", "问题三模型")
    append_text(body, pmap, "h3", "数据口径与预处理")
    append_text(body, pmap, "body", "以beer作为唯一连接键，beer_id不参与建模。保留250个样本、231个字段和原始缺失；对正值成对比对识别37个log10字段和33个原尺度字段，其余字段保持附件存储尺度。另行保存缺失掩码、校准零、参考冲突和可比观测状态。生成与迁移中的分位、尺度、相关阵和边界均在训练折内估计，全量描述量只用于最终画像和图表。")
    append_caption(body, pmap, "表1  问题三数据口径与处理规则")
    body.append(clone_data_table(tmap["data"], ["对象", "处理规则", "保留的边界"], [
        ["主键", "按beer对齐化学谱、风格和感官数据", "beer_id不连接、不建模"],
        ["化学字段", "按字段核验结果保留log10或原尺度", "不对所有字段统一指数化"],
        ["缺失/地板", "保留缺失掩码和地板状态", "未知浓度不作为安全上限"],
        ["感官画像", "使用第一问最终感官组收缩NNLS", "0–6仅作训练折经验映射"],
        ["迁移约束", "已知直接浓度逐项比较，未知项返回未知", "未知不计入通过分母"],
    ], [1700, 3700, 3500]))
    append_text(body, pmap, "h3", "风格条件收缩Copula")
    append_text(body, pmap, "body", "设风格为s，第k个字段的风格内经验分位函数为Qsk。对风格内观测按字段取平均秩，并把秩映射到标准正态空间。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[0](), 1))
    append_text(body, pmap, "body", "在潜在空间中，对风格内协方差向总体对角结构收缩；αs由训练折内的稳定性准则确定，Ds为协方差的对角阵，p为字段数。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[1](), 2))
    body.append(clone_formula_table(tmap["formula"], FORMULAS[2](), 3))
    append_text(body, pmap, "body", "从标准正态相关分布采样后，按训练风格的经验边缘分位函数反变换，保留每个字段的非正态边缘并共享稳定的依赖结构。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[3](), 4))
    append_text(body, pmap, "body", "该模型不把三个目标风格的均值简单平移到同一空间，也不将生成尾部解释为实测浓度；评价同时使用边缘W1、联合energy、留出覆盖率、对数离散误差、秩相关误差和近训练复制率。")
    append_text(body, pmap, "h3", "感官画像与相对等级")
    append_text(body, pmap, "body", "每个生成谱通过第一问最终感官组收缩NNLS得到50维标准化感官预测，再按对应训练折的经验分布做单调0–6映射。每种风格汇总300份生成谱的均值和第10–90百分位范围，该区间描述合成样本散布，不是均值置信区间。")
    append_caption(body, pmap, "表2  三种风格生成样本的六维相对画像均值")
    body.append(clone_data_table(tmap["data"], ["维度", "Tripel", "Kriek", "Lambic"], [
        ["酒精感", "4.4", "0.9", "1.1"], ["苦味", "3.9", "0.6", "1.1"],
        ["酒花味", "3.7", "0.8", "1.6"], ["4VG", "4.6", "0.8", "1.3"],
        ["酸度", "3.0", "5.6", "5.6"], ["乳酸感", "2.9", "5.6", "5.7"],
    ], [1700, 2350, 2350, 2500]))
    append_text(body, pmap, "h3", "受约束迁移")
    append_text(body, pmap, "body", "设源谱为x0，目标风格候选端点为a，在化学设计空间中沿线性路径搜索候选谱。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[4](), 5))
    append_text(body, pmap, "body", "对DMS、双乙酰、4VG和trans-2-nonenal，仅在源浓度可核验时施加不升约束；未知源值不填补为安全上限。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[5](), 6))
    append_text(body, pmap, "body", "目标接近用目标风格相关结构下的距离衡量，谱改动用标准差归一化RMS衡量；两者是化学设计空间距离，不等同于酿造成本或实测迁移真值。")
    body.append(clone_formula_table(tmap["formula"], FORMULAS[6](), 7))
    body.append(clone_formula_table(tmap["formula"], FORMULAS[7](), 8))

    append_text(body, pmap, "h2", "问题三求解")
    append_text(body, pmap, "h3", "交叉验证与模型比较")
    append_text(body, pmap, "body", "所有生成方法复用相同的10次重复、每次3折外层划分；边缘、相关阵、缺失处理和风格内参数均只由训练折估计。全数据最终生成仅用于三种目标风格的画像，不回写外层评价。")
    append_caption(body, pmap, "表3  生成方法的稳健尺度比较")
    body.append(clone_data_table(tmap["metrics"], ["方法", "W1↓", "energy↓", "覆盖率↑", "离散误差↓"], [
        ["风格收缩Copula", "0.4762", "0.5813", "0.5580", "0.2475"],
        ["整行局部混合", "0.4740", "0.5772", "0.6599", "0.2925"],
        ["PCA邻域混合", "0.4768", "0.5805", "0.6660", "0.3026"],
        ["基函数邻域混合", "0.4786", "0.5834", "0.6580", "0.3014"],
    ], [2700, 1450, 1600, 1700, 1800]))
    append_text(body, pmap, "body", "收缩Copula在W1、联合energy和对数离散误差之间取得较均衡的结果；局部混合覆盖率更高，但离散误差同步增加。因此正文主线保留风格条件收缩Copula，把局部混合作为覆盖敏感性比较。")
    append_figure(body, pmap, "图1  风格条件生成方法在外层重复验证中的分布保真与多样性", "rId10")
    append_note(body, pmap, "注：箱线为外层重复分布，点为重复结果，菱形为均值；W1、energy和离散误差越低越好，覆盖率越高越好。")
    append_text(body, pmap, "h3", "三风格感官画像")
    append_text(body, pmap, "body", "Tripel在酒精感、苦味、酒花味和4VG上较高，Kriek与Lambic在酸度和乳酸感上较高。图中误差棒表示300份生成样本的第10–90百分位范围；4VG的第一问外折解释度约为0.255，画像用于相对比较，不能替代真实品评。")
    append_figure(body, pmap, "图2  三目标风格生成样本的相对感官画像", "rId11")
    append_note(body, pmap, "注：柱高为生成样本均值，黑色误差棒为第10–90百分位范围；0–6为训练折经验映射后的相对等级。")
    append_text(body, pmap, "h3", "跨风格迁移与缺陷约束")
    append_text(body, pmap, "body", "33组源—目标任务中，固定最小改动规则得到29条有谱候选：2条进入待实测统计候选，26条直接化学证据不足，1条感官代理不稳定，4组未找到搜索可行谱。两条候选均来自源啤酒230，分别指向Tripel和Kriek；目标距离分别下降38.36%和37.36%，谱RMS改动分别为0.4862和0.5822。两条候选均未取得物理安全认证。")
    append_caption(body, pmap, "表4  两条待实测统计候选的迁移结果")
    body.append(clone_data_table(tmap["metrics"], ["目标风格", "目标距离下降", "谱RMS改动", "代理模型状态", "化学状态"], [
        ["Tripel", "38.36%", "0.4862", "两模型预测不升", "仍有待核验状态"],
        ["Kriek", "37.36%", "0.5822", "两模型预测不升", "仍有待核验状态"],
    ], [1900, 1900, 1500, 2600, 2600]))
    append_figure(body, pmap, "图3  跨风格迁移候选的目标接近程度与证据状态", "rId12")
    append_note(body, pmap, "注：目标距离下降越大表示越接近目标风格；颜色和点形同时标记统计候选、直接证据不足、代理不稳定及无可行谱。")

    append_text(body, pmap, "h2", "模型检验与结果讨论")
    append_text(body, pmap, "h3", "泄漏与稳健性检验")
    append_text(body, pmap, "body", "验证折不参与缺失填补、边缘分位、尺度估计、相关阵、风格均值或迁移边界计算；近训练复制率为0，完整谱重构误差处于10⁻¹²量级。未知浓度、检测地板和绝对阈值缺失分别保留未知状态，不能把缺失值计入缺陷约束通过率。")
    append_text(body, pmap, "h3", "模型对比与消融")
    append_text(body, pmap, "body", "经验重采样和收缩高斯分别用于检查边缘与相关结构的必要性；局部混合、共享低秩、地板感知边缘和神经残差作为消融。局部混合提高覆盖率但增加离散误差，共享低秩和神经残差没有形成稳定的联合指标增益，故保留收缩Copula作为正文主线。公开文献中的Copula生成、协方差收缩、Mahalanobis距离和约束搜索只用于说明方法边界，文献数字与本题任务不可直接横比。")
    append_caption(body, pmap, "表5  生成与迁移方法的功能对照及消融结论")
    body.append(clone_data_table(tmap["literature"], ["方法家族", "回答的问题", "本文对应结果", "采用边界"], [
        ["经验边缘与Copula", "如何保持非正态边缘和联合依赖", "收缩Copula作为主生成方法", "目标风格样本稀疏时依赖收缩"],
        ["协方差收缩", "如何稳定估计高维相关结构", "风格内协方差向对角结构收缩", "不把收缩后的相关解释为机理"],
        ["局部混合/低秩结构", "如何扩大覆盖或降维采样", "作为消融，覆盖提升伴随离散误差", "不能只凭覆盖率替换主模型"],
        ["Mahalanobis约束搜索", "如何度量目标接近与限制谱改动", "用于迁移候选排序", "候选须经过直接化学核验"],
        ["强化学习或深度生成", "如何学习长期动作或复杂生成策略", "当前数据缺少动作—状态转移和独立实测", "登记为后续研究，不进入本问主结果"],
    ], [1800, 2350, 2600, 2100]))
    append_text(body, pmap, "h3", "合理性与局限")
    append_text(body, pmap, "body", "生成画像复现了三种目标风格在酒精感、苦味、酒花味、4VG、酸度和乳酸感上的相对差异，说明化学空间到感官空间的映射具有可解释的方向性。迁移候选的“预测不升”只针对冻结感官预测器；Stale hops的外层解释度约0.03、平均绝对误差约0.30 z，不能把预测变化直接等同于真实感官改善。")
    append_text(body, pmap, "body", "本问仍受250个样本、22种不均衡风格、缺失/地板状态和缺少完整配方工艺的限制。两条迁移候选应作为待补测的统计设计，而不是可直接执行的酿造处方；若增加独立批次、原始浓度、检测限和感官评价，应重新进行端到端验证。")
    append_text(body, pmap, "h2", "问题三方法参考文献")
    for ref in [
        "[1] Sklar A. Fonctions de répartition à n dimensions et leurs marges[J]. Publications de l'Institut de Statistique de l'Université de Paris, 1959.",
        "[2] Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices[J]. Journal of Multivariate Analysis, 2004, 88(2): 365–411.",
        "[3] Mahalanobis P C. On the generalized distance in statistics[J]. Proceedings of the National Institute of Sciences of India, 1936, 2: 49–55.",
        "[4] Nelsen R B. An Introduction to Copulas[M]. 2nd ed. New York: Springer, 2006.",
        "[5] Friedman J, Hastie T, Tibshirani R. Sparse inverse covariance estimation with the graphical lasso[J]. Biostatistics, 2008, 9(3): 432–441.",
    ]:
        append_text(body, pmap, "body", ref)
    body.append(sect)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def rewrite_short(template_path: Path, output_path: Path, title: str, paragraphs: list[str]):
    with ZipFile(template_path) as zin:
        files = {name: zin.read(name) for name in zin.namelist()}
    root = etree.fromstring(files["word/document.xml"])
    body = root.find("./w:body", NS)
    simple_paras = body.xpath("./w:p", namespaces=NS)
    h1 = next(p for p in simple_paras if p.find("./w:pPr/w:pStyle", NS) is not None and p.find("./w:pPr/w:pStyle", NS).get(qn(W_NS, "val")) == "Heading1")
    body_template = next(p for p in simple_paras if p.find("./w:pPr/w:pStyle", NS) is not None and p.find("./w:pPr/w:pStyle", NS).get(qn(W_NS, "val")) == "BodyText")
    sect = body.find("./w:sectPr", NS)
    body.remove(sect)
    for child in list(body):
        body.remove(child)
    body.append(para_from(h1, title))
    for text in paragraphs:
        body.append(para_from(body_template, text))
    body.append(sect)
    files["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)


def build_main():
    with ZipFile(Q2_TEMPLATE) as zin:
        files = {name: zin.read(name) for name in zin.namelist()}
    files["word/document.xml"] = replace_body(files["word/document.xml"])
    for index, filename in enumerate([
        "图1_生成质量与多样性.png",
        "图2_三风格感官画像.png",
        "图3_迁移收益与审计状态.png",
    ], start=1):
        files[f"word/media/image{index}.png"] = (FIGDIR / filename).read_bytes()
    with ZipFile(OUT_MAIN, "w", ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)


def main():
    QDIR.mkdir(parents=True, exist_ok=True)
    build_main()
    rewrite_short(
        Q2_TEMPLATE.parent / "模型假设_Q2_v5.docx", OUT_ASSUMP,
        "模型假设（问题三）",
        [
            "为简化问题，本文做出以下假设：",
            "假设1：同一化学字段在250款啤酒之间具有可比的测量含义。依据是附件1字段、附件2相关矩阵和附件8参考字段可以按beer对齐；若单位或批次漂移无法校准，则该字段只保留相对分布，不做绝对阈值判断。",
            "假设2：风格内经验边缘与收缩相关结构足以描述生成任务所需的联合变化。若留出联合距离或覆盖率明显恶化，则回退到经验重采样，并降低该风格画像的解释强度。",
            "假设3：检测地板、缺失值和参考冲突可以作为观测状态单独保留。若获得检测限、原始浓度或批次校准资料，应重建边缘并重新计算化学约束。",
            "假设4：第一问最终感官模型在训练支持域附近可提供带误差的相对画像。若迁移候选离开支持域或代理模型方向不稳定，则只保留为探索性统计候选，不写成真实感官改善。",
            "假设5：已知直接浓度约束下的最小改动路径可作为迁移候选的统计设计。该假设不代表配方、发酵工艺或物理安全；若要形成生产建议，必须增加独立批次、实测浓度和感官验证。",
        ],
    )
    rewrite_short(
        Q2_TEMPLATE.parent / "问题分析_Q2_v5.docx", OUT_ANALYSIS,
        "问题三分析",
        [
            "问题三要求以啤酒风格为条件生成合理的化合物谱，给出生成样本的感官画像，并将源啤酒迁移到目标风格，同时控制缺陷或代理指标不恶化。附件1提供250款啤酒的化学及理化字段，附件4提供标准化感官值，风格标签用于条件化建模。任务本质是高维联合分布生成、感官映射和带约束迁移的连续流程。",
            "本题的难点有三：目标风格样本数较少而字段维数高，完整协方差容易不稳定；检测地板、缺失和参考冲突使部分绝对浓度不可比；迁移候选的感官变化依赖第一问预测器，模型外推不能直接当作真实改善。对此，本文在训练折内拟合经验边缘和收缩相关结构，以同一外层协议评价生成质量，再用感官画像、化学约束和证据状态分层报告迁移结果。",
        ],
    )
    print(OUT_MAIN)
    print(OUT_ASSUMP)
    print(OUT_ANALYSIS)


if __name__ == "__main__":
    main()
