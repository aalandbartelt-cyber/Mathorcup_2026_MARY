#!/usr/bin/env python3
"""按国赛 C 题第一问模板生成 MathorCup A 题第一问论文正文。

正文中的结果数字全部从 03_结果/Q1_感官评分 读取，并在生成时校验样本数、
目标数和重复数。这样可避免人工转录造成图、表和正文口径不一致。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


REPO = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE = Path(
    "/mnt/d/上海大学__/26年数学建模/26数模国赛/"
    "2026年全国大学生数学建模竞赛/C题研究/第一问/"
    "可与论文手交接的文档、图片等/第一问_论文正文_最终版.docx"
)
DEFAULT_OUTPUT = REPO / "05_论文/Q1_感官评分/第一问_论文正文_v1.docx"
RESULT_DIR = REPO / "03_结果/Q1_感官评分"
FIG_DIR = REPO / "04_图表/Q1_感官评分"


def set_cell_margins(cell, top=45, start=70, bottom=45, end=70):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        # tcMar 必须位于 textDirection/fitText/vAlign/hideMark 之前。
        tc_pr.insert_element_before(
            tc_mar, "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark"
        )
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, **edges):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        # tcBorders 在 OOXML 顺序中位于 shd/noWrap/tcMar/vAlign 之前。
        tc_pr.insert_element_before(
            borders,
            "w:shd",
            "w:noWrap",
            "w:tcMar",
            "w:textDirection",
            "w:tcFitText",
            "w:vAlign",
            "w:hideMark",
        )
    for edge, opts in edges.items():
        tag = f"w:{edge}"
        elem = borders.find(qn(tag))
        if elem is None:
            elem = OxmlElement(tag)
            borders.append(elem)
        for k, v in opts.items():
            elem.set(qn(f"w:{k}"), str(v))


def set_run_font(run, size=10.5, bold=None, italic=None, east_asia="宋体", latin="Times New Roman"):
    run.font.name = latin
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:eastAsia"), east_asia)
    r_fonts.set(qn("w:ascii"), latin)
    r_fonts.set(qn("w:hAnsi"), latin)


def omml_run(text: str):
    """Create a native Office Math run."""
    run = OxmlElement("m:r")
    node = OxmlElement("m:t")
    node.text = text
    run.append(node)
    return run


def omml_box(tag: str, parts):
    box = OxmlElement(f"m:{tag}")
    for part in parts:
        box.append(omml_run(part) if isinstance(part, str) else part)
    return box


def omml_sub(base, subscript):
    node = OxmlElement("m:sSub")
    node.append(omml_box("e", base))
    node.append(omml_box("sub", subscript))
    return node


def omml_sup(base, superscript):
    node = OxmlElement("m:sSup")
    node.append(omml_box("e", base))
    node.append(omml_box("sup", superscript))
    return node


def omml_subsup(base, subscript, superscript):
    node = OxmlElement("m:sSubSup")
    node.append(omml_box("e", base))
    node.append(omml_box("sub", subscript))
    node.append(omml_box("sup", superscript))
    return node


def omml_fraction(numerator, denominator):
    node = OxmlElement("m:f")
    node.append(omml_box("num", numerator))
    node.append(omml_box("den", denominator))
    return node


def omml_accent(parts, char="̂"):
    node = OxmlElement("m:acc")
    prop = OxmlElement("m:accPr")
    mark = OxmlElement("m:chr")
    mark.set(qn("m:val"), char)
    prop.append(mark)
    node.append(prop)
    node.append(omml_box("e", parts))
    return node


def omml_bar(parts):
    node = OxmlElement("m:bar")
    prop = OxmlElement("m:barPr")
    pos = OxmlElement("m:pos")
    pos.set(qn("m:val"), "top")
    prop.append(pos)
    node.append(prop)
    node.append(omml_box("e", parts))
    return node


def omml_delimiter(parts, begin="(", end=")"):
    node = OxmlElement("m:d")
    prop = OxmlElement("m:dPr")
    begin_node = OxmlElement("m:begChr")
    begin_node.set(qn("m:val"), begin)
    end_node = OxmlElement("m:endChr")
    end_node.set(qn("m:val"), end)
    prop.extend([begin_node, end_node])
    node.append(prop)
    node.append(omml_box("e", parts))
    return node


def omml_nary(parts, lower, upper=None, char="∑"):
    """Create a native n-ary operator with real lower/upper limits."""
    node = OxmlElement("m:nary")
    prop = OxmlElement("m:naryPr")
    chr_node = OxmlElement("m:chr")
    chr_node.set(qn("m:val"), char)
    lim = OxmlElement("m:limLoc")
    lim.set(qn("m:val"), "undOvr")
    grow = OxmlElement("m:grow")
    grow.set(qn("m:val"), "1")
    prop.extend([chr_node, lim, grow])
    if upper is None:
        hide = OxmlElement("m:supHide")
        hide.set(qn("m:val"), "1")
        prop.append(hide)
    node.append(prop)
    node.append(omml_box("sub", lower))
    if upper is not None:
        node.append(omml_box("sup", upper))
    else:
        node.append(omml_box("sup", []))
    node.append(omml_box("e", parts))
    return node


def formula_model_relation():
    return [
        omml_sub(["y"], ["i"]),
        " = F(",
        omml_sub(["x"], ["i"]),
        ") + ",
        omml_sub(["ε"], ["i"]),
        ",    F: ",
        omml_sup(["ℝ"], ["231"]),
        " → ",
        omml_sup(["ℝ"], ["50"]),
        ",    i = 1,…,250",
    ]


def formula_extra_trees():
    y_hat = omml_sub([omml_accent(["y"])], ["j"])
    tree = omml_sub(["T"], ["bj"])
    total = omml_nary([tree, "(x)"], ["b=1"], ["B"])
    return [y_hat, "(x) = ", omml_fraction(["1"], ["B"]), " ", total]


def formula_grade_mapping():
    g_hat = omml_sub([omml_accent(["g"])], ["ij"])
    f_hat = omml_sub([omml_accent(["F"])], ["j,tr"])
    y_hat = omml_sub([omml_accent(["y"])], ["ij"])
    return [g_hat, " = 6·", f_hat, "(", y_hat, ")", ",    0 ≤ ", copy.deepcopy(g_hat), " ≤ 6"]


def formula_r_squared():
    y = omml_sub(["y"], ["ij"])
    y_hat = omml_sub([omml_accent(["y"])], ["ij"])
    y_bar = omml_sub([omml_bar(["y"])], ["j"])
    residual_sq = omml_sup([omml_delimiter([y, " − ", y_hat])], ["2"])
    total_sq = omml_sup([omml_delimiter([copy.deepcopy(y), " − ", y_bar])], ["2"])
    numerator = omml_nary([residual_sq], ["i=1"], ["n"])
    denominator = omml_nary([total_sq], ["i=1"], ["n"])
    return [omml_subsup(["R"], ["j"], ["2"]), " = 1 − ", omml_fraction([numerator], [denominator])]


def formula_mae():
    y = omml_sub(["y"], ["ij"])
    y_hat = omml_sub([omml_accent(["y"])], ["ij"])
    absolute = omml_delimiter([y, " − ", y_hat], begin="|", end="|")
    total = omml_nary([absolute], ["i=1"], ["n"])
    return [omml_sub(["MAE"], ["j"]), " = ", omml_fraction(["1"], ["n"]), " ", total]


def formula_paired_delta():
    left = omml_sub(["MAE"], ["i"])
    right = omml_sub(["MAE"], ["i"])
    difference = omml_delimiter(
        [left, "(ExtraTrees) − ", right, "(comparator)"], begin="[", end="]"
    )
    total = omml_nary([difference], ["i=1"], ["N"])
    return ["ΔMAE = ", omml_fraction(["1"], ["N"]), " ", total]


def clear_document_body(doc: Document):
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def add_body(doc: Document, text: str, *, bold_lead: str | None = None):
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control = True
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(text[len(bold_lead) :])
    else:
        p.add_run(text)
    return p


def add_caption(doc: Document, text: str, *, above=False):
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(3 if above else 2)
    p.paragraph_format.space_after = Pt(2 if above else 5)
    p.paragraph_format.keep_with_next = above
    run = p.add_run(text)
    set_run_font(run, size=10.5)
    return p


def add_heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    return p


def add_equation(doc: Document, formula_parts, number: int, formula_proto):
    """复制模板的 1×3 无边框公式表，公式在中间，编号在右侧。"""
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # 直接沿用模板公式表的表格属性与列宽。
    new_pr = copy.deepcopy(formula_proto.tblPr)
    table._tbl.replace(table._tbl.tblPr, new_pr)
    old_grid = table._tbl.tblGrid
    table._tbl.replace(old_grid, copy.deepcopy(formula_proto.tblGrid))
    widths = (Cm(1.02), Cm(13.20), Cm(1.02))
    for cell, width in zip(table.rows[0].cells, widths):
        cell.width = width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell, top=30, bottom=30, start=0, end=0)
    for cell in table.rows[0].cells:
        p = cell.paragraphs[0]
        for child in list(p._p):
            if child.tag != qn("w:pPr"):
                p._p.remove(child)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
    center = table.rows[0].cells[1].paragraphs[0]
    center.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # 公式由 m:f、m:sSub、m:sSup、m:nary 等 OMML 结构组成，
    # 是 Word 可编辑的原生专业公式，禁止用图片或单一 m:t 伪装。
    omath = OxmlElement("m:oMath")
    for part in formula_parts:
        omath.append(omml_run(part) if isinstance(part, str) else part)
    center._p.append(omath)
    right = table.rows[0].cells[2].paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = right.add_run(f"({number})")
    set_run_font(run, size=10.5)
    return table


def add_three_line_table(doc: Document, title: str, headers, rows, widths=None, font_size=9.0):
    add_caption(doc, title, above=True)
    table = doc.add_table(rows=1, cols=len(headers), style="Table")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    if widths is None:
        widths = [Cm(15.6 / len(headers))] * len(headers)
    all_rows = [headers] + list(rows)
    # 表格不使用竖线，仅保留顶线、表头线和底线。
    for r_idx, values in enumerate(all_rows):
        row = table.rows[0] if r_idx == 0 else table.add_row()
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        for c_idx, value in enumerate(values):
            cell = row.cells[c_idx]
            cell.width = widths[c_idx]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.style = doc.styles["Body Text"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            run = p.add_run(str(value))
            set_run_font(run, size=font_size, bold=(r_idx == 0))
            nil = {"val": "nil"}
            set_cell_border(cell, left=nil, right=nil, insideH=nil, insideV=nil)
            if r_idx == 0:
                set_cell_border(
                    cell,
                    top={"val": "single", "sz": "10", "color": "000000"},
                    bottom={"val": "single", "sz": "6", "color": "000000"},
                )
            elif r_idx == len(all_rows) - 1:
                set_cell_border(cell, bottom={"val": "single", "sz": "10", "color": "000000"})
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    return table


def add_figure(doc: Document, path: Path, caption: str, width_cm=15.6):
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph(style="Body Text")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption, above=False)


def add_page_break(doc: Document):
    p = doc.add_paragraph()
    p.add_run().add_break()


def normalize_ooxml_order(doc: Document):
    """修正原模板经 python-docx 保存后的少量 OOXML 顺序问题。"""
    # 严格验证器不接受模板尾部的 m:mathPr；删除后使用 Office 默认数学属性。
    settings = doc.settings.element
    math_pr = settings.find(qn("m:mathPr"))
    if math_pr is not None:
        settings.remove(math_pr)

    sect_pr = doc.sections[0]._sectPr
    pg_mar = sect_pr.find(qn("w:pgMar"))
    if pg_mar is not None and pg_mar.get(qn("w:gutter")) is None:
        pg_mar.set(qn("w:gutter"), "0")

    # 模板公式表的 tblBorders 位于 tblLayout 之后，调整为规范顺序。
    for tbl_pr in doc._element.findall(".//" + qn("w:tblPr")):
        borders = tbl_pr.find(qn("w:tblBorders"))
        if borders is not None:
            tbl_pr.remove(borders)
            before = None
            for tag in ("w:shd", "w:tblLayout", "w:tblLook"):
                before = tbl_pr.find(qn(tag))
                if before is not None:
                    break
            if before is None:
                tbl_pr.append(borders)
            else:
                tbl_pr.insert(list(tbl_pr).index(before), borders)

    # 同一边框集内各边也有固定次序。
    edge_order = ("top", "left", "bottom", "right", "insideH", "insideV", "tl2br", "tr2bl")
    for borders_tag in ("w:tcBorders", "w:tblBorders"):
        for borders in doc._element.findall(".//" + qn(borders_tag)):
            by_name = {child.tag: child for child in list(borders)}
            for child in list(borders):
                borders.remove(child)
            for edge in edge_order:
                child = by_name.get(qn(f"w:{edge}"))
                if child is not None:
                    borders.append(child)


def validate_native_omml(path: Path, expected_equations=6):
    """Reject flat-text or image equations after saving the DOCX."""
    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    equations = root.findall(f".//{{{math_ns}}}oMath")
    if len(equations) != expected_equations:
        raise ValueError(f"Expected {expected_equations} native equations, found {len(equations)}")
    structural = {
        f"{{{math_ns}}}f",
        f"{{{math_ns}}}sSub",
        f"{{{math_ns}}}sSup",
        f"{{{math_ns}}}sSubSup",
        f"{{{math_ns}}}nary",
        f"{{{math_ns}}}acc",
        f"{{{math_ns}}}bar",
        f"{{{math_ns}}}d",
    }
    for index, equation in enumerate(equations, start=1):
        tags = {node.tag for node in equation.iter()}
        if not tags.intersection(structural):
            raise ValueError(f"Equation ({index}) is flat m:t text, not structured OMML")


def load_evidence():
    contract = json.loads((RESULT_DIR / "q1_data_contract.json").read_text(encoding="utf-8"))
    models = pd.read_csv(RESULT_DIR / "figure_data/model_comparison.csv")
    targets = pd.read_csv(RESULT_DIR / "q1_metrics_by_target_E003.csv")
    styles = pd.read_csv(RESULT_DIR / "q1_metrics_by_style_E003.csv")
    pairwise = pd.read_csv(RESULT_DIR / "q1_pairwise_bootstrap_E003.csv")
    params = pd.read_csv(RESULT_DIR / "q1_selected_parameters_E003.csv")
    npz = np.load(RESULT_DIR / "q1_oof_predictions_E003.npz", allow_pickle=True)

    assert contract["n_samples"] == 250
    assert contract["n_features"] == 231
    assert contract["n_targets"] == 50
    assert models["repeat"].nunique() == 10
    assert npz["pred_extra_trees"].shape == (10, 250, 50)

    model_summary = (
        models.groupby(["model", "display_name"], sort=False)
        .agg(
            r2_mean=("macro_r2", "mean"),
            r2_sd=("macro_r2", "std"),
            mae_mean=("macro_mae", "mean"),
            mae_sd=("macro_mae", "std"),
            rmse_mean=("macro_rmse", "mean"),
            spearman_mean=("macro_spearman", "mean"),
        )
        .reset_index()
    )
    mean_oof = targets[targets["repeat"].astype(str) == "mean_oof"].copy()
    assert len(mean_oof) == 50
    pred_sd = np.std(npz["pred_extra_trees"], axis=0)
    return {
        "contract": contract,
        "model_summary": model_summary,
        "targets": mean_oof,
        "styles": styles,
        "pairwise": pairwise,
        "params": params,
        "prediction_sd_mean": float(pred_sd.mean()),
        "prediction_sd_median": float(np.median(pred_sd)),
        "prediction_sd_p95": float(np.quantile(pred_sd, 0.95)),
    }


def build_document(template: Path, output: Path):
    ev = load_evidence()
    c = ev["contract"]
    ms = ev["model_summary"].set_index("model")
    tm = ev["targets"]
    sm = ev["styles"]
    bs = ev["pairwise"].set_index("comparator")
    params = ev["params"]

    doc = Document(str(template))
    formula_proto = copy.deepcopy(doc.tables[0]._tbl)
    clear_document_body(doc)
    cp = doc.core_properties
    cp.title = "问题一模型建立与求解"
    cp.subject = "基于化学特征的啤酒多维感官强度预测"
    cp.author = ""
    cp.last_modified_by = ""
    cp.comments = ""
    cp.keywords = ""
    cp.category = ""

    add_heading(doc, "问题一模型建立与求解", 1)
    add_heading(doc, "问题一分析", 2)
    add_body(
        doc,
        f"附件1与附件4按 beer 字段对齐后，共得到 {c['n_samples']} 款啤酒、"
        f"{c['n_features']} 项化学与理化特征以及 {c['n_targets']} 个连续感官目标。"
        "输入维数与样本数接近，不同化合物存在共线性，多个感官目标又共享香气、风味与口感结构，"
        "因此该问题属于小样本、高维、强共线的多输出回归问题。",
    )
    add_equation(
        doc,
        formula_model_relation(),
        1,
        formula_proto,
    )
    add_body(
        doc,
        "问题的关键并非追求单次划分上的最高分数，而是在严格隔离验证信息的条件下完成化学—感官映射。"
        "其一，附件4是评委内标准化后汇总的连续值，无法从现有附件唯一恢复原始 0—6 平均分；"
        "其二，样本涵盖 22 种细分风格且样本量不均，风格信息可能掩盖真正的化学关系；"
        "其三，缺陷味和细分香气信号较弱，单一宏平均指标会掩盖局部失败。"
        "为此，本文采用“数据口径固化—多类基线对照—非线性多输出建模—重复折外评价”的技术路线。",
    )

    add_heading(doc, "问题一模型", 2)
    add_heading(doc, "数据口径与预处理", 3)
    add_body(
        doc,
        "建模基于以下假设：同一化学字段在所有样本中采用一致的单位和测量流程；"
        "在缺少酒厂与批次标识时，将每款啤酒视为一个独立观测；"
        "缺失值允许携带信息，故中位数填补后保留缺失指示；"
        "化学—感官关系在当前商业啤酒样本范围内近似稳定，结论不直接外推至新测量平台或完全未见的风格。",
    )
    add_three_line_table(
        doc,
        "表1 第一问数据契约与建模口径",
        ["项目", "审计结果", "建模处理", "理由"],
        [
            ("样本与特征", "250款，231特征", "全部保留", "按实际附件而非题面约数"),
            ("感官目标", "50个连续标准化值", "直接建模", "附件4不是原始0—6评分"),
            ("细分风格", "22类", "只用于分层与基线", "主模型保持 chemistry-only"),
            ("缺失", "73个单元，63行，9列", "折内填补+指示", "不删除含缺失样本"),
            ("log10字段", "37个可核验字段", "保持附件尺度", "负值表示原浓度小于1"),
            ("主键", "beer完整且唯一", "按beer连接", "beer_id在附件4有浮点精度损失"),
        ],
        widths=[Cm(2.7), Cm(3.5), Cm(3.7), Cm(5.7)],
        font_size=8.5,
    )
    add_body(
        doc,
        "预处理的原则是不删除可能代表真实风格差异的极端样本，且任何需估计的统计量只来自当前训练折。"
        "附件2的 231 个英文化合物名与附件1字段代码建立了一对一映射，并由附件8的字段代码全量复核，"
        "从而保证后续化学解释的字段身份一致。表2列出从读取到输出的处理步骤。",
    )
    add_three_line_table(
        doc,
        "表2 数据预处理步骤及维度变化",
        ["步骤", "处理", "处理前后", "作用域", "目的"],
        [
            ("P0", "读取附件1、4并核对类型", "250×234；250×53，不变", "全数据契约", "确定附件规模"),
            ("P1", "按beer一一对齐", "250+250→250", "全数据键约束", "避免beer_id精度问题"),
            ("P2", "移除标识列并拆分X、Y", "250×231；250×50", "字段白名单", "防止标识信息入模"),
            ("P3", "保留附件给定尺度", "X维度不变", "附件8量纲核验", "不对log10字段重复取对数"),
            ("P4", "中位数填补+缺失指示", "231→231+k，0≤k≤9", "每个训练折", "保留样本并防止泄漏"),
            ("P5", "主分析不删样、不全局截尾", "250行不变", "主分析", "保留真实风格差异"),
            ("P6", "z尺度建模，折内CDF映射0—6", "50维z+50维等级", "每个训练折", "保证量表映射不窥视验证折"),
        ],
        widths=[Cm(1.0), Cm(4.2), Cm(3.3), Cm(3.2), Cm(3.9)],
        font_size=8.1,
    )

    add_heading(doc, "模型建立与 0—6 量表映射", 3)
    add_body(
        doc,
        "为区分风格先验、线性共享结构和非线性化学关系，依次设置全局均值、风格收缩均值、PLS2、Ridge 和 ExtraTrees 五类模型。"
        "PLS2 与 Ridge 对折内秩高斯变换后的特征建模；ExtraTrees 保留附件尺度，通过多棵随机化回归树拟合非线性、阈值效应与特征交互。"
        "对第 j 个感官目标，树集成预测为",
    )
    add_equation(doc, formula_extra_trees(), 2, formula_proto)
    add_body(
        doc,
        "其中，B 为树的数量，T_bj(x) 为第 b 棵树对目标 j 的输出。"
        "每个外层训练折内再使用 3 折交叉验证，以宏平均 MAE 选择 max_features 和 min_samples_leaf，"
        "避免直接根据外层结果调参。表3给出各模型在实验中的作用。",
    )
    add_three_line_table(
        doc,
        "表3 第一问候选模型及对照作用",
        ["模型", "输入处理", "表达能力", "对照作用"],
        [
            ("全局均值", "无化学特征", "常数", "最低性能基准"),
            ("风格收缩均值", "风格+全局收缩", "类别先验", "检验化学模型是否超过风格查表"),
            ("PLS2（秩高斯）", "折内变换与缩放", "低秩共享结构", "检验多目标公共潜变量"),
            ("Ridge（秩高斯）", "折内变换与缩放", "线性带收缩", "检验非线性模型的增益"),
            ("ExtraTrees", "折内填补，保留原尺度", "非线性与交互", "最终主模型"),
        ],
        widths=[Cm(3.2), Cm(4.2), Cm(3.5), Cm(4.7)],
        font_size=8.6,
    )
    add_body(
        doc,
        "由于附件未给出每位评委在每个属性上的原始均值和标准差，标准化聚合值无法唯一逆变换为原始 0—6 平均分。"
        "本文对每个外层测试样本，仅利用对应训练折中第 j 个感官目标的经验分布函数 F̂_j,tr，"
        "将标准化预测值单调映射为 0—6 的相对强度等级：",
    )
    add_equation(doc, formula_grade_mapping(), 3, formula_proto)
    add_body(
        doc,
        "该映射保持同一感官维度上的强弱次序，但数值含义是“相对强度等级”，不解释为原始评委平均分。"
        "该口径同时避免了使用全体目标分布定义映射所导致的信息泄漏。",
    )

    add_heading(doc, "问题一求解", 2)
    add_heading(doc, "交叉验证与 OOF 结果", 3)
    add_body(
        doc,
        "将 tasting_category_fine 作为分层字段，实施 10×3 重复分层外层交叉验证，"
        "使每款啤酒在每次重复中恰好作为一次验证样本。训练折中再设置 3 折内层验证完成超参数选择。"
        "填补值、缺失指示列、秩高斯变换及 0—6 映射均在外层训练折内拟合。"
        "对第 j 个目标，主要评价指标为 R² 与 MAE，并同时报告 RMSE 和 Spearman 秩相关。",
    )
    add_equation(
        doc,
        formula_r_squared(),
        4,
        formula_proto,
    )
    add_equation(doc, formula_mae(), 5, formula_proto)

    model_rows = []
    for key in ["global_mean", "shrunk_style_mean", "pls2_quantile", "ridge_quantile", "extra_trees"]:
        r = ms.loc[key]
        model_rows.append(
            (
                r["display_name"],
                f"{r['r2_mean']:.4f} ± {r['r2_sd']:.4f}",
                f"{r['mae_mean']:.4f} ± {r['mae_sd']:.4f}",
                f"{r['rmse_mean']:.4f}",
                f"{r['spearman_mean']:.4f}",
            )
        )
    add_three_line_table(
        doc,
        "表4 五类模型在 10 次重复外层验证中的宏平均结果",
        ["模型", "R²（均值±SD）", "MAE（均值±SD）", "RMSE", "Spearman ρ"],
        model_rows,
        widths=[Cm(3.3), Cm(3.3), Cm(3.4), Cm(2.5), Cm(3.1)],
        font_size=8.5,
    )
    et = ms.loc["extra_trees"]
    ridge = ms.loc["ridge_quantile"]
    add_body(
        doc,
        f"表4与图1表明，ExtraTrees 的重复折宏平均 R² 为 {et['r2_mean']:.4f}±{et['r2_sd']:.4f}，"
        f"MAE 为 {et['mae_mean']:.4f}±{et['mae_sd']:.4f}，RMSE 为 {et['rmse_mean']:.4f}，"
        f"Spearman ρ 为 {et['spearman_mean']:.4f}。与表现最好的线性模型 Ridge 相比，"
        f"其重复折宏平均 MAE 从 {ridge['mae_mean']:.4f} 降至 {et['mae_mean']:.4f}，"
        f"相对下降 {(ridge['mae_mean']-et['mae_mean'])/ridge['mae_mean']*100:.2f}%。",
    )
    add_figure(
        doc,
        FIG_DIR / "图1_模型对比.png",
        "图1 五类基线在重复分层外层交叉验证中的性能比较",
    )

    macro = tm[["r2", "mae", "rmse", "spearman"]].mean()
    add_body(
        doc,
        f"为得到每款啤酒唯一的折外预测，再对 10 次折外预测取平均。"
        f"此时 50 个目标的 mean-OOF 宏平均 R² 为 {macro['r2']:.4f}，MAE 为 {macro['mae']:.4f}，"
        f"RMSE 为 {macro['rmse']:.4f}，Spearman ρ 为 {macro['spearman']:.4f}。"
        f"50 个目标的 R² 均大于 0，其中 {(tm['r2']>=0.25).sum()} 个不低于 0.25，"
        f"{(tm['r2']>=0.50).sum()} 个不低于 0.50。图2将目标分布与逐目标解释度按同一顺序并列，"
        "可见目标间的可预测性差异显著。",
    )
    add_figure(
        doc,
        FIG_DIR / "图2_目标分布与逐目标R2.png",
        "图2 50 个感官目标的分布与 ExtraTrees 逐目标解释度",
    )
    top = tm.nlargest(5, "r2").set_index("target_code")["r2"]
    low = tm.nsmallest(2, "r2").set_index("target_code")["r2"]
    add_body(
        doc,
        f"可预测性较高的目标为 acidity（R²={top['acidity']:.4f}）、alcohol（{top['alcohol']:.4f}）、"
        f"lactic（{top['lactic']:.4f}）、acetic（{top['acetic']:.4f}）与 bitternes（{top['bitternes']:.4f}）；"
        f"stale_hops（{low['stale_hops']:.4f}）和 F_esters_flower（{low['F_esters_flower']:.4f}）仅略优于均值预测。"
        "图3选取高、中、低可预测性目标展示观测—预测关系。"
        "连续理化感受与总体属性的化学信号更强，而稀疏缺陷味与细分香气明显向总体均值收缩。",
    )
    add_figure(
        doc,
        FIG_DIR / "图3_OOF观测预测.png",
        "图3 高、中、低可预测感官目标的 OOF 观测—预测关系",
    )

    add_heading(doc, "模型检验与结果讨论", 2)
    add_heading(doc, "泄漏与稳健性检验", 3)
    add_body(
        doc,
        "泄漏审计从字段、预处理与评价三层展开。"
        "字段层不使用 beer、beer_id 及风格标签训练主模型；"
        "预处理层的中位数、缺失指示、特征变换和量表映射均在当前训练折内拟合；"
        "评价层只用每个样本未参与训练时得到的 OOF 预测。"
        "附件2、5、6的全数据相关矩阵仅用于数据核验与描述，未进入特征筛选或调参流程。",
    )
    best_style = sm.loc[sm["mae_all_targets"].idxmin()]
    worst_style = sm.loc[sm["mae_all_targets"].idxmax()]
    add_body(
        doc,
        f"图4左侧给出 22 个细分风格的 OOF 误差，MAE 从 {best_style['mae_all_targets']:.4f}"
        f"（{best_style['tasting_category_fine']}，n={int(best_style['n_beers'])}）到 {worst_style['mae_all_targets']:.4f}"
        f"（{worst_style['tasting_category_fine']}，n={int(worst_style['n_beers'])}）不等。"
        "小风格的误差易受单个样本影响，故本文将样本量与误差同图报告。"
        "该结果反映已见风格内的折外稳健性，不等同于对完全未见风格的外推性能。"
        "图4右侧则显示折内经验分布的单调量表映射。",
    )
    add_figure(
        doc,
        FIG_DIR / "图4_风格误差与量表映射.png",
        "图4 不同啤酒风格的 OOF 误差与训练折经验分布的 0—6 映射",
    )

    add_heading(doc, "模型对比与消融", 3)
    add_body(
        doc,
        "五类模型形成逐层结构对照：全局均值衡量无信息输出，风格收缩均值衡量风格先验，"
        "PLS2 检验低秩共享结构，Ridge 衡量稳健线性映射，ExtraTrees 进一步引入非线性与交互。"
        "ExtraTrees 同时超过风格基线和两类线性/低秩模型，说明化学特征中的非线性信息对预测有实质贡献。"
        "为减少不同样本划分带来的偶然性，每款啤酒先将 50 个目标的绝对误差取平均，"
        "再对模型差值进行 20,000 次样本对自助重抽样。",
    )
    add_equation(doc, formula_paired_delta(), 6, formula_proto)
    comparator_labels = {
        "shrunk_style_mean": "风格收缩均值",
        "pls2_quantile": "PLS2（秩高斯）",
        "ridge_quantile": "Ridge（秩高斯）",
    }
    boot_rows = []
    for key in ["shrunk_style_mean", "pls2_quantile", "ridge_quantile"]:
        r = bs.loc[key]
        boot_rows.append(
            (
                comparator_labels[key],
                f"{r['mean_oof_mae_comparator']:.4f}",
                f"{r['delta_mae_extra_minus_comparator']:.4f}",
                f"[{r['beer_bootstrap_ci_lower']:.4f}, {r['beer_bootstrap_ci_upper']:.4f}]",
                f"{r['relative_mae_improvement']*100:.2f}%",
                f"{int(r['beers_with_lower_mae'])}/250",
            )
        )
    add_three_line_table(
        doc,
        "表5 ExtraTrees 与对照模型的 mean-OOF 配对比较",
        ["对照模型", "对照MAE", "ΔMAE", "95%自助区间", "相对改善", "优势样本数"],
        boot_rows,
        widths=[Cm(3.2), Cm(2.1), Cm(2.0), Cm(3.5), Cm(2.4), Cm(2.4)],
        font_size=8.3,
    )
    ridge_bs = bs.loc["ridge_quantile"]
    style_bs = bs.loc["shrunk_style_mean"]
    add_body(
        doc,
        f"表5显示，ExtraTrees 的 mean-OOF MAE 为 {ridge_bs['mean_oof_mae_extra_trees']:.4f}。"
        f"与 Ridge 相比，配对差值为 {ridge_bs['delta_mae_extra_minus_comparator']:.4f}，"
        f"95% 自助区间为 [{ridge_bs['beer_bootstrap_ci_lower']:.4f}, {ridge_bs['beer_bootstrap_ci_upper']:.4f}]，"
        f"{int(ridge_bs['beers_with_lower_mae'])}/250 款啤酒的平均绝对误差更低；"
        f"与风格收缩均值相比，配对差值的 95% 区间为 "
        f"[{style_bs['beer_bootstrap_ci_lower']:.4f}, {style_bs['beer_bootstrap_ci_upper']:.4f}]。"
        "三组区间均不跨过 0，且 10/10 次外层重复的宏平均 MAE 均有改善，因此最终选择 ExtraTrees。",
    )

    add_heading(doc, "参数敏感性与不确定性", 3)
    counts = params["selected_candidate"].value_counts()
    add_body(
        doc,
        f"在 30 个外层训练折中，内层选择结果集中于五种参数组合："
        f"max_features=1.00、min_samples_leaf=2 出现 {int(counts.get('max_features=1.00,min_samples_leaf=2',0))} 次，"
        f"1.00/1 出现 {int(counts.get('max_features=1.00,min_samples_leaf=1',0))} 次，"
        f"0.60/2 出现 {int(counts.get('max_features=0.60,min_samples_leaf=2',0))} 次，"
        f"0.60/1 出现 {int(counts.get('max_features=0.60,min_samples_leaf=1',0))} 次，"
        f"1.00/4 出现 {int(counts.get('max_features=1.00,min_samples_leaf=4',0))} 次。"
        "多个相近复杂度的组合被选中，表明局部参数有一定敏感性；但宏平均 R² 与 MAE 的跨重复标准差仅为 "
        f"{et['r2_sd']:.4f} 和 {et['mae_sd']:.4f}，说明最终性能对随机划分较稳定。",
    )
    add_body(
        doc,
        f"对每个“啤酒—感官目标”组合，10 次 OOF 预测的标准差平均为 {ev['prediction_sd_mean']:.4f}，"
        f"中位数为 {ev['prediction_sd_median']:.4f}，95% 分位数为 {ev['prediction_sd_p95']:.4f}。"
        "附图S1中，高信号目标的残差中心接近 0，但在标签分布两端存在回归到均值的倾向；"
        "低信号目标的残差离散更大。这意味着单个弱目标的精确分数不宜过度解读。",
    )
    add_figure(
        doc,
        FIG_DIR / "附图S1_残差诊断.png",
        "附图S1 代表感官目标的折外残差诊断",
    )
    add_body(
        doc,
        "综合评价表明，ExtraTrees 在当前样本上能稳定超过风格、低秩与线性基线，"
        "其优势主要来自对化学特征间非线性与交互的表达。"
        "模型的实用边界也较清楚：附件缺少酒厂和批次标识，无法检验潜在聚类泄漏；"
        "相对等级不是原始 0—6 平均分；风格分层 OOF 只支持已见风格内的内部泛化；"
        "弱目标和小风格的不确定性较高；相关或特征重要性均不构成化学因果证据。",
    )

    normalize_ooxml_order(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))
    validate_native_omml(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="生成第一问论文正文 DOCX")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    out = build_document(args.template, args.output)
    print(out)


if __name__ == "__main__":
    main()
