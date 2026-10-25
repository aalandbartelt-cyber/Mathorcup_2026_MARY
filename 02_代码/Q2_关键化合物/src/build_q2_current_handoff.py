#!/usr/bin/env python3
"""Build the frozen-result Q2 v3 handoff from the required Word template."""

from __future__ import annotations

import copy
import json
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm
from lxml import etree


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
Q1_SRC = ROOT / "02_代码" / "Q1_感官评分" / "src"
sys.path.insert(0, str(Q1_SRC))
from build_q1_docx import (  # noqa: E402
    add_body,
    add_equation,
    add_figure,
    add_heading,
    add_three_line_table,
    clear_document_body,
    normalize_ooxml_order,
)
from q2_omml import (  # noqa: E402
    conditional_importance,
    graph_elastic_objective,
    model_relation,
    oav_formula,
    rank_correlation,
    rrf_score,
    selection_frequency,
)


TEMPLATE = CANONICAL / "docs" / "reference" / "agent初稿" / "分小问独立文档" / "Q2.docx"
OUT = ROOT / "05_论文" / "Q2_关键化合物"
RESULT = ROOT / "03_结果" / "Q2_关键化合物"
PUBLICATION = CANONICAL / "03_结果" / "Q2_关键化合物" / "publication_current"
Q1_FINAL = CANONICAL / "03_结果" / "Q1_感官评分" / "publication_final"
FIG = ROOT / "04_图表" / "Q2_关键化合物" / "v3"


ANALYSIS = [
    "问题二要求对50个感官维度分别给出3个关键单体化合物，并同时论证结果的统计稳定性与化学合理性。附件1提供250款啤酒的231项化学及理化测量，附件2给出同一批字段的Spearman共表达矩阵，附件4提供50维标准化感官值。候选变量数与样本量接近，化合物又因共同代谢途径而高度共线，因此边际相关靠前并不等同于独立贡献，更不能直接解释为因果作用。",
    "本题的难点有三：共线字段会在重采样中相互替代；啤酒风格同时影响化学组成和感官评分，容易形成风格间关联；部分缺陷味与细分香气的折外可预测性较弱。本文先冻结218个严格单体候选，在相同的10次重复、每次3折外层划分下比较边际相关、稀疏回归和图正则回归，并对第一问最终感官模型做模块与字段两级条件归因。最终排序仍采用经分层自助、风格内校正和留一风格检验的稳健秩融合；统计排名冻结后，再用阈值、OAV、同名锚点和化学文献分级。",
]

ABSTRACT_Q2 = (
    "针对50个标准化感官维度的关键化合物识别，本文从250款啤酒的231项化学及理化测量中预先界定218个严格单体候选，"
    "在10次重复、每次3折的外层验证下比较边际关联、PLS/VIP、ElasticNet、图正则回归和非线性树模型，并对第一问最终感官模型开展模块与字段两级条件归因。"
    "最终答案采用总关联、500次分层自助、风格校正和22次留一风格删除的四路稳健秩融合，每个感官维度输出3项，共150条；自助入选频率中位数为0.428。"
    "第一问最终感官模型的宏平均R²、MAE和Spearman为0.2564、0.3361和0.4302，其归因仅作为独立预测依赖证据。"
    "化学核验得到A/B/C/D级关系8/16/78/48条；补充的三证据等权融合稳定性下降，故不替换正式Top-3。"
)

ASSUMPTIONS = [
    "假设1：同一字段在250款啤酒之间具有可比的测量含义。依据是附件1采用统一字段且附件2与其Spearman矩阵一致；若存在批次或平台漂移，应先做批次校准，并按批次分组重新验证。",
    "假设2：在缺少酒厂、批次与重复配方标识时，每款啤酒可作为一个观测单位。beer字段完整且唯一，支持一一对齐；若后续获得同厂或同批信息，应改用分组交叉验证，当前误差可能偏乐观。",
    "假设3：73个缺失单元不使整款样本失去分析价值。缺失仅涉及63行9列，故所有填补量均由当前训练子样本估计；若缺失依赖未观测真实浓度，则需补做删失模型或多重插补。",
    "假设4：附件4的标准化连续值保留各感官维度内部的相对变化，可用于关联筛选与折外评价。若获得原始评委评分，应在原始量表重建感官模型，并重新计算关键化合物排名。",
    "假设5：跨重复折、分层自助与留一风格检验均稳定的关系可作为关键预测关联。该假设不赋予因果含义；若要提出调控或配方建议，必须增加加标、干预或独立批次实验。",
]


def _formula_proto(document: Document):
    proto = copy.deepcopy(document.tables[0]._tbl)
    if proto.find(qn("w:tblGrid")) is None:
        grid = OxmlElement("w:tblGrid")
        for width in (Cm(1.02), Cm(13.20), Cm(1.02)):
            node = OxmlElement("w:gridCol")
            node.set(qn("w:w"), str(width.twips))
            grid.append(node)
        proto.insert(1, grid)
    return proto


def base_document(title: str) -> tuple[Document, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    working = ROOT / "outputs" / "qa" / "q2_current" / "template_working_copy.docx"
    working.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE, working)
    document = Document(working)
    proto = _formula_proto(document)
    clear_document_body(document)
    properties = document.core_properties
    properties.title = title
    properties.subject = "啤酒关键化合物的稳健识别与化学核验"
    for field in ("author", "last_modified_by", "comments", "keywords", "category", "identifier"):
        setattr(properties, field, "")
    return document, proto


def clean_unused_parts(path: Path) -> None:
    """Remove unused sample media and personal/custom parts from the template."""

    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    with ZipFile(TEMPLATE) as archive:
        parts["word/styles.xml"] = archive.read("word/styles.xml")
    root = etree.fromstring(parts["word/document.xml"])
    used = set(
        root.xpath(
            "//@r:embed | //@r:link | //@r:id",
            namespaces={"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"},
        )
    )
    relationships = etree.fromstring(parts["word/_rels/document.xml.rels"])
    keep_media: set[str] = set()
    for node in list(relationships):
        target = node.get("Target", "")
        relation_type = node.get("Type", "")
        if relation_type.endswith("/image"):
            if node.get("Id") in used:
                keep_media.add("word/" + target)
            else:
                relationships.remove(node)
        elif any(relation_type.endswith("/" + suffix) for suffix in ("comments", "footnotes", "endnotes", "customXml")):
            relationships.remove(node)
    parts["word/_rels/document.xml.rels"] = etree.tostring(
        relationships, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    removed: list[str] = []
    for name in list(parts):
        if (
            (name.startswith("word/media/") and name not in keep_media)
            or name.startswith("customXml/")
            or name in {"word/footnotes.xml", "word/endnotes.xml", "word/comments.xml", "docProps/custom.xml"}
        ):
            removed.append(name)
            del parts[name]
    for name in ("[Content_Types].xml", "_rels/.rels"):
        tree = etree.fromstring(parts[name])
        for node in list(tree):
            target = node.get("PartName", "").lstrip("/") or node.get("Target", "")
            if target in removed:
                tree.remove(node)
        parts[name] = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


def save(document: Document, path: Path) -> Path:
    """Preserve the template's direct heading/body formatting while replacing content."""

    template = Document(TEMPLATE)
    prototypes = {
        name: next(p for p in template.paragraphs if p.style.name == name and p.text)
        for name in ("Heading 1", "Heading 2", "Heading 3", "Body Text")
    }
    for paragraph in document.paragraphs:
        if not paragraph.text or paragraph.style.name not in prototypes:
            continue
        if re.match(r"^(表\d|图\d)", paragraph.text) and paragraph.alignment == 1:
            continue
        prototype = prototypes[paragraph.style.name]
        old = paragraph._p.find(qn("w:pPr"))
        if old is not None:
            paragraph._p.remove(old)
        paragraph._p.insert(0, copy.deepcopy(prototype._p.pPr))
        source_run = next(run for run in prototype.runs if run.text.strip())._r.rPr
        for run in paragraph.runs:
            old = run._r.find(qn("w:rPr"))
            if old is not None:
                run._r.remove(old)
            run._r.insert(0, copy.deepcopy(source_run))
    normalize_ooxml_order(document)
    table_property_sequence = (
        "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
        "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
        "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption",
        "tblDescription", "tblPrChange",
    )
    order = {qn("w:" + name): index for index, name in enumerate(table_property_sequence)}
    for table in document.tables:
        properties = table._tbl.tblPr
        children = list(properties)
        for node in children:
            properties.remove(node)
        for node in sorted(children, key=lambda item: order.get(item.tag, 99)):
            properties.append(node)
        if table.rows:
            for grid, cell in zip(table._tbl.tblGrid, table.rows[0].cells):
                if cell.width is not None:
                    grid.set(qn("w:w"), str(cell.width.twips))
    document.save(path)
    clean_unused_parts(path)
    return path


def load_evidence() -> dict:
    linear = pd.read_csv(RESULT / "q2_model_summary_E101_E104.csv")
    nonlinear = pd.read_csv(RESULT / "q2_model_summary_E105.csv")
    top3 = pd.read_csv(PUBLICATION / "q2_final_top3.csv")
    manifest = json.loads((RESULT / "q2_run_manifest_E112.json").read_text(encoding="utf-8"))
    consensus = json.loads((RESULT / "q2_run_manifest_E111.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(RESULT / "q2_bootstrap_selection_E107.csv")
    style = pd.read_csv(RESULT / "q2_style_adjustment_jaccard_E108.csv")
    q1_final = json.loads((Q1_FINAL / "final_model_summary.json").read_text(encoding="utf-8"))
    nnls_audit = json.loads((RESULT / "q2_audit_E115_final_nnls.json").read_text(encoding="utf-8"))
    cpss_audit = json.loads((RESULT / "q2_audit_E117.json").read_text(encoding="utf-8"))
    fusion_decision = json.loads((RESULT / "q2_replacement_decision_E118.json").read_text(encoding="utf-8"))
    if len(top3) != 150 or top3["target_code"].nunique() != 50:
        raise ValueError("current Q2 Top-3 table must contain 150 rows and 50 targets")
    if manifest["grade_counts"] != {"C": 78, "D": 48, "B": 16, "A": 8}:
        raise ValueError("current chemistry-grade counts changed unexpectedly")
    if q1_final.get("model") != "感官组收缩NNLS":
        raise ValueError("Q1 final model registry is not the frozen grouped-shrinkage NNLS")
    if nnls_audit.get("audit_status") != "pass" or cpss_audit.get("audit_status") != "pass":
        raise ValueError("final NNLS attribution or CPSS audit is incomplete")
    if fusion_decision.get("promoted") is not False:
        raise ValueError("v2 text assumes the three-family sensitivity candidate was not adopted")
    return {
        "models": pd.concat([linear, nonlinear], ignore_index=True),
        "top3": top3,
        "manifest": manifest,
        "consensus": consensus,
        "bootstrap": bootstrap,
        "style": style,
        "q1_final": q1_final,
        "nnls_audit": nnls_audit,
        "cpss_audit": cpss_audit,
        "fusion_decision": fusion_decision,
    }


def build_main() -> Path:
    evidence = load_evidence()
    models = evidence["models"]
    top3 = evidence["top3"]
    document, proto = base_document("问题二模型建立与求解")

    add_heading(document, "问题二模型建立与求解", 1)
    add_heading(document, "摘要式结论", 2)
    add_body(document, ABSTRACT_Q2)
    add_heading(document, "问题二分析", 2)
    add_heading(document, "研究背景与任务界定", 3)
    add_body(document, ANALYSIS[0])
    add_heading(document, "难点与证据链设计", 3)
    add_body(document, ANALYSIS[1])

    add_heading(document, "问题二模型", 2)
    add_heading(document, "数据口径与候选范围", 3)
    add_body(document, "附件1与附件4按beer字段一一对齐，形成250×231的输入矩阵和250×50的连续感官矩阵。问题二不使用beer_id连接，不把风格字段作为候选化合物，也不对附件1中已给出的log10字段重复取对数。73个缺失单元仅在相应训练子样本内用中位数填补，缺失位置在预测模型中保留指示。")
    add_body(document, "候选资格先于统计建模冻结。色度、pH、苦度、蛋白质、总糖、热量、总酯、总酸、总硫及其他聚合或理化指标共13项不属于严格单体化合物，只参与全因素敏感性检查，不进入题面要求的最终Top-3。严格候选因此为218项；化学名称由字段映射表与附件8逐项核对。表1概括数据和证据边界。")
    add_three_line_table(
        document,
        "表1 问题二数据契约与证据边界",
        ["项目", "审计结果", "处理方式", "解释边界"],
        [
            ("样本与目标", "250款、50维", "固定10×3外层划分", "逐目标折外评价"),
            ("输入字段", "231项、73个缺失", "训练子样本内填补", "不删除含缺失样本"),
            ("严格候选", "218个单体", "排除13个聚合/理化指标", "全因素仅作敏感性"),
            ("共表达矩阵", "231×231 Spearman", "描述图与折内图分开", "验证时不复用全量图"),
            ("感官值", "标准化连续值", "按附件尺度建模", "不声称恢复原始0—6分"),
            ("化学证据", "阈值、来源、异味、文献", "排名冻结后追加", "不作因果解释"),
        ],
        widths=[Cm(2.5), Cm(3.5), Cm(4.2), Cm(5.4)],
        font_size=8.5,
    )
    add_body(document, "记第i款啤酒的化学向量为x_i，第j个标准化感官值为y_ij。关键化合物识别以式(1)所示预测关系为共同评价对象，但最终名次由相互独立的统计证据融合决定，而不是直接把某一模型的重要性当作因果效应。")
    add_equation(document, model_relation(), 1, proto)

    add_heading(document, "边际关联、稀疏回归与折内图正则", 3)
    add_body(document, "第一类证据为逐目标Spearman边际关联。它保留单调非线性且对异常值较稳健，式(2)将化学字段和感官值的样本秩相关标准化。所有排名均先过滤为218个严格单体候选。")
    add_equation(document, rank_correlation(), 2, proto)
    add_body(document, "第二类证据为ElasticNet条件稀疏系数；第三类为加入有符号归一化图拉普拉斯的稀疏回归。图矩阵在每个外层训练折内重新估计，负边保留方向，缺失填补和标准化也只使用训练折。对目标j，图正则目标函数写为式(3)。其中L_s为有符号图拉普拉斯，λ_1控制稀疏度，λ_G控制共表达邻居的系数平滑。")
    add_equation(document, graph_elastic_objective(), 3, proto)
    add_body(document, "图正则模型用于检验代谢共表达结构是否提高预测与排名稳定性。若折外误差没有改善，图结构仍可用于模块联合置换和解释，但不据此宣称预测增益。")

    add_heading(document, "最终感官模型的分层条件归因", 3)
    add_body(document, "第一问最终模型由ExtraTrees、直方图梯度提升、秩高斯Ridge和秩高斯PLS2组成，并用带自由截距的非负最小二乘在香气、风味和味觉/口感/总体三个感官组内估计融合权重，再向共享权重收缩0.5。问题二沿用第一问的10×3外层划分，在每个外层训练折中固定基学习器参数和融合权重，重建同一折预测器；归因阶段不重新选参，也不重新估计非负融合权重。")
    add_body(document, "归因分两级进行。先将折内共表达模块整组置换，测量模块对50维感官预测的联合影响；再在模块内逐字段条件置换，只打乱观测值并保留原缺失掩码，以减弱模块内替代造成的重要性错配。若置换字段k后目标j的验证MAE上升，则式(4)的条件误差增量为正。")
    add_equation(document, conditional_importance(), 4, proto)
    add_body(document, "30个外层折的预测重建最大绝对误差为1.998×10⁻¹⁵。模块置换和模块内条件置换分别形成13,900条与327,000条折级汇总记录。两类误差增量只刻画冻结预测器对字段或模块的依赖，不直接等同于感官机理或因果效应。")

    add_heading(document, "四路稳健秩融合", 3)
    add_body(document, "三类基础方法先形成总关联共识，随后与分层自助稳定排名、风格校正排名、留一风格保留排名共同组成四路证据。对每个目标和候选，先比较四路Top-10与Top-3支持数，再计算式(5)的倒数秩分数；r_ℓkj表示候选k在第ℓ类证据中对目标j的名次。")
    add_equation(document, rrf_score(), 5, proto)
    add_body(document, "稳定性检验在22种啤酒风格内保持样本数进行500次有放回抽样。每次重新完成预处理、候选排序和方法融合，候选的入选频率按式(6)计算。与此同时，逐一删除22种风格并重算排名，以检验某一风格是否主导关系。")
    add_equation(document, selection_frequency(), 6, proto)

    add_heading(document, "化学合理性与证据分级", 3)
    add_body(document, "统计Top-3冻结后，才依据同名锚点、描述词特异性、附件来源/老化/异味注释及同行评议文献进行化学核验。仅当附件8同时提供原始校准浓度c_k和同字段、同单位感知阈值τ_k时，才按式(7)计算气味活性值。附件1中的log10数值不直接除以阈值。")
    add_equation(document, oav_formula(), 7, proto)
    add_body(document, "A级表示统计关系较稳健且有直接身份或特异描述词证据；B级表示统计与化学类别证据较完整；C级至少有一侧证据但特异性不足；D级为弱预测背景或缺少特异化学支持。OAV达到1仅说明单体浓度达到阈值量级，不排除混合物协同、掩蔽、基质与个体差异。")

    add_heading(document, "问题二求解", 2)
    add_heading(document, "交叉验证与模型比较", 3)
    add_body(document, "所有监督模型复用问题一冻结的10次重复、每次3折外层划分；每个重复中250款啤酒各作为一次验证样本。参数、缺失填补、图结构、风格均值和条件分组均只由外层训练折估计。表2报告六种候选方法及最终感官模型的重复宏平均结果。图1突出相关阈值变化对正负边和网络连通结构的影响；图2分开呈现候选模型性能与正式Top-3对四路证据的一致性。")
    names = {
        "E100_GlobalMean": "全局均值",
        "E101_SpearmanTop3": "Spearman Top-3",
        "E102_PLS_VIP": "PLS/VIP",
        "E103_ElasticNet": "ElasticNet",
        "E104_GraphElasticNet": "有符号图ElasticNet",
        "E105_ExtraTreesConditionalPermutation": "ExtraTrees条件归因",
    }
    rows = []
    for row in models.itertuples(index=False):
        rows.append(
            (
                names[row.method],
                f"{row.macro_r2_mean:.4f}±{row.macro_r2_sd:.4f}",
                f"{row.macro_mae_mean:.4f}±{row.macro_mae_sd:.4f}",
                f"{row.macro_rmse_mean:.4f}",
                f"{row.macro_spearman_mean:.4f}",
            )
        )
    final_metrics = evidence["q1_final"]["repeat_mean"]
    rows.append(
        (
            "感官组收缩NNLS",
            f"{final_metrics['r2_mean']:.4f}±{final_metrics['r2_sd']:.4f}",
            f"{final_metrics['mae_mean']:.4f}±{final_metrics['mae_sd']:.4f}",
            f"{final_metrics['rmse_mean']:.4f}",
            f"{final_metrics['spearman_mean']:.4f}",
        )
    )
    add_three_line_table(
        document,
        "表2 相同外层划分下的候选模型与最终感官模型性能",
        ["方法", "R²（均值±SD）", "MAE（均值±SD）", "RMSE", "Spearman"],
        rows,
        widths=[Cm(3.8), Cm(3.3), Cm(3.3), Cm(2.5), Cm(2.7)],
        font_size=8.2,
    )
    add_body(document, "最终感官模型的宏平均R²、MAE和Spearman分别为0.2564、0.3361和0.4302，优于独立ExtraTrees验证器的0.2359、0.3422和0.4144。图正则模型相对ElasticNet的MAE增加0.00011，因此图结构只用于模块与敏感性分析。最终感官模型承担预测归因，关键化合物答案仍由跨方法、重采样和风格扰动下更稳定的四路秩融合给出。")
    add_figure(document, FIG / "图1_共表达图阈值敏感性_v3.png", "图1 化合物共表达网络的阈值选择与结构敏感性")
    add_figure(document, FIG / "图2_模型性能与四路一致性_v3.png", "图2 候选模型的折外性能与稳健排序的四路一致性")

    add_heading(document, "稳定性与风格敏感性", 3)
    add_body(document, "500次分层自助中，150条聚合Top-3的入选频率范围为0.182—0.992，中位数为0.428；达到0.60、0.70和0.80的候选分别有43、31和23条。逐目标Top-3两两Jaccard均值为0.245，说明共线字段替代和样本扰动不可忽略。稳定共聚类得到170个模块，其中155个为单字段模块，最大模块含15个字段。")
    add_body(document, "相同参数下，未校正与风格校正Top-3的平均Jaccard为0.264，只有2/50个目标完全一致。逐一删除22种风格后，1,100个“风格—目标”组合的Top-3平均Jaccard为0.523；当前150条候选在22次删除中的平均保留率为0.641。图3将入选频率、留一风格保留率、预测层级与化学等级放在同一证据框架下。")
    add_body(document, "补充的互补半样本检验按风格分层构造100对互补子样本，每个子样本重新填补、风格中心化并排序。入选频率不低于0.60的关系有24条；在0.60—0.90四个预设阈值下，风格内目标置换均未产生入选关系。互补Top-3的平均Jaccard仅为0.085，说明排名尾部仍有明显替代性。因此，论文同时报告确认关系与探索性关系，不把强制输出的150条关系写成同等可靠。")
    add_figure(document, FIG / "图3_候选稳定性与证据等级_v3.png", "图3 最终候选的重采样稳定性、跨风格保留率与证据等级")

    add_heading(document, "关键化合物与化学核验", 3)
    add_body(document, "四路融合的150条统计Top-3中，强稳健、中等稳健和探索性候选分别为38、52和60条；50个目标中，22个折外R²不低于0.20，12个介于0.10与0.20之间，16个低于0.10。题面要求每个维度输出3项，因此弱预测目标仍保留候选，但必须标明探索性。")
    target_zh = {
        "A_esters_ethac": "指甲油样香气",
        "F_esters_ethac": "指甲油样风味",
        "X4vg": "4-乙烯基愈创木酚",
        "acidity": "酸度",
        "alcohol": "酒精感",
        "body": "酒体",
        "co2": "碳酸感",
        "lactic": "乳酸感",
    }
    compound_zh = {
        "ethyl_acetate": "乙酸乙酯",
        "X4VG": "4-乙烯基愈创木酚",
        "lactic_acid.mg.L.": "乳酸",
        "ethanol..v.v.": "乙醇",
        "CO2.PSI.": "二氧化碳",
    }
    grade_a = top3.loc[top3["chemical_evidence_grade"].eq("A")].copy()
    table_rows = []
    for row in grade_a.itertuples(index=False):
        oav = "不可比" if pd.isna(row.oav_median) else f"{row.oav_median:.3f}"
        table_rows.append(
            (
                target_zh[row.target_code],
                str(row.consensus_rank),
                compound_zh[row.feature_code],
                f"{row.bootstrap_selection_frequency:.3f}",
                f"{row.leave_style_top3_frequency:.3f}",
                oav,
            )
        )
    add_three_line_table(
        document,
        "表3 化学证据较充分的代表性关键关系",
        ["感官维度", "名次", "关键化合物", "自助入选频率", "留一风格保留率", "中位OAV"],
        table_rows,
        widths=[Cm(3.1), Cm(1.3), Cm(3.8), Cm(2.6), Cm(2.6), Cm(2.2)],
        font_size=8.0,
    )
    add_body(document, "150条候选的A/B/C/D级数量分别为8、16、78和48，其中73条具备可比的浓度与阈值，23条的中位OAV不低于1。表3列出的8条A级关系包括乙酸乙酯—指甲油样酯香、4-乙烯基愈创木酚—同名感官、乳酸—酸度/乳酸感、乙醇—酒精感/酒体以及二氧化碳—碳酸感。图4展示这些关系在四路证据中的名次和可计算的中位OAV。")
    add_figure(document, FIG / "图4_A级关系四路排名与OAV_v3.png", "图4 A级化合物—感官关系的四路名次与气味活性值")

    add_heading(document, "模型检验与结果讨论", 2)
    add_heading(document, "泄漏与稳健性检验", 3)
    add_body(document, "验证折不参与缺失填补、尺度估计、图构建、风格均值、条件分组或参数选择；全量附件2仅用于数据定义核验与描述图。分层自助、风格校正和留一风格三种扰动从不同方向检验名次稳定性，字段级结果同时报告所属稳定模块，避免把同模块替代误写为唯一关键成分。")
    add_heading(document, "模型对比与消融", 3)
    add_body(document, "边际相关的宏平均R²为0.1064，说明单变量排序不足以概括多维感官信号；PLS/VIP与ElasticNet提高到约0.209，而加入有符号图惩罚后R²为0.2084、MAE为0.3530，未获得额外预测收益。独立ExtraTrees验证器达到0.2359，最终感官模型提高到0.2564。图结构因此只承担模块、条件依赖和敏感性解释，不被写成预测增益。")
    add_body(document, "将最终模型条件归因、折内偏相关和稳定模块三类证据等权融合后，候选Top-3的CPSS频率中位数、互补半样本Jaccard和留一风格生存率分别由0.1425、0.1669、0.6591降至0.0750、0.1151、0.4091，并遗漏碳酸感的直接锚点。该方案作为失败消融保留，不进入正式排序；后续自动化化学分级也不生成正式答案。问题二最终结论仍是四路稳健统计Top-3，最终感官模型归因只作为独立预测依赖证据。")
    add_body(document, "表4把本文各方法家族与公开文献中的相应技术目的并列。文献数据的研究对象、样本尺度和评价任务与本题不同，因此文献指标不可直接横比；本表只比较方法能回答的问题、本文中的证据角色及采用边界。")
    add_three_line_table(
        document,
        "表4 公开方法与本文方法链的功能对照及消融结论",
        ["方法家族", "回答的问题", "本文对应结果", "差异与采用判断"],
        [
            (
                "Spearman/加权相关网络\nZhang与Horvath（2005）",
                "描述成对共变及模块结构",
                "边际基线R²=0.1064；|ρ|=0.60时保留266条描述性边",
                "相关网络不区分直接与间接联系；全量图只作描述，监督验证中的图均由训练折估计。",
            ),
            (
                "稀疏回归与graphical lasso\nFriedman等（2008）",
                "估计条件线性效应和稀疏条件依赖",
                "ElasticNet R²=0.2086；图正则R²=0.2084，折内偏相关模块用于结构审计",
                "稀疏精度矩阵有助于排除部分间接边，但未改善正式排名的综合稳定性，故不替换四路排序。",
            ),
            (
                "树模型条件置换与SHAP\nStrobl等（2008）；Lundberg与Lee（2017）",
                "解释非线性预测器依赖哪些变量",
                "ExtraTrees R²=0.2359；最终感官模型R²=0.2564，并完成模块—字段两级条件归因",
                "SHAP回答加性预测归因；本文共线性较强，采用模块联合与条件置换，归因只作预测依赖证据。",
            ),
            (
                "稳定选择与互补半样本\nMeinshausen与Bühlmann（2010）；Shah与Samworth（2013）",
                "评价变量选择在重采样下的可重复性与误选风险",
                "500次分层自助入选频率中位数0.428；互补半样本与风格内置换用于补充检验",
                "理论误选界依赖相应假设；本文将稳定频率用于分层和不确定性说明，不单独决定强制Top-3。",
            ),
            (
                "四路稳健秩融合（本文）",
                "综合总关联、样本扰动、风格内关系与跨风格生存性",
                "输出50×3完整答案；R1/R2/R3为38/52/60条，A/B/C/D为8/16/78/48条",
                "各路径只融合名次，不直接相加异质量纲的重要性；三证据等权方案稳定性下降，未被采用。",
            ),
        ],
        widths=[Cm(3.6), Cm(3.2), Cm(4.4), Cm(5.0)],
        font_size=7.6,
    )
    add_heading(document, "合理性与局限", 3)
    add_body(document, "4-乙烯基愈创木酚、乙醇、乳酸和二氧化碳等直接锚点进入最终Top-3，支持方法能够恢复部分已知关系。DMS、双乙酰和trans-2-nonenal的同名字段未进入前三，分别排在较后位置，因此相应目标的前三只能解释为当前样本中的预测关联，不能冒充已知缺陷化合物。")
    add_body(document, "本问仍受250个样本、22种不均衡风格、共线替代和标准化感官信号限制。16个弱预测目标的Top-3为题面要求下的探索性输出；OAV仅在单位可比时计算，且不代表混合物中的实际贡献。若增加独立批次、原始评委评分或加标实验，应首先复核弱目标、风格敏感关系与缺陷味候选，再讨论可干预的工艺变量。")
    add_body(document, "最终感官模型的模块与字段两级归因已完成30折复核。归因结果支持把预测依赖与统计稳健性分开报告：前者回答冻结模型依赖哪些化学信息，后者决定每个感官维度的Top-3。两类证据结论不一致时，以重采样和风格扰动下更稳定的排序作为题面答案。")

    add_heading(document, "问题二方法参考文献", 2)
    references = [
        "[1] Zhang B, Horvath S. A general framework for weighted gene co-expression network analysis[J]. Statistical Applications in Genetics and Molecular Biology, 2005, 4: Article 17. DOI: 10.2202/1544-6115.1128.",
        "[2] Friedman J, Hastie T, Tibshirani R. Sparse inverse covariance estimation with the graphical lasso[J]. Biostatistics, 2008, 9(3): 432-441. DOI: 10.1093/biostatistics/kxm045.",
        "[3] Strobl C, Boulesteix A L, Kneib T, et al. Conditional variable importance for random forests[J]. BMC Bioinformatics, 2008, 9: 307. DOI: 10.1186/1471-2105-9-307.",
        "[4] Meinshausen N, Bühlmann P. Stability selection[J]. Journal of the Royal Statistical Society: Series B, 2010, 72(4): 417-473. DOI: 10.1111/j.1467-9868.2010.00740.x.",
        "[5] Shah R D, Samworth R J. Variable selection with error control: another look at stability selection[J]. Journal of the Royal Statistical Society: Series B, 2013, 75(1): 55-80. DOI: 10.1111/j.1467-9868.2011.01034.x.",
        "[6] Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems 30. 2017: 4765-4774.",
    ]
    for reference in references:
        add_body(document, reference)

    output = OUT / "第二问_论文正文_v3.docx"
    return save(document, output)


def build_assumptions() -> Path:
    document, _ = base_document("模型假设（问题二）")
    add_heading(document, "模型假设（问题二）", 1)
    add_body(document, "为简化问题，本文做出以下假设：")
    for assumption in ASSUMPTIONS:
        add_body(document, assumption)
    return save(document, OUT / "模型假设_Q2_v3.docx")


def build_analysis() -> Path:
    document, _ = base_document("问题二分析")
    add_heading(document, "问题二分析", 1)
    for paragraph in ANALYSIS:
        add_body(document, paragraph)
    return save(document, OUT / "问题分析_Q2_v3.docx")


def validate_native_math(path: Path, expected: int = 7) -> None:
    namespace = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    word_namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with ZipFile(path) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
    namespaces = {"m": namespace, "w": word_namespace}
    equations = root.xpath(".//m:oMath", namespaces=namespaces)
    if len(equations) != expected:
        raise ValueError(f"expected {expected} native equations, found {len(equations)}")
    required_structures = (
        {"sSub"},
        {"f", "nary", "sSub", "sSup", "bar", "d"},
        {"f", "sSub", "sSup", "d"},
        {"sSub", "sSup", "d"},
        {"f", "nary", "sSub"},
        {"f", "nary", "sSubSup", "sSub", "d"},
        {"f", "sSub"},
    )
    required_text = (
        ("y", "f", "ε"),
        ("ρ", "R", "n"),
        ("β", "λ", "L"),
        ("Δ", "MAE", "f"),
        ("S", "60", "r"),
        ("π", "B", "b", "I", "∈", "T", "k", "j"),
        ("OAV", "c", "τ"),
    )
    for number, (equation, expected_tags, expected_tokens) in enumerate(
        zip(equations, required_structures, required_text), start=1
    ):
        tags = {etree.QName(node).localname for node in equation.iter()}
        if not expected_tags <= tags:
            raise ValueError(f"equation {number} is missing OMML structures: {expected_tags - tags}")
        text = "".join(equation.xpath(".//m:t/text()", namespaces=namespaces))
        missing_tokens = [token for token in expected_tokens if token not in text]
        if missing_tokens:
            raise ValueError(f"equation {number} is missing tokens: {missing_tokens}")
        for run in equation.xpath(".//m:r", namespaces=namespaces):
            values = run.xpath("./m:t/text()", namespaces=namespaces)
            if not values or not "".join(values).strip():
                raise ValueError(f"equation {number} contains an empty math run")
            fonts = run.xpath("./w:rPr/w:rFonts", namespaces=namespaces)
            if not fonts or fonts[0].get(f"{{{word_namespace}}}ascii") != "Cambria Math":
                raise ValueError(f"equation {number} contains a non-Cambria Math run")


def prepare_sanitized_handoff_data() -> None:
    code_dir = OUT / "附录核心代码"
    data_dir = OUT / "数据与结果"
    for folder in (code_dir, data_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    sources = {
        "01_数据契约.py": ROOT / "02_代码/Q2_关键化合物/src/q2_contract.py",
        "02_候选选择模型.py": ROOT / "02_代码/Q2_关键化合物/src/selection_models.py",
        "03_条件置换重要性.py": ROOT / "02_代码/Q2_关键化合物/src/permutation_importance.py",
        "04_秩融合.py": ROOT / "02_代码/Q2_关键化合物/src/rank_consensus.py",
        "05_稳健性汇总.py": ROOT / "02_代码/Q2_关键化合物/src/robust_consensus.py",
        "06_化学证据.py": ROOT / "02_代码/Q2_关键化合物/src/chemical_evidence.py",
    }
    for target, source in sources.items():
        text = source.read_text(encoding="utf-8")
        # The appendix is a public handoff.  Remove internal stage identifiers
        # from copied source while leaving the executable project sources intact.
        # The negative look-arounds avoid changing scientific notation or longer
        # alphanumeric identifiers that merely contain the letter E.
        text = re.sub(r"\bQ2\s+E\d{3}\b", "Q2稳健秩融合", text)
        text = re.sub(r"(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])", "registered_stage", text)
        (code_dir / target).write_text(text, encoding="utf-8")
    (code_dir / "附录代码说明.md").write_text(
        "# 第二问附录核心代码\n\n"
        "本目录仅供论文手编排核心代码附录，依次覆盖数据契约、候选选择、条件置换、秩融合、稳健性汇总和化学证据。"
        "完整可运行流程与配置仍保存在仓库支撑材料中；论文中只摘录能够说明模型定义和关键检验的函数。\n",
        encoding="utf-8",
    )

    top3 = pd.read_csv(PUBLICATION / "q2_final_top3.csv")
    robust = pd.read_csv(PUBLICATION / "q2_robust_consensus_rankings.csv")
    chemistry = pd.read_csv(PUBLICATION / "q2_chemical_evidence_audit.csv")
    rename = {
        "e105_mean_oof_r2": "predictive_validator_mean_oof_r2",
        "e105_mean_oof_mae": "predictive_validator_mean_oof_mae",
        "e105_mean_oof_spearman": "predictive_validator_mean_oof_spearman",
        "rank__E106_total_association": "rank_total_association",
        "rank__E107_bootstrap_stability": "rank_bootstrap_stability",
        "rank__E108_leave_one_style_survival": "rank_leave_one_style_survival",
        "rank__E108_style_adjusted": "rank_style_adjusted",
    }
    top3.rename(columns=rename).to_csv(data_dir / "关键化合物Top3_当前版.csv", index=False)
    robust.rename(columns=rename).to_csv(data_dir / "全部候选稳健排名_当前版.csv", index=False)
    chemistry.rename(columns=rename).to_csv(data_dir / "化学证据核验_当前版.csv", index=False)
    evidence = load_evidence()["models"].copy()
    evidence["method"] = evidence["method"].map(
        {
            "E100_GlobalMean": "全局均值",
            "E101_SpearmanTop3": "Spearman Top-3",
            "E102_PLS_VIP": "PLS/VIP",
            "E103_ElasticNet": "ElasticNet",
            "E104_GraphElasticNet": "有符号图ElasticNet",
            "E105_ExtraTreesConditionalPermutation": "ExtraTrees条件归因",
        }
    )
    final_metrics = load_evidence()["q1_final"]["repeat_mean"]
    evidence = pd.concat(
        [
            evidence,
            pd.DataFrame(
                [
                    {
                        "method": "感官组收缩NNLS",
                        "macro_r2_mean": final_metrics["r2_mean"],
                        "macro_r2_sd": final_metrics["r2_sd"],
                        "macro_mae_mean": final_metrics["mae_mean"],
                        "macro_mae_sd": final_metrics["mae_sd"],
                        "macro_rmse_mean": final_metrics["rmse_mean"],
                        "macro_rmse_sd": final_metrics["rmse_sd"],
                        "macro_spearman_mean": final_metrics["spearman_mean"],
                        "macro_spearman_sd": final_metrics["spearman_sd"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    evidence.to_csv(data_dir / "模型性能对比_当前版.csv", index=False)

    conditional = pd.read_csv(RESULT / "q2_final_nnls_conditional_importance_E115.csv.gz")
    conditional_summary = (
        conditional.groupby(["target_code", "feature_code"], as_index=False)
        .agg(
            conditional_delta_mae_median=("median_delta_mae", "median"),
            positive_fold_fraction=("median_delta_mae", lambda values: float((values > 0).mean())),
            fold_count=("median_delta_mae", "size"),
        )
        .sort_values(
            ["target_code", "conditional_delta_mae_median", "positive_fold_fraction", "feature_code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    conditional_summary["predictive_dependence_rank"] = (
        conditional_summary.groupby("target_code", sort=False).cumcount().add(1)
    )
    conditional_summary.to_csv(data_dir / "最终感官模型条件归因汇总.csv", index=False)

    modules = pd.read_csv(RESULT / "q2_final_nnls_module_importance_E115.csv.gz").drop(
        columns=["module_source"]
    )
    modules.to_csv(data_dir / "最终感官模型模块归因_30折.csv.gz", index=False, compression="gzip")
    deletion = pd.read_csv(RESULT / "q2_final_nnls_sufficiency_E115.csv")
    deletion = deletion[
        [
            "repeat",
            "fold",
            "target_code",
            "top3_features",
            "baseline_mae",
            "masked_mae",
            "delta_mae_masked_minus_baseline",
        ]
    ].rename(columns={"top3_features": "matching_fold_top3_features"})
    deletion.to_csv(data_dir / "最终感官模型Top3删除敏感性.csv", index=False)

    (OUT / "当前版本说明.md").write_text(
        "# 第二问当前版本说明\n\n"
        "本版本已经接入第一问最终感官模型的30折模块归因与模块内条件归因。补充的三证据融合在互补半样本、留一风格和直接锚点检验中均未超过现行稳健秩融合，因此关键化合物Top-3与化学核验结果保持不变。"
        "论文手应把最终感官模型归因写成预测依赖证据，不把置换重要性解释为因果效应，也不把Top-3删除敏感性写成充分性检验。\n",
        encoding="utf-8",
    )


def main() -> None:
    main_doc = build_main()
    assumptions = build_assumptions()
    analysis = build_analysis()
    validate_native_math(main_doc)
    prepare_sanitized_handoff_data()
    print(
        json.dumps(
            {
                "main": str(main_doc),
                "assumptions": str(assumptions),
                "analysis": str(analysis),
                "native_equations": 7,
                "figures": 4,
                "tables": 4,
                "status": "frozen_results_paper_quality_handoff_v3",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
