#!/usr/bin/env python3
"""Build Q1 paper v6: add calibration/coverage/reliability and ablation figure S2."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path.home() / "mathorcup"
PAPER = ROOT / "05_论文/Q1_感官评分"
VAL = PAPER / "数据与结果/validation_v1"
ABL = ROOT / ".worktrees/q1-nnls-opt/03_结果/Q1_感官评分/validation_ablation_v1"
FIG = ROOT / "04_图表/Q1_感官评分/validation_v1"
SOURCE = PAPER / "第一问_论文正文_v5.docx"
TARGET = PAPER / "第一问_论文正文_v6.docx"


def set_cell(cell, value: str, bold: bool = False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(value)
    run.bold = bold
    run.font.size = Pt(8.5)
    run.font.name = "SimSun"


def compact_table(table, row_twips: int = 20, col_twips: int = 60):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, value in (("top", row_twips), ("bottom", row_twips),
                        ("left", col_twips), ("right", col_twips)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tbl_pr.append(mar)


def main():
    shutil.copy2(SOURCE, TARGET)
    doc = Document(TARGET)
    target = next(p for p in doc.paragraphs if p.text.strip() == "结果解释与模型评价")
    fig4_caption = next(p for p in doc.paragraphs if p.text.strip().startswith("图4 "))
    caption_style = fig4_caption.style

    cov = pd.read_csv(VAL / "q1_interval_coverage.csv")
    rel = pd.read_csv(VAL / "q1_target_reliability.csv")
    cal = pd.read_csv(VAL / "q1_calibration_summary.csv")
    abl = pd.read_csv(ABL / "q1_ablation_summary.csv")
    leave = pd.read_csv(ABL / "q1_leave_style_out.csv")

    def get_cov(method, nominal):
        x = cov[(cov["method"] == method) & (cov["nominal"] == nominal)]["coverage"].mean()
        return float(x)

    ece = float(cal["ece"].mean())
    r2_min, r2_med, r2_max = float(rel["r2"].min()), float(rel["r2"].median()), float(rel["r2"].max())
    tier_counts = rel["reliability_tier"].value_counts().to_dict()
    full = abl[abl["setting"].str.startswith("完整模型")].iloc[0]
    summary = abl[abl["setting"].str.contains("汇总")].iloc[0]
    perm = abl[abl["setting"].str.contains("置换")].iloc[0]
    leave_sp_min, leave_sp_max = float(leave["spearman"].min()), float(leave["spearman"].max())
    leave_mae_min, leave_mae_max = float(leave["mae"].min()), float(leave["mae"].max())
    leave_r2_min, leave_r2_max = float(leave["r2"].min()), float(leave["r2"].max())

    def add_before(text, style):
        return target.insert_paragraph_before(text, style=style)

    h = add_before("补充检验：校准、区间与逐目标可靠性", "Heading 3")
    for r in h.runs:
        r.bold = True
        r.font.size = Pt(12)
        r.font.name = "Times New Roman"

    add_before(
        f"10次重复的宏平均R²为0.2564±0.0070、MAE为0.3361±0.0016；逐目标5分位箱校准的平均ECE为{ece:.4f}。"
        f"以重复标准差构造95%区间时，实际覆盖率仅{100*get_cov('repeat_sd',0.95):.1f}%；"
        f"以其余重复残差校准的95%区间覆盖{100*get_cov('cross_repeat_residual',0.95):.1f}%，"
        "说明重复波动远小于真实预测误差，不能把重复标准差当作预测区间。",
        "Body Text",
    )
    add_before(
        f"50个目标的折外R²范围为{r2_min:.3f}–{r2_max:.3f}，中位数{r2_med:.3f}；"
        f"按三分位分为较高/中等/较低各{tier_counts.get('较高',0)}/{tier_counts.get('中等',0)}/{tier_counts.get('较低',0)}项。"
        "强目标可用于配方筛查，弱目标只作探索性排序，不能由宏平均替代逐维可靠性。",
        "Body Text",
    )
    add_before(
        f"小规模重训显示：移除4个汇总特征后R²由{full['r2']:.4f}变为{summary['r2']:.4f}、"
        f"MAE由{full['mae']:.4f}变为{summary['mae']:.4f}，汇总特征贡献有限；"
        f"标签置换负对照的R²为{perm['r2']:.4f}、MAE为{perm['mae']:.4f}，"
        "说明流程没有把目标结构泄漏进特征。22次留一风格训练的Spearman为"
        f"{leave_sp_min:.2f}–{leave_sp_max:.2f}、MAE为{leave_mae_min:.2f}–{leave_mae_max:.2f}；"
        "小样本风格的R²因留出方差接近0而失去意义，不使用R²作风格外推的主指标。",
        "Body Text",
    )

    add_before(
        "表7与图5汇总上述检验，图5(d)显示重复标准差区间明显低于对角线。",
        "Body Text",
    )

    tcap = add_before("表7 补充检验的数值结果", caption_style)
    tcap.alignment = fig4_caption.alignment

    rows = [
        ("检验项", "结果", "口径"),
        ("OOF 完整性", "10×250×50；全部有限", "重复×样本×目标"),
        ("重复稳定性", "R² 0.2564±0.0070；MAE 0.3361±0.0016", "10次重复"),
        ("校准", f"平均ECE {ece:.4f}", "逐目标5分位箱"),
        ("重复SD区间覆盖", f"80/90/95%名义：{100*get_cov('repeat_sd',0.80):.1f}/{100*get_cov('repeat_sd',0.90):.1f}/{100*get_cov('repeat_sd',0.95):.1f}%", "10次重复预测SD"),
        ("残差校准区间覆盖", f"80/90/95%名义：{100*get_cov('cross_repeat_residual',0.80):.1f}/{100*get_cov('cross_repeat_residual',0.90):.1f}/{100*get_cov('cross_repeat_residual',0.95):.1f}%", "其余9次重复残差"),
        ("逐目标R²", f"{r2_min:.3f}–{r2_max:.3f}；中位{r2_med:.3f}", "50目标"),
        ("汇总特征消融", f"R² {full['r2']:.4f}→{summary['r2']:.4f}", "移除4个汇总列"),
        ("标签置换", f"R² {perm['r2']:.4f}；MAE {perm['mae']:.4f}", "负对照"),
        ("留一风格", f"Spearman {leave_sp_min:.2f}–{leave_sp_max:.2f}；MAE {leave_mae_min:.2f}–{leave_mae_max:.2f}", "22风格；R²不作主指标"),
    ]
    table = doc.add_table(rows=len(rows), cols=3)
    table.style = "Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Cm(3.4), Cm(7.2), Cm(4.8)]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            set_cell(table.cell(i, j), value, bold=(i == 0))
            table.cell(i, j).width = widths[j]
    compact_table(table)
    target._p.addprevious(table._tbl)

    pic = add_before("", "Body Text")
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.add_run().add_picture(str(FIG / "图5_重复验证稳定性与预测校准.png"), width=Cm(15.5))
    cap = add_before("图5 重复验证稳定性、预测校准与区间覆盖", caption_style)
    cap.alignment = fig4_caption.alignment

    add_before(
        "注：重复SD区间只反映重复拟合波动，不是新样本预测区间；残差校准覆盖率是校准检查，不构成独立验证。"
        "弱目标与留一风格结果只作描述，不改变Q1主结果。",
        "Body Text",
    )

    # Appendix S2/S3: append after the existing S1 caption.
    s1 = next(p for p in doc.paragraphs if p.text.strip().startswith("附图S1 "))
    s2pic = doc.add_paragraph()
    s2pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s2pic.add_run().add_picture(str(FIG / "附图S2_消融与负对照.png"), width=Cm(15.5))
    s2cap = doc.add_paragraph(
        "附图S2 特征消融、标签置换与留一风格压力测试（蓝色为n≥10，橙色为n<10）",
        style=s1.style,
    )
    s2cap.alignment = s1.alignment
    s3pic = doc.add_paragraph()
    s3pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s3pic.add_run().add_picture(str(FIG / "附图S3_逐目标预测可靠性.png"), width=Cm(16.0))
    s3cap = doc.add_paragraph(
        "附图S3 逐目标预测可靠性（按折外R²排序，左、右各25项；误差线为95% bootstrap区间）",
        style=s1.style,
    )
    s3cap.alignment = s1.alignment

    doc.save(TARGET)
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "target": str(TARGET.relative_to(ROOT)),
        "added": ["补充检验 Heading 3", "图5", "表7", "附图S2", "附图S3"],
        "values": {
            "ece": ece,
            "repeat_sd_coverage_95": get_cov("repeat_sd", 0.95),
            "residual_coverage_95": get_cov("cross_repeat_residual", 0.95),
            "r2_range": [r2_min, r2_max],
            "leave_style_spearman_range": [leave_sp_min, leave_sp_max],
            "leave_style_mae_range": [leave_mae_min, leave_mae_max],
            "leave_style_r2_range_unreliable": [leave_r2_min, leave_r2_max],
        },
    }
    (VAL / "q1_v6_build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(TARGET)


if __name__ == "__main__":
    main()
