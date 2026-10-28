#!/usr/bin/env python3
"""Package Q1 v6 paper, validation evidence, figures and program for handoff."""
from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from audit_paper_language import audit

ROOT = Path.home() / "mathorcup"
QDIR = ROOT / "05_论文/Q1_感官评分"
FIG = ROOT / "04_图表/Q1_感官评分"
OUT = ROOT / "05_论文/交付给论文手"
FIGURES = [
    ("final_v4", "图1_模型对比"),
    ("final_v4", "图2_逐目标预测性能"),
    ("final_v4", "图3_OOF观测预测"),
    ("final_v4", "图4_风格与难目标稳健性"),
    ("final_v4", "附图S1_残差诊断与量表映射"),
    ("validation_v1", "图5_重复验证稳定性与预测校准"),
    ("validation_v1", "附图S2_消融与负对照"),
    ("validation_v1", "附图S3_逐目标预测可靠性"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    docx = QDIR / "第一问_论文正文_v6.docx"
    pdf = QDIR / "第一问_论文正文_v6.pdf"
    bundle = QDIR / "Q1_论文审计合订本_v6.pdf"
    assumptions = QDIR / "模型假设_Q1_v3.docx"
    analysis = QDIR / "问题分析_Q1_v3.docx"
    qlist = FIG / "Q1_图清单_v6.csv"
    required = [docx, pdf, bundle, assumptions, analysis, qlist]
    required += [FIG / folder / f"{stem}.png" for folder, stem in FIGURES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("缺项：\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT / "Q1_感官评分_交付包_v6.zip"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "Q1_感官评分"
        (stage / "图片").mkdir(parents=True)
        shutil.copy2(docx, stage / docx.name)
        shutil.copy2(pdf, stage / pdf.name)
        shutil.copy2(bundle, stage / bundle.name)
        shutil.copy2(assumptions, stage / assumptions.name)
        shutil.copy2(assumptions.with_suffix(".pdf"), stage / assumptions.with_suffix(".pdf").name)
        shutil.copy2(analysis, stage / analysis.name)
        shutil.copy2(analysis.with_suffix(".pdf"), stage / analysis.with_suffix(".pdf").name)
        shutil.copy2(QDIR / "摘要结论与模型评价.md", stage / "摘要结论与模型评价.md")
        for name in ["程序", "数据与结果", "附录核心代码"]:
            shutil.copytree(QDIR / name, stage / name)

        for folder, stem in FIGURES:
            for suffix in (".png", ".pdf", ".svg"):
                src = FIG / folder / f"{stem}{suffix}"
                if src.is_file():
                    shutil.copy2(src, stage / "图片" / src.name)
        shutil.copy2(qlist, stage / "图片" / "Q1_图清单.csv")

        manifest = [
            "# 交付说明（Q1_感官评分，v6）",
            f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
            "- 正文：第一问_论文正文_v6.docx / 同步 PDF",
            "- 模型假设：模型假设_Q1_v3.docx；问题分析：问题分析_Q1_v3.docx",
            "- 图片：正文图1—图5、附图S1—S3，PNG 300 dpi；图1—图4与S1附SVG矢量源，图5与S2/S3附PDF矢量源，另附图片清单",
            "- 数据：原折外结果、逐目标与逐风格指标，以及 validation_v1、validation_ablation_v1",
            "- 程序：完整复现脚本；附录核心代码保留入口与组合权重逻辑",
            "",
            "## 版本变化",
            "- 新增“补充检验：校准、区间与逐目标可靠性”，正文表7与图5对应",
            "- 新增附图S2：汇总特征消融、标签置换负对照与22次留一风格压力测试",
            "- 新增附图S3：50个目标的折外R²及95% bootstrap区间，左右分栏展示",
            "- 主结果仍为感官组收缩NNLS；新增内容只补充可靠性与边界，不改写主结论",
            "",
            "## 使用边界",
            "- 重复标准差区间只反映重复拟合波动，不能当作新样本预测区间；残差校准覆盖率是校准检查，不构成独立验证",
            "- 逐目标可靠性显示弱目标只能作探索性排序，不能用宏平均替代逐维判断",
            "- 留一风格结果以小样本风格为主，R²在留出方差接近0时不稳定，正文以Spearman与MAE报告",
            "- 0—6相对等级仅为训练折经验分布上的单调映射，不等同于原始评委评分",
            "- 数据与结果中的指标与正文数值一致；图片与清单中的图号、图题与正文一致",
        ]
        (stage / "交付说明.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

        hashes = []
        for item in sorted(stage.rglob("*")):
            if item.is_file() and item.name != "SHA256SUMS.txt":
                hashes.append(f"{sha256(item)}  {item.relative_to(stage)}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

        report = audit([stage])
        if report["hits"] or report["unhandled"]:
            (OUT / "Q1_v6_language_audit.json").write_text(
                __import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise SystemExit(f"语言检查未通过：hits={len(report['hits'])} unhandled={len(report['unhandled'])}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for item in sorted(stage.rglob("*")):
                if item.is_file():
                    z.write(item, item.relative_to(stage.parent))

    print(f"已生成：{zip_path}")
    print(f"大小：{zip_path.stat().st_size} bytes")
    with zipfile.ZipFile(zip_path) as z:
        print(f"条目：{len(z.namelist())}")
        for name in z.namelist():
            if name.endswith((".docx", ".pdf", ".zip")) or "图5" in name:
                print("  ", name)


if __name__ == "__main__":
    main()
