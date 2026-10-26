#!/usr/bin/env python3
"""把某一问的论文交付物打包成 zip（给论文手）。

用法：python scripts/package_paper_handoff.py --question Q1_感官评分
输出：05_论文/交付给论文手/<问题>_交付包.zip
"""
import argparse
import shutil
import subprocess
import tempfile
import zipfile
import hashlib
import csv
from datetime import datetime
from pathlib import Path

QNUM = {"Q1": "第一问", "Q2": "第二问", "Q3": "第三问"}
Q1_REQUIRED_FIGURES = [
    "图1_模型对比.png",
    "图2_目标分布与逐目标R2.png",
    "图3_OOF观测预测.png",
    "图4_风格误差与量表映射.png",
    "附图S1_残差诊断.png",
]


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def find_one(folder: Path, pattern: str):
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--question", required=True, help="例如 Q1_感官评分")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    qdir = root / "05_论文" / args.question
    figdir = root / "04_图表" / args.question
    outdir = root / "05_论文" / "交付给论文手"
    outdir.mkdir(parents=True, exist_ok=True)

    key = args.question.split("_")[0]
    qnum = QNUM.get(key, key)

    docx = find_one(qdir, f"{qnum}_论文正文*.docx")
    assumptions = find_one(qdir, "模型假设*.docx") or find_one(qdir, "模型假设*.md")
    analysis = find_one(qdir, "问题分析*.docx") or find_one(qdir, "问题分析*.md")
    figs = sorted([p for p in figdir.glob("*") if p.suffix.lower() in {".png", ".pdf", ".svg"}]) if figdir.exists() else []
    qlist = figdir / f"{key}_图清单.csv" if figdir.exists() else None

    if key == "Q1":
        docx = qdir / "第一问_论文正文_v1.docx"
        assumptions = qdir / "模型假设_Q1.docx"
        analysis = qdir / "问题分析_Q1.docx"
        figs = [figdir / name for name in Q1_REQUIRED_FIGURES]
    elif key == "Q3":
        figdir = figdir / "draft_v2"
        docx = qdir / "第三问_论文正文_v1.docx"
        assumptions = qdir / "模型假设_Q3_v1.docx"
        analysis = qdir / "问题分析_Q3_v1.docx"
        figs = [figdir / "图1_生成质量与多样性.png",
                figdir / "图2_三风格感官画像.png",
                figdir / "图3_迁移收益与审计状态.png"]
        qlist = figdir / "Q3_图清单_draft_v2.csv"

    missing = []
    if not docx or not docx.is_file(): missing.append(f"论文正文 docx（{qdir}/{qnum}_论文正文_v1.docx）")
    if not assumptions or not assumptions.is_file(): missing.append(f"模型假设（{qdir}/模型假设_*.docx|md）")
    if not analysis or not analysis.is_file(): missing.append(f"问题分析（{qdir}/问题分析_*.docx|md）")
    if not figs: missing.append(f"图片源文件（{figdir}/*.png|pdf|svg）")
    for fig in figs:
        if not fig.is_file(): missing.append(f"指定图片（{fig}）")
    if not qlist or not qlist.is_file(): missing.append(f"图清单（{figdir}/{key}_图清单.csv）")
    if key == "Q3":
        for required in [qdir / "第三问_论文正文_v1.pdf", qdir / "模型假设_Q3_v1.pdf",
                         qdir / "问题分析_Q3_v1.pdf", qdir / "Q3_论文审计合订本_v1.pdf",
                         qdir / "当前版本说明.md", qdir / "数据与结果", qdir / "附录核心代码"]:
            if not required.exists(): missing.append(f"Q3交付项（{required}）")
    if missing:
        raise SystemExit("交付包缺项：\n- " + "\n- ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d")
    zip_path = outdir / f"{args.question}_交付包.zip"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / args.question
        (stage / "图片").mkdir(parents=True, exist_ok=True)
        shutil.copy2(docx, stage / docx.name)
        shutil.copy2(assumptions, stage / assumptions.name)
        shutil.copy2(analysis, stage / analysis.name)
        if key == "Q3":
            for pdf in [qdir / "第三问_论文正文_v1.pdf", qdir / "模型假设_Q3_v1.pdf",
                        qdir / "问题分析_Q3_v1.pdf", qdir / "Q3_论文审计合订本_v1.pdf"]:
                shutil.copy2(pdf, stage / pdf.name)
            shutil.copy2(qdir / "当前版本说明.md", stage / "当前版本说明.md")
            shutil.copytree(qdir / "数据与结果", stage / "数据与结果")
            shutil.copytree(qdir / "附录核心代码", stage / "附录核心代码")
        for f in figs:
            shutil.copy2(f, stage / "图片" / f.name)
            vector = f.with_suffix('.svg')
            if vector.exists(): shutil.copy2(vector, stage / "图片" / vector.name)
        qlist_out = stage / "图片" / ("Q3_图清单.csv" if key == "Q3" else qlist.name)
        if key == "Q3":
            with qlist.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames; rows = list(reader)
            for row in rows:
                if "路径" in row: row["路径"] = "图片/" + Path(row["路径"]).name
                if "path" in row: row["path"] = "图片/" + Path(row["path"]).name
            with qlist_out.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        else:
            shutil.copy2(qlist, qlist_out)
        manifest = [
            f"# 交付说明（{args.question}）",
            f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
            f"- 仓库 HEAD：{git_head(root)}",
            f"- 正文：{docx.name}",
            f"- 模型假设：{assumptions.name}",
            f"- 问题分析：{analysis.name}",
            f"- 图片：{len(figs)} 张 + {qlist_out.name}（图片/）",
            "",
            "## 交付状态",
            "- 正文、图稿和机器可读结果使用同一数据口径",
            "- 0–6 相对强度等级仅按训练折经验分布映射",
        ]
        if key == "Q3":
            manifest += [
                "- 主生成方法为风格条件收缩Copula，画像使用第一问最终感官模型",
                "- 迁移结果保留待实测统计候选、直接证据不足、代理不稳定和无可行谱四类状态",
                "- 统计候选不等同于实测结果、酿造处方或物理安全认证",
                "- 正文同步PDF、审计合订PDF、核心代码和精简结果表均已包含",
            ]
        (stage / "交付说明.md").write_text("\n".join(manifest), encoding="utf-8")
        sums=[]
        for item in sorted(stage.rglob('*')):
            if item.is_file() and item.name != 'SHA256SUMS.txt':
                sums.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.relative_to(stage)}")
        (stage/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf-8')
        if key == "Q3":
            from audit_paper_language import audit
            report = audit([stage])
            if report['hits'] or report['unhandled']:
                raise SystemExit(f"交付文本检查失败：命中{len(report['hits'])}，未处理{len(report['unhandled'])}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(stage.parent))

    print("已生成：", zip_path)
    print("内容：")
    for f in sorted(zip_path.parent.glob("*.zip")):
        if f == zip_path:
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    print("  ", n)


if __name__ == "__main__":
    main()
