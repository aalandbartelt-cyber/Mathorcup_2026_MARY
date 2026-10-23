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
from datetime import datetime
from pathlib import Path

QNUM = {"Q1": "第一问", "Q2": "第二问", "Q3": "第三问"}


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

    missing = []
    if not docx: missing.append(f"论文正文 docx（{qdir}/{qnum}_论文正文_v1.docx）")
    if not assumptions: missing.append(f"模型假设（{qdir}/模型假设_*.docx|md）")
    if not analysis: missing.append(f"问题分析（{qdir}/问题分析_*.docx|md）")
    if not figs: missing.append(f"图片源文件（{figdir}/*.png|pdf|svg）")
    if missing:
        raise SystemExit("交付包缺项：\n- " + "\n- ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d")
    zip_path = outdir / f"{args.question}_交付包.zip"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / args.question
        (stage / "图片" / "附图").mkdir(parents=True, exist_ok=True)
        shutil.copy2(docx, stage / docx.name)
        shutil.copy2(assumptions, stage / assumptions.name)
        shutil.copy2(analysis, stage / analysis.name)
        for f in figs:
            shutil.copy2(f, stage / "图片" / f.name)
        qlist = figdir / f"{key}_图清单.csv" if figdir.exists() else None
        if qlist and qlist.exists():
            shutil.copy2(qlist, stage / "图片" / qlist.name)
        manifest = [
            f"# 交付说明（{args.question}）",
            f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
            f"- 仓库 HEAD：{git_head(root)}",
            f"- 正文：{docx.name}",
            f"- 模型假设：{assumptions.name}",
            f"- 问题分析：{analysis.name}",
            f"- 图片：{len(figs)} 张（图片/）；附图放 图片/附图/",
            "",
            "## 待确认",
            "- 图质量是否通过用户检查",
            "- 正文数字是否与 03_结果 最新指标一致",
            "- 0–6 映射口径是否按训练折经验分布",
        ]
        (stage / "交付说明.md").write_text("\n".join(manifest), encoding="utf-8")
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
