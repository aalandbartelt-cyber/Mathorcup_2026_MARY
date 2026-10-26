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


def find_latest(folder: Path, pattern: str):
    """Return the highest numeric vN artifact without overwriting older handoffs."""

    hits = list(folder.glob(pattern))
    if not hits:
        return None

    def version(path: Path) -> int:
        import re

        match = re.search(r"_v(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(hits, key=lambda path: (version(path), path.name))


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
    elif key in {"Q2", "Q3"}:
        docx = find_latest(qdir, f"{qnum}_论文正文_v*.docx")
        assumptions = find_latest(qdir, f"模型假设_{key}_v*.docx")
        analysis = find_latest(qdir, f"问题分析_{key}_v*.docx")
        if key == "Q2":
            import re

            match = re.search(r"_v(\d+)", docx.name) if docx else None
            version = f"v{match.group(1)}" if match else "v0"
            if match:
                figdir = figdir / version
                assumptions = qdir / f"模型假设_Q2_{version}.docx"
                analysis = qdir / f"问题分析_Q2_{version}.docx"
                figs = sorted(figdir.glob(f"*_{version}.png"))
                qlist = figdir / f"Q2_图清单_{version}.csv"

    missing = []
    if not docx or not docx.is_file(): missing.append(f"论文正文 docx（{qdir}/{qnum}_论文正文_v1.docx）")
    if not assumptions or not assumptions.is_file(): missing.append(f"模型假设（{qdir}/模型假设_*.docx|md）")
    if not analysis or not analysis.is_file(): missing.append(f"问题分析（{qdir}/问题分析_*.docx|md）")
    if not figs: missing.append(f"图片源文件（{figdir}/*.png|pdf|svg）")
    for fig in figs:
        if not fig.is_file(): missing.append(f"指定图片（{fig}）")
    if not qlist or not qlist.is_file(): missing.append(f"图清单（{figdir}/{key}_图清单.csv）")
    if key=="Q1":
        if docx and not docx.with_suffix('.pdf').is_file():missing.append('正文同步PDF')
        if not list(qdir.glob('Q1_论文审计合订本_v*.pdf')):missing.append('审计合订本PDF')
        for name in ['程序','数据与结果','附录核心代码']:
            if not (qdir/name).is_dir():missing.append(name)
    elif key == "Q2":
        if docx and not docx.with_suffix('.pdf').is_file():missing.append('正文同步PDF')
        audit_pdf = qdir / f'Q2_论文审计合订本_{version}.pdf'
        if not audit_pdf.is_file():missing.append('同版本审计合订本PDF')
        for name in ['数据与结果','附录核心代码']:
            if not (qdir/name).is_dir():missing.append(name)
        if not (qdir/'当前版本说明.md').is_file():missing.append('当前版本说明.md')
    if missing:
        raise SystemExit("交付包缺项：\n- " + "\n- ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d")
    version_match = __import__("re").search(r"_v(\d+)", docx.name)
    suffix = f"_v{version_match.group(1)}" if version_match else ""
    zip_path = outdir / f"{args.question}_交付包{suffix}.zip"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / args.question
        (stage / "图片").mkdir(parents=True, exist_ok=True)
        shutil.copy2(docx, stage / docx.name)
        shutil.copy2(assumptions, stage / assumptions.name)
        shutil.copy2(analysis, stage / analysis.name)
        if key=="Q1":
            for name in ['程序','数据与结果','附录核心代码']:
                shutil.copytree(qdir/name,stage/name)
            for name in ['摘要结论与模型评价.md']:
                if (qdir/name).exists():shutil.copy2(qdir/name,stage/name)
            for pdf in qdir.glob('*.pdf'):shutil.copy2(pdf,stage/pdf.name)
        elif key == "Q2":
            for name in ['数据与结果','附录核心代码']:
                shutil.copytree(qdir/name,stage/name)
            shutil.copy2(qdir/'当前版本说明.md',stage/'当前版本说明.md')
            for source in (docx.with_suffix('.pdf'), assumptions.with_suffix('.pdf'), analysis.with_suffix('.pdf'), audit_pdf):
                shutil.copy2(source, stage / source.name)
        for f in figs:
            shutil.copy2(f, stage / "图片" / f.name)
        shutil.copy2(qlist, stage / "图片" / qlist.name)
        manifest = [
            f"# 交付说明（{args.question}）",
            f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
            f"- 仓库 HEAD：{git_head(root)}",
            f"- 正文：{docx.name}",
            f"- 模型假设：{assumptions.name}",
            f"- 问题分析：{analysis.name}",
            f"- 图片：{len(figs)} 张 + {qlist.name}（图片/）",
            "",
            "## 交付状态",
        ]
        if key == "Q1":
            manifest += [
                "- 最终方法为感官组收缩NNLS；基线与融合方法均按学术名称呈现",
                "- 正文、图表与附带的折外指标和预测表一致",
                "- 0–6 相对强度等级仅按训练折经验分布映射",
                "- 七档分位代理一致性不等于原始评分准确率",
            ]
        elif key == "Q2":
            manifest += [
                "- 本包已接入最终感官模型的30折模块归因与模块内条件归因",
                "- 正文、图表与机器可读表统一采用四路稳健秩融合和化学核验口径",
                "- 完整150条Top-3及稳健排名放在“数据与结果”，正文只列代表性关系",
                "- 补充的三证据融合在三类稳健性检验中均未改善，故不替换现行Top-3",
                "- 置换归因只解释冻结预测器的依赖，不解释化学因果关系",
            ]
        manifest += [
            "- 图片为300dpi PNG；图题在正文中另排",
            "- 模板保留原有标题、正文直接格式与页面设置；总论文组装时统一版心",
        ]
        (stage / "交付说明.md").write_text("\n".join(manifest), encoding="utf-8")
        hashes=[]
        import hashlib
        for item in sorted(stage.rglob('*')):
            if item.is_file() and item.name!='SHA256SUMS.txt':
                hashes.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.relative_to(stage)}")
        (stage/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='utf-8')
        from audit_paper_language import audit
        language=audit([stage])
        if language['hits'] or language['unhandled']:
            raise SystemExit('交付文本检查发现内部用语或未处理文件；请先核对内部报告。')
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
