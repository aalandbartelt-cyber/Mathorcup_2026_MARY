#!/usr/bin/env python3
"""Curate the flat support folders to the final methods and paper assets only."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "06_交付/04_支撑材料"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
APPENDIX = ROOT / "05_论文/交付给论文手/三问附录核心代码_v2"
Q1 = ROOT / "05_论文/Q1_感官评分/程序"
Q2 = ROOT / ".worktrees/q2-evidence-opt/02_代码/Q2_关键化合物/src"

spec = importlib.util.spec_from_file_location("submission_helpers", ROOT / "scripts/build_final_submission_materials.py")
helpers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helpers)

PY_SOURCES = [
    ("Q1_01_基础学习器.py", Q1 / "baseline_models.py", "模型与折内预处理"),
    ("Q1_02_梯度提升学习器.py", Q1 / "ensemble_models.py", "模型"),
    ("Q1_03_感官组收缩NNLS.py", Q1 / "nnls_stacking.py", "模型"),
    ("Q1_04_嵌套折内求解.py", Q1 / "stacking_runner.py", "求解"),
    ("Q1_05_预测评价.py", Q1 / "q1_metrics.py", "评价"),
    ("Q1_06_感官预测运行.py", Q1 / "run_nnls.py", "求解入口"),
    ("Q2_01_候选选择与图正则.py", Q2 / "selection_models.py", "模型与折内预处理"),
    ("Q2_02_条件置换与模块.py", Q2 / "permutation_importance.py", "独立解释"),
    ("Q2_03_重采样稳定性.py", Q2 / "bootstrap_stability.py", "稳定性"),
    ("Q2_04_风格效应校正.py", Q2 / "style_adjustment.py", "稳健性"),
    ("Q2_05_四路稳健排序.py", APPENDIX / "Q2_关键化合物/rank_core.py", "正式排序求解"),
    ("Q2_06_冻结感官模型.py", APPENDIX / "Q2_关键化合物/frozen_nnls_core.py", "冻结预测接口"),
    ("Q2_07_最终模型条件归因.py", APPENDIX / "Q2_关键化合物/conditional_core.py", "独立解释"),
    ("Q2_08_化学核验与OAV.py", APPENDIX / "Q2_关键化合物/chemical_core.py", "化学核验"),
    ("Q3_01_风格收缩Copula.py", APPENDIX / "Q3_风格生成/style_copula.py", "生成模型"),
    ("Q3_02_浓度转换与缺陷约束.py", APPENDIX / "Q3_风格生成/concentration_constraints.py", "约束"),
    ("Q3_03_受约束迁移求解.py", APPENDIX / "Q3_风格生成/migration_search.py", "迁移求解"),
    ("Q3_04_感官画像推理.py", APPENDIX / "Q3_风格生成/sensory_inference.py", "画像推理"),
    ("Q3_05_迁移证据状态.py", APPENDIX / "Q3_风格生成/evidence_status.py", "证据审计"),
    ("Q3_06_分布一致性检验.py", APPENDIX / "Q3_风格生成/distribution_diagnostics.py", "分布检验"),
]
R_SOURCES = [
    ("Q1_07_正式主图.R", Q1 / "plot_final.R", "Q1"),
    ("Q1_08_补充检验图.R", ROOT / "04_图表/Q1_感官评分/待审_20261027/plot_validation_drafts.R", "Q1"),
    ("Q1_09_附图字体统一.R", ROOT / "05_论文/总论文/附录图_v2/plot_supplements_fonts.R", "Q1"),
    ("Q2_09_正式主图.R", ROOT / ".worktrees/q2-evidence-opt/04_图表/Q2_关键化合物/v4/plot_q2_paper_v4.R", "Q2"),
    ("Q2_10_负对照与锚点图.R", ROOT / "04_图表/Q2_关键化合物/待审_20261027/plot_q2_validation.R", "Q2"),
    ("Q3_07_生成画像与迁移图.R", ROOT / ".worktrees/q3-foundation/04_图表/Q3_风格生成/draft_v2/plot_q3_figures.R", "Q3"),
    ("Q3_08_联合结构检验图.R", ROOT / "04_图表/Q3_风格生成/待审_20261027/plot_q3_validation.R", "Q3"),
    ("COM_02_三问数据概览.R", ROOT / "scripts/plot_total_data_overviews.R", "COM"),
    ("COM_03_技术路线.R", ROOT / "scripts/plot_total_technical_route.R", "COM"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict]) -> None:
    helpers.write_csv(path, list(rows[0]), rows)


def adapt_imports(text: str, names: dict[str, str]) -> str:
    for old, new in names.items():
        text = re.sub(rf"(from\s+\.?)\b{re.escape(old)}\b(?=\s+import)", rf"\g<1>{new}", text)
    return text


FOUR_LENS_SOLVER = '''

def solve_four_lens_consensus(total, bootstrap, adjusted, deletion_folds):
    """正式四路秩融合；输入为总关联、重采样、风格校正、留一风格结果。"""
    deletion = aggregate_leave_one_style_ranks(deletion_folds, expected_styles=22)
    lenses = pd.concat([
        _as_lens(total, "consensus_rank", "total_association"),
        _as_lens(bootstrap, "bootstrap_rank", "bootstrap_stability"),
        _as_lens(adjusted, "consensus_rank", "style_adjusted"),
        _as_lens(deletion, "aggregate_rank", "leave_one_style_survival"),
    ], ignore_index=True)
    return reciprocal_rank_consensus(lenses, expected_methods=4, rrf_k=60).rename(
        columns={"method_top3_support": "lens_top3_support",
                 "method_top10_support": "lens_top10_support"})
'''


def write_python_core(stage: Path, retained: list[dict], source_rows: list[dict]) -> None:
    directory = stage / "01_源程序"
    directory.mkdir()
    names = {src.stem: Path(name).stem for name, src, _ in PY_SOURCES}
    q1_config = json.loads((Q1 / "model_config.json").read_text())
    data_rows = [r for r in retained if r["区段"] == "02_数据_不含赛题原始数据"]
    assets = {
        "target_dictionary.csv": next(r["平铺文件名"] for r in data_rows if r["问题"] == "Q1" and "感官目标字典" in r["平铺文件名"]),
        "cross_validation_splits.csv": next(r["平铺文件名"] for r in data_rows if r["问题"] == "Q1" and "外层交叉验证划分" in r["平铺文件名"]),
    }
    for name, source, purpose in PY_SOURCES:
        text = helpers.sanitize_text(source.read_text(encoding="utf-8-sig"))
        text = adapt_imports(text, names)
        if source.name == "run_nnls.py":
            original = "config=json.loads((Path(__file__).parent/'model_config.json').read_text())"
            assert original in text
            text = text.replace(original, "config=MODEL_CONFIG")
            anchor = "def load_tables(folder):"
            text = text.replace(anchor, "MODEL_CONFIG = " + pprint.pformat(q1_config, width=100) + "\n\nFLAT_ASSETS = " + repr(assets) + "\n\n" + anchor)
            text = text.replace("Path(__file__).resolve().parent.parent/'数据与结果'", "Path(__file__).resolve().parent.parent/'02_数据_不含赛题原始数据'")
            text = text.replace("args.assets/'target_dictionary.csv'", "args.assets/FLAT_ASSETS['target_dictionary.csv']")
            text = text.replace("args.assets/'cross_validation_splits.csv'", "args.assets/FLAT_ASSETS['cross_validation_splits.csv']")
        if source.name == "rank_core.py":
            text += FOUR_LENS_SOLVER
        text = f"# {name[:-3]}：{purpose}。软件：Python，NumPy/SciPy/pandas/scikit-learn。\n" + text
        ast.parse(text, filename=name)
        out = directory / name
        out.write_text(text, encoding="utf-8")
        source_rows.append({"问题": name.split("_")[0], "类别": purpose, "是否核心": "是", "原目录路径": str(source.relative_to(ROOT)), "文件": f"01_源程序/{name}", "大小_字节": out.stat().st_size, "SHA256": helpers.sha256(out)})


def select_assets(old: Path, stage: Path, rows: list[dict]) -> list[dict]:
    retained = []
    plot_data_seen: set[tuple[str, str, str]] = set()
    for section in ["02_数据_不含赛题原始数据", "03_结果", "04_论文图表"]:
        (stage / section).mkdir()
    for row in rows:
        section, name = row["区段"], row["平铺文件名"]
        if section == "01_源程序":
            continue
        path = old / section / name
        if path.suffix.lower() in {".md", ".yaml", ".yml", ".txt", ".py", ".r"}:
            continue
        if section == "03_结果" and path.suffix == ".json" and not any(t in name for t in ["final_model_summary", "补充检验摘要"]):
            continue
        if section == "04_论文图表":
            formal = row["类别"] == "正式用图"
            data = path.suffix == ".csv" and not any(t in name for t in ["图清单", "逐行排版核验"])
            if not (formal or data):
                continue
            if data:
                key = (row["问题"], Path(row["原目录路径"]).name, helpers.sha256(path))
                if key in plot_data_seen:
                    continue
                plot_data_seen.add(key)
                row = dict(row)
                row["区段"] = "03_结果"
                row["类别"] = "绘图数据"
                section = "03_结果"
        if section == "03_结果" and row["类别"] == "绘图数据":
            key = (row["问题"], Path(row["原目录路径"]).name, helpers.sha256(path))
            plot_data_seen.add(key)
        shutil.copy2(path, stage / section / name)
        retained.append(row)
    # Exact source tables used by the final supplemental plots.
    for index, source in enumerate(sorted((ROOT / "04_图表/Q1_感官评分/待审_20261027").glob("*.csv")), 1):
        if "图清单" in source.name:
            continue
        name = f"Q1_40_正式检验图源_{index:03d}_{source.name}"
        dst = stage / "03_结果" / name
        if dst.exists():
            continue
        shutil.copy2(source, dst)
        retained.append({"区段": "03_结果", "问题": "Q1", "类别": "绘图数据", "是否核心": "否", "排序前缀": f"Q1_40_{index:03d}", "原目录路径": f"正式检验图源/{source.name}", "平铺文件名": name, "SHA256": helpers.sha256(dst)})
    return retained


def r_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_r_core(stage: Path, rows: list[dict], source_rows: list[dict]) -> None:
    directory = stage / "01_源程序"
    aliases: dict[str, str] = {}
    # Final result tables have priority over historical plotting snapshots.
    ordered = sorted(rows, key=lambda r: (r["区段"] != "03_结果", "01_最终结果" not in r["原目录路径"], r["平铺文件名"]))
    for row in ordered:
        if not str(row["平铺文件名"]).endswith((".csv", ".csv.gz")):
            continue
        basename = Path(str(row["原目录路径"])).name
        question = str(row["问题"])
        flat_path = f"{row['区段']}/{row['平铺文件名']}"
        aliases.setdefault(f"{question}|{basename}", flat_path)
        aliases.setdefault(f"COM|{basename}", flat_path)

    assignments = "\n".join(f".mcup_csv_map[{r_string(key)}] <- {r_string(value)}" for key, value in aliases.items())
    helpers_r = '''

# Read the frozen flat source tables; all outputs go to a separate replay folder.
.mcup_csv_map <- character()
__MAP__
.mcup_csv_path <- function(path) {
  key <- paste(.mcup_question, basename(path), sep="|")
  if (!(key %in% names(.mcup_csv_map))) key <- paste("COM", basename(path), sep="|")
  if (!(key %in% names(.mcup_csv_map))) stop(paste("缺少绘图输入:", path))
  file.path(.mcup_support, unname(.mcup_csv_map[[key]]))
}
.mcup_read_csv <- function(file, ...) utils::read.csv(.mcup_csv_path(file), ...)
.mcup_readr_csv <- function(file, ...) readr::read_csv(.mcup_csv_path(file), ...)
.mcup_read_lines <- function(path, ...) {
  name <- if (grepl("plot_validation_drafts",path)) "Q1_08_补充检验图.R" else "Q1_07_正式主图.R"
  readLines(file.path(.mcup_code, name), ...)
}
'''.replace("__MAP__", assignments)
    theme = helpers.sanitize_text((ROOT / "scripts/plot_style.R").read_text()) + helpers_r
    out = directory / "COM_01_统一绘图主题.R"
    out.write_text(theme, encoding="utf-8")
    source_rows.append({"问题": "COM", "类别": "正式绘图", "是否核心": "是", "原目录路径": "scripts/plot_style.R", "文件": f"01_源程序/{out.name}", "大小_字节": out.stat().st_size, "SHA256": helpers.sha256(out)})

    for name, source, question in R_SOURCES:
        text = helpers.sanitize_text(source.read_text(encoding="utf-8-sig"))
        bootstrap = f'''.mcup_args <- grep("^--file=", commandArgs(FALSE), value=TRUE)
.mcup_script <- if (length(.mcup_args)) sub("^--file=", "", .mcup_args[1]) else Sys.getenv("MCUP_R_SCRIPT")
.mcup_code <- dirname(normalizePath(.mcup_script, mustWork=TRUE))
.mcup_support <- dirname(.mcup_code)
.mcup_question <- "{question}"
source(file.path(.mcup_code, "COM_01_统一绘图主题.R"))
.mcup_out <- file.path(.mcup_support, "绘图复算输出", "{Path(name).stem}")
dir.create(.mcup_out, recursive=TRUE, showWarnings=FALSE)'''
        text = re.sub(r'source\("(?:scripts/)?plot_style[.]R"\)', bootstrap, text)
        text = re.sub(r'source\(file.path\(root,\s*"scripts/plot_style[.]R"\)\)', bootstrap, text)
        text = text.replace("readr::read_csv(", ".mcup_readr_csv(")
        text = re.sub(r"\bread_csv\(", ".mcup_readr_csv(", text)
        text = re.sub(r"\bread[.]csv\(", ".mcup_read_csv(", text)
        text = re.sub(r"\breadLines\(", ".mcup_read_lines(", text)
        # Output location changes only. Input lookup uses the frozen table map.
        text = re.sub(r'^(OUT|figure_dir|out_dir)\s*<-\s*"[^"\n]+"', r"\1 <- .mcup_out", text, flags=re.MULTILINE)
        text = re.sub(r'^out\s*<-\s*file.path\(root,"[^"\n]+"\)', "out <- .mcup_out", text, flags=re.MULTILINE)
        text = re.sub(r'^(source_dir <- file.path\(figure_dir, "source_data"\))', r"\1\ndir.create(source_dir, recursive=TRUE, showWarnings=FALSE)", text, flags=re.MULTILINE)
        if source.name == "plot_final.R":
            text = re.sub(r'figure_dir <- Sys.getenv\(\s*"Q1_FIGURE_DIR",\s*unset = "[^"\n]+"\s*\)', "figure_dir <- .mcup_out", text)
        if question == "Q2" and "OUT <- file.path(Q2" in text:
            text = re.sub(r'OUT <- file.path\(Q2, "[^"\n]+"\)', "OUT <- .mcup_out", text)
        out = directory / name
        out.write_text(text, encoding="utf-8")
        source_rows.append({"问题": question, "类别": "正式绘图", "是否核心": "是", "原目录路径": str(source.relative_to(ROOT)), "文件": f"01_源程序/{name}", "大小_字节": out.stat().st_size, "SHA256": helpers.sha256(out)})


def refresh(stage: Path, retained: list[dict], source_rows: list[dict], figures: list[dict]) -> None:
    directory = stage / "06_清单与校验"
    # Rebuild metadata rather than leave references to files moved to the archive.
    for path in list(directory.iterdir()):
        if path.is_file():
            path.unlink()
    mappings = retained[:]
    for row in source_rows:
        mappings.append({"区段": "01_源程序", "问题": row["问题"], "类别": row["类别"], "是否核心": "是", "排序前缀": Path(str(row["文件"])).name.split("_", 2)[0], "原目录路径": row["原目录路径"], "平铺文件名": Path(str(row["文件"])).name, "SHA256": row["SHA256"]})
    write_rows(directory / "平铺文件映射.csv", mappings)
    write_rows(directory / "程序清单.csv", source_rows)
    write_rows(directory / "论文用图清单.csv", figures)
    for section, manifest in [("02_数据_不含赛题原始数据", "数据清单.csv"), ("03_结果", "结果清单.csv"), ("04_论文图表", "图源文件清单.csv")]:
        listing = []
        for row in retained:
            if row["区段"] != section:
                continue
            path = stage / section / str(row["平铺文件名"])
            listing.append({"问题": row["问题"], "类别": row["类别"], "文件": f"{section}/{path.name}", "大小_字节": path.stat().st_size, "SHA256": helpers.sha256(path)})
        write_rows(directory / manifest, listing)
    # Replace the former restoration workflow; renamed modules import each other directly.
    restore = stage / "05_运行环境/还原平铺文件.py"
    restore.unlink(missing_ok=True)
    (stage / "05_运行环境/平铺目录使用说明.md").write_text(
        "# 核心程序使用说明\n\n源程序仅保留20个Python核心文件及10个R正式绘图文件，全部平铺。参数直接嵌入Python求解入口，模块导入已随命名同步。\n\n"
        "Q1运行：`python 01_源程序/Q1_06_感官预测运行.py --help`。指定官方附件目录后可进行训练或新样本预测。\n\n"
        "Q2/Q3文件提供正式方法的核心计算函数，直接导入调用；完整历史实验工程留在原仓库及归档。\n\n"
        "R运行示例：`Rscript 01_源程序/Q2_09_正式主图.R`。图源CSV通过主题文件中的相对路径映射定位，输出到新建的绘图复算输出目录。\n",
        encoding="utf-8")
    (stage / "05_运行环境/软件与命令说明.md").write_text(
        "# 实际软件与核心程序命令\n\n"
        "实际环境：WSL2 Ubuntu 22.04，Python 3.10.19，NumPy 2.2.6，SciPy 1.15.3，pandas 2.3.3，scikit-learn 1.7.2；R 4.5.3，绘图使用ggplot2、patchwork及ragg。\n\n"
        "在支撑材料根目录运行：\n\n"
        "```bash\nconda run -n mathorcup python 01_源程序/Q1_06_感官预测运行.py --help\n"
        "conda run -n mathorcup python 01_源程序/Q1_06_感官预测运行.py --attachments 官方附件目录 --execute --output 运行结果\n"
        "conda run -n r Rscript 01_源程序/Q1_07_正式主图.R\n"
        "conda run -n r Rscript 01_源程序/Q2_09_正式主图.R\n"
        "conda run -n r Rscript 01_源程序/Q3_07_生成画像与迁移图.R\n```\n\n"
        "Q1其余五个Python模块由运行入口调用；Q2八个文件给出候选选择、重采样、风格校正、四路排序、冻结预测、条件归因和化学核验核心函数；Q3六个文件给出Copula、浓度约束、迁移路径、画像推理和检验核心函数。模块间导入已随平铺文件名同步。\n\n"
        "三问核心计算使用CPU。配置参数已写入Python源代码，源程序目录不需要YAML或Markdown。绘图直接读取本交付的平铺CSV；复算图单独输出，不覆盖正式用图。\n",
        encoding="utf-8")
    (stage / "05_运行环境/版本核验.txt").write_text(
        "conda run -n mathorcup python --version\nPython 3.10.19\n\n"
        "NumPy 2.2.6; SciPy 1.15.3; pandas 2.3.3; scikit-learn 1.7.2\n\n"
        "conda run -n r Rscript --version\nRscript (R) version 4.5.3 (2026-03-11)\n",
        encoding="utf-8")
    (stage / "README_支撑材料说明.md").write_text(
        "# 支撑材料\n\n前四个目录均为单层平铺。文件名前缀Q1/Q2/Q3/COM/DOC分别表示问题一、二、三、公共工具和总论文。\n\n"
        "源程序只保留核心.py和.R；数据不含赛题原始附件；结果保留正式结果、论文检验数据和必要绘图CSV；论文图表目录仅保留21张正式PNG图片。\n\n"
        "软件与命令见05_运行环境，文件清单和SHA256见06_清单与校验。三问模型、结果与排序沿用冻结版本。\n",
        encoding="utf-8")
    counts = {section: len(list((stage / section).iterdir())) for section in ["01_源程序", "02_数据_不含赛题原始数据", "03_结果", "04_论文图表"]}
    (directory / "交付统计.json").write_text(json.dumps({"形式": "核心文件平铺", "各目录文件数": counts, "Python核心数": 20, "R绘图数": 10, "论文正式用图": len(figures), "原始赛题附件": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = directory / "SHA256SUMS.txt"
    hashes.write_text("".join(f"{helpers.sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in sorted(stage.rglob("*")) if p.is_file() and p != hashes), encoding="utf-8")


def main() -> None:
    old_rows = read_rows(TARGET / "06_清单与校验/平铺文件映射.csv")
    figures = read_rows(TARGET / "06_清单与校验/论文用图清单.csv")
    build = Path(tempfile.mkdtemp(prefix="core_support_", dir=ROOT / "_archive"))
    stage = build / "04_支撑材料"
    stage.mkdir()
    for directory in ["05_运行环境", "06_清单与校验"]:
        shutil.copytree(TARGET / directory, stage / directory)
    retained = select_assets(TARGET, stage, old_rows)
    sources: list[dict] = []
    write_python_core(stage, retained, sources)
    write_r_core(stage, retained, sources)
    refresh(stage, retained, sources, figures)
    code = list((stage / "01_源程序").iterdir())
    assert len(code) == 30 and all(p.is_file() and p.suffix.lower() in {".py", ".r"} for p in code)
    for section in ["01_源程序", "02_数据_不含赛题原始数据", "03_结果", "04_论文图表"]:
        assert all(p.is_file() for p in (stage / section).iterdir())
    for row in figures:
        assert helpers.sha256(stage / row["文件"]) == row["SHA256"]
    backup = ROOT / f"_archive/20261028_最终交付整理/精简前支撑材料_{STAMP}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    TARGET.rename(backup)
    stage.rename(TARGET)
    build.rmdir()
    simplifier = ROOT / "scripts/simplify_submission_names.py"
    if simplifier.is_file():
        subprocess.run([sys.executable, str(simplifier)], cwd=ROOT, check=True)
    print(json.dumps({"目录": str(TARGET), "核心源程序": len(code), "备份": str(backup), "各目录文件数": json.loads((TARGET / "06_清单与校验/交付统计.json").read_text())["各目录文件数"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
