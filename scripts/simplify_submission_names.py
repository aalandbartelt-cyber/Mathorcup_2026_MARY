#!/usr/bin/env python3
"""Shorten support filenames and update code references and integrity lists."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "06_交付/04_支撑材料"
SECTIONS = ["01_源程序", "02_数据_不含赛题原始数据", "03_结果", "04_论文图表"]

LABELS = {
    "compound_name_map": "化合物名称映射",
    "感官目标中文与模板列对照": "感官目标对照",
    "all_target_deltas": "逐目标性能变化",
    "final_model_oof": "最终折外预测",
    "final_model_summary": "最终模型指标",
    "final_oof_predictions": "折外预测明细",
    "final_oof_with_relative_grades": "相对评分明细",
    "fixed_tail_13": "尾部13目标",
    "model_comparison": "模型对比",
    "model_metrics_by_repeat": "逐重复指标",
    "model_summary": "模型指标汇总",
    "paired_beer_bootstrap": "配对重采样",
    "preregistered_hard_targets": "难目标检验",
    "representative_oof": "代表目标折外预测",
    "scale_mapping_example": "量表映射",
    "stacking_weights_by_fold": "逐折堆叠权重",
    "stacking_weights_summary": "堆叠权重汇总",
    "style_metrics_final": "逐风格指标",
    "styles_22": "22风格指标",
    "target_metrics_by_repeat": "逐目标逐重复指标",
    "target_metrics_final": "逐目标指标",
    "weight_structure_ablation": "权重结构消融",
    "calibration_bins": "校准分箱",
    "calibration_summary": "校准汇总",
    "interval_coverage": "区间覆盖率",
    "repeat_metrics_summary": "重复指标汇总",
    "repeat_model_ranks": "重复模型名次",
    "target_reliability": "目标可靠性",
    "ablation_folds": "逐折消融",
    "ablation_summary": "消融汇总",
    "leave_style_out": "留一风格检验",
    "interval_summary": "区间汇总",
    "model_repeats": "模型重复指标",
    "style_metrics_extra_trees": "基线风格指标",
    "target_dictionary": "感官目标字典",
    "target_metrics_extra_trees": "基线目标指标",
    "validation_anchor": "直接锚点核验",
    "validation_completeness": "完整性核验",
    "validation_deletion": "删除敏感性",
    "validation_negative_controls": "负对照",
    "validation_sensitivity": "敏感性检验",
    "validation_stability": "稳定性核验",
    "missing_nine_fields_zh": "9字段缺失分布",
    "style_counts": "风格样本数",
    "target_families": "感官家族数量",
    "节点": "路线节点",
    "连线": "路线连线",
}
SOURCE_NAMES = {
    "COM_01_统一绘图主题.R": "绘图主题.R",
    "COM_02_三问数据概览.R": "数据概览.R",
    "COM_03_技术路线.R": "技术路线.R",
    "Q1_07_正式主图.R": "Q1_07_主图.R",
    "Q1_08_补充检验图.R": "Q1_08_检验图.R",
    "Q1_09_附图字体统一.R": "Q1_09_附图排版.R",
    "Q2_09_正式主图.R": "Q2_09_主图.R",
    "Q3_07_生成画像与迁移图.R": "Q3_07_生成与迁移图.R",
}
FIGURE_NAMES = {
    "图01": "技术路线",
    "图02": "Q1数据概览",
    "图03": "预测模型对比",
    "图04": "逐目标预测性能",
    "图05": "观测与预测",
    "图06": "风格与难目标误差",
    "图07": "重复验证与校准",
    "图08": "Q2数据概览",
    "图09": "网络阈值敏感性",
    "图10": "模型性能与方法一致性",
    "图11": "候选稳定性",
    "图12": "A级关系与OAV",
    "图13": "负对照与锚点",
    "图14": "Q3数据概览",
    "图15": "生成质量与多样性",
    "图16": "三风格感官画像",
    "图17": "迁移收益与状态",
    "图18": "联合结构检验",
    "附图S1": "残差与量表映射",
    "附图S2": "消融与留一风格",
    "附图S3": "逐目标可靠性",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_suffix(name: str) -> tuple[str, str]:
    for suffix in [".csv.gz", ".csv", ".json", ".npz", ".png", ".py", ".R"]:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError(name)


def propose(section: str, name: str) -> str:
    if section == "01_源程序":
        return SOURCE_NAMES.get(name, name)
    if section == "04_论文图表":
        name = re.sub(r"^DOC_10_正式用图_\d+_", "", name)
        code = name.split("_", 1)[0]
        return f"{code}_{FIGURE_NAMES[code]}.png"
    match = re.match(r"^(Q[123]|COM|DOC)_\d+_.+?_\d+_(.*)$", name)
    if not match:
        return name
    question, body = match.groups()
    stem, suffix = split_suffix(body)
    stem = stem.replace("_当前版", "").replace("_可复用", "")
    inner = re.match(r"^(Q[123]|q[123])_(.+)$", stem)
    if inner:
        question, stem = inner.group(1).upper(), inner.group(2)
    stem = LABELS.get(stem, stem)
    prefix = f"{question}_" if question in {"Q1", "Q2", "Q3"} else ""
    return prefix + stem + suffix


def compact_environment(stage: Path) -> None:
    """Keep exactly one plain-text environment and command reference."""
    folder = stage / "05_运行环境"
    output_name = "软件与运行说明.txt"
    existing = list(folder.iterdir())
    if len(existing) == 1 and existing[0].name == output_name:
        return
    sections = [
        ("一、实际软件与核心程序命令", "软件与命令说明.md"),
        ("二、版本核验", "版本核验.txt"),
        ("三、Python环境依赖快照", "env_py.txt"),
        ("四、R环境会话快照", "env_r.txt"),
    ]
    parts = [
        "软件与运行说明\n\n运行位置：04_支撑材料根目录。\n"
        "核心程序为平铺的Python与R文件；原始赛题附件需自行提供。\n"
        "依赖快照记录现有环境，并不表示其中所有软件均用于三问最终模型。\n"
    ]
    for heading, name in sections:
        path = folder / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        text = re.sub(r"^#+\s*.*\n", "", text, count=1)
        text = re.sub(r"^```[^\n]*\n?", "", text, flags=re.MULTILINE)
        text = text.replace("`", "").strip()
        parts.append(heading + "\n\n" + text + "\n")
    # This is a staging copy; the complete prior folder is archived on install.
    shutil.rmtree(folder)
    folder.mkdir()
    (folder / output_name).write_text("\n".join(parts), encoding="utf-8", newline="\r\n")
    readme = stage / "README_支撑材料说明.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("软件与命令见05_运行环境，", "软件与命令见05_运行环境/软件与运行说明.txt，")
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp = Path(tempfile.mkdtemp(prefix="simple_names_", dir=ROOT / "_archive"))
    stage = temp / "04_支撑材料"
    shutil.copytree(SUPPORT, stage)
    meta = stage / "06_清单与校验"
    map_fields, provenance = read_csv(meta / "平铺文件映射.csv")
    provenance_lookup = {(row["区段"], row["平铺文件名"]): row for row in provenance}
    relative_names: dict[str, str] = {}
    rows = []
    byte_hashes: dict[str, str] = {}
    for section in SECTIONS:
        paths = sorted((stage / section).iterdir())
        assert all(p.is_file() for p in paths)
        used: set[str] = set()
        pending = []
        for index, path in enumerate(paths):
            name = propose(section, path.name)
            normal = unicodedata.normalize("NFC", name).casefold()
            if normal in used:
                record = provenance_lookup[(section, path.name)]
                stem, suffix = split_suffix(name)
                if "绘图" in record["类别"]:
                    stem += "_图源"
                elif "模型解释" in record["类别"]:
                    stem += "_明细"
                elif "补充" in record["类别"]:
                    stem += "_汇总"
                else:
                    stem += "_明细"
                name = stem + suffix
                normal = unicodedata.normalize("NFC", name).casefold()
            assert normal not in used, (section, name)
            used.add(normal)
            old_rel, new_rel = f"{section}/{path.name}", f"{section}/{name}"
            relative_names[old_rel] = new_rel
            if section != "01_源程序":
                byte_hashes[new_rel] = digest(path)
            rows.append({"目录": section, "原文件名": path.name, "新文件名": name, "SHA256": digest(path)})
            temporary = path.with_name(f".__rename_{index:03d}")
            path.rename(temporary)
            pending.append((temporary, path.with_name(name)))
        for old, new in pending:
            old.rename(new)

    # Rewrite code and documentation references without touching table values.
    replacements: dict[str, str] = {}
    for old, new in relative_names.items():
        replacements[old] = new
        replacements[Path(old).name] = Path(new).name
    for old, new in SOURCE_NAMES.items():
        replacements[Path(old).stem] = Path(new).stem
    ordered = sorted(((old, new) for old, new in replacements.items() if old != new), key=lambda pair: -len(pair[0]))
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        is_code = path.parent == stage / "01_源程序"
        is_meta = path.parent == meta and path.suffix in {".csv", ".json"}
        is_doc = path.suffix == ".md"
        if not (is_code or is_meta or is_doc):
            continue
        text = path.read_text(encoding="utf-8-sig")
        for old, new in ordered:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8-sig" if is_meta and path.suffix == ".csv" else "utf-8")
        if is_code and path.suffix == ".py":
            ast.parse(text, filename=path.name)

    # Fix all manifest paths and hashes, including changed source imports/maps.
    for path in meta.glob("*.csv"):
        fields, listing = read_csv(path)
        if "文件" in fields and "SHA256" in fields:
            for row in listing:
                actual = stage / row["文件"]
                assert actual.is_file(), actual
                row["SHA256"] = digest(actual)
                if "大小_字节" in fields:
                    row["大小_字节"] = actual.stat().st_size
            write_csv(path, fields, listing)
    _, provenance = read_csv(meta / "平铺文件映射.csv")
    for row in provenance:
        path = stage / row["区段"] / row["平铺文件名"]
        assert path.is_file(), path
        row["SHA256"] = digest(path)
        row["排序前缀"] = row["平铺文件名"].split("_")[0]
    write_csv(meta / "平铺文件映射.csv", map_fields, provenance)
    for row in rows:
        row["SHA256"] = digest(stage / row["目录"] / row["新文件名"])
    write_csv(meta / "简短文件名对照.csv", list(rows[0]), rows)
    compact_environment(stage)
    for relative, before in byte_hashes.items():
        assert digest(stage / relative) == before, relative
    report = {
        "命名规则": "图片用图号和名称；数据结果用Q1/Q2/Q3加用途；代码用问题编号和两位顺序",
        "各目录文件数": {s: len(list((stage / s).iterdir())) for s in SECTIONS},
        "数据结果与图片字节变化": 0,
        "命名冲突": 0,
        "平铺": True,
        "运行环境文件": "05_运行环境/软件与运行说明.txt",
    }
    (meta / "文件名简化验收.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checksum = meta / "SHA256SUMS.txt"
    checksum.write_text("".join(f"{digest(p)}  {p.relative_to(stage).as_posix()}\n" for p in sorted(stage.rglob("*")) if p.is_file() and p != checksum), encoding="utf-8")
    backup = ROOT / f"_archive/20261028_最终交付整理/简化命名前_{stamp}"
    SUPPORT.rename(backup)
    stage.rename(SUPPORT)
    temp.rmdir()
    print(json.dumps({**report, "备份": str(backup)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
