#!/usr/bin/env python3
"""Export frozen Q2 results into paper-facing figure tables.

This script performs no fitting, ranking, or threshold selection.  It only
renames and subsets already frozen result tables for the R plotting layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果/Q2_关键化合物"
PAPER_DATA = ROOT / "05_论文/Q2_关键化合物/数据与结果"
OUT = ROOT / "04_图表/Q2_关键化合物/v3/source_data"
PUBLIC_MANIFEST = OUT.parent / "source_data_manifest.json"
INTERNAL_LINEAGE = RESULT / "publication_audit/q2_v3_figure_lineage.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INTERNAL_LINEAGE.parent.mkdir(parents=True, exist_ok=True)

    graph_source = RESULT / "q2_graph_threshold_audit.csv"
    graph = pd.read_csv(graph_source).rename(
        columns={
            "abs_rho_threshold": "相关阈值",
            "n_edges": "总边数",
            "positive_edges": "正相关边",
            "negative_edges": "负相关边",
            "isolated_nodes": "孤立节点",
            "n_components": "连通分量",
            "largest_component": "最大连通分量节点数",
        }
    )
    graph["选定阈值"] = graph["相关阈值"].eq(0.60)
    graph_path = write_table(
        graph[
            [
                "相关阈值",
                "总边数",
                "正相关边",
                "负相关边",
                "孤立节点",
                "连通分量",
                "最大连通分量节点数",
                "选定阈值",
            ]
        ],
        "图1_阈值与网络结构.csv",
    )

    model_source = PAPER_DATA / "模型性能对比_当前版.csv"
    model = pd.read_csv(model_source)
    model = model.loc[~model["method"].eq("全局均值")].copy()
    model = model.rename(
        columns={
            "method": "方法",
            "macro_r2_mean": "R2均值",
            "macro_r2_sd": "R2标准差",
            "macro_mae_mean": "MAE均值",
            "macro_mae_sd": "MAE标准差",
        }
    )
    model["方法角色"] = model["方法"].map(
        {
            "Spearman Top-3": "边际基线",
            "PLS/VIP": "线性降维",
            "ElasticNet": "稀疏回归",
            "有符号图ElasticNet": "图正则消融",
            "ExtraTrees条件归因": "非线性验证器",
            "感官组收缩NNLS": "最终感官模型",
        }
    )
    model_path = write_table(
        model[["方法", "方法角色", "R2均值", "R2标准差", "MAE均值", "MAE标准差"]],
        "图2a_模型性能.csv",
    )

    lens_source = RESULT / "q2_final_lens_agreement_E111.csv"
    lens = pd.read_csv(lens_source)
    lens["证据路径"] = lens["lens"].map(
        {
            "E106_total_association": "总关联",
            "E107_bootstrap_stability": "分层自助",
            "E108_style_adjusted": "风格校正",
            "E108_leave_one_style_survival": "逐风格删除",
        }
    )
    lens = (
        lens.groupby("证据路径", as_index=False, sort=False)
        .agg(平均Jaccard=("jaccard", "mean"), 完全一致目标数=("exact_top3", "sum"))
    )
    lens_path = write_table(lens, "图2b_四路一致性.csv")

    top3_source = PAPER_DATA / "关键化合物Top3_当前版.csv"
    top3 = pd.read_csv(top3_source)
    stability = top3[
        [
            "target_code",
            "feature_code",
            "bootstrap_selection_frequency",
            "leave_style_top3_frequency",
            "statistical_robustness_tier",
            "chemical_evidence_grade",
            "predictability_tier",
        ]
    ].rename(
        columns={
            "target_code": "感官编码",
            "feature_code": "化合物编码",
            "bootstrap_selection_frequency": "自助入选频率",
            "leave_style_top3_frequency": "留一风格保留率",
            "statistical_robustness_tier": "稳健等级",
            "chemical_evidence_grade": "化学证据等级",
            "predictability_tier": "目标可预测性",
        }
    )
    stability["稳健等级"] = stability["稳健等级"].map(
        {"R1_strong": "R1 强", "R2_moderate": "R2 中", "R3_exploratory": "R3 探索"}
    )
    stability_path = write_table(stability, "图3_候选稳定性.csv")

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
    grade_a["关系"] = grade_a.apply(
        lambda row: f"{target_zh[row.target_code]}—{compound_zh[row.feature_code]}", axis=1
    )
    grade_a = grade_a.rename(
        columns={
            "rank__三方法秩共识_total_association": "总关联",
            "rank__分层自助稳定性分析_bootstrap_stability": "分层自助",
            "rank__风格校正与留一风格分析_style_adjusted": "风格校正",
            "rank__风格校正与留一风格分析_leave_one_style_survival": "逐风格删除",
            "oav_q25": "OAV下四分位",
            "oav_median": "OAV中位数",
            "oav_q75": "OAV上四分位",
        }
    )
    grade_a["OAV状态"] = grade_a["OAV中位数"].map(
        lambda value: "N/A" if pd.isna(value) else ("零值" if value <= 0 else "可计算")
    )
    grade_a_path = write_table(
        grade_a[
            [
                "关系",
                "总关联",
                "分层自助",
                "风格校正",
                "逐风格删除",
                "OAV下四分位",
                "OAV中位数",
                "OAV上四分位",
                "OAV状态",
            ]
        ],
        "图4_A级关系名次与OAV.csv",
    )

    exports = [graph_path, model_path, lens_path, stability_path, grade_a_path]
    public_manifest = {
        "status": "frozen_result_export",
        "numeric_results_changed": False,
        "tables": [
            {
                "name": path.stem,
                "rows": int(len(pd.read_csv(path))),
                "columns": int(len(pd.read_csv(path).columns)),
                "sha256": sha256(path),
            }
            for path in exports
        ],
    }
    PUBLIC_MANIFEST.write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lineage = {
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (graph_source, model_source, lens_source, top3_source)
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in exports
        ],
    }
    INTERNAL_LINEAGE.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
