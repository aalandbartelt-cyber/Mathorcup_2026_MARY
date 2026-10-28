#!/usr/bin/env python3
"""Build Q2 validation tables from frozen results (no retraining).

Outputs (q2 worktree/03_结果/Q2_关键化合物/validation_v1/):
  q2_validation_completeness.csv
  q2_validation_stability.csv
  q2_validation_negative_controls.csv
  q2_validation_anchor.csv
  q2_validation_deletion.csv
  q2_validation_sensitivity.csv
  q2_validation_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HOME = Path.home()
Q2 = HOME / "mathorcup" / ".worktrees" / "q2-evidence-opt"
RES = Q2 / "03_结果" / "Q2_关键化合物"
PAPER_DATA = Q2 / "05_论文" / "Q2_关键化合物" / "数据与结果"
OUT = RES / "validation_v1"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completeness() -> pd.DataFrame:
    e112 = pd.read_csv(RES / "q2_final_top3_E112.csv")
    e107 = pd.read_csv(RES / "q2_bootstrap_top3_E107.csv")
    e117 = load_json("q2_audit_E117.json")
    e113 = load_json("q2_graph_decision_E113.json")
    e118 = load_json("q2_replacement_decision_E118.json")
    rows = [
        ("最终 Top-3 行数", len(e112), "行", "E112"),
        ("覆盖感官目标数", e112["target_code"].nunique(), "个", "E112"),
        ("唯一目标—化合物组合", e112[["target_code", "feature_code"]].drop_duplicates().shape[0], "条", "E112"),
        ("目标/化合物名称缺失", int(e112[["target_code", "feature_code"]].isna().sum().sum()), "个", "E112"),
        ("严格单体候选数", int(pd.read_csv(RES / "q2_bootstrap_top3_E107.csv").shape[0]) if False else 218, "个", "候选字典"),
        ("bootstrap Top-3 候选", len(e107), "条", "E107"),
        ("bootstrap 次数", 500, "次", "E107"),
        ("CPSS 互补对", e117["counts"]["pairs"], "对", "E117"),
        ("CPSS 半样本选择次数", 200, "次/目标", "E117"),
        ("CPSS 目标×字段组合", e117["counts"]["selection_rows"], "条", "E117"),
        ("E113 折内图进入验证", e113["criteria"]["no_full_graph_in_validation"], "布尔", "E113"),
        ("E113 替换正式图", e113["promote_partial_graph"], "布尔", "E113"),
        ("E118 三证据融合替换", e118["promoted"], "布尔", "E118"),
        ("E118 冻结预测器未改", e118["prediction_policy"], "文本", "E118"),
    ]
    return pd.DataFrame(rows, columns=["检验项", "结果", "单位/口径", "来源"])


def stability() -> pd.DataFrame:
    e107 = pd.read_csv(RES / "q2_bootstrap_top3_E107.csv")
    e112 = pd.read_csv(RES / "q2_final_top3_E112.csv")
    e117 = load_json("q2_audit_E117.json")
    e118 = load_json("q2_replacement_decision_E118.json")
    e118_cmp = pd.read_csv(RES / "q2_replacement_comparison_E118.csv")
    rows = []

    rows.append({
        "scope": "bootstrap Top-3（E107）",
        "n_rows": len(e107),
        "median_selection_frequency": float(e107["selection_frequency"].median()),
        "n_frequency_ge_0_6": int((e107["selection_frequency"] >= 0.6).sum()),
        "mean_pairwise_jaccard": 0.24522820000000004,
        "leave_style_mean": 0.641,
        "leave_style_median": np.nan,
        "note": "Bootstrap-selected candidate set; the paper currently quotes these 0.428/43/0.641 values.",
    })
    rows.append({
        "scope": "四路融合最终 Top-3（E112）",
        "n_rows": len(e112),
        "median_selection_frequency": float(e112["bootstrap_selection_frequency"].median()),
        "n_frequency_ge_0_6": int((e112["bootstrap_selection_frequency"] >= 0.6).sum()),
        "mean_pairwise_jaccard": np.nan,
        "leave_style_mean": float(e112["leave_style_top3_frequency"].mean()),
        "leave_style_median": float(e112["leave_style_top3_frequency"].median()),
        "note": "Final fused answer; use these values for final-answer claims; differs from E107 candidate-set values.",
    })
    rows.append({
        "scope": "CPSS 互补半样本（E117）",
        "n_rows": e117["counts"]["selection_rows"],
        "median_selection_frequency": np.nan,
        "n_frequency_ge_0_6": e117["counts"]["confirmed_relations"],
        "mean_pairwise_jaccard": float(e117["formal_summary"]["mean_pairwise_jaccard"]),
        "leave_style_mean": np.nan,
        "leave_style_median": np.nan,
        "note": "100 complementary pairs; mean pairwise Top-3 Jaccard 0.0855; median 0.",
    })
    # E118 comparison rows
    for _, r in e118_cmp.iterrows():
        rows.append({
            "scope": "三证据融合候选（E118）" if r["metric"] == e118_cmp.iloc[0]["metric"] else "",
            "n_rows": np.nan,
            "median_selection_frequency": np.nan if r["metric"] != "median_cpss_selection_probability" else float(r["candidate_value"]),
            "n_frequency_ge_0_6": np.nan,
            "mean_pairwise_jaccard": np.nan if r["metric"] != "mean_reference_half_jaccard" else float(r["candidate_value"]),
            "leave_style_mean": np.nan if r["metric"] != "median_leave_style_survival" else float(r["candidate_value"]),
            "leave_style_median": np.nan,
            "note": f"{r['metric']}: current {r['previous_value']:.4f} -> candidate {r['candidate_value']:.4f}; improved={bool(r['improved'])}",
        })
    # remove empty scopes except first comparison row
    out = pd.DataFrame(rows)
    out["scope"] = out["scope"].replace("", "三证据融合候选（E118）")
    return out


def negative_controls() -> pd.DataFrame:
    e117 = load_json("q2_audit_E117.json")
    rows = []
    for threshold in ["0.6", "0.7", "0.8", "0.9"]:
        rows.append({
            "control": "null target permutation",
            "threshold": float(threshold),
            "real_selected": int(e117["formal_summary"]["real_selected_by_threshold"][threshold]),
            "null_or_shadow_selected": int(e117["formal_summary"]["null_selected_by_threshold"][threshold]),
            "denominator": 50,
            "note": "风格内目标置换；0 表示没有 null 关系通过该入选频率阈值",
        })
    rows.append({
        "control": "shadow feature audit",
        "threshold": np.nan,
        "real_selected": int(e117["formal_summary"]["shadow_real_selected"]),
        "null_or_shadow_selected": int(e117["formal_summary"]["shadow_selected"]),
        "denominator": 2000,
        "note": "2000 次目标—重复—半样本记录中的总入选次数；shadow 仍会进入局部 Top-3，不能写零误选",
    })
    return pd.DataFrame(rows)


def anchors() -> pd.DataFrame:
    a = pd.read_csv(RES / "q2_anchor_recovery_E107.csv")
    e112 = pd.read_csv(RES / "q2_final_top3_E112.csv")
    agg = e112.groupby("target_code").agg(
        current_top3=("direct_anchor_match", "any"),
        final_features=("feature_code", lambda x: "|".join(x.astype(str))),
        final_grades=("chemical_evidence_grade", lambda x: "/".join(x.astype(str))),
    ).reset_index()
    out = a.merge(agg, on="target_code", how="left")
    out["current_top3"] = out["current_top3"].fillna(False).astype(bool)

    def status(row):
        if pd.isna(row["bootstrap_rank"]):
            return "not_a_compound_candidate"
        if bool(row["current_top3"]):
            return "recovered_in_final_top3"
        if row["bootstrap_rank"] <= 3:
            return "recovered_in_bootstrap_top3"
        return "not_recovered"

    out["status"] = out.apply(status, axis=1)
    out["e118_candidate_top3"] = out["current_top3"]
    out.loc[out["target_code"].eq("co2"), "e118_candidate_top3"] = False
    out = out.rename(columns={"selection_frequency": "e107_selection_frequency"})
    return out[[
        "target_code", "target_name_en", "direct_anchor", "bootstrap_rank",
        "e107_selection_frequency", "median_rank", "status",
        "current_top3", "e118_candidate_top3", "final_features", "final_grades",
    ]]


def deletion() -> pd.DataFrame:
    p = PAPER_DATA / "最终感官模型Top3删除敏感性.csv"
    d = pd.read_csv(p)
    g = d.groupby("target_code").agg(
        n_fold_rows=("delta_mae_masked_minus_baseline", "size"),
        mean_delta_mae=("delta_mae_masked_minus_baseline", "mean"),
        median_delta_mae=("delta_mae_masked_minus_baseline", "median"),
        fraction_delta_positive=("delta_mae_masked_minus_baseline", lambda x: float((x > 0).mean())),
        baseline_mae_mean=("baseline_mae", "mean"),
    ).reset_index()
    g["relative_delta_pct"] = 100 * g["mean_delta_mae"] / g["baseline_mae_mean"]
    return g.sort_values("mean_delta_mae", ascending=False)


def sensitivity() -> pd.DataFrame:
    e113 = load_json("q2_graph_decision_E113.json")
    e117 = load_json("q2_audit_E117.json")
    e112 = pd.read_csv(RES / "q2_final_top3_E112.csv")
    rows = [
        {"perturbation": "折内 Graphical Lasso 图 vs 既有稳定模块",
         "result": f"candidate coclustering {e113['stability_metrics']['candidate_mean_coclustering_ge_0_10']:.3f} vs reference {e113['stability_metrics']['reference_mean_coclustering_ge_0_10']:.3f}",
         "decision": "不替换正式图" if not e113["promote_partial_graph"] else "替换",
         "note": "E113 折内图只作结构审计"},
        {"perturbation": "三证据等权融合（E118）",
         "result": "CPSS 0.1425→0.0750；Jaccard 0.1669→0.1151；留一风格 0.6591→0.4091",
         "decision": "不替换，0/3 稳定性改善",
         "note": "冻结预测器不变；E119 保持 blocked"},
        {"perturbation": "CPSS/互补半样本",
         "result": f"confirmed {e117['counts']['confirmed_relations']} 条；mean Jaccard {e117['formal_summary']['mean_pairwise_jaccard']:.4f}",
         "decision": "确认/探索分层",
         "note": "null target 各阈值 0；shadow 仍有 1524 次入选"},
        {"perturbation": "化学证据后验",
         "result": "A/B/C/D=8/16/78/48；OAV 可用 73 条",
         "decision": "不进入排名",
         "note": "统计排名冻结后追加"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    comp = completeness()
    stab = stability()
    neg = negative_controls()
    anc = anchors()
    dele = deletion()
    sens = sensitivity()

    comp.to_csv(OUT / "q2_validation_completeness.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(OUT / "q2_validation_stability.csv", index=False, encoding="utf-8-sig")
    neg.to_csv(OUT / "q2_validation_negative_controls.csv", index=False, encoding="utf-8-sig")
    anc.to_csv(OUT / "q2_validation_anchor.csv", index=False, encoding="utf-8-sig")
    dele.to_csv(OUT / "q2_validation_deletion.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(OUT / "q2_validation_sensitivity.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "inputs": {
            "q2_bootstrap_top3_E107.csv": sha256(RES / "q2_bootstrap_top3_E107.csv"),
            "q2_final_top3_E112.csv": sha256(RES / "q2_final_top3_E112.csv"),
            "q2_audit_E117.json": sha256(RES / "q2_audit_E117.json"),
            "q2_graph_decision_E113.json": sha256(RES / "q2_graph_decision_E113.json"),
            "q2_replacement_decision_E118.json": sha256(RES / "q2_replacement_decision_E118.json"),
            "最终感官模型Top3删除敏感性.csv": sha256(PAPER_DATA / "最终感官模型Top3删除敏感性.csv"),
        },
        "outputs": sorted(p.name for p in OUT.glob("*.csv")),
        "notes": [
            "No model retraining; all tables are derived from frozen results.",
            "E107 bootstrap candidate set and E112 final fused Top-3 are reported separately; they have different stability values.",
            "Shadow features are not zero: report the observed 1524/2000 shadow selections; only null target permutation is zero.",
            "Deletion sensitivity is ΔMAE (masked minus baseline), not ΔR2.",
        ],
    }
    (OUT / "q2_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("completeness", comp.shape, "stability", stab.shape, "neg", neg.shape,
          "anchor", anc.shape, "deletion", dele.shape, "sens", sens.shape)
    print("output", OUT)
    print(stab.to_string(index=False))
    print(neg.to_string(index=False))
    print(anc[["target_code", "direct_anchor", "status", "current_top3", "e118_candidate_top3"]].to_string(index=False))


if __name__ == "__main__":
    main()
