"""Read frozen Q3 artifacts and write small, traceable figure tables.

This script does not fit, tune, select, or modify any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
RESULT = ROOT / "03_结果" / "Q3_风格生成"
Q1_TARGET = ROOT.parents[1] / "03_结果" / "Q1_感官评分" / "publication_final" / "target_metrics_final.csv"

METHODS = {
    "style_copula_shrunk": "风格收缩 Copula",
    "raw_mixture": "整行局部混合",
    "pca_mixture": "PCA 邻域混合",
    "basis_mixture": "基函数邻域混合",
}
METRICS = {
    "mean_w1": "边缘 W1",
    "energy_v": "联合 energy",
    "coverage": "覆盖率",
    "log_spread_error": "对数离散误差",
}
TARGETS = {
    "alcohol": "酒精感",
    "bitternes": "苦味",
    "F_hops_all": "酒花味",
    "X4vg": "4VG",
    "acidity": "酸度",
    "lactic": "乳酸感",
}
STATUSES = {
    "eligible_statistical_candidate": "待实测统计候选",
    "needs_direct_verification": "直接化学证据不足",
    "proxy_variation_uncertain": "感官代理不稳定",
    "rejected_no_primary_feasible_candidate": "未找到搜索可行谱",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    inputs = {
        "generation_repeat_metrics": RESULT / "q3_metrics_by_repeat_E211.csv",
        "generated_profiles": RESULT / "q3_generated_profiles_E205.csv.gz",
        "migration_primary": RESULT / "q3_primary_candidates_E207.csv",
        "q1_target_reliability": Q1_TARGET,
    }
    if not all(p.is_file() for p in inputs.values()):
        raise FileNotFoundError({k: str(p) for k, p in inputs.items() if not p.is_file()})

    metrics = pd.read_csv(inputs["generation_repeat_metrics"])
    metrics = metrics.loc[
        (metrics.metric_scale == "robust") & metrics.method.isin(METHODS)
    ].copy()
    counts = metrics.groupby(["method", "repeat"]).style.nunique()
    assert len(counts) == 40 and counts.eq(3).all(), "Every method/repeat needs three styles"
    assert set(metrics["repeat"]) == set(range(10))
    agg = metrics.groupby(["method", "repeat"], as_index=False)[list(METRICS)].mean()
    long = agg.melt(id_vars=["method", "repeat"], var_name="metric", value_name="value")
    long["method_cn"] = long.method.map(METHODS)
    long["metric_cn"] = long.metric.map(METRICS)
    long.to_csv(SOURCE / "图1_十次重复稳健指标.csv", index=False)

    profiles = pd.read_csv(inputs["generated_profiles"])
    profiles = profiles.loc[profiles.target_code.isin(TARGETS)].copy()
    assert profiles.groupby(["style", "target_code"]).sample_id.nunique().eq(300).all()
    q1 = pd.read_csv(inputs["q1_target_reliability"])
    reliability = q1.set_index("target_code").loc[list(TARGETS), "r2"]
    assert reliability.ge(0.25).all(), reliability.to_dict()
    summary = (
        profiles.groupby(["style", "target_code"])
        .grade_0_6.agg(
            n="size", mean="mean", q10=lambda x: x.quantile(0.10), q90=lambda x: x.quantile(0.90)
        )
        .reset_index()
    )
    summary["target_cn"] = summary.target_code.map(TARGETS)
    summary["q1_oof_r2"] = summary.target_code.map(reliability)
    assert summary.shape[0] == 18 and summary.n.eq(300).all()
    summary.to_csv(SOURCE / "图2_三风格六维感官画像.csv", index=False)

    candidates = pd.read_csv(inputs["migration_primary"])
    assert len(candidates) == 33 and candidates.beer.nunique() == 11
    assert candidates.groupby("target_style").size().eq(11).all()
    assert set(candidates.audited_status) == set(STATUSES)
    candidates["status_cn"] = candidates.audited_status.map(STATUSES)
    plot_cols = [
        "beer", "target_style", "candidate_available", "relative_target_gain",
        "source_change_rms", "audited_status", "status_cn", "eligible", "physical_safety_certified",
    ]
    candidates[plot_cols].to_csv(SOURCE / "图3_固定主规则迁移状态.csv", index=False)
    counts = candidates.groupby(["target_style", "audited_status", "status_cn"]).size().rename("n").reset_index()
    counts.to_csv(SOURCE / "图3_各风格状态计数.csv", index=False)
    assert candidates.eligible.sum() == 2 and not candidates.physical_safety_certified.any()

    manifest = {
        "scope": "figure_presentation_only; frozen_results; no_training",
        "figures": {
            "图1": "10 repeated outer splits; 3 styles equal weight; robust scale; descriptive repeat variation",
            "图2": "full-data postselection; 300 synthetic profiles per style; descriptive 0-6 mapping",
            "图3": "11 sources x 3 target styles; frozen minimum-change primary rule; statistical status only",
        },
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in inputs.items()},
        "outputs": {
            p.name: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
            for p in sorted(SOURCE.glob("*.csv"))
        },
    }
    (HERE / "source_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Q3 figure source data ready:", len(manifest["outputs"]), "tables")


if __name__ == "__main__":
    main()
