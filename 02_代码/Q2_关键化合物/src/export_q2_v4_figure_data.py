#!/usr/bin/env python3
"""Create the Q2 v4 figure-facing tables from frozen result files.

This is a read-only export.  It adds repeat-level Spearman summaries to the
existing v3 model table so the 2x2 performance figure can show all requested
metrics without fitting, ranking, or changing any Q2 result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果/Q2_关键化合物"
V3 = ROOT / "04_图表/Q2_关键化合物/v3/source_data"
OUT = ROOT / "04_图表/Q2_关键化合物/v4/source_data"
Q1_SUMMARY = Path("/home/mary/mathorcup/03_结果/Q1_感官评分/publication_final/final_model_summary.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source in V3.glob("*.csv"):
        target = OUT / source.name
        target.write_bytes(source.read_bytes())

    # Repeat summaries provide the same frozen ten-repeat statistics used in
    # the v3 table.  SD is computed across repeats only for figure error bars.
    repeat_frames = [
        pd.read_csv(RESULT / "q2_model_summary_by_repeat_E101_E104.csv"),
        pd.read_csv(RESULT / "q2_model_summary_by_repeat_E105.csv"),
    ]
    repeat = pd.concat(repeat_frames, ignore_index=True)
    metric = (
        repeat.groupby("method", as_index=False)
        .agg(
            Spearman均值=("macro_spearman", "mean"),
            Spearman标准差=("macro_spearman", "std"),
        )
    )
    model_path = OUT / "图2a_模型性能.csv"
    # The public table uses translated labels, while the frozen repeat table
    # uses registered method identifiers.  Fill those rows explicitly.
    method_map = {
        "Spearman Top-3": "E101_SpearmanTop3",
        "PLS/VIP": "E102_PLS_VIP",
        "ElasticNet": "E103_ElasticNet",
        "有符号图ElasticNet": "E104_GraphElasticNet",
        "ExtraTrees条件归因": "E105_ExtraTreesConditionalPermutation",
    }
    metric_public = metric.rename(columns={"method": "方法"})
    metric_public["方法"] = metric_public["方法"].map({v: k for k, v in method_map.items()})
    metric_public = metric_public.dropna(subset=["方法"])
    model = pd.read_csv(model_path).merge(metric_public, on="方法", how="left")
    summary = json.loads(Q1_SUMMARY.read_text(encoding="utf-8"))
    final = summary["repeat_mean"]
    final_mask = model["方法"].eq("感官组收缩NNLS")
    model.loc[final_mask, "Spearman均值"] = final["spearman_mean"]
    model.loc[final_mask, "Spearman标准差"] = final["spearman_sd"]
    model.to_csv(model_path, index=False)

    files = sorted(OUT.glob("*.csv"))
    manifest = {
        "status": "frozen_result_export_v4",
        "numeric_results_changed": False,
        "source_scope": "v3_sanitized_tables_plus_frozen_repeat_summaries",
        "tables": [
            {
                "name": path.name,
                "rows": int(len(pd.read_csv(path))),
                "columns": int(len(pd.read_csv(path).columns)),
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (OUT.parent / "source_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
