"""Add a matched 10x3 raw-data calibration to the E108 style sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from q2_contract import RESULT_DIR, build_candidate_dictionary, load_q2
from q2_pipeline import load_frozen_outer_folds
from run_e108_style_sensitivity import (
    RANDOM_STATE,
    _aggregate,
    _jaccard_by_target,
    _one_fold,
)


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    x, y, _, _ = load_q2()
    candidates = build_candidate_dictionary(x.columns)
    feature_codes = (
        candidates.loc[candidates["eligible_strict"], "feature_code"]
        .astype(str)
        .tolist()
    )
    feature_positions = {feature: x.columns.get_loc(feature) for feature in feature_codes}
    target_codes = y.columns.astype(str).tolist()
    rows = []
    folds = load_frozen_outer_folds(len(x), 10)
    for position, split in enumerate(folds):
        rows.extend(
            _one_fold(
                x.iloc[split.train_index].reset_index(drop=True),
                y.iloc[split.train_index].reset_index(drop=True),
                x.iloc[split.validation_index].reset_index(drop=True),
                y.iloc[split.validation_index].reset_index(drop=True),
                feature_codes=feature_codes,
                feature_positions=feature_positions,
                target_codes=target_codes,
                split_id=f"raw10x3_{split.split_id}",
                seed=RANDOM_STATE + 800_000 + position * 10_000,
            )
        )
        print(
            json.dumps(
                {"stage": "raw_10x3_calibration", "fold": position + 1, "total": 30},
                ensure_ascii=False,
            ),
            flush=True,
        )
    raw, methods = _aggregate(rows)
    e106 = pd.read_csv(output_dir / "q2_consensus_rankings_E106.csv")
    adjusted = pd.read_csv(output_dir / "q2_style_adjusted_rankings_E108.csv")
    versus_e106 = _jaccard_by_target(
        e106, raw, label="E106_vs_fixed_parameter_raw_10x3"
    )
    versus_adjusted = _jaccard_by_target(
        raw, adjusted, label="matched_raw_10x3_vs_style_adjusted_10x3"
    )
    comparison = pd.concat([versus_e106, versus_adjusted], ignore_index=True)
    methods.to_csv(
        output_dir / "q2_fixed_parameter_10x3_method_rankings_E108.csv.gz",
        index=False,
        compression="gzip",
    )
    raw.to_csv(output_dir / "q2_fixed_parameter_10x3_rankings_E108.csv", index=False)
    comparison.to_csv(output_dir / "q2_matched_style_calibration_E108.csv", index=False)

    manifest_path = output_dir / "q2_run_manifest_E108.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matched_raw_10x3_vs_E106"] = {
        "mean_jaccard": float(versus_e106["jaccard"].mean()),
        "median_jaccard": float(versus_e106["jaccard"].median()),
        "exact_top3_targets": int(versus_e106["exact_top3"].sum()),
    }
    manifest["matched_raw_vs_style_adjusted_10x3"] = {
        "mean_jaccard": float(versus_adjusted["jaccard"].mean()),
        "median_jaccard": float(versus_adjusted["jaccard"].median()),
        "exact_top3_targets": int(versus_adjusted["exact_top3"].sum()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
