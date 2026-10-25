"""Write a descriptive Q3 contract. No predictive/generative model is fitted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from q3_contract import META, align_by_beer, defect_registry, feature_contract, source_defect_table

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parents[1] / "configs/E200_contract.yaml"
ATTACHMENT_PATH = Path("data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件")


def validate_cv(chemistry: pd.DataFrame, splits: pd.DataFrame, *,
                n_repeats: int = 10, n_folds: int = 3) -> None:
    columns = {"repeat", "fold", "row_position", "beer", "tasting_category_fine"}
    if not columns.issubset(splits) or splits[list(columns)].isna().any().any():
        raise ValueError("CV fields are missing")
    if (len(splits) != len(chemistry) * n_repeats or
            set(splits["repeat"]) != set(range(n_repeats))):
        raise ValueError("CV repeat/row coverage differs from frozen protocol")
    for repeat, block in splits.groupby("repeat"):
        block = block.sort_values("row_position")
        if (block.beer.duplicated().any() or block.row_position.duplicated().any()
                or set(block.fold) != set(range(n_folds))):
            raise ValueError(f"CV repeat {repeat} contains invalid or duplicate slots")
        if not np.array_equal(block.row_position, np.arange(len(chemistry))):
            raise ValueError(f"CV repeat {repeat} row positions disagree")
        for code in ["beer", "tasting_category_fine"]:
            if not np.array_equal(block[code], chemistry[code]):
                raise ValueError(f"CV repeat {repeat} {code} disagrees with chemistry")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_contract(root: Path, output_dir: Path, *, settings: dict | None = None) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    config = yaml.safe_load(CONFIG.read_text()) if settings is None else dict(settings)
    paths = {f"attachment_{name}": root / ATTACHMENT_PATH / name for name in ["1.csv", "4.csv", "8.xlsx"]}
    paths["frozen_cv"] = root / "03_结果/Q1_感官评分/q1_cv_splits.csv"
    paths["compound_dictionary"] = root / "01_数据/compound_name_map.csv"
    hashes_before = {name: _hash(path) for name, path in paths.items()}
    chemistry = pd.read_csv(paths["attachment_1.csv"])
    sensory = align_by_beer(chemistry, pd.read_csv(paths["attachment_4.csv"]), "sensory")
    calibrated = pd.read_excel(paths["attachment_8.xlsx"], sheet_name="Calibrated chemical dataset")
    metadata = pd.read_excel(paths["attachment_8.xlsx"], sheet_name="Sensory thresholds")
    calibrated = align_by_beer(chemistry, calibrated, "calibrated")
    for label, table, n_fields in [("chemistry", chemistry, config["expected_features"]),
                                   ("sensory", sensory, config["expected_targets"])]:
        if not set(META).issubset(table) or table.shape != (config["expected_beers"], n_fields + 3):
            raise ValueError(f"{label}: unexpected shape/metadata {table.shape}")
    x, y = chemistry.drop(columns=META), sensory.drop(columns=META)
    if np.isinf(x.to_numpy(float)).any() or not np.isfinite(y.to_numpy(float)).all():
        raise ValueError("infinite chemistry or missing/nonfinite sensory data")
    if chemistry.tasting_category_fine.nunique() != config["expected_styles"]:
        raise ValueError("unexpected number of styles")
    if int(x.isna().sum().sum()) != config["expected_missing_cells"]:
        raise ValueError("chemistry missing-cell count differs from contract")
    names = pd.read_csv(paths["compound_dictionary"])
    if ("feature_code" not in names or names.feature_code.duplicated().any()
            or names.feature_code.tolist() != x.columns.tolist()):
        raise ValueError("compound dictionary and full feature order disagree")
    splits = pd.read_csv(paths["frozen_cv"])
    validate_cv(chemistry, splits, n_repeats=config["n_repeats"], n_folds=config["n_folds"])
    fields, cells = feature_contract(chemistry, calibrated, metadata,
                                    minimum_pairs=config["minimum_scale_pairs"],
                                    minimum_match=config["minimum_scale_match"])
    counts = {str(k): int(v) for k, v in fields.scale_in_1_csv.value_counts().items()}
    if counts != config["expected_scale_counts"]:
        raise ValueError(f"scale recognition changed: {counts}")
    registry = defect_registry(metadata)
    source = source_defect_table(chemistry, calibrated, fields, config["source_style"])
    if source.beer.nunique() != config["expected_source_beers"]:
        raise ValueError("unexpected source-style sample count")
    conflicts = cells.loc[cells.status.isin(["reference_conflict", "reference_negative"])].reset_index(drop=True)
    states = cells.groupby(["feature_code", "status"], as_index=False).size().rename(columns={"size": "n_cells"})
    styles = chemistry.tasting_category_fine.value_counts().rename_axis("style").reset_index(name="n_beers")
    steps = pd.DataFrame([
        {"step": "P0_read", "rows_before": len(chemistry), "cols_before": chemistry.shape[1],
         "rows_after": len(chemistry), "cols_after": chemistry.shape[1], "reason": "读取并核对实际样本和字段"},
        {"step": "P1_beer_align", "rows_before": len(sensory), "cols_before": sensory.shape[1],
         "rows_after": len(sensory), "cols_after": sensory.shape[1], "reason": "beer唯一连接，beer_id不参与"},
        {"step": "P2_feature_split", "rows_before": len(x), "cols_before": chemistry.shape[1],
         "rows_after": len(x), "cols_after": x.shape[1], "reason": "分离元数据，不增删样本"},
        {"step": "P3_scale_audit", "rows_before": len(x), "cols_before": x.shape[1],
         "rows_after": len(x), "cols_after": x.shape[1], "reason": "识别存储尺度和冲突，不改原值"},
        {"step": "P4_missing_zero_audit", "rows_before": len(x), "cols_before": x.shape[1],
         "rows_after": len(x), "cols_after": x.shape[1], "reason": "缺失/零/地板保留状态，全数据不插补"},
        {"step": "P5_frozen_cv", "rows_before": len(splits), "cols_before": splits.shape[1],
         "rows_after": len(splits), "cols_after": splits.shape[1], "reason": "复用既有分折，不重新划分"},
    ])
    steps["scope"] = "descriptive_schema_audit_no_model_fit"
    outputs = {"q3_field_contract_E200.csv": fields,
               "q3_calibration_conflicts_E200.csv": conflicts,
               "q3_measurement_states_E200.csv": states,
               "q3_defect_registry_E200.csv": registry,
               "q3_source_defects_E200.csv": source,
               "q3_style_counts_E200.csv": styles,
               "q3_preprocessing_steps_E200.csv": steps}
    hashes_after = {name: _hash(path) for name, path in paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("inputs changed during E200 audit; discard this run")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        temporary = output_dir / (name + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(output_dir / name)
    manifest = {
        "experiment_id": "E200", "status": "complete", "model_fitted": False,
        "gpu_used": False, "data_values_changed": False, "input_hashes_unchanged": True,
        "scope": "full_data_descriptive_contract_not_training_statistics",
        "counts": {"n_beers": len(x), "n_features": x.shape[1], "n_targets": y.shape[1],
                   "n_styles": len(styles), "scale_counts": counts,
                   "missing_chemistry_cells": int(x.isna().sum().sum()),
                   "calibration_conflicts": len(conflicts), "n_source_beers": source.beer.nunique(),
                   "positive_reference_conflicts": int(cells.status.eq("reference_conflict").sum()),
                   "negative_reference_cells": int(cells.status.eq("reference_negative").sum()),
                   "n_source_defect_rows": len(source),
                   "source_measurement_states": {str(k): int(v) for k, v in source.status.value_counts().items()}},
        "settings": config, "input_sha256": hashes_before,
        "input_files": {name: str(path.relative_to(root)) for name, path in paths.items()},
        "source_sha256": {f.name: _hash(f) for f in [Path(__file__), CONFIG,
                           Path(__file__).with_name("q3_contract.py"),
                           Path(__file__).with_name("q3_constraints.py"),
                           Path(__file__).with_name("q3_evaluation.py")]},
        "output_sha256": {name: _hash(output_dir / name) for name in outputs},
        "limitations": ["reference zero is not a certified true zero",
                        "source missing/conflicting values cannot certify nonworsening",
                        "stale_hops is a sensory proxy; no chemical threshold",
                        "fold-local statistics and model validation remain required"],
    }
    temporary = output_dir / "q3_run_manifest_E200.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_dir / "q3_run_manifest_E200.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write data-audit tables only; no model fitting")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.write:
        parser.error("use --write to export the descriptive E200 contract")
    output = args.output_dir or args.root / "03_结果/Q3_风格生成"
    result = run_contract(args.root, output)
    print(json.dumps({"status": result["status"], "model_fitted": False, "counts": result["counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
