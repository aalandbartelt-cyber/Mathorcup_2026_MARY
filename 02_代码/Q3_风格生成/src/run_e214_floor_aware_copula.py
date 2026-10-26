"""Formal E214 floor-state Copula candidate with frozen baseline reuse."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from floor_aware_copula import FloorAwareStyleCopula
from q1_interface import FrozenQ1V5OOF
from run_e200_contract import validate_cv
from run_e208_joint_generation import METRICS, metrics_for_scales
from audit_e208 import replacement_decision


CONFIG = Path(__file__).resolve().parents[1] / "configs/E214_floor_aware_copula.yaml"
ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
RESULT = ROOT / "03_结果/Q3_风格生成"


def validate_formal_config(config):
    if config.get("experiment") != "E214":
        raise ValueError("E214 experiment id changed")
    if config.get("methods") != ["floor_copula"]:
        raise ValueError("formal E214 candidate changed")
    if config.get("target_styles") != ["Tripel", "Kriek", "Lambic"]:
        raise ValueError("formal E214 target styles changed")
    if config.get("n_repeats") != 10 or config.get("n_folds") != 3:
        raise ValueError("formal E214 requires frozen 10x3 outer CV")
    seeds = config.get("seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("formal E214 requires five unique seeds")
    if config.get("floor_fields", {}).keys() != {"X4VG"}:
        raise ValueError("only the registered X4VG floor is allowed")
    spec = config["floor_fields"]["X4VG"]
    if float(spec.get("value")) != -2.013157381:
        raise ValueError("registered X4VG floor value changed")
    if spec.get("detection") != "registered_official_floor":
        raise ValueError("floor detection must remain registered, not inferred")
    if config.get("q1_retrained") is not False or config.get("nnls_refit") is not False:
        raise ValueError("E214 cannot refit Q1")
    if config.get("gpu_used") is not False or config.get("kaggle_used_for_training") is not False:
        raise ValueError("formal E214 is a CPU-only local candidate")
    rules = config.get("replacement_rules", {})
    required = {
        "candidate", "raw_w1_max_increase", "raw_energy_max_increase",
        "robust_w1_max_increase", "robust_energy_min_decrease",
        "robust_energy_better_repeats", "robust_coverage_min_increase",
        "robust_log_spread_error_max_increase", "each_style_robust_w1_max_increase",
        "each_style_robust_energy_max_increase", "near_duplicate_fraction_max",
    }
    if set(rules) != required or rules.get("candidate") != "floor_copula":
        raise ValueError("E214 replacement rules changed")
    return True


def floor_specs_from_config(config, features):
    validate_formal_config(config)
    if "X4VG" not in features:
        raise ValueError("registered X4VG field is absent")
    return {features.index("X4VG"): dict(config["floor_fields"]["X4VG"])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_row(repeat, fold, style, seed_index, scope, method, metrics, *, diagnostics=None):
    diagnostics = diagnostics or {}
    row = {
        "repeat": int(repeat), "fold": int(fold), "style": str(style),
        "method": str(method), "alpha": np.nan, "seed_index": int(seed_index),
        "metric_scale": str(scope), "scope": "outer_validation_evaluation",
        "selected_candidate": str(method),
        **diagnostics,
    }
    row.update(metrics)
    return row


def _decision(metrics, repeats, config):
    rules = dict(config["replacement_rules"])
    rules["candidate"] = "joint_selected"
    mapped_metrics = metrics.copy()
    mapped_repeats = repeats.copy()
    mapped_metrics["method"] = mapped_metrics["method"].replace({"floor_copula": "joint_selected"})
    mapped_repeats["method"] = mapped_repeats["method"].replace({"floor_copula": "joint_selected"})
    decision, paired, by_style = replacement_decision(
        mapped_repeats, mapped_metrics, {"replacement_rules": rules}
    )
    decision["candidate_method"] = "floor_copula"
    decision["primary_distribution"] = "floor_copula" if decision["replace_copula"] else "style_copula_shrunk"
    decision["selection_status"] = (
        "candidate_pending_full_fit_and_migration_audit"
        if decision["replace_copula"] else "retain_frozen_distribution"
    )
    decision["q1_retrained"] = False
    decision["nnls_refit"] = False
    decision["gpu_used"] = False
    decision["kaggle_used_for_training"] = False
    decision["physical_safety_certified"] = False
    decision["notes"] = [
        "X4VG 仅按官方登记地板值拆分状态；其他重复最小值未自动解释为检出限",
        "候选仅替换分布生成器，迁移约束、感官画像和缺陷安全仍需单独审计",
    ]
    paired["method_new"] = paired["method_new"].replace({"joint_selected": "floor_copula"})
    by_style.insert(0, "method", "floor_copula")
    return decision, paired, by_style


def run(max_folds=None):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_formal_config(config)
    RESULT.mkdir(parents=True, exist_ok=True)
    q1 = FrozenQ1V5OOF(CANONICAL)
    baseline_manifest_path = RESULT / "q3_run_manifest_E211.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("status") != "complete" or baseline_manifest.get("n_outer_folds") != 30:
        raise ValueError("complete frozen E211 manifest required")
    if baseline_manifest.get("signature", {}).get("q1_final_oof_sha256") != q1.signature:
        raise ValueError("Q1 final OOF identity changed after E211")

    attachment = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    split_path = ROOT / "03_结果/Q1_感官评分/q1_cv_splits.csv"
    chemistry = pd.read_csv(attachment)
    splits = pd.read_csv(split_path)
    validate_cv(chemistry, splits, n_repeats=10, n_folds=3)
    features = [c for c in chemistry.columns if c not in ["beer", "beer_id", "tasting_category_fine"]]
    if len(chemistry) != 250 or len(features) != 231:
        raise ValueError("frozen Q3 input contract changed")
    x = chemistry[features].to_numpy(float)
    styles = chemistry.tasting_category_fine.to_numpy(str)
    floor_specs = floor_specs_from_config(config, features)
    baseline_metrics = pd.read_csv(RESULT / "q3_distribution_metrics_E211.csv")
    baseline_metrics = baseline_metrics[baseline_metrics.method.eq("style_copula_shrunk")].copy()
    if len(baseline_metrics) != 900:
        raise ValueError("frozen E211 Copula baseline must contain 900 rows")
    signature = {
        "input_sha256": {str(p.resolve()): sha(p) for p in [CONFIG, Path(__file__), Path(__file__).with_name("floor_aware_copula.py"), Path(__file__).with_name("q3_evaluation.py"), attachment, split_path, baseline_manifest_path]},
        "q1_final_oof_sha256": q1.signature,
        "e211_manifest_sha256": sha(baseline_manifest_path),
    }
    checkpoint_root = RESULT / "q3_checkpoints_E214"
    checkpoint_root.mkdir(exist_ok=True)
    all_metrics = []
    all_floor = []
    completed = []
    started = time.perf_counter()
    for number, ((repeat, fold), group) in enumerate(splits.groupby(["repeat", "fold"], sort=True), 1):
        if max_folds is not None and number > int(max_folds):
            break
        repeat, fold = int(repeat), int(fold)
        split_id = f"R{repeat:02d}F{fold}"
        slot = checkpoint_root / split_id
        slot.mkdir(exist_ok=True)
        commit = slot / "manifest.json"
        valid = np.sort(group.row_position.to_numpy(int))
        train = np.setdiff1d(np.arange(len(x)), valid)
        if commit.exists():
            meta = json.loads(commit.read_text(encoding="utf-8"))
            if meta.get("signature") != signature:
                raise ValueError("E214 checkpoint signature changed")
            for name, digest in meta["files"].items():
                if sha(slot / name) != digest:
                    raise ValueError("E214 checkpoint damaged")
            all_metrics.append(pd.read_csv(slot / "metrics.csv"))
            all_floor.append(pd.read_csv(slot / "floor_summary.csv"))
            completed.append(split_id)
            print(f"E214 {number}/30 {split_id} cached=True", flush=True)
            continue
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(x[train])
        valid_x = imputer.transform(x[valid])
        model = FloorAwareStyleCopula().fit(train_x, styles[train], floor_specs=floor_specs)
        rows = []
        floor_rows = []
        generated_frames = []
        baseline_fold = baseline_metrics[
            baseline_metrics.repeat.eq(repeat) & baseline_metrics.fold.eq(fold)
        ].copy()
        if len(baseline_fold) != 30:
            raise ValueError("E211 frozen baseline fold coverage changed")
        rows.extend(baseline_fold.to_dict("records"))
        for style_index, style in enumerate(config["target_styles"]):
            local = np.flatnonzero(styles[train] == style)
            valid_style = styles[valid] == style
            real = valid_x[valid_style]
            if len(real) < 2 or len(local) < 2:
                raise ValueError("registered style fold has too few rows")
            local_x = train_x[local]
            diagnostics = model.diagnostics(style)
            for seed_index, base_seed in enumerate(config["seeds"]):
                seed = int(base_seed) + repeat * 1000 + fold * 100 + style_index * 10
                sample = model.generate(style, len(real), seed=seed)
                metrics = metrics_for_scales(real, sample["values"], train_x, local_x)
                for scope, values in metrics.items():
                    rows.append(_metric_row(repeat, fold, style, seed_index, scope, "floor_copula", values, diagnostics=diagnostics))
                floor_column = floor_specs[next(iter(floor_specs))]
                floor_index = next(iter(floor_specs))
                real_floor = float(np.isclose(real[:, floor_index], floor_column["value"], atol=floor_column["tolerance"], rtol=0).mean())
                generated_floor = float(sample["floor_fraction"].get(floor_index, 0.0))
                train_floor = float(np.isclose(local_x[:, floor_index], floor_column["value"], atol=floor_column["tolerance"], rtol=0).mean())
                floor_rows.append({"repeat": repeat, "fold": fold, "style": style, "seed_index": seed_index,
                    "floor_field": features[floor_index], "validation_floor_fraction": real_floor,
                    "training_floor_fraction": train_floor, "generated_floor_fraction": generated_floor,
                    "floor_fraction_absolute_error": abs(generated_floor - real_floor),
                    "n_training_style": len(local), "n_validation_style": len(real)})
                frame = pd.DataFrame(sample["values"], columns=features)
                frame.insert(0, "sample_position", np.arange(len(real)))
                for key, value in [("seed_index", seed_index), ("style", style), ("method", "floor_copula"),
                                   ("fold", fold), ("repeat", repeat), ("scope", "outer_training_generated")]:
                    frame.insert(0, key, value)
                generated_frames.append(frame)
        metrics_path = slot / "metrics.csv"
        floor_path = slot / "floor_summary.csv"
        generated_path = slot / "generated.csv.gz"
        state_path = slot / "model_state.pkl"
        table = pd.DataFrame(rows)
        table.to_csv(metrics_path, index=False)
        pd.DataFrame(floor_rows).to_csv(floor_path, index=False)
        pd.concat(generated_frames, ignore_index=True).to_csv(generated_path, index=False, compression="gzip")
        with state_path.open("wb") as handle:
            pickle.dump({"imputer_statistics": imputer.statistics_, "features": features, "model": model.state_dict()}, handle, protocol=4)
        meta = {
            "signature": signature,
            "train_beer": chemistry.beer.iloc[train].astype(int).tolist(),
            "validation_beer": chemistry.beer.iloc[valid].astype(int).tolist(),
            "files": {p.name: sha(p) for p in [metrics_path, floor_path, generated_path, state_path]},
        }
        commit.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_metrics.append(table)
        all_floor.append(pd.DataFrame(floor_rows))
        completed.append(split_id)
        (RESULT / "q3_progress_E214.json").write_text(json.dumps({"experiment_id": "E214", "status": "running", "completed_folds": len(completed), "n_outer_folds": 30, "last_split_id": split_id, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"E214 {number}/30 {split_id} complete", flush=True)
    if len(completed) < 30:
        return {"status": "checkpoint_only", "completed_folds": len(completed)}
    metrics = pd.concat(all_metrics, ignore_index=True)
    floors = pd.concat(all_floor, ignore_index=True)
    expected = {"style_copula_shrunk": 900, "floor_copula": 900}
    if len(metrics) != 1800 or metrics.groupby("method").size().to_dict() != expected:
        raise ValueError("E214 metric coverage changed")
    repeat_metrics = metrics.groupby(["method", "style", "repeat", "metric_scale"], as_index=False)[METRICS].mean()
    outputs = []
    for name, frame in [("q3_distribution_metrics_E214.csv", metrics), ("q3_metrics_by_repeat_E214.csv", repeat_metrics), ("q3_floor_summary_E214.csv", floors)]:
        path = RESULT / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    decision, paired, by_style = _decision(metrics, repeat_metrics, config)
    decision_path = RESULT / "q3_replacement_decision_E214.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs.append(decision_path)
    for name, frame in [("q3_paired_comparisons_E214.csv", paired), ("q3_comparisons_by_style_E214.csv", by_style)]:
        path = RESULT / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    manifest = {
        "experiment_id": "E214", "status": "complete", "n_outer_folds": 30, "n_repeats": 10,
        "n_seeds": 5, "methods": ["floor_copula", "style_copula_shrunk"], "signature": signature,
        "q1_retrained": False, "nnls_refit": False, "gpu_used": False, "kaggle_used_for_training": False,
        "decision": decision, "output_sha256": {p.name: sha(p) for p in outputs},
    }
    manifest_path = RESULT / "q3_run_manifest_E214.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULT / "q3_progress_E214.json").write_text(json.dumps({"experiment_id": "E214", "status": "complete", "completed_folds": 30, "n_outer_folds": 30}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()
    run(max_folds=args.max_folds)
