"""Formal replay and frozen-baseline audit for E214."""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from q1_interface import FrozenQ1V5OOF
from run_e208_joint_generation import METRICS, metrics_for_scales
from run_e214_floor_aware_copula import CANONICAL, CONFIG, RESULT, sha
from floor_aware_copula import FloorAwareStyleCopula


def _assert_frame_metrics_equal(left, right, keys):
    merged = left.merge(right, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("metric key coverage changed")
    errors = {}
    for metric in METRICS:
        a = merged[f"{metric}_new"].to_numpy(float)
        b = merged[f"{metric}_old"].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise ValueError(f"finite pattern changed for {metric}")
        error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        if error > 1e-8:
            raise ValueError(f"metric mismatch for {metric}: {error}")
        errors[metric] = error
    return errors


def validate_manifest_structure(manifest, *, expected_folds=30):
    if manifest.get("experiment_id") != "E214" or manifest.get("status") != "complete":
        raise ValueError("complete E214 manifest required")
    if manifest.get("n_outer_folds") != expected_folds:
        raise ValueError("E214 must contain all 30 outer folds")
    if manifest.get("q1_retrained") or manifest.get("nnls_refit") or manifest.get("gpu_used"):
        raise ValueError("E214 interface/GPU flags changed")
    return True


def _replay_checkpoint(slot, meta, chemistry, splits, features):
    name = slot.name
    repeat, fold = int(name[1:3]), int(name[4:])
    valid = np.sort(splits.loc[(splits.repeat == repeat) & (splits.fold == fold), "row_position"].to_numpy(int))
    train = np.setdiff1d(np.arange(len(chemistry)), valid)
    if meta["train_beer"] != chemistry.beer.iloc[train].astype(int).tolist() or meta["validation_beer"] != chemistry.beer.iloc[valid].astype(int).tolist():
        raise ValueError(f"{name}: fold identities changed")
    for filename, digest in meta["files"].items():
        if sha(slot / filename) != digest:
            raise ValueError(f"{name}: checkpoint file hash changed")
    state_bytes = (slot / "model_state.pkl").read_bytes()
    state = pickle.loads(gzip.decompress(state_bytes) if state_bytes[:2] == b"\x1f\x8b" else state_bytes)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(chemistry.iloc[train][features])
    valid_x = imputer.transform(chemistry.iloc[valid][features])
    np.testing.assert_allclose(imputer.statistics_, state["imputer_statistics"], atol=1e-12, rtol=0.0)
    model = FloorAwareStyleCopula.from_state(state["model"])
    generated = pd.read_csv(slot / "generated.csv.gz")
    metrics = pd.read_csv(slot / "metrics.csv")
    floor = pd.read_csv(slot / "floor_summary.csv")
    generated = generated[generated.method.eq("floor_copula")]
    max_error = 0.0
    floor_error = 0.0
    styles = chemistry.tasting_category_fine.to_numpy(str)
    for (style, seed_index), frame in generated.groupby(["style", "seed_index"], sort=True):
        style = str(style)
        style_index = ["Tripel", "Kriek", "Lambic"].index(style)
        seed = 20261025 + int(seed_index) + repeat * 1000 + fold * 100 + style_index * 10
        replay = model.generate(style, len(frame), seed=seed)["values"]
        ordered = frame.sort_values("sample_position")
        error = float(np.max(np.abs(ordered[features].to_numpy(float) - replay)))
        max_error = max(max_error, error)
        if error > 1e-8:
            raise ValueError(f"{name}/{style}/{seed_index}: generated state replay differs")
        real = valid_x[styles[valid] == style]
        local = train_x[styles[train] == style]
        metric_rows = metrics[(metrics.method == "floor_copula") & (metrics["style"] == style) & (metrics.seed_index == seed_index)]
        for scope, values in metrics_for_scales(real, replay, train_x, local).items():
            row = metric_rows[metric_rows.metric_scale == scope]
            if len(row) != 1:
                raise ValueError(f"{name}: missing replay metric row")
            for metric in METRICS:
                if not np.isclose(float(row.iloc[0][metric]), float(values[metric]), atol=1e-8, rtol=0.0, equal_nan=True):
                    raise ValueError(f"{name}: replay metric differs for {metric}")
        floor_row = floor[(floor["style"] == style) & (floor.seed_index == seed_index)]
        if len(floor_row) != 1:
            raise ValueError(f"{name}: missing floor summary")
        floor_index = features.index("X4VG")
        observed = float(np.isclose(replay[:, floor_index], -2.013157381, atol=1e-9, rtol=0).mean())
        floor_error = max(floor_error, abs(observed - float(floor_row.generated_floor_fraction.iloc[0])))
    return {"fold_name": name, "max_spectrum_error": max_error, "max_floor_summary_error": floor_error, "saved_state_replay_without_training": True}


def run():
    manifest_path = RESULT / "q3_run_manifest_E214.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_structure(manifest)
    for path, digest in manifest["signature"]["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"E214 frozen input changed: {path}")
    q1 = FrozenQ1V5OOF(CANONICAL)
    if manifest["signature"]["q1_final_oof_sha256"] != q1.signature:
        raise ValueError("Q1 final OOF interface changed")
    for name, digest in manifest.get("output_sha256", {}).items():
        if sha(RESULT / name) != digest:
            raise ValueError(f"E214 output changed: {name}")
    e211 = json.loads((RESULT / "q3_run_manifest_E211.json").read_text(encoding="utf-8"))
    if sha(RESULT / "q3_distribution_metrics_E211.csv") != e211["output_sha256"]["q3_distribution_metrics_E211.csv"]:
        raise ValueError("E211 frozen baseline metrics changed")
    chemistry = pd.read_csv(CANONICAL / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv")
    splits = pd.read_csv(CANONICAL / "03_结果/Q1_感官评分/q1_cv_splits.csv")
    features = [c for c in chemistry.columns if c not in ["beer", "beer_id", "tasting_category_fine"]]
    reconstruction = []
    for repeat in range(10):
        for fold in range(3):
            name = f"R{repeat:02d}F{fold}"
            slot = RESULT / "q3_checkpoints_E214" / name
            meta = json.loads((slot / "manifest.json").read_text(encoding="utf-8"))
            if meta.get("signature") != manifest["signature"]:
                raise ValueError(f"{name}: checkpoint signature changed")
            reconstruction.append(_replay_checkpoint(slot, meta, chemistry, splits, features))
    metrics = pd.read_csv(RESULT / "q3_distribution_metrics_E214.csv")
    if len(metrics) != 1800 or metrics.duplicated(["repeat", "fold", "style", "method", "seed_index", "metric_scale"]).any():
        raise ValueError("E214 metric coverage incomplete")
    old = pd.read_csv(RESULT / "q3_distribution_metrics_E211.csv")
    new_base = metrics[metrics.method.eq("style_copula_shrunk")].copy()
    old_base = old[old.method.eq("style_copula_shrunk")].copy()
    baseline_errors = _assert_frame_metrics_equal(
        new_base, old_base, ["repeat", "fold", "style", "method", "seed_index", "metric_scale"]
    )
    report = {
        "experiment_id": "E214", "status": "complete", "n_outer_folds": 30,
        "n_metric_rows": len(metrics), "frozen_baseline_reproduced": True,
        "baseline_max_absolute_metric_error": baseline_errors,
        "checkpoint_replay": reconstruction,
        "max_spectrum_replay_error": float(max(x["max_spectrum_error"] for x in reconstruction)),
        "max_floor_summary_error": float(max(x["max_floor_summary_error"] for x in reconstruction)),
        "q1_retrained": False, "nnls_refit": False, "gpu_used": False,
        "n_frozen_inputs_verified": len(manifest["signature"]["input_sha256"]),
    }
    out = RESULT / "q3_e214_audit_manifest.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
