"""E116 runner; actual-data execution requires --execute and E113--E115 first."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from grouped_multitask import fit_grouped_multitask, aggregate_grouped_rankings
from q2_contract import ATTACHMENTS, Q1_RESULT_DIR, RESULT_DIR, load_q2
from q2_pipeline import load_frozen_outer_folds, metrics_by_target
from run_e114_partial_effects import _strict_codes
from target_partial_effects import load_target_groups, rank_signed_effects

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E116_grouped_multitask.yaml"


def fit_outer_fold(x_train, y_train, x_validation, *, feature_codes, target_codes,
                   groups, config) -> dict:
    """Fit and rank with outer-training rows only; no validation labels accepted."""
    fitted = fit_grouped_multitask(
        x_train, y_train, target_codes=target_codes, groups=groups,
        alpha_grid=config["alpha_grid"], l1_ratio_grid=config["l1_ratio_grid"],
        inner_splits=int(config["inner_splits"]),
        random_state=int(config["random_state"]),
    )
    frames = []
    for position, code in enumerate(target_codes):
        frame = rank_signed_effects(fitted.coefficients[position], feature_codes)
        frame.insert(0, "target_code", code)
        frames.append(frame)
    return {"prediction": fitted.predict(x_validation),
            "coefficients": fitted.coefficients,
            "rankings": pd.concat(frames, ignore_index=True),
            "selection_rows": fitted.selection_rows}


def save_fold_checkpoint(path, *, signature, prediction, coefficients, selection_rows):
    path = Path(path)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=np.asarray(signature),
                            prediction=prediction, coefficients=coefficients,
                            selection_json=np.asarray(json.dumps(selection_rows, sort_keys=True)))
    temporary.replace(path)


def load_fold_checkpoint(path, *, signature, prediction_shape, coefficient_shape):
    with np.load(path, allow_pickle=False) as archive:
        prediction, coef = archive["prediction"].copy(), archive["coefficients"].copy()
        if (str(archive["signature"].item()) != signature
                or prediction.shape != prediction_shape or coef.shape != coefficient_shape
                or not np.isfinite(prediction).all() or not np.isfinite(coef).all()):
            raise RuntimeError(f"E116 checkpoint signature/shape/value mismatch: {path}")
        return {"prediction": prediction, "coefficients": coef,
                "selection_rows": json.loads(str(archive["selection_json"].item()))}


def run_e116() -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if (config.get("experiment") != "E116"
            or config.get("outer_folds") != "frozen_10x3"
            or config.get("candidate_scope") != "strict_compounds_218"):
        raise ValueError("E116 requires its registered experiment/fold/candidate scope")
    for experiment in ["E113", "E114", "E115"]:
        path = RESULT_DIR / f"q2_run_manifest_{experiment}.json"
        if not path.exists():
            raise RuntimeError(f"E116 requires prior completed {experiment}: {path}")
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "complete":
            raise RuntimeError(f"E116 prerequisite is incomplete: {path}")
    x, y, style, beer = load_q2()
    features = _strict_codes(x)
    targets = y.columns.astype(str).tolist()
    groups = load_target_groups(CONFIG_PATH)
    sources = [CONFIG_PATH, Path(__file__), Path(__file__).with_name("grouped_multitask.py"),
               Path(__file__).with_name("target_partial_effects.py"),
               Path(__file__).with_name("run_e114_partial_effects.py"),
               Q1_RESULT_DIR / "q1_cv_splits.csv", RESULT_DIR / "q2_candidate_dictionary.csv",
               ATTACHMENTS / "1.csv", ATTACHMENTS / "4.csv"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    folds = load_frozen_outer_folds(len(x), repeats=10)
    predictions = np.full((10, len(x), len(targets)), np.nan)
    assignments = np.full((10, len(x)), -1, dtype=int)
    checkpoint_dir = RESULT_DIR / "q2_checkpoint_E116"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    effect_frames, parameter_rows = [], []
    for number, outer in enumerate(folds, start=1):
        path = checkpoint_dir / f"repeat_{outer.repeat}_fold_{outer.fold}.npz"
        if path.exists():
            result = load_fold_checkpoint(
                path, signature=signature, prediction_shape=(len(outer.validation_index), len(targets)),
                coefficient_shape=(len(targets), len(features)),
            )
        else:
            result = fit_outer_fold(
                x[features].iloc[outer.train_index].to_numpy(float),
                y.iloc[outer.train_index].to_numpy(float),
                x[features].iloc[outer.validation_index].to_numpy(float),
                feature_codes=features, target_codes=targets, groups=groups,
                config={**config, "random_state": int(config["random_state"])
                        + 100 * outer.repeat + outer.fold},
            )
            save_fold_checkpoint(path, signature=signature, prediction=result["prediction"],
                                 coefficients=result["coefficients"],
                                 selection_rows=result["selection_rows"])
        predictions[outer.repeat, outer.validation_index] = result["prediction"]
        assignments[outer.repeat, outer.validation_index] = outer.fold
        for position, target in enumerate(targets):
            frame = rank_signed_effects(result["coefficients"][position], features)
            frame.insert(0, "target_code", target)
            frame.insert(0, "fold", outer.fold)
            frame.insert(0, "repeat", outer.repeat)
            effect_frames.append(frame)
        parameter_rows.append({"repeat": outer.repeat, "fold": outer.fold,
                               "selection_json": json.dumps(result["selection_rows"], sort_keys=True)})
        print(f"[{number:02d}/30] {outer.split_id} complete", flush=True)
    if not np.isfinite(predictions).all() or (assignments < 0).any():
        raise RuntimeError("E116 OOF predictions are incomplete")
    metrics = []
    for repeat in range(10):
        frame = metrics_by_target(y.to_numpy(float), predictions[repeat], targets, "grouped_multitask")
        frame.insert(0, "repeat", repeat)
        metrics.append(frame)
    effects = pd.concat(effect_frames, ignore_index=True)
    rankings = aggregate_grouped_rankings(effects)
    paths = {
        "oof": RESULT_DIR / "q2_oof_predictions_E116.npz",
        "fold_effects": RESULT_DIR / "q2_fold_grouped_effects_E116.csv.gz",
        "rankings": RESULT_DIR / "q2_grouped_rankings_E116.csv",
        "metrics": RESULT_DIR / "q2_metrics_by_target_E116.csv",
        "parameters": RESULT_DIR / "q2_selected_parameters_E116.csv",
        "decision": RESULT_DIR / "q2_decision_E116.json",
    }
    np.savez_compressed(paths["oof"], y_true=y.to_numpy(float), prediction=predictions,
                        beer=beer.to_numpy(int), style=style.to_numpy(str),
                        target_code=np.asarray(targets), repeat=np.arange(10),
                        fold_assignment=assignments)
    effects.to_csv(paths["fold_effects"], index=False, compression="gzip")
    rankings.to_csv(paths["rankings"], index=False)
    pd.concat(metrics, ignore_index=True).to_csv(paths["metrics"], index=False)
    pd.DataFrame(parameter_rows).to_csv(paths["parameters"], index=False)
    paths["decision"].write_text(json.dumps({
        "status": "awaiting_stability_and_anchor_audit", "promoted": False,
        "reason": "predictive metrics alone cannot satisfy E116 gate; compare E117 stability and anchors",
        "gate_config": config["promotion_tolerance"], "metric_path": str(paths["metrics"]),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"experiment_id": "E116", "status": "complete", "promoted": False,
                "n_outer_folds": 30, "n_features": len(features), "n_targets": len(targets),
                "prediction_shape": list(predictions.shape), "input_source_hashes": hashes,
                "run_signature": signature, "gpu_used": False,
                "elapsed_seconds": time.perf_counter() - started,
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "outputs": [str(path) for path in paths.values()]}
    (RESULT_DIR / "q2_run_manifest_E116.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run registered actual-data E116 after prerequisites")
    args = parser.parse_args()
    if not args.execute:
        parser.error("actual-data execution is opt-in; use --execute only after authorization")
    print(json.dumps(run_e116(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
