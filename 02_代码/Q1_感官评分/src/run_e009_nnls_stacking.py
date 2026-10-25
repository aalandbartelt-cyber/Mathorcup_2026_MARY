"""Gated E009 NNLS runner; E007 must be complete and formally audited first."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit_q1_second_round import compare_full_frames, compare_screening_frames
from data_contract import ATTACHMENTS, RESULT_DIR, load_q1
from nnls_stacking import fit_nnls_stacking
from q1_metrics import matrix_metrics
from run_e005_continuous_ensemble import _candidates, _fit_base, _parameter_key
from second_round_runner import (
    load_config, load_run_checkpoint, make_inner_splits, outer_indices,
    repeats_for_stage, save_prediction_archive, save_run_checkpoint, write_metric_artifacts,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E009_nnls_stacking.yaml"


def require_e007_audit(prior_results: Path) -> dict:
    path = Path(prior_results) / "q1_run_manifest_E007.json"
    if not path.exists():
        raise RuntimeError("E009 cannot start: E007 complete manifest is missing")
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run.get("status") != "complete" or run.get("outer_folds") != 30
            or run.get("repeats") != list(range(10))):
        raise RuntimeError("E009 cannot start: E007 is not a complete frozen 10x3 run")
    path = Path(prior_results) / "q1_second_round_decision.json"
    if not path.exists():
        raise RuntimeError("E009 requires the formal E007 frozen audit")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or "E007" not in audit.get("completed_candidates", []):
        raise RuntimeError("E009 requires the formal E007 frozen audit")
    if audit.get("selected_experiment") is not None:
        raise RuntimeError("a prior candidate was promoted; stop expanding candidates")
    return audit


def fit_outer_nnls(x_train, y_train, x_validation, *, base_config, groups,
                   scopes, shrinkage, inner_splits, seed) -> dict:
    """Rebuild base-model OOF entirely inside outer training, then fit NNLS."""
    y = np.asarray(y_train, dtype=float)
    splits = make_inner_splits(len(y), int(inner_splits), int(seed))
    assignment = np.full(len(y), -1, dtype=int)
    for fold, (_, valid) in enumerate(splits):
        assignment[valid] = fold
    inner_predictions, outer_predictions, selections = [], [], {}
    for model_position, model in enumerate(base_config["base_models"]):
        candidate_rows = []
        for candidate_position, parameter in enumerate(_candidates(base_config, model)):
            prediction = np.full_like(y, np.nan)
            for fold, (train, valid) in enumerate(splits):
                prediction[valid] = _fit_base(
                    model, parameter, x_train[train], y[train], x_train[valid], base_config,
                    random_state=int(seed) + model_position * 100000 + candidate_position * 1000 + fold,
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError("E009 base inner OOF is incomplete")
            candidate_rows.append({"parameter": parameter, "prediction": prediction,
                                   "metrics": matrix_metrics(y, prediction)})
        selected = min(candidate_rows, key=lambda row: _parameter_key(row["metrics"]))
        inner_predictions.append(selected["prediction"])
        outer_predictions.append(_fit_base(
            model, selected["parameter"], x_train, y, x_validation, base_config,
            random_state=int(seed) + model_position * 100000,
        ))
        selections[model] = {
            "parameter": selected["parameter"],
            "candidate_metrics": [{"parameter": row["parameter"], "metrics": row["metrics"]}
                                  for row in candidate_rows],
        }
    inner_stack, outer_stack = np.stack(inner_predictions, axis=-1), np.stack(outer_predictions, axis=-1)
    predictions, parameters = {}, {}
    for scope in scopes:
        fitted = fit_nnls_stacking(inner_stack, y, groups=groups, scope=scope, shrinkage=shrinkage)
        predictions[str(scope)] = fitted.predict(outer_stack)
        parameters[str(scope)] = {"weights": fitted.weights.tolist(),
                                  "intercept": fitted.intercept.tolist(),
                                  "global_weights": fitted.global_weights.tolist()}
    return {"predictions": predictions, "parameters": parameters, "base_selection": selections,
            "inner_fold_assignment": assignment, "inner_prediction_stack": inner_stack,
            "outer_base_stack": outer_stack}


def run(stage: str, *, prior_results: Path) -> dict:
    started = time.perf_counter()
    require_e007_audit(prior_results)
    diagnostic_path = RESULT_DIR / "diagnostics_nnls" / "q1_complementarity_manifest.json"
    if not diagnostic_path.exists():
        raise RuntimeError("E009 requires existing OOF complementarity evidence")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("diagnostic_only") is not True or diagnostic.get("n_repeats") != 10:
        raise RuntimeError("E009 diagnostic evidence is invalid")
    config = load_config(CONFIG_PATH)
    base_path = CONFIG_PATH.parent / config["source_config"]
    source = load_config(base_path)
    if (config["base_models"] != source["base_models"]
            or config["scopes"] != ["shared", "group", "target"] or config["primary_scope"] != "group"
            or float(config["local_shrinkage"]) != .5):
        raise ValueError("E009 source models/scopes/shrinkage differ from registration")
    if stage == "full":
        decision_path = RESULT_DIR / "q1_screening_decision_E009.json"
        if not decision_path.exists() or not json.loads(decision_path.read_text())["promote_to_full"]:
            raise RuntimeError("E009 full requires promoted pre-registered group screening")
    x_frame, y_frame, style = load_q1()
    x, y = x_frame.to_numpy(float), y_frame.to_numpy(float)
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    split_path = RESULT_DIR / "q1_cv_splits.csv"
    group_path = RESULT_DIR / "q1_target_dictionary.csv"
    dictionary = pd.read_csv(group_path).set_index("target_code")
    groups = dictionary.loc[y_frame.columns, config["target_group_column"]].astype(str).tolist()
    splits = pd.read_csv(split_path)
    if (split_path.read_bytes() != (Path(prior_results) / "q1_cv_splits.csv").read_bytes()):
        raise RuntimeError("E009 outer splits disagree with E007")
    repeats = repeats_for_stage(config, stage)
    sources = [CONFIG_PATH, base_path, Path(__file__), Path(__file__).with_name("nnls_stacking.py"),
               Path(__file__).with_name("run_e005_continuous_ensemble.py"),
               Path(__file__).with_name("baseline_models.py"),
               Path(__file__).with_name("ensemble_models.py"),
               ATTACHMENTS / "1.csv", ATTACHMENTS / "4.csv", split_path, group_path, diagnostic_path]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    signature = hashlib.sha256(json.dumps({"stage": stage, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
    suffix = "_screening" if stage == "screening" else ""
    checkpoint_path = RESULT_DIR / f"q1_checkpoint_E009{suffix}.npz"
    predictions = np.full((len(repeats), len(y), y.shape[1], 3), np.nan)
    assignment = np.full((len(repeats), len(y)), -1, dtype=np.int8)
    rows = []
    if checkpoint_path.exists():
        restored = load_run_checkpoint(checkpoint_path, run_signature=signature,
                                       prediction_shape=predictions.shape, assignment_shape=assignment.shape)
        predictions, assignment, rows = restored["predictions"], restored["fold_assignment"], restored["parameter_rows"]
    cache_dir = RESULT_DIR / f"q1_base_cache_E009{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected = [(int(rep), int(fold)) for rep, fold in splits.groupby(["repeat", "fold"]).groups if int(rep) in repeats]
    for number, (repeat, fold) in enumerate(selected, start=1):
        train, valid = outer_indices(splits, repeat, fold, len(y))
        position = repeats.index(repeat)
        if (np.isfinite(predictions[position, valid]).all()
                and np.all(assignment[position, valid] == fold)
                and any(row["repeat"] == repeat and row["fold"] == fold for row in rows)):
            print(f"[{number:02d}/{len(selected)}] repeat={repeat} fold={fold} resumed", flush=True)
            continue
        result = fit_outer_nnls(
            x[train], y[train], x[valid], base_config=source, groups=groups,
            scopes=config["scopes"], shrinkage=float(config["local_shrinkage"]),
            inner_splits=int(config["inner_splits"]), seed=int(config["random_state"]) + repeat * 10 + fold,
        )
        for model_position, scope in enumerate(config["scopes"]):
            predictions[position, valid, :, model_position] = result["predictions"][scope]
        assignment[position, valid] = fold
        rows.append({"repeat": repeat, "fold": fold,
                     "base_selection_json": json.dumps(result["base_selection"], sort_keys=True),
                     "nnls_parameters_json": json.dumps(result["parameters"], sort_keys=True)})
        cache_path = cache_dir / f"repeat_{repeat}_fold_{fold}.npz"
        with cache_path.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, run_signature=np.asarray(signature), train_index=train, validation_index=valid,
                                inner_fold_assignment=result["inner_fold_assignment"],
                                inner_prediction_stack=result["inner_prediction_stack"],
                                outer_base_stack=result["outer_base_stack"])
        cache_path.with_suffix(".tmp").replace(cache_path)
        save_run_checkpoint(checkpoint_path, run_signature=signature, predictions=predictions,
                            fold_assignment=assignment, parameter_rows=rows)
        print(f"[{number:02d}/{len(selected)}] repeat={repeat} fold={fold} complete", flush=True)
    prediction_map = {f"pred_nnls_{scope}": predictions[:, :, :, i] for i, scope in enumerate(config["scopes"])}
    archive_path = RESULT_DIR / f"q1_oof_predictions_E009{suffix}.npz"
    save_prediction_archive(archive_path, y_true=y, beer=chemistry["beer"].to_numpy(int),
                            style=style.to_numpy(str), target_code=np.asarray(y_frame.columns),
                            repeats=np.asarray(repeats), fold_assignment=assignment, predictions=prediction_map)
    summary_path, target_path = write_metric_artifacts("E009", archive_path, list(prediction_map), suffix.removeprefix("_"))
    parameter_path = RESULT_DIR / f"q1_selected_parameters_E009{suffix}.csv"
    pd.DataFrame(rows).to_csv(parameter_path, index=False)
    compare = compare_screening_frames if stage == "screening" else compare_full_frames
    summary, paired, decision = compare(
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv"), pd.read_csv(summary_path),
        experiment_id="E009", model="nnls_group", candidate_prefix="pred_nnls_group:repeat_",
    )
    (RESULT_DIR / f"q1_{stage}_decision_E009.json").write_text(json.dumps(decision, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(RESULT_DIR / f"q1_{stage}_comparison_E009.csv", index=False)
    paired.to_csv(RESULT_DIR / f"q1_{stage}_by_repeat_E009.csv", index=False)
    manifest = {"experiment_id": "E009", "status": "complete", "stage": stage,
                "repeats": repeats, "outer_folds": len(selected), "primary_scope": "group",
                "selection_scope": "group only; shared and target are pre-registered ablations",
                "run_signature": signature, "input_source_hashes": hashes, "gpu_used": False,
                "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
                "outputs": [str(p) for p in [archive_path, summary_path, target_path, parameter_path, checkpoint_path]],
                "formal_robustness_audit_required": True}
    (RESULT_DIR / f"q1_run_manifest_E009{suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=["screening", "full"], default="screening")
    parser.add_argument("--prior-results", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    if not args.execute:
        parser.error("use --execute only after E007 formal audit and complementarity review")
    print(json.dumps(run(args.stage, prior_results=args.prior_results), indent=2))


if __name__ == "__main__":
    main()
