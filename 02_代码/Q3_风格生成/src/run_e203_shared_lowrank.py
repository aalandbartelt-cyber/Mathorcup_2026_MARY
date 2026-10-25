"""E203 shared-low-rank generator with fold-local tuning and floor states."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from lowrank_generator import METHODS, SharedLowRankStyleGenerator
from q1_interface import FrozenQ1V5OOF
from q3_evaluation import distribution_diagnostics
from run_e200_contract import validate_cv


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
RESULT = ROOT / "03_结果/Q3_风格生成"
CONFIG = Path(__file__).resolve().parents[1] / "configs/E203_shared_lowrank.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_e202_gate(result_dir, *, expected_q1_oof_sha256):
    path = Path(result_dir) / "q3_run_manifest_E202.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "E202"
        or manifest.get("status") != "complete"
        or manifest.get("n_outer_folds") != 30
    ):
        raise ValueError("complete 30-fold E202 manifest required")
    if manifest.get("signature", {}).get("q1_final_oof_sha256") != expected_q1_oof_sha256:
        raise ValueError("Q1 final OOF identity changed after E202")
    return {"n_outer_folds": 30, "e202_manifest_sha256": sha(path)}


def validate_formal_config(config):
    if config.get("methods") != list(METHODS):
        raise ValueError("formal E203 methods changed")
    if config.get("target_styles") != ["Tripel", "Kriek", "Lambic"]:
        raise ValueError("formal E203 target styles changed")
    if config.get("n_repeats") != 10 or config.get("n_folds") != 3:
        raise ValueError("formal E203 requires frozen 10x3 outer CV")
    if len(config.get("seeds", [])) != 5 or len(set(config["seeds"])) != 5:
        raise ValueError("formal E203 requires five unique seeds")
    if config.get("inner_splits") != 2:
        raise ValueError("formal E203 requires two training-only inner splits")
    ranks = config.get("rank_candidates", [])
    kappas = config.get("covariance_kappa_candidates", [])
    if not ranks or not kappas or any(int(rank) < 1 for rank in ranks) or any(float(kappa) < 0 for kappa in kappas):
        raise ValueError("rank and covariance shrinkage grids required")
    floor = config.get("floor_fields", {})
    if set(floor) != {"X4VG"}:
        raise ValueError("registered observed-floor field changed")
    if config.get("gpu_used") is not False:
        raise ValueError("formal E203 is a CPU statistical model")


def floor_specs_from_config(config, features):
    specs = {}
    for name, values in config["floor_fields"].items():
        if name not in features:
            raise ValueError(f"registered floor field missing: {name}")
        specs[features.index(name)] = dict(values)
    return specs


def fit_model(train_x, train_styles, *, rank, covariance_kappa, config, floor_specs):
    return SharedLowRankStyleGenerator().fit(
        train_x,
        train_styles,
        rank=int(rank),
        covariance_kappa=float(covariance_kappa),
        mean_kappa=float(config["mean_kappa"]),
        floor_specs=floor_specs,
        random_state=int(config["inner_seed"]),
    )


def select_hyperparameters(train_x, train_styles, config, floor_specs):
    """Select rank and residual shrinkage using only the current outer training fold."""
    candidates = [
        (int(rank), float(kappa))
        for rank in config["rank_candidates"]
        for kappa in config["covariance_kappa_candidates"]
        if int(rank) < min(train_x.shape)
    ]
    if not candidates:
        raise ValueError("no admissible low-rank candidates")
    splitter = KFold(
        n_splits=int(config["inner_splits"]),
        shuffle=True,
        random_state=int(config["inner_seed"]),
    )
    scores = []
    for rank, kappa in candidates:
        fold_scores = []
        for inner_fold, (inner_train, inner_valid) in enumerate(splitter.split(train_x)):
            inner_train_styles = train_styles[inner_train]
            model = fit_model(
                train_x[inner_train],
                inner_train_styles,
                rank=rank,
                covariance_kappa=kappa,
                config=config,
                floor_specs=floor_specs,
            )
            scale = np.std(train_x[inner_train], axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            style_scores = []
            for style_index, style in enumerate(config["target_styles"]):
                real = train_x[inner_valid][train_styles[inner_valid] == style]
                local = train_x[inner_train][inner_train_styles == style]
                if len(real) < 2 or len(local) < 2:
                    continue
                generated = model.generate(
                    style,
                    len(real),
                    method="style_residual_floor",
                    seed=int(config["inner_seed"]) + inner_fold * 100 + style_index,
                )["values"]
                metrics = distribution_diagnostics(real, generated, local, scale)
                style_scores.append(metrics["mean_w1"] + metrics["energy_v"])
            if style_scores:
                fold_scores.append(float(np.mean(style_scores)))
        score = float(np.mean(fold_scores)) if len(fold_scores) == int(config["inner_splits"]) else np.inf
        scores.append({"rank": rank, "covariance_kappa": kappa, "selection_score": score})
    table = pd.DataFrame(scores).sort_values(
        ["selection_score", "rank", "covariance_kappa"], kind="mergesort"
    )
    if not np.isfinite(table.iloc[0]["selection_score"]):
        raise ValueError("training-only inner selection produced no valid candidate")
    selected = table.iloc[0]
    return int(selected["rank"]), float(selected["covariance_kappa"]), table


def compose_metric_row(base, metrics):
    metrics = dict(metrics)
    diagnostic_scope = metrics.pop("scope", None)
    metrics.pop("w1_by_feature", None)
    return {**metrics, **base, "evaluation_scope": "outer_validation_evaluation", "diagnostic_scope": diagnostic_scope}


def run():
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_formal_config(config)
    RESULT.mkdir(parents=True, exist_ok=True)
    q1 = FrozenQ1V5OOF(CANONICAL)
    entry_gate = verify_e202_gate(RESULT, expected_q1_oof_sha256=q1.signature)
    attachment = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    split_path = ROOT / "03_结果/Q1_感官评分/q1_cv_splits.csv"
    chemistry = pd.read_csv(attachment)
    splits = pd.read_csv(split_path)
    validate_cv(chemistry, splits, n_repeats=10, n_folds=3)
    features = [column for column in chemistry.columns if column not in ["beer", "beer_id", "tasting_category_fine"]]
    if len(chemistry) != 250 or len(features) != 231:
        raise ValueError("Q3 input contract changed")
    if any(field not in features for field in config["defect_fields"]):
        raise ValueError("registered defect field missing")
    defect_indices = [features.index(field) for field in config["defect_fields"]]
    floor_specs = floor_specs_from_config(config, features)
    x = chemistry[features].to_numpy(float)
    styles = chemistry.tasting_category_fine.to_numpy(str)
    signature = {
        "input_sha256": sha(attachment),
        "cv_sha256": sha(split_path),
        "config_sha256": sha(CONFIG),
        "source_sha256": {
            path.name: sha(path)
            for path in [Path(__file__), Path(__file__).with_name("lowrank_generator.py"), Path(__file__).with_name("q3_evaluation.py")]
        },
        "q1_final_model": q1.registry["final_model"],
        "q1_final_oof_sha256": q1.signature,
        "entry_gate": entry_gate,
    }
    checkpoint_root = RESULT / "q3_checkpoints_E203"
    checkpoint_root.mkdir(exist_ok=True)
    tables = []
    parameter_tables = []
    generated_files = []
    for number, ((repeat, fold), group) in enumerate(splits.groupby(["repeat", "fold"], sort=True), start=1):
        repeat, fold = int(repeat), int(fold)
        split_id = f"R{repeat:02d}F{fold}"
        slot = checkpoint_root / split_id
        slot.mkdir(exist_ok=True)
        checkpoint = slot / "manifest.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if record.get("signature") != signature:
                raise ValueError("Q3 E203 checkpoint signature changed")
            for filename, digest in record["files"].items():
                if sha(slot / filename) != digest:
                    raise ValueError("Q3 E203 checkpoint damaged")
            tables.append(pd.read_csv(slot / "metrics.csv"))
            parameter_tables.append(pd.read_csv(slot / "selection.csv"))
            generated_files.append(slot / "generated.csv.gz")
            print(f"[{number:02d}/30] {split_id} resumed", flush=True)
            continue

        valid = np.sort(group.row_position.to_numpy(int))
        train = np.setdiff1d(np.arange(len(x)), valid)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(x[train])
        valid_x = imputer.transform(x[valid])
        selected_rank, selected_kappa, selection = select_hyperparameters(
            train_x, styles[train], config, floor_specs
        )
        selection.insert(0, "fold", fold)
        selection.insert(0, "repeat", repeat)
        selection["selected"] = (
            (selection["rank"] == selected_rank)
            & (selection["covariance_kappa"] == selected_kappa)
        )
        model = fit_model(
            train_x,
            styles[train],
            rank=selected_rank,
            covariance_kappa=selected_kappa,
            config=config,
            floor_specs=floor_specs,
        )
        scale = np.std(train_x, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        rows = []
        generated_tables = []
        for style_index, style in enumerate(config["target_styles"]):
            local = np.flatnonzero(styles[train] == style)
            real = valid_x[styles[valid] == style]
            if len(real) < 2 or len(local) < 2:
                raise ValueError("style fold too small for registered diagnostics")
            diagnostic = model.diagnostics(style)
            validation_floor = {
                column: float(np.mean(np.isclose(real[:, column], spec["value"], atol=spec["tolerance"], rtol=0)))
                for column, spec in floor_specs.items()
            }
            for seed_index, base_seed in enumerate(config["seeds"]):
                seed = int(base_seed) + repeat * 1000 + fold * 100 + style_index * 10
                for method in config["methods"]:
                    sample = model.generate(style, len(real), method=method, seed=seed)
                    metrics = distribution_diagnostics(real, sample["values"], train_x[local], scale)
                    defect_w1 = float(
                        np.mean(
                            [
                                wasserstein_distance(real[:, index] / scale[index], sample["values"][:, index] / scale[index])
                                for index in defect_indices
                            ]
                        )
                    )
                    floor_column = next(iter(floor_specs))
                    generated_floor = float(
                        np.mean(
                            np.isclose(
                                sample["values"][:, floor_column],
                                floor_specs[floor_column]["value"],
                                atol=floor_specs[floor_column]["tolerance"],
                                rtol=0,
                            )
                        )
                    )
                    base = {
                        "repeat": repeat,
                        "fold": fold,
                        "style": style,
                        "method": method,
                        "seed_index": seed_index,
                        "status": "evaluated",
                        "selected_rank": selected_rank,
                        "selected_covariance_kappa": selected_kappa,
                        "defect_mean_w1": defect_w1,
                        "generated_floor_fraction": generated_floor,
                        "validation_floor_fraction": validation_floor[floor_column],
                        "floor_fraction_absolute_error": abs(generated_floor - validation_floor[floor_column]),
                        "pre_clip_cell_fraction": sample["pre_clip_cell_fraction"],
                        "pre_clip_row_fraction": sample["pre_clip_row_fraction"],
                        **diagnostic,
                    }
                    rows.append(compose_metric_row(base, metrics))
                    generated = pd.DataFrame(sample["values"], columns=features)
                    generated.insert(0, "sample_position", np.arange(len(real)))
                    for key, value in [
                        ("style", style),
                        ("seed_index", seed_index),
                        ("method", method),
                        ("fold", fold),
                        ("repeat", repeat),
                        ("scope", "outer_training_generated"),
                    ]:
                        generated.insert(0, key, value)
                    generated_tables.append(generated)
        metrics_table = pd.DataFrame(rows)
        metrics_path = slot / "metrics.csv"
        selection_path = slot / "selection.csv"
        generated_path = slot / "generated.csv.gz"
        metrics_table.to_csv(metrics_path, index=False)
        selection.to_csv(selection_path, index=False)
        pd.concat(generated_tables, ignore_index=True).to_csv(generated_path, index=False, compression="gzip")
        record = {
            "signature": signature,
            "train_beer": chemistry.beer.iloc[train].astype(int).tolist(),
            "validation_beer": chemistry.beer.iloc[valid].astype(int).tolist(),
            "selected_rank": selected_rank,
            "selected_covariance_kappa": selected_kappa,
            "files": {path.name: sha(path) for path in [metrics_path, selection_path, generated_path]},
        }
        temporary = slot / "manifest.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(checkpoint)
        tables.append(metrics_table)
        parameter_tables.append(selection)
        generated_files.append(generated_path)
        progress = {
            "experiment_id": "E203",
            "status": "running",
            "completed_folds": number,
            "n_outer_folds": 30,
            "last_split_id": split_id,
            "elapsed_seconds": time.perf_counter() - started,
        }
        temporary = RESULT / "q3_progress_E203.json.tmp"
        temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(RESULT / "q3_progress_E203.json")
        print(f"[{number:02d}/30] {split_id} complete rank={selected_rank} kappa={selected_kappa:g}", flush=True)

    metrics = pd.concat(tables, ignore_index=True)
    metrics_path = RESULT / "q3_distribution_metrics_E203.csv"
    metrics.to_csv(metrics_path, index=False)
    parameters = pd.concat(parameter_tables, ignore_index=True)
    parameters_path = RESULT / "q3_hyperparameter_selection_E203.csv"
    parameters.to_csv(parameters_path, index=False)
    names = [
        "mean_w1",
        "energy_v",
        "coverage",
        "near_duplicate_fraction",
        "spearman_pair_rmse",
        "defect_mean_w1",
        "floor_fraction_absolute_error",
        "pre_clip_cell_fraction",
        "pre_clip_row_fraction",
    ]
    repeat_table = metrics.groupby(["method", "style", "repeat"], as_index=False)[names].mean()
    repeat_path = RESULT / "q3_metrics_by_repeat_E203.csv"
    repeat_table.to_csv(repeat_path, index=False)
    summary = repeat_table.groupby(["method", "style"])[names].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary_path = RESULT / "q3_baseline_summary_E203.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    decision = {
        "status": "shared_lowrank_complete_no_migration",
        "formal_methods": config["methods"],
        "sensory_profiles_generated": False,
        "chemical_safety_certified": False,
        "migration_performed": False,
        "q1_retrained": False,
        "gpu_used": False,
        "notes": [
            "秩与风格残差收缩只在当前外层训练折内选择",
            "地板状态表示观测地板而非真实零或已知检出限",
            "E204入口需与E202统计基线正式比较后另行判定",
        ],
    }
    decision_path = RESULT / "q3_baseline_decision_E203.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs = [metrics_path, parameters_path, repeat_path, summary_path, decision_path]
    manifest = {
        "experiment_id": "E203",
        "status": "complete",
        "n_outer_folds": 30,
        "target_styles": config["target_styles"],
        "methods": config["methods"],
        "signature": signature,
        "gpu_used": False,
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregation": "first average folds/seeds within repeat/style, then mean and SD across repeats; repeats and seeds are not independent samples",
        "decision": decision,
        "checkpoint_files": [str(path) for path in generated_files],
        "output_sha256": {path.name: sha(path) for path in outputs},
    }
    manifest_path = RESULT / "q3_run_manifest_E203.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress = json.loads((RESULT / "q3_progress_E203.json").read_text(encoding="utf-8"))
    progress.update(status="complete", completed_folds=30)
    (RESULT / "q3_progress_E203.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run()
