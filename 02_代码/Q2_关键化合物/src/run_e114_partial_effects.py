"""Run E114 fold-local Ledoit-Wolf target partial effects."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer

from partial_graph import FoldGaussianizer
from q2_contract import Q1_RESULT_DIR, RESULT_DIR, load_q2
from q2_pipeline import load_frozen_outer_folds
from target_partial_effects import (
    aggregate_partial_effect_rankings,
    ledoit_wolf_target_partial,
    rank_signed_effects,
)


EXPERIMENT_ID = "E114"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E114_partial_effects.yaml"


def _load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if str(config.get("experiment")) != EXPERIMENT_ID:
        raise ValueError("E114 config experiment ID is invalid")
    if config.get("outer_folds") != "frozen_10x3":
        raise ValueError("E114 must use the frozen 10x3 outer folds")
    if config.get("candidate_scope") != "strict_compounds_218":
        raise ValueError("E114 must use the strict 218-compound scope")
    return config


def _strict_codes(x: pd.DataFrame) -> list[str]:
    dictionary = pd.read_csv(RESULT_DIR / "q2_candidate_dictionary.csv")
    eligible = dictionary["eligible_strict"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().map({"true": True, "false": False})
    codes = dictionary.loc[eligible, "feature_code"].astype(str).tolist()
    if len(codes) != 218 or len(set(codes)) != 218:
        raise ValueError(f"expected 218 unique strict compounds, got {len(codes)}")
    if set(codes) - set(x.columns.astype(str)):
        raise ValueError("strict compound dictionary disagrees with Q2 predictors")
    return codes


def _fold_effects(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_codes: list[str],
    target_codes: list[str],
    repeat: int,
    fold: int,
    split_id: str,
    seed: int,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(x_train)
    transformed = FoldGaussianizer(random_state=seed).fit_transform(imputed)
    target_mean = np.mean(y_train, axis=0)
    target_scale = np.std(y_train, axis=0, ddof=0)
    if np.any(target_scale <= 0) or not np.isfinite(target_scale).all():
        raise RuntimeError("E114 encountered a constant or invalid training target")
    standardized_targets = (y_train - target_mean) / target_scale
    frames = []
    for target_position, target_code in enumerate(target_codes):
        effect = ledoit_wolf_target_partial(
            transformed, standardized_targets[:, target_position]
        )
        ranked = rank_signed_effects(effect, feature_codes)
        ranked.insert(0, "target_code", target_code)
        ranked.insert(0, "split_id", split_id)
        ranked.insert(0, "fold", int(fold))
        ranked.insert(0, "repeat", int(repeat))
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def _anchor_recovery(rankings: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(RESULT_DIR / "q2_anchor_recovery_E107.csv")
    anchors = reference[["target_code", "target_name_en", "direct_anchor"]].drop_duplicates()
    anchors = anchors.loc[anchors["direct_anchor"].notna()].copy()
    selected = rankings.merge(
        anchors,
        left_on=["target_code", "feature_code"],
        right_on=["target_code", "direct_anchor"],
        how="right",
        validate="many_to_one",
    )
    return selected[
        [
            "target_code",
            "target_name_en",
            "direct_anchor",
            "partial_effect_rank",
            "median_fold_rank",
            "top3_frequency",
            "top10_frequency",
            "median_effect",
            "direction",
            "sign_consistency",
        ]
    ].sort_values("target_code", kind="mergesort")


def run_e114(*, output_dir: Path = RESULT_DIR) -> dict:
    started = time.perf_counter()
    config = _load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, _, _ = load_q2()
    feature_codes = _strict_codes(x)
    target_codes = y.columns.astype(str).tolist()
    x_values = x[feature_codes].to_numpy(float)
    y_values = y.to_numpy(float)
    folds = load_frozen_outer_folds(len(x), repeats=10)
    checkpoint_dir = output_dir / "q2_checkpoint_E114"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_frames = []
    for number, outer in enumerate(folds, start=1):
        checkpoint = checkpoint_dir / f"repeat_{outer.repeat}_fold_{outer.fold}.csv.gz"
        if checkpoint.exists():
            frame = pd.read_csv(checkpoint)
            expected = len(feature_codes) * len(target_codes)
            if len(frame) != expected:
                raise RuntimeError(f"invalid E114 checkpoint row count: {checkpoint}")
            print(f"[{number:02d}/{len(folds)}] {outer.split_id} resumed", flush=True)
        else:
            frame = _fold_effects(
                x_values[outer.train_index],
                y_values[outer.train_index],
                feature_codes=feature_codes,
                target_codes=target_codes,
                repeat=outer.repeat,
                fold=outer.fold,
                split_id=outer.split_id,
                seed=int(config["random_state"]) + outer.repeat * 100 + outer.fold,
            )
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(checkpoint)
            print(f"[{number:02d}/{len(folds)}] {outer.split_id} complete", flush=True)
        fold_frames.append(frame)

    fold_effects = pd.concat(fold_frames, ignore_index=True)
    expected_rows = len(folds) * len(feature_codes) * len(target_codes)
    if len(fold_effects) != expected_rows:
        raise RuntimeError(f"E114 expected {expected_rows} fold rows, got {len(fold_effects)}")
    rankings = aggregate_partial_effect_rankings(fold_effects)
    if len(rankings) != len(feature_codes) * len(target_codes):
        raise RuntimeError("E114 aggregate ranking is incomplete")
    anchors = _anchor_recovery(rankings)

    paths = {
        "fold_effects": output_dir / "q2_fold_partial_effects_E114.csv.gz",
        "rankings": output_dir / "q2_partial_effect_rankings_E114.csv",
        "anchors": output_dir / "q2_anchor_recovery_E114.csv",
        "manifest": output_dir / "q2_run_manifest_E114.json",
    }
    fold_effects.to_csv(paths["fold_effects"], index=False, compression="gzip")
    rankings.to_csv(paths["rankings"], index=False)
    anchors.to_csv(paths["anchors"], index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "n_outer_folds": len(folds),
        "n_features": len(feature_codes),
        "n_targets": len(target_codes),
        "fold_rows": len(fold_effects),
        "ranking_rows": len(rankings),
        "random_state": int(config["random_state"]),
        "gpu_used": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "cv_sha256": hashlib.sha256(
            (Q1_RESULT_DIR / "q1_cv_splits.csv").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "outputs": [str(path) for key, path in paths.items() if key != "manifest"],
        "checkpoint_directory": str(checkpoint_dir),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(run_e114(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
