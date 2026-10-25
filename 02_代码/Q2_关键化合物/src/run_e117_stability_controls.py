"""Run Q2 complementary-pairs stability selection and null controls.

The committed configuration is fail-closed (``formal_run: false``).  The
default CLI performs a read-only readiness audit.  Formal execution is only
possible after the E113 and final-NNLS E115 manifests are complete and the
configuration is deliberately switched after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from complementary_stability import (
    complementary_pair_jaccard,
    confirmed_threshold,
    mb_pfer_bound,
    selection_probability,
    stratified_complementary_pairs,
)
from q2_contract import Q1_RESULT_DIR, RESULT_DIR, load_q2


EXPERIMENT_ID = "E117"
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "02_代码" / "Q2_关键化合物" / "configs" / "E117_stability_controls.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("experiment") != EXPERIMENT_ID:
        raise ValueError("configuration experiment must be E117")
    if config.get("candidate_scope") != "strict_compounds_218":
        raise ValueError("E117 candidate scope must remain strict_compounds_218")
    if config.get("target_permutation_scope") != "within_style":
        raise ValueError("target permutations must remain within style")
    if not config.get("ranking_recomputed_each_half", False):
        raise ValueError("each complementary half must recompute its ranking")
    if config.get("full_data_graph_reused") is not False:
        raise ValueError("E117 must not reuse a full-data graph")
    thresholds = [float(value) for value in config["stability_thresholds"]]
    if thresholds != sorted(set(thresholds)) or not all(0.5 < value <= 1 for value in thresholds):
        raise ValueError("stability thresholds must be unique, increasing, and >0.5")
    return config


def audit_inputs(result_dir: Path = RESULT_DIR, root: Path = ROOT) -> dict:
    """Read-only gate that rejects pilot/incomplete upstream artifacts."""

    result_dir, root = Path(result_dir), Path(root)
    problems: list[str] = []

    candidates_path = result_dir / "q2_candidate_dictionary.csv"
    if candidates_path.is_file():
        candidates = pd.read_csv(candidates_path)
        strict_count = int(candidates.get("eligible_strict", pd.Series(dtype=bool)).eq(True).sum())
        if strict_count != 218:
            problems.append(f"strict candidate count is {strict_count}, expected 218")
    else:
        strict_count = None
        problems.append("missing q2_candidate_dictionary.csv")

    e113_path = result_dir / "q2_run_manifest_E113.json"
    e115_path = result_dir / "q2_run_manifest_E115_final_nnls.json"
    e113 = _load_json(e113_path) if e113_path.is_file() else {}
    e115 = _load_json(e115_path) if e115_path.is_file() else {}
    if e113.get("status") != "complete" or e113.get("n_outer_folds") != 30:
        problems.append("E113 must have status=complete and n_outer_folds=30")
    if e113.get("n_features") != 218:
        problems.append("E113 must use 218 strict candidates")
    if e115.get("status") != "complete":
        problems.append("E115 pilot/incomplete output is not accepted")
    if e115.get("repeat_ids") != list(range(10)) or e115.get("n_outer_folds") != 30:
        problems.append("E115 must contain repeats 0..9 and 30 outer folds")
    if e115.get("n_features_strict") != 218 or e115.get("n_targets") != 50:
        problems.append("E115 dimensions must be 218 strict fields x 50 targets")
    for flag in ("q1_retrained", "nnls_refit", "hyperparameter_search"):
        if e115 and e115.get(flag) is not False:
            problems.append(f"E115 {flag} must be false")
    for output in e115.get("outputs", []):
        candidate = Path(output)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            problems.append(f"missing E115 output: {candidate}")

    registry_path = root / "03_结果" / "Q1_感官评分" / "final_model_registry.json"
    registry = _load_json(registry_path) if registry_path.is_file() else {}
    if registry.get("final_model") != "感官组收缩NNLS":
        problems.append("Q1 registry does not point to the frozen final NNLS model")
    oof_path = root / str(registry.get("oof_path", ""))
    if not oof_path.is_file():
        problems.append("Q1 final OOF is missing")
        actual_oof_hash = None
    else:
        actual_oof_hash = _sha256(oof_path)
        if actual_oof_hash != registry.get("oof_sha256"):
            problems.append("Q1 final OOF hash mismatch")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "ready" if not problems else "blocked",
        "formal_run_started": False,
        "problems": problems,
        "candidate_strict_count": strict_count,
        "e113_manifest": str(e113_path),
        "e113_status": e113.get("status"),
        "e115_manifest": str(e115_path),
        "e115_status": e115.get("status"),
        "q1_final_model": registry.get("final_model"),
        "q1_final_oof_sha256": actual_oof_hash,
    }


def _median_impute(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("a selected half contains an all-missing feature")
    rows, columns = np.where(~np.isfinite(values))
    values[rows, columns] = medians[columns]
    return values


def _within_style_center(values: np.ndarray, styles: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for style in pd.unique(styles):
        mask = styles == style
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result


def _absolute_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_rank = pd.DataFrame(x).rank(axis=0, method="average").to_numpy(float)
    y_rank = pd.DataFrame(y).rank(axis=0, method="average").to_numpy(float)
    x_rank -= x_rank.mean(axis=0, keepdims=True)
    y_rank -= y_rank.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(x_rank * x_rank, axis=0))
    y_norm = np.sqrt(np.sum(y_rank * y_rank, axis=0))
    denominator = np.outer(x_norm, y_norm)
    correlation = np.divide(
        x_rank.T @ y_rank,
        denominator,
        out=np.zeros((x.shape[1], y.shape[1]), dtype=float),
        where=denominator > 0,
    )
    return np.abs(correlation)


def select_style_adjusted_topk(
    x: np.ndarray, y: np.ndarray, styles: np.ndarray, top_k: int
) -> tuple[list[set[int]], np.ndarray]:
    """Select features after half-local imputation and style centering."""

    x = _median_impute(x)
    y = np.asarray(y, dtype=float)
    styles = np.asarray(styles, dtype=object)
    if len(x) != len(y) or len(x) != len(styles):
        raise ValueError("x, y, and styles must contain the same rows")
    if not 1 <= int(top_k) <= x.shape[1]:
        raise ValueError("top_k is out of range")
    score = _absolute_spearman(
        _within_style_center(x, styles), _within_style_center(y, styles)
    )
    # Stable mergesort makes feature position the deterministic tie-breaker.
    order = np.argsort(-score, axis=0, kind="stable")
    selected = [set(order[: int(top_k), target].tolist()) for target in range(y.shape[1])]
    return selected, score


def _permute_within_style(values: np.ndarray, styles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(values).copy()
    for style in pd.unique(styles):
        rows = np.flatnonzero(styles == style)
        result[rows] = result[rows[rng.permutation(len(rows))]]
    return result


def run_e117(
    output_dir: Path = RESULT_DIR,
    config_path: Path = CONFIG_PATH,
    root: Path = ROOT,
) -> dict:
    """Execute formal E117 after all fail-closed gates are satisfied."""

    started = time.perf_counter()
    output_dir = Path(output_dir)
    config = load_config(config_path)
    if config.get("formal_run") is not True:
        raise RuntimeError("formal E117 is disabled; reviewed config must set formal_run: true")
    readiness = audit_inputs(output_dir, root)
    if readiness["status"] != "ready":
        raise RuntimeError("E117 input gate failed: " + "; ".join(readiness["problems"]))

    x_frame, y_frame, style_series, _ = load_q2()
    candidates = pd.read_csv(output_dir / "q2_candidate_dictionary.csv")
    strict_codes = candidates.loc[candidates["eligible_strict"].eq(True), "feature_code"].astype(str).tolist()
    x = x_frame[strict_codes].to_numpy(float)
    y = y_frame.to_numpy(float)
    styles = style_series.to_numpy(object)
    target_codes = y_frame.columns.astype(str).tolist()
    n_pairs = int(config["formal_pairs"])
    top_k = int(config["selection_size_per_target"])
    seed = int(config["random_state"])

    pairs = stratified_complementary_pairs(styles, n_pairs, seed)
    selections: list[list[set[int]]] = [[] for _ in target_codes]
    jaccard_rows: list[dict] = []
    for pair_id, (left, right) in enumerate(pairs):
        left_sets, _ = select_style_adjusted_topk(x[left], y[left], styles[left], top_k)
        right_sets, _ = select_style_adjusted_topk(x[right], y[right], styles[right], top_k)
        for target_id, target in enumerate(target_codes):
            selections[target_id].extend([left_sets[target_id], right_sets[target_id]])
            jaccard_rows.append(
                {
                    "pair": pair_id,
                    "target_code": target,
                    "jaccard": float(complementary_pair_jaccard([left_sets[target_id]], [right_sets[target_id]])[0]),
                }
            )

    cpss_rows: list[dict] = []
    real_frequency: dict[str, np.ndarray] = {}
    for target_id, target in enumerate(target_codes):
        frequency = selection_probability(selections[target_id], len(strict_codes))
        real_frequency[target] = frequency
        cpss_rows.extend(
            {
                "target_code": target,
                "feature_code": feature,
                "selection_probability": float(frequency[feature_id]),
                "selection_count": int(round(frequency[feature_id] * 2 * n_pairs)),
                "selection_trials": 2 * n_pairs,
            }
            for feature_id, feature in enumerate(strict_codes)
        )

    null_selections: list[list[set[int]]] = [[] for _ in target_codes]
    rng = np.random.default_rng(seed + 117000)
    for _ in range(int(config["target_permutation_repeats"])):
        null_y = _permute_within_style(y, styles, rng)
        selected, _ = select_style_adjusted_topk(x, null_y, styles, top_k)
        for target_id, chosen in enumerate(selected):
            null_selections[target_id].append(chosen)
    null_rows, error_rows = [], []
    for target_id, target in enumerate(target_codes):
        null_frequency = selection_probability(null_selections[target_id], len(strict_codes))
        null_rows.extend(
            {
                "target_code": target,
                "feature_code": feature,
                "null_selection_probability": float(null_frequency[feature_id]),
            }
            for feature_id, feature in enumerate(strict_codes)
        )
        gate = confirmed_threshold(
            real_frequency[target], null_frequency,
            config["stability_thresholds"], config["gates"]["max_confirmed_fdp"],
        )
        for threshold in config["stability_thresholds"]:
            n_real = int((real_frequency[target] >= float(threshold)).sum())
            n_null = int((null_frequency >= float(threshold)).sum())
            error_rows.append(
                {
                    "target_code": target,
                    "threshold": float(threshold),
                    "real_selected": n_real,
                    "null_selected": n_null,
                    "empirical_fdp": min(1.0, n_null / max(n_real, 1)),
                    "mb_pfer_bound": mb_pfer_bound(top_k, float(threshold), len(strict_codes)),
                    "is_confirmed_threshold": gate["threshold"] == float(threshold),
                }
            )

    # Shadow audit does not alter real rankings.  It records how often a
    # within-style-permuted copy enters a joint Top-k screen.
    shadow_rows = []
    for shadow_repeat in range(int(config["shadow_feature_repeats"])):
        shadow = _permute_within_style(x, styles, rng)
        joint = np.concatenate([x, shadow], axis=1)
        chosen, _ = select_style_adjusted_topk(joint, y, styles, top_k)
        for target_id, target in enumerate(target_codes):
            shadow_rows.append(
                {
                    "repeat": shadow_repeat,
                    "target_code": target,
                    "real_selected": sum(index < len(strict_codes) for index in chosen[target_id]),
                    "shadow_selected": sum(index >= len(strict_codes) for index in chosen[target_id]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": output_dir / "q2_cpss_selection_E117.csv",
        "jaccard": output_dir / "q2_cpss_pair_jaccard_E117.csv",
        "null": output_dir / "q2_null_target_permutation_E117.csv.gz",
        "shadow": output_dir / "q2_shadow_feature_audit_E117.csv",
        "error": output_dir / "q2_error_control_E117.csv",
        "manifest": output_dir / "q2_run_manifest_E117.json",
    }
    pd.DataFrame(cpss_rows).to_csv(paths["selection"], index=False)
    pd.DataFrame(jaccard_rows).to_csv(paths["jaccard"], index=False)
    pd.DataFrame(null_rows).to_csv(paths["null"], index=False, compression="gzip")
    pd.DataFrame(shadow_rows).to_csv(paths["shadow"], index=False)
    pd.DataFrame(error_rows).to_csv(paths["error"], index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "formal_run": True,
        "n_pairs": n_pairs,
        "selection_trials_per_target": 2 * n_pairs,
        "n_features_strict": len(strict_codes),
        "n_targets": len(target_codes),
        "sampling": config["sampling"],
        "target_permutation_scope": config["target_permutation_scope"],
        "selection_method": "half_local_median_style_centered_absolute_spearman",
        "upstream": readiness,
        "config_sha256": _sha256(Path(config_path)),
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "gpu_used": False,
        "outputs": [str(path) for key, path in paths.items() if key != "manifest"],
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run only after formal_run is reviewed and enabled")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if args.execute:
        payload = run_e117(args.output_dir, args.config)
    else:
        payload = audit_inputs(args.output_dir, ROOT)
        payload["config_formal_run"] = bool(load_config(args.config).get("formal_run"))
        payload["status"] = "code_ready_not_run" if payload["status"] == "ready" else "blocked_not_run"
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
