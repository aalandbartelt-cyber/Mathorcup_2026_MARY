"""Run E113 fold-local Graphical Lasso graphs and Louvain modules."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from graph_checkpoint import save_fold_checkpoint, load_fold_checkpoint
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from partial_graph import (
    FoldGaussianizer,
    coclustering_matrix,
    edge_table,
    fit_graphical_lasso,
    graph_structure_summary,
    heldout_gaussian_log_likelihood,
    louvain_membership,
    stable_edge_table,
)
from q2_contract import Q1_RESULT_DIR, RESULT_DIR, load_q2
from q2_pipeline import load_frozen_outer_folds


EXPERIMENT_ID = "E113"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E113_partial_graph.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if str(config.get("experiment")) != EXPERIMENT_ID:
        raise ValueError("E113 config experiment ID is invalid")
    alpha_grid = [float(value) for value in config.get("alpha_grid", [])]
    if not alpha_grid or any(value <= 0 for value in alpha_grid):
        raise ValueError("E113 alpha_grid must contain positive values")
    if alpha_grid != sorted(set(alpha_grid)):
        raise ValueError("E113 alpha_grid must be unique and increasing")
    return config


def select_graph_alpha(rows: pd.DataFrame) -> float:
    """Apply 1-SE likelihood feasibility, then stability, then sparsity."""
    required = {"alpha", "mean_loglik", "se_loglik", "mean_edge_jaccard"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"alpha table misses {sorted(missing)}")
    table = rows[list(required)].copy()
    if table.empty or table["alpha"].duplicated().any():
        raise ValueError("alpha table must be non-empty with unique alphas")
    if (table["alpha"] <= 0).any() or (table["se_loglik"].dropna() < 0).any():
        raise ValueError("alpha must be positive and SE non-negative")
    finite = np.isfinite(
        table[["mean_loglik", "se_loglik", "mean_edge_jaccard"]].to_numpy(float)
    ).all(axis=1)
    table = table.loc[finite].copy()
    if table.empty:
        raise RuntimeError("no alpha completed every inner fold")
    best = table.sort_values(
        ["mean_loglik", "alpha"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    threshold = float(best["mean_loglik"] - best["se_loglik"])
    feasible = table.loc[table["mean_loglik"] >= threshold].copy()
    selected = feasible.sort_values(
        ["mean_edge_jaccard", "alpha", "mean_loglik"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    return float(selected["alpha"])


def _preprocess_train_validation(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train)
    validation_imputed = imputer.transform(validation)
    gaussianizer = FoldGaussianizer(random_state=seed)
    return (
        gaussianizer.fit_transform(train_imputed),
        gaussianizer.transform(validation_imputed),
    )


def _edge_set(partial: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    upper_i, upper_j = np.triu_indices(len(partial), k=1)
    keep = np.abs(partial[upper_i, upper_j]) >= float(threshold)
    return set(zip(upper_i[keep].tolist(), upper_j[keep].tolist()))


def _mean_pairwise_jaccard(edge_sets: list[set[tuple[int, int]]]) -> float:
    if len(edge_sets) < 2:
        raise ValueError("at least two edge sets are required")
    values = []
    for left, right in combinations(edge_sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))


def _score_alpha_grid(
    x_train: np.ndarray,
    *,
    alpha_grid: list[float],
    inner_splits: int,
    edge_threshold: float,
    seed: int,
) -> pd.DataFrame:
    splitter = KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(seed))
    loglik = {alpha: [] for alpha in alpha_grid}
    edge_sets = {alpha: [] for alpha in alpha_grid}
    for inner_fold, (fit_index, validation_index) in enumerate(splitter.split(x_train)):
        inner_train, inner_validation = _preprocess_train_validation(
            x_train[fit_index], x_train[validation_index], seed=seed + inner_fold
        )
        for alpha in alpha_grid:
            try:
                fitted = fit_graphical_lasso(inner_train, alpha=alpha)
                loglik[alpha].append(
                    heldout_gaussian_log_likelihood(
                        inner_validation,
                        mean=fitted.location,
                        precision=fitted.precision,
                    )
                )
                edge_sets[alpha].append(
                    _edge_set(fitted.partial_correlation, edge_threshold)
                )
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                loglik[alpha].append(np.nan)
                edge_sets[alpha].append(set())
    rows = []
    for alpha in alpha_grid:
        values = np.asarray(loglik[alpha], dtype=float)
        complete = bool(np.isfinite(values).all())
        rows.append(
            {
                "alpha": alpha,
                "mean_loglik": float(np.mean(values)) if complete else np.nan,
                "se_loglik": (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if complete
                    else np.nan
                ),
                "mean_edge_jaccard": (
                    _mean_pairwise_jaccard(edge_sets[alpha]) if complete else np.nan
                ),
                "mean_edge_count": (
                    float(np.mean([len(edges) for edges in edge_sets[alpha]]))
                    if complete
                    else np.nan
                ),
                "failed_inner_folds": int(np.sum(~np.isfinite(values))),
            }
        )
    return pd.DataFrame(rows)


def _strict_feature_codes(x: pd.DataFrame, result_dir: Path) -> list[str]:
    dictionary = pd.read_csv(result_dir / "q2_candidate_dictionary.csv")
    required = {"feature_code", "eligible_strict"}
    missing = required - set(dictionary.columns)
    if missing:
        raise ValueError(f"candidate dictionary misses {sorted(missing)}")
    eligible = dictionary["eligible_strict"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().map({"true": True, "false": False})
    if eligible.isna().any():
        raise ValueError("eligible_strict must be Boolean")
    codes = dictionary.loc[eligible, "feature_code"].astype(str).tolist()
    if len(codes) != 218 or len(set(codes)) != 218:
        raise ValueError(f"expected 218 unique strict compounds, got {len(codes)}")
    missing_codes = set(codes) - set(x.columns.astype(str))
    if missing_codes:
        raise ValueError(f"strict compounds missing from X: {sorted(missing_codes)}")
    return codes


def _membership_table(
    feature_codes: list[str], labels: np.ndarray, *, repeat: int, fold: int, scope: str
) -> pd.DataFrame:
    sizes = pd.Series(labels).value_counts().to_dict()
    return pd.DataFrame(
        {
            "scope": scope,
            "repeat": int(repeat),
            "fold": int(fold),
            "feature_code": feature_codes,
            "module_id": [f"M{int(value) + 1:03d}" for value in labels],
            "module_size": [int(sizes[int(value)]) for value in labels],
        }
    )


def _write_coclustering(
    matrix: np.ndarray, feature_codes: list[str], output_path: Path
) -> None:
    frame = pd.DataFrame(matrix, columns=feature_codes)
    frame.insert(0, "feature_code", feature_codes)
    frame.to_csv(output_path, index=False, compression="gzip")


def _descriptive_alpha(selected_alphas: Sequence[float]) -> float:
    counts = pd.Series(np.asarray(selected_alphas, dtype=float)).value_counts()
    maximum = int(counts.max())
    return float(max(counts.index[counts.eq(maximum)].tolist()))


def _stability_decision(
    coclustering: np.ndarray,
    *,
    reference_path: Path,
) -> tuple[bool, dict[str, float]]:
    upper = coclustering[np.triu_indices(len(coclustering), k=1)]
    candidate_values = upper[upper >= 0.10]
    reference = pd.read_csv(reference_path)["coclustering_frequency"].to_numpy(float)
    if not len(candidate_values) or not len(reference):
        return False, {
            "candidate_mean_coclustering_ge_0_10": 0.0,
            "reference_mean_coclustering_ge_0_10": float(np.mean(reference)) if len(reference) else 0.0,
        }
    candidate_mean = float(np.mean(candidate_values))
    reference_mean = float(np.mean(reference))
    return candidate_mean > reference_mean, {
        "candidate_mean_coclustering_ge_0_10": candidate_mean,
        "reference_mean_coclustering_ge_0_10": reference_mean,
    }


def run_e113(*, output_dir: Path = RESULT_DIR) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    x, _, _, _ = load_q2()
    feature_codes = _strict_feature_codes(x, RESULT_DIR)
    values = x[feature_codes].to_numpy(float)
    folds = load_frozen_outer_folds(len(x), repeats=10)
    alpha_grid = [float(value) for value in config["alpha_grid"]]
    edge_threshold = float(config["partial_edge_threshold"])
    seed = int(config["random_state"])
    identity = {
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "cv_sha256": hashlib.sha256((Q1_RESULT_DIR/'q1_cv_splits.csv').read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "feature_order": feature_codes,
        "source_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__),Path(__file__).with_name('partial_graph.py'),
                                    Path(__file__).with_name('graph_checkpoint.py')]},
    }
    checkpoint_dir=output_dir/'q2_graph_checkpoints_E113'

    summary_rows = []
    alpha_score_frames = []
    fold_edge_frames = []
    membership_frames = []
    label_vectors = []
    selected_alphas = []
    for number, outer in enumerate(folds, start=1):
        restored=load_fold_checkpoint(checkpoint_dir,outer.split_id,identity)
        if restored is not None:
            summary_rows.append(restored['summary'].iloc[0].to_dict())
            alpha_score_frames.append(restored['alpha_scores'])
            fold_edge_frames.append(restored['edges'])
            membership_frames.append(restored['membership'])
            labels=restored['membership'].set_index('feature_code').loc[feature_codes,'module_id'].astype(str)
            label_vectors.append(pd.factorize(labels,sort=True)[0])
            selected_alphas.append(float(summary_rows[-1]['selected_alpha']))
            print(f'[{number:02d}/{len(folds)}] {outer.split_id} resumed',flush=True)
            continue
        fold_started = time.perf_counter()
        outer_seed = seed + outer.repeat * 100 + outer.fold
        alpha_scores = _score_alpha_grid(
            values[outer.train_index],
            alpha_grid=alpha_grid,
            inner_splits=int(config["inner_splits"]),
            edge_threshold=edge_threshold,
            seed=outer_seed,
        )
        selected_alpha = select_graph_alpha(alpha_scores)
        alpha_scores.insert(0, "fold", outer.fold)
        alpha_scores.insert(0, "repeat", outer.repeat)
        alpha_scores.insert(0, "split_id", outer.split_id)
        alpha_score_frames.append(alpha_scores)
        selected_alphas.append(selected_alpha)
        train, validation = _preprocess_train_validation(
            values[outer.train_index], values[outer.validation_index], seed=outer_seed
        )
        fitted = fit_graphical_lasso(train, alpha=selected_alpha)
        validation_loglik = heldout_gaussian_log_likelihood(
            validation, mean=fitted.location, precision=fitted.precision
        )
        edges = edge_table(
            fitted.partial_correlation,
            feature_codes,
            edge_threshold=edge_threshold,
        )
        edges.insert(0, "fold", outer.fold)
        edges.insert(0, "repeat", outer.repeat)
        edges.insert(0, "split_id", outer.split_id)
        edges.insert(0, "scope", "outer_train_fold")
        fold_edge_frames.append(edges)
        labels = louvain_membership(
            fitted.partial_correlation,
            feature_codes,
            edge_threshold=edge_threshold,
            seed=int(config["module_seed"]),
        )
        label_vectors.append(labels)
        membership_frames.append(
            _membership_table(
                feature_codes,
                labels,
                repeat=outer.repeat,
                fold=outer.fold,
                scope="outer_train_fold",
            )
        )
        selected_scores = alpha_scores.loc[alpha_scores["alpha"].eq(selected_alpha)].iloc[0]
        summary_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "scope": "outer_train_fold",
                "split_id": outer.split_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "n_train": len(outer.train_index),
                "n_validation": len(outer.validation_index),
                "selected_alpha": selected_alpha,
                "inner_mean_loglik": float(selected_scores["mean_loglik"]),
                "inner_se_loglik": float(selected_scores["se_loglik"]),
                "inner_mean_edge_jaccard": float(selected_scores["mean_edge_jaccard"]),
                "validation_loglik": validation_loglik,
                "n_iter": fitted.n_iter,
                **graph_structure_summary(
                    fitted.partial_correlation,
                    feature_codes,
                    edge_threshold=edge_threshold,
                    membership=labels,
                ),
                "elapsed_seconds": float(time.perf_counter() - fold_started),
            }
        )
        print(
            f"[{number:02d}/{len(folds)}] {outer.split_id} "
            f"alpha={selected_alpha:.2f} edges={len(edges)}",
            flush=True,
        )
        save_fold_checkpoint(checkpoint_dir,outer.split_id,identity,{
            'summary':pd.DataFrame([summary_rows[-1]]),'alpha_scores':alpha_score_frames[-1],
            'edges':fold_edge_frames[-1],'membership':membership_frames[-1]})
        progress={'experiment_id':EXPERIMENT_ID,'status':'running','completed_folds':number,
                  'n_outer_folds':len(folds),'n_features':len(feature_codes),
                  'last_split_id':outer.split_id,'elapsed_seconds':float(time.perf_counter()-started),
                  'scope':'outer_train_fold','identity':identity}
        temp=output_dir/'q2_progress_E113.json.tmp'
        temp.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
        temp.replace(output_dir/'q2_progress_E113.json')

    fold_summary = pd.DataFrame(summary_rows)
    alpha_score_table = pd.concat(alpha_score_frames, ignore_index=True)
    fold_edges = pd.concat(fold_edge_frames, ignore_index=True)
    memberships = pd.concat(membership_frames, ignore_index=True)
    coclustering = coclustering_matrix(label_vectors)
    stable_edges = stable_edge_table(
        [
            frame[["source", "target", "partial_correlation"]]
            for frame in fold_edge_frames
        ]
    )

    descriptive_alpha = _descriptive_alpha(selected_alphas)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    full_imputed = imputer.fit_transform(values)
    gaussianizer = FoldGaussianizer(random_state=seed)
    full_transformed = gaussianizer.fit_transform(full_imputed)
    descriptive_fit = fit_graphical_lasso(full_transformed, alpha=descriptive_alpha)
    descriptive_edges = edge_table(
        descriptive_fit.partial_correlation,
        feature_codes,
        edge_threshold=edge_threshold,
    )
    descriptive_edges.insert(0, "scope", "full_data_descriptive")
    descriptive_labels = louvain_membership(
        descriptive_fit.partial_correlation,
        feature_codes,
        edge_threshold=edge_threshold,
        seed=int(config["module_seed"]),
    )
    descriptive_membership = _membership_table(
        feature_codes,
        descriptive_labels,
        repeat=-1,
        fold=-1,
        scope="full_data_descriptive",
    )

    paths = {
        "fold_summary": output_dir / "q2_graph_fold_summary_E113.csv",
        "alpha_scores": output_dir / "q2_graph_alpha_scores_E113.csv.gz",
        "fold_edges": output_dir / "q2_partial_edges_E113.csv.gz",
        "stable_edges": output_dir / "q2_stable_edges_E113.csv",
        "memberships": output_dir / "q2_module_membership_E113.csv.gz",
        "coclustering": output_dir / "q2_coclustering_E113.csv.gz",
        "descriptive_edges": output_dir / "q2_partial_edges_full_descriptive_E113.csv.gz",
        "descriptive_membership": output_dir / "q2_module_membership_full_descriptive_E113.csv",
        "decision": output_dir / "q2_graph_decision_E113.json",
        "manifest": output_dir / "q2_run_manifest_E113.json",
    }
    fold_summary.to_csv(paths["fold_summary"], index=False)
    alpha_score_table.to_csv(paths["alpha_scores"], index=False, compression="gzip")
    fold_edges.to_csv(paths["fold_edges"], index=False, compression="gzip")
    stable_edges.to_csv(paths["stable_edges"], index=False)
    memberships.to_csv(paths["memberships"], index=False, compression="gzip")
    _write_coclustering(coclustering, feature_codes, paths["coclustering"])
    descriptive_edges.to_csv(
        paths["descriptive_edges"], index=False, compression="gzip"
    )
    descriptive_membership.to_csv(paths["descriptive_membership"], index=False)

    stability_improves, stability_metrics = _stability_decision(
        coclustering,
        reference_path=RESULT_DIR / "q2_bootstrap_coclustering_E107.csv.gz",
    )
    criteria = {
        "heldout_loglik_finite": bool(np.isfinite(fold_summary["validation_loglik"]).all()),
        "edge_or_module_stability_improves": bool(stability_improves),
        "no_full_graph_in_validation": bool(
            set(fold_edges["scope"]) == {"outer_train_fold"}
            and set(descriptive_edges["scope"]) == {"full_data_descriptive"}
        ),
    }
    decision = {
        "promote_partial_graph": bool(all(criteria.values())),
        "criteria": criteria,
        "stability_metrics": stability_metrics,
        "descriptive_alpha": descriptive_alpha,
    }
    paths["decision"].write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config_bytes = CONFIG_PATH.read_bytes()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "candidate_scope": config["candidate_scope"],
        "n_outer_folds": len(folds),
        "n_features": len(feature_codes),
        "alpha_grid": alpha_grid,
        "selected_alpha_counts": {
            str(key): int(value)
            for key, value in pd.Series(selected_alphas).value_counts().sort_index().items()
        },
        "decision": decision,
        "random_state": seed,
        "gpu_used": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint_identity":identity,
        "cv_sha256": hashlib.sha256(
            (Q1_RESULT_DIR / "q1_cv_splits.csv").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "outputs": [str(path) for path in paths.values() if path != paths["manifest"]],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    progress_path=output_dir/'q2_progress_E113.json'
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());progress.update(status='complete',completed_folds=len(folds))
        progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main() -> None:
    print(json.dumps(run_e113(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
