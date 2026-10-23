"""Run Q2 E101--E104 with frozen outer folds and fold-local preprocessing.

E101 ranks strict compounds by training-fold Spearman and predicts each target
with a Ridge model using its training-fold Top-3. E102 fits PLS2 and ranks by
target-specific signed VIP. E103 selects a shared ElasticNet penalty by inner
macro MAE. E104 adds a signed graph penalty, with the graph rebuilt inside
every outer and inner training split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from q2_contract import (
    RESULT_DIR,
    build_candidate_dictionary,
    fold_spearman_graph,
    load_q2,
    signed_normalized_laplacian,
)
from q2_pipeline import (
    aggregate_fold_rankings,
    extend_laplacian,
    load_frozen_outer_folds,
    metrics_by_target,
)
from selection_models import (
    build_quadratic_penalty_rows,
    fit_fold_preprocessor,
    fit_multi_target_with_penalty_rows,
    fit_pls,
    rank_score_matrix,
    spearman_target_scores,
    target_vip_scores,
)


RANDOM_STATE = 20261024
INNER_SPLITS = 3
PLS_COMPONENTS = (1, 2, 4, 8, 12, 16)
L1_GRID = (0.01, 0.03, 0.06, 0.10, 0.20, 0.30)
L2_GRID = (0.00, 0.01, 0.10, 1.00)
GRAPH_BETAS = (1, 2, 4, 6)
GRAPH_LAMBDAS = (0.0, 0.01, 0.10, 1.00, 10.00)

METHOD_MEAN = "E100_GlobalMean"
METHOD_SPEARMAN = "E101_SpearmanTop3"
METHOD_PLS = "E102_PLS_VIP"
METHOD_ELASTIC = "E103_ElasticNet"
METHOD_GRAPH = "E104_GraphElasticNet"
RANKING_METHODS = (METHOD_SPEARMAN, METHOD_PLS, METHOD_ELASTIC, METHOD_GRAPH)
PREDICTION_METHODS = (METHOD_MEAN,) + RANKING_METHODS


def resolve_grid_profile(profile: str) -> dict[str, tuple[float, ...]]:
    """Return registered grids or the boundary-triggered E109 sensitivity grids."""
    registered = {
        "pls_components": tuple(PLS_COMPONENTS),
        "lambda_l1": tuple(L1_GRID),
        "lambda_l2": tuple(L2_GRID),
        "graph_beta": tuple(GRAPH_BETAS),
        "lambda_graph": tuple(GRAPH_LAMBDAS),
    }
    if profile == "registered":
        return registered
    if profile == "e109_wide":
        return {
            **registered,
            "lambda_l1": (0.003, 0.006, 0.01, 0.03, 0.06, 0.10, 0.20, 0.30),
            "lambda_l2": (0.0, 0.01, 0.10, 1.0, 3.0, 10.0),
            "graph_beta": (1, 2, 4, 6, 8, 12),
        }
    raise ValueError(f"unknown grid profile: {profile}")


def artifact_suffix(run_label: str) -> str:
    """Return a safe optional suffix used by every artifact in one run."""
    if run_label == "":
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_label):
        raise ValueError("run label must contain only letters, digits, '_' or '-'")
    return f"_{run_label}"


def select_best_candidate(
    candidates: list[dict[str, Any]], scores: list[float]
) -> dict[str, Any]:
    """Select minimum score, preserving registration order for exact ties."""
    if len(candidates) != len(scores) or not candidates:
        raise ValueError("candidate and score lists must be non-empty and aligned")
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("candidate scores must be finite")
    return candidates[int(np.argmin(values))].copy()


def _macro_mae(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_prediction))))


def _inner_indices(n_rows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return [(train, validation) for train, validation in splitter.split(np.arange(n_rows))]


def _tune_pls(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    inner: list[tuple[np.ndarray, np.ndarray]],
    components: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [{"n_components": int(value)} for value in components]
    scores = []
    for candidate in candidates:
        fold_scores = []
        for inner_train, inner_validation in inner:
            processed = fit_fold_preprocessor(
                x_train.iloc[inner_train], x_train.iloc[inner_validation]
            )
            _, prediction = fit_pls(
                processed.x_train,
                y_train[inner_train],
                processed.x_validation,
                candidate["n_components"],
            )
            fold_scores.append(_macro_mae(y_train[inner_validation], prediction))
        scores.append(float(np.mean(fold_scores)))
    selected = select_best_candidate(candidates, scores)
    audit = [
        {**candidate, "macro_mae": score, "selected": candidate == selected}
        for candidate, score in zip(candidates, scores)
    ]
    return selected, audit


def _tune_elasticnet(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    inner: list[tuple[np.ndarray, np.ndarray]],
    l1_grid: tuple[float, ...],
    l2_grid: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        {"lambda_l1": l1, "lambda_l2": l2}
        for l1, l2 in itertools.product(l1_grid, l2_grid)
    ]
    fold_cache = []
    for inner_train, inner_validation in inner:
        processed = fit_fold_preprocessor(
            x_train.iloc[inner_train], x_train.iloc[inner_validation]
        )
        zero_graph = np.zeros((processed.x_train.shape[1],) * 2, dtype=float)
        fold_cache.append((inner_train, inner_validation, processed, zero_graph))
    scores = []
    for candidate in candidates:
        fold_scores = []
        for inner_train, inner_validation, processed, zero_graph in fold_cache:
            penalty = build_quadratic_penalty_rows(
                zero_graph,
                n_samples=len(inner_train),
                lambda_l2=candidate["lambda_l2"],
                lambda_graph=0.0,
            )
            model = fit_multi_target_with_penalty_rows(
                processed.x_train,
                y_train[inner_train],
                penalty,
                lambda_l1=candidate["lambda_l1"],
            )
            prediction = model.predict(processed.x_validation)
            fold_scores.append(_macro_mae(y_train[inner_validation], prediction))
        scores.append(float(np.mean(fold_scores)))
    selected = select_best_candidate(candidates, scores)
    audit = [
        {**candidate, "macro_mae": score, "selected": candidate == selected}
        for candidate, score in zip(candidates, scores)
    ]
    return selected, audit


def _tune_graph(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    inner: list[tuple[np.ndarray, np.ndarray]],
    elastic_parameters: dict[str, Any],
    graph_betas: tuple[float, ...],
    graph_lambdas: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        {
            **elastic_parameters,
            "beta": beta,
            "lambda_graph": lambda_graph,
        }
        for beta, lambda_graph in itertools.product(graph_betas, graph_lambdas)
    ]
    fold_cache = []
    for inner_train, inner_validation in inner:
        processed = fit_fold_preprocessor(
            x_train.iloc[inner_train], x_train.iloc[inner_validation]
        )
        correlation = fold_spearman_graph(x_train, inner_train)
        laplacians = {}
        for beta in graph_betas:
            base = signed_normalized_laplacian(correlation, beta)
            laplacians[beta] = extend_laplacian(
                base, processed.x_train.shape[1]
            )
        fold_cache.append((inner_train, inner_validation, processed, laplacians))
    scores = []
    for candidate in candidates:
        fold_scores = []
        for inner_train, inner_validation, processed, laplacians in fold_cache:
            penalty = build_quadratic_penalty_rows(
                laplacians[candidate["beta"]],
                n_samples=len(inner_train),
                lambda_l2=candidate["lambda_l2"],
                lambda_graph=candidate["lambda_graph"],
            )
            model = fit_multi_target_with_penalty_rows(
                processed.x_train,
                y_train[inner_train],
                penalty,
                lambda_l1=candidate["lambda_l1"],
            )
            prediction = model.predict(processed.x_validation)
            fold_scores.append(_macro_mae(y_train[inner_validation], prediction))
        scores.append(float(np.mean(fold_scores)))
    selected = select_best_candidate(candidates, scores)
    audit = [
        {**candidate, "macro_mae": score, "selected": candidate == selected}
        for candidate, score in zip(candidates, scores)
    ]
    return selected, audit


def _spearman_top3_prediction(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    eligible: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame]:
    scores = spearman_target_scores(x_train, y_train)
    rankings = rank_score_matrix(scores, METHOD_SPEARMAN, eligible)
    prediction = np.zeros((len(x_validation), y_train.shape[1]), dtype=float)
    for target_position, target in enumerate(y_train.columns):
        selected = (
            rankings.loc[rankings["target_code"].eq(target)]
            .nsmallest(3, "rank")["feature_code"]
            .tolist()
        )
        processed = fit_fold_preprocessor(
            x_train[selected], x_validation[selected]
        )
        model = Ridge(alpha=1.0)
        model.fit(processed.x_train, y_train.iloc[:, target_position])
        prediction[:, target_position] = model.predict(processed.x_validation)
    return prediction, rankings


def _enrich_rankings(
    rankings: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [
        "feature_code",
        "compound_name",
        "chemical_name_short",
        "unit",
        "factor_type",
        "eligible_strict",
    ]
    target_columns = ["target_code", "target_name_en", "target_group", "direct_anchor"]
    return (
        rankings.merge(
            candidates[feature_columns], on="feature_code", how="left", validate="many_to_one"
        )
        .merge(
            targets[target_columns], on="target_code", how="left", validate="many_to_one"
        )
    )


def _metric_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_repeat = (
        metrics.groupby(["method", "repeat"], sort=False)
        .agg(
            macro_r2=("r2", "mean"),
            macro_mae=("mae", "mean"),
            macro_rmse=("rmse", "mean"),
            macro_spearman=("spearman", "mean"),
            positive_r2_targets=("r2", lambda values: int((values > 0).sum())),
            n_targets=("target_code", "nunique"),
        )
        .reset_index()
    )
    summary = (
        by_repeat.groupby("method", sort=False)
        .agg(
            macro_r2_mean=("macro_r2", "mean"),
            macro_r2_sd=("macro_r2", "std"),
            macro_mae_mean=("macro_mae", "mean"),
            macro_mae_sd=("macro_mae", "std"),
            macro_rmse_mean=("macro_rmse", "mean"),
            macro_spearman_mean=("macro_spearman", "mean"),
            positive_r2_targets_mean=("positive_r2_targets", "mean"),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )
    return by_repeat, summary


def run_experiments(
    *,
    repeats: int,
    run_label: str,
    grid_profile: str = "registered",
    output_dir: Path = RESULT_DIR,
) -> dict[str, Any]:
    """Run E101--E104 and write OOF metrics and fold-stable rankings."""
    started = time.perf_counter()
    suffix = artifact_suffix(run_label)
    grids = resolve_grid_profile(grid_profile)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, _, beer = load_q2()
    candidates = build_candidate_dictionary(x.columns)
    targets = pd.read_csv(RESULT_DIR / "q2_target_dictionary.csv")
    eligible = candidates.set_index("feature_code")["eligible_strict"].astype(bool)
    folds = load_frozen_outer_folds(len(x), repeats)

    predictions = {
        method: np.full((repeats, len(x), y.shape[1]), np.nan, dtype=np.float32)
        for method in PREDICTION_METHODS
    }
    fold_rankings = []
    parameter_rows = []
    inner_score_rows = []

    for outer_position, fold in enumerate(folds):
        fold_started = time.perf_counter()
        train = fold.train_index
        validation = fold.validation_index
        x_train = x.iloc[train].reset_index(drop=True)
        x_validation = x.iloc[validation].reset_index(drop=True)
        y_train_frame = y.iloc[train].reset_index(drop=True)
        y_train = y_train_frame.to_numpy(float)

        predictions[METHOD_MEAN][fold.repeat, validation] = y_train.mean(axis=0)

        spearman_prediction, spearman_ranking = _spearman_top3_prediction(
            x_train, y_train_frame, x_validation, eligible
        )
        predictions[METHOD_SPEARMAN][fold.repeat, validation] = spearman_prediction
        spearman_ranking["split_id"] = fold.split_id
        spearman_ranking["repeat"] = fold.repeat
        spearman_ranking["fold"] = fold.fold
        fold_rankings.append(spearman_ranking)

        inner = _inner_indices(len(train), RANDOM_STATE + outer_position)
        pls_selected, pls_audit = _tune_pls(
            x_train, y_train, inner, grids["pls_components"]
        )
        elastic_selected, elastic_audit = _tune_elasticnet(
            x_train,
            y_train,
            inner,
            grids["lambda_l1"],
            grids["lambda_l2"],
        )
        graph_selected, graph_audit = _tune_graph(
            x_train,
            y_train,
            inner,
            elastic_selected,
            grids["graph_beta"],
            grids["lambda_graph"],
        )
        for experiment, selected in (
            ("E102", pls_selected),
            ("E103", elastic_selected),
            ("E104", graph_selected),
        ):
            parameter_rows.append(
                {
                    "experiment_id": experiment,
                    "split_id": fold.split_id,
                    "repeat": fold.repeat,
                    "fold": fold.fold,
                    **selected,
                }
            )
        for experiment, audit in (
            ("E102", pls_audit),
            ("E103", elastic_audit),
            ("E104", graph_audit),
        ):
            for row in audit:
                inner_score_rows.append(
                    {
                        "experiment_id": experiment,
                        "split_id": fold.split_id,
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        **row,
                    }
                )

        processed = fit_fold_preprocessor(x_train, x_validation)
        pls_model, pls_prediction = fit_pls(
            processed.x_train,
            y_train,
            processed.x_validation,
            pls_selected["n_components"],
        )
        predictions[METHOD_PLS][fold.repeat, validation] = pls_prediction
        vip = target_vip_scores(pls_model)[: x.shape[1], :]
        coefficient_sign = np.sign(np.asarray(pls_model.coef_, dtype=float)[:, : x.shape[1]].T)
        signed_vip = vip * coefficient_sign
        pls_scores = pd.DataFrame(signed_vip, index=x.columns, columns=y.columns)
        pls_ranking = rank_score_matrix(pls_scores, METHOD_PLS, eligible)
        pls_ranking["split_id"] = fold.split_id
        pls_ranking["repeat"] = fold.repeat
        pls_ranking["fold"] = fold.fold
        fold_rankings.append(pls_ranking)

        zero_graph = np.zeros((processed.x_train.shape[1],) * 2, dtype=float)
        elastic_penalty = build_quadratic_penalty_rows(
            zero_graph,
            n_samples=len(train),
            lambda_l2=elastic_selected["lambda_l2"],
            lambda_graph=0.0,
        )
        elastic_model = fit_multi_target_with_penalty_rows(
            processed.x_train,
            y_train,
            elastic_penalty,
            lambda_l1=elastic_selected["lambda_l1"],
        )
        predictions[METHOD_ELASTIC][fold.repeat, validation] = elastic_model.predict(
            processed.x_validation
        )
        elastic_scores = pd.DataFrame(
            elastic_model.coef_[:, : x.shape[1]].T,
            index=x.columns,
            columns=y.columns,
        )
        elastic_ranking = rank_score_matrix(
            elastic_scores, METHOD_ELASTIC, eligible
        )
        elastic_ranking["split_id"] = fold.split_id
        elastic_ranking["repeat"] = fold.repeat
        elastic_ranking["fold"] = fold.fold
        fold_rankings.append(elastic_ranking)

        outer_correlation = fold_spearman_graph(x_train, np.arange(len(x_train)))
        base_laplacian = signed_normalized_laplacian(
            outer_correlation, graph_selected["beta"]
        )
        graph_laplacian = extend_laplacian(
            base_laplacian, processed.x_train.shape[1]
        )
        graph_penalty = build_quadratic_penalty_rows(
            graph_laplacian,
            n_samples=len(train),
            lambda_l2=graph_selected["lambda_l2"],
            lambda_graph=graph_selected["lambda_graph"],
        )
        graph_model = fit_multi_target_with_penalty_rows(
            processed.x_train,
            y_train,
            graph_penalty,
            lambda_l1=graph_selected["lambda_l1"],
        )
        predictions[METHOD_GRAPH][fold.repeat, validation] = graph_model.predict(
            processed.x_validation
        )
        graph_scores = pd.DataFrame(
            graph_model.coef_[:, : x.shape[1]].T,
            index=x.columns,
            columns=y.columns,
        )
        graph_ranking = rank_score_matrix(graph_scores, METHOD_GRAPH, eligible)
        graph_ranking["split_id"] = fold.split_id
        graph_ranking["repeat"] = fold.repeat
        graph_ranking["fold"] = fold.fold
        fold_rankings.append(graph_ranking)

        print(
            json.dumps(
                {
                    "split_id": fold.split_id,
                    "pls": pls_selected,
                    "elastic": elastic_selected,
                    "graph": graph_selected,
                    "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for method, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"OOF predictions incomplete for {method}")

    metric_frames = []
    for method, values in predictions.items():
        for repeat in range(repeats):
            frame = metrics_by_target(
                y.to_numpy(float), values[repeat], y.columns.tolist(), method
            )
            frame.insert(1, "repeat", repeat)
            metric_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary_by_repeat, summary = _metric_summaries(metrics)

    fold_ranking_table = pd.concat(fold_rankings, ignore_index=True)
    aggregate = aggregate_fold_rankings(fold_ranking_table)
    fold_ranking_table = _enrich_rankings(fold_ranking_table, candidates, targets)
    aggregate = _enrich_rankings(aggregate, candidates, targets)
    sparse_method = aggregate["method"].isin([METHOD_ELASTIC, METHOD_GRAPH])
    aggregate["available_for_method_top3"] = (
        ~sparse_method | aggregate["selection_frequency"].gt(0)
    )
    top3 = aggregate.loc[
        aggregate["available_for_method_top3"]
        & aggregate["aggregate_rank"].le(3)
    ].copy()

    anchors = targets.loc[targets["direct_anchor"].fillna("").ne("")].copy()
    anchor_audit = aggregate.merge(
        anchors[["target_code", "direct_anchor"]],
        left_on=["target_code", "feature_code"],
        right_on=["target_code", "direct_anchor"],
        how="inner",
        suffixes=("", "_target"),
    )
    anchor_audit = anchor_audit[
        [
            "method",
            "target_code",
            "target_name_en",
            "direct_anchor",
            "aggregate_rank",
            "median_rank",
            "median_score",
            "median_nonzero_score",
            "selection_frequency",
            "direction",
        ]
    ].sort_values(["target_code", "method"])

    np.savez_compressed(
        output_dir / f"q2_oof_predictions_E101_E104{suffix}.npz",
        beer=beer.to_numpy(int),
        target_codes=np.asarray(y.columns, dtype=str),
        **{method: values for method, values in predictions.items()},
    )
    metrics.to_csv(
        output_dir / f"q2_metrics_by_target_E101_E104{suffix}.csv", index=False
    )
    summary_by_repeat.to_csv(
        output_dir / f"q2_model_summary_by_repeat_E101_E104{suffix}.csv", index=False
    )
    summary.to_csv(
        output_dir / f"q2_model_summary_E101_E104{suffix}.csv", index=False
    )
    fold_ranking_table.to_csv(
        output_dir / f"q2_fold_rankings_E101_E104{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(
        output_dir / f"q2_method_rankings_E101_E104{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    top3.to_csv(output_dir / f"q2_top3_by_method_E101_E104{suffix}.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / f"q2_selected_parameters_E101_E104{suffix}.csv", index=False
    )
    pd.DataFrame(inner_score_rows).to_csv(
        output_dir / f"q2_inner_cv_scores_E101_E104{suffix}.csv", index=False
    )
    anchor_audit.to_csv(
        output_dir / f"q2_anchor_recovery_E101_E104{suffix}.csv", index=False
    )

    manifest = {
        "experiments": ["E101", "E102", "E103", "E104"],
        "status": "complete" if repeats == 10 else "pilot_complete",
        "run_label": run_label,
        "grid_profile": grid_profile,
        "outer_repeats": repeats,
        "outer_folds": len(folds),
        "inner_splits": INNER_SPLITS,
        "ranking_scope": "strict single compounds",
        "strict_candidate_count": int(eligible.sum()),
        "graph_recomputed_inside_outer_and_inner_training_splits": True,
        "preprocessing": {
            "winsor_quantiles": [0.01, 0.99],
            "bounds_fitted_on_training_fold_only": True,
            "order": ["winsorize", "median_impute", "missing_indicator", "standardize"],
        },
        "solver": {
            "lasso_max_iter": 50000,
            "automatic_retry_multiplier": 4,
            "unresolved_convergence_warning": False,
        },
        "gpu_used": False,
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "grids": {
            key: list(values) for key, values in grids.items()
        },
    }
    (output_dir / f"q2_run_manifest_E101_E104{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--run-label", default="pilot_R1")
    parser.add_argument(
        "--grid-profile",
        default="registered",
        choices=("registered", "e109_wide"),
    )
    args = parser.parse_args()
    manifest = run_experiments(
        repeats=args.repeats,
        run_label=args.run_label,
        grid_profile=args.grid_profile,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
