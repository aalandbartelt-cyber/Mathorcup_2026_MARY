"""Run Q2 E105 ExtraTrees with outer-fold conditional/module permutation importance."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from permutation_importance import (
    aggregate_feature_importance,
    assign_quantile_groups,
    connected_modules,
    fit_quantile_edges,
    grouped_permutation_order,
    strongest_neighbor,
)
from q2_contract import RESULT_DIR, build_candidate_dictionary, load_q2
from q2_pipeline import load_frozen_outer_folds, metrics_by_target


EXPERIMENT_ID = "E105"
METHOD = "E105_ExtraTreesConditionalPermutation"
RANDOM_STATE = 20261024
N_ESTIMATORS = 300
N_CONDITIONING_BINS = 4
MODULE_THRESHOLD = 0.70
Q1_PARAMETER_FILE = Path("03_结果/Q1_感官评分/q1_selected_parameters_E003.csv")


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


def _fold_parameter_table() -> pd.DataFrame:
    table = pd.read_csv(Q1_PARAMETER_FILE)
    required = {
        "repeat",
        "fold",
        "selected_max_features",
        "selected_min_samples_leaf",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Q1 E003 parameter table misses: {sorted(missing)}")
    return table


def _indicator_column_map(imputer: SimpleImputer, n_original: int) -> dict[int, int]:
    if imputer.indicator_ is None:
        return {}
    return {
        int(feature): n_original + position
        for position, feature in enumerate(imputer.indicator_.features_)
    }


def _permuted_columns(
    original_positions: list[int], indicator_map: dict[int, int]
) -> list[int]:
    columns = list(original_positions)
    columns.extend(
        indicator_map[position]
        for position in original_positions
        if position in indicator_map
    )
    return columns


def _rank_fold_importance(
    scores: np.ndarray,
    feature_codes: list[str],
    target_codes: list[str],
    split_id: str,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    rows = []
    scores = np.asarray(scores, dtype=float)
    for target_position, target in enumerate(target_codes):
        table = pd.DataFrame(
            {
                "feature_code": feature_codes,
                "score": scores[:, target_position],
            }
        ).sort_values(
            ["score", "feature_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        table["rank"] = np.arange(1, len(table) + 1)
        table["selected_top3"] = table["rank"].le(3) & table["score"].gt(0)
        table.insert(0, "target_code", target)
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "method", METHOD)
    result["split_id"] = split_id
    result["repeat"] = int(repeat)
    result["fold"] = int(fold)
    return result


def run_e105(*, repeats: int, output_dir: Path = RESULT_DIR) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, _, beer = load_q2()
    target_codes = y.columns.astype(str).tolist()
    candidates = build_candidate_dictionary(x.columns)
    targets = pd.read_csv(RESULT_DIR / "q2_target_dictionary.csv")
    feature_codes = (
        candidates.loc[candidates["eligible_strict"], "feature_code"]
        .astype(str)
        .tolist()
    )
    feature_position = {feature: x.columns.get_loc(feature) for feature in feature_codes}
    folds = load_frozen_outer_folds(len(x), repeats)
    q1_parameters = _fold_parameter_table()
    predictions = np.full((repeats, len(x), y.shape[1]), np.nan, dtype=np.float32)
    fold_rankings: list[pd.DataFrame] = []
    module_importance: list[pd.DataFrame] = []
    parameter_rows = []

    for outer_position, split in enumerate(folds):
        fold_started = time.perf_counter()
        train, validation = split.train_index, split.validation_index
        x_train = x.iloc[train]
        x_validation = x.iloc[validation]
        y_train = y.iloc[train].to_numpy(float)
        y_validation = y.iloc[validation].to_numpy(float)
        selected = q1_parameters.loc[
            q1_parameters["repeat"].eq(split.repeat)
            & q1_parameters["fold"].eq(split.fold)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"missing unique Q1 E003 parameters for {split.split_id}")
        max_features = float(selected.iloc[0]["selected_max_features"])
        min_samples_leaf = int(selected.iloc[0]["selected_min_samples_leaf"])
        model_seed = 20261023 + int(split.repeat) * 10 + int(split.fold)
        permutation_seed = RANDOM_STATE + outer_position * 10_000

        imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        train_processed = imputer.fit_transform(x_train)
        validation_processed = imputer.transform(x_validation)
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=None,
            random_state=model_seed,
            n_jobs=4,
        )
        model.fit(train_processed, y_train)
        baseline_prediction = np.asarray(model.predict(validation_processed), dtype=float)
        predictions[split.repeat, validation] = baseline_prediction
        baseline_mae = np.mean(np.abs(y_validation - baseline_prediction), axis=0)
        indicator_map = _indicator_column_map(imputer, x.shape[1])

        training_correlation = x_train.corr(method="spearman")
        conditional_batch = np.repeat(
            validation_processed[None, :, :], len(feature_codes), axis=0
        )
        for feature_index, feature in enumerate(feature_codes):
            neighbor = strongest_neighbor(training_correlation, feature)
            edges = fit_quantile_edges(
                x_train[neighbor].to_numpy(float), n_bins=N_CONDITIONING_BINS
            )
            groups = assign_quantile_groups(
                x_validation[neighbor].to_numpy(float), edges
            )
            rng = np.random.default_rng(permutation_seed + feature_index + 1)
            order = grouped_permutation_order(groups, rng)
            position = feature_position[feature]
            columns = _permuted_columns([position], indicator_map)
            conditional_batch[feature_index][:, columns] = validation_processed[order][
                :, columns
            ]
        conditional_predictions = np.asarray(
            model.predict(conditional_batch.reshape(-1, validation_processed.shape[1])),
            dtype=float,
        ).reshape(len(feature_codes), len(validation), y.shape[1])
        conditional_scores = (
            np.mean(
                np.abs(y_validation[None, :, :] - conditional_predictions), axis=1
            )
            - baseline_mae[None, :]
        )
        fold_rankings.append(
            _rank_fold_importance(
                conditional_scores,
                feature_codes,
                target_codes,
                split.split_id,
                split.repeat,
                split.fold,
            )
        )

        modules = connected_modules(
            training_correlation,
            eligible_features=feature_codes,
            threshold=MODULE_THRESHOLD,
        )
        module_batch = np.repeat(validation_processed[None, :, :], len(modules), axis=0)
        module_metadata = []
        for module_position, members in enumerate(modules, start=1):
            rng = np.random.default_rng(permutation_seed + 1000 + module_position)
            order = rng.permutation(len(validation))
            positions = [feature_position[feature] for feature in members]
            columns = _permuted_columns(positions, indicator_map)
            module_batch[module_position - 1][:, columns] = validation_processed[order][
                :, columns
            ]
            module_metadata.append((module_position, members))
        module_predictions = np.asarray(
            model.predict(module_batch.reshape(-1, validation_processed.shape[1])),
            dtype=float,
        ).reshape(len(modules), len(validation), y.shape[1])
        module_scores = (
            np.mean(np.abs(y_validation[None, :, :] - module_predictions), axis=1)
            - baseline_mae[None, :]
        )
        module_frames = []
        for (module_position, members), delta in zip(module_metadata, module_scores):
            module_frames.append(
                pd.DataFrame(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "split_id": split.split_id,
                        "repeat": int(split.repeat),
                        "fold": int(split.fold),
                        "module_id": f"{split.split_id}_M{module_position:03d}",
                        "module_size": len(members),
                        "module_members": "|".join(members),
                        "target_code": target_codes,
                        "delta_mae": delta,
                    }
                )
            )
        module_importance.append(pd.concat(module_frames, ignore_index=True))
        parameter_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "split_id": split.split_id,
                "repeat": int(split.repeat),
                "fold": int(split.fold),
                "n_estimators": N_ESTIMATORS,
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
                "n_modules_abs_rho_ge_0_70": len(modules),
                "largest_module": max(map(len, modules)),
            }
        )
        print(
            json.dumps(
                {
                    "split_id": split.split_id,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "modules": len(modules),
                    "largest_module": max(map(len, modules)),
                    "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("E105 OOF predictions are incomplete or non-finite")
    ranking_folds = pd.concat(fold_rankings, ignore_index=True)
    aggregate = aggregate_feature_importance(ranking_folds)
    candidate_columns = [
        "feature_code",
        "compound_name",
        "chemical_name_short",
        "unit",
        "factor_type",
    ]
    aggregate = aggregate.merge(
        candidates[candidate_columns], on="feature_code", how="left", validate="many_to_one"
    )
    aggregate = aggregate.merge(
        targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
        on="target_code",
        how="left",
        validate="many_to_one",
    )
    top3 = aggregate.loc[aggregate["aggregate_rank"].le(3)].copy()
    anchor_audit = (
        targets.loc[targets["direct_anchor"].notna()]
        .merge(
            aggregate,
            left_on=["target_code", "direct_anchor"],
            right_on=["target_code", "feature_code"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_ranking"),
        )[
            [
                "target_code",
                "target_name_en",
                "direct_anchor",
                "aggregate_rank",
                "median_rank",
                "median_score",
                "positive_fraction",
                "top3_frequency",
            ]
        ]
        .sort_values("target_code")
    )
    metrics = []
    truth = y.to_numpy(float)
    for repeat in range(repeats):
        frame = metrics_by_target(truth, predictions[repeat], target_codes, METHOD)
        frame.insert(1, "repeat", repeat)
        metrics.append(frame)
    metrics = pd.concat(metrics, ignore_index=True)
    summary_by_repeat, summary = _metric_summaries(metrics)
    modules = pd.concat(module_importance, ignore_index=True)

    np.savez_compressed(
        output_dir / "q2_oof_predictions_E105.npz",
        beer=beer.to_numpy(int),
        target_codes=np.asarray(target_codes, dtype=str),
        **{METHOD: predictions},
    )
    metrics.to_csv(output_dir / "q2_metrics_by_target_E105.csv", index=False)
    summary_by_repeat.to_csv(
        output_dir / "q2_model_summary_by_repeat_E105.csv", index=False
    )
    summary.to_csv(output_dir / "q2_model_summary_E105.csv", index=False)
    ranking_folds.to_csv(
        output_dir / "q2_conditional_importance_folds_E105.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(output_dir / "q2_conditional_importance_E105.csv", index=False)
    top3.to_csv(output_dir / "q2_top3_E105.csv", index=False)
    anchor_audit.to_csv(output_dir / "q2_anchor_recovery_E105.csv", index=False)
    modules.to_csv(
        output_dir / "q2_module_importance_folds_E105.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "q2_selected_parameters_E105.csv", index=False
    )
    manifest = {
        "experiment": EXPERIMENT_ID,
        "status": "complete" if repeats == 10 else "pilot_complete",
        "outer_repeats": repeats,
        "outer_folds": len(folds),
        "model": "native multi-output ExtraTreesRegressor",
        "n_estimators": N_ESTIMATORS,
        "hyperparameters": "reuse leakage-safe Q1 E003 selections on identical folds",
        "conditional_permutation": {
            "conditioning_variable": "strongest absolute training-fold Spearman neighbor",
            "training_quantile_bins": N_CONDITIONING_BINS,
            "permutations_per_outer_fold": 1,
        },
        "module_permutation": {
            "training_fold_abs_spearman_threshold": MODULE_THRESHOLD,
            "joint_row_permutation_preserves_within_module_structure": True,
        },
        "strict_candidate_count": len(feature_codes),
        "gpu_used": False,
        "model_random_state_base": 20261023,
        "permutation_random_state_base": RANDOM_STATE,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output_dir / "q2_run_manifest_E105.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 11))
    args = parser.parse_args()
    print(json.dumps(run_e105(repeats=args.repeats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
