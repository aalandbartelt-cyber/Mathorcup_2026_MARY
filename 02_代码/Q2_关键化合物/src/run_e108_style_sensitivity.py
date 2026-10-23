"""Run Q2 E108 style adjustment and leave-one-style-out sensitivity."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from bootstrap_stability import rank_scores_descending
from permutation_importance import aggregate_feature_importance
from q2_contract import RESULT_DIR, build_candidate_dictionary, load_q2
from q2_pipeline import aggregate_fold_rankings, load_frozen_outer_folds
from rank_consensus import reciprocal_rank_consensus
from run_e107_bootstrap import _tree_conditional_scores
from selection_models import (
    build_quadratic_penalty_rows,
    fit_fold_preprocessor,
    fit_multi_target_with_penalty_rows,
    spearman_target_scores,
)
from style_adjustment import fit_style_residualizer


EXPERIMENT_ID = "E108"
RANDOM_STATE = 20261024
METHOD_SPEARMAN = "E101_SpearmanTop3"
METHOD_ELASTIC = "E103_ElasticNet"
METHOD_TREE = "E105_ExtraTreesConditionalPermutation"
LAMBDA_L1 = 0.03
LAMBDA_L2 = 1.0


def _rank_rows(
    scores: np.ndarray,
    *,
    method: str,
    feature_codes: list[str],
    target_codes: list[str],
    split_id: str,
    absolute: bool,
    tree: bool = False,
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float)
    ranked_values = np.abs(values) if absolute else values
    ranks = rank_scores_descending(ranked_values, feature_codes)
    feature_grid = np.repeat(np.asarray(feature_codes, dtype=object), len(target_codes))
    target_grid = np.tile(np.asarray(target_codes, dtype=object), len(feature_codes))
    frame = pd.DataFrame(
        {
            "method": method,
            "target_code": target_grid,
            "feature_code": feature_grid,
            "score": values.reshape(-1),
            "rank": ranks.reshape(-1),
            "split_id": split_id,
        }
    )
    if tree:
        frame["selected_top3"] = frame["rank"].le(3) & frame["score"].gt(0)
    return frame


def _one_fold(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    *,
    feature_codes: list[str],
    feature_positions: dict[str, int],
    target_codes: list[str],
    split_id: str,
    seed: int,
) -> list[pd.DataFrame]:
    correlation = x_train.corr(method="spearman")
    spearman = spearman_target_scores(x_train, y_train).loc[
        feature_codes, target_codes
    ].to_numpy(float)

    processed = fit_fold_preprocessor(x_train, x_validation)
    penalty = build_quadratic_penalty_rows(
        np.zeros((processed.x_train.shape[1],) * 2, dtype=float),
        n_samples=len(x_train),
        lambda_l2=LAMBDA_L2,
        lambda_graph=0.0,
    )
    elastic_model = fit_multi_target_with_penalty_rows(
        processed.x_train,
        y_train.to_numpy(float),
        penalty,
        lambda_l1=LAMBDA_L1,
    )
    strict_positions = np.asarray([feature_positions[feature] for feature in feature_codes])
    elastic = elastic_model.coef_[:, strict_positions].T

    tree = _tree_conditional_scores(
        x_train,
        y_train.to_numpy(float),
        x_validation,
        y_validation.to_numpy(float),
        feature_codes,
        feature_positions,
        correlation,
        seed=seed,
    )
    return [
        _rank_rows(
            spearman,
            method=METHOD_SPEARMAN,
            feature_codes=feature_codes,
            target_codes=target_codes,
            split_id=split_id,
            absolute=True,
        ),
        _rank_rows(
            elastic,
            method=METHOD_ELASTIC,
            feature_codes=feature_codes,
            target_codes=target_codes,
            split_id=split_id,
            absolute=True,
        ),
        _rank_rows(
            tree,
            method=METHOD_TREE,
            feature_codes=feature_codes,
            target_codes=target_codes,
            split_id=split_id,
            absolute=False,
            tree=True,
        ),
    ]


def _aggregate(fold_rows: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.concat(fold_rows, ignore_index=True)
    signed = aggregate_fold_rankings(
        table.loc[table["method"].isin([METHOD_SPEARMAN, METHOD_ELASTIC])]
    )
    tree = aggregate_feature_importance(table.loc[table["method"].eq(METHOD_TREE)])
    method_rankings = pd.concat([signed, tree], ignore_index=True, sort=False)
    consensus = reciprocal_rank_consensus(
        method_rankings, expected_methods=3, rrf_k=60
    )
    return consensus, method_rankings


def _jaccard_by_target(
    reference: pd.DataFrame,
    alternate: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    rows = []
    targets = sorted(reference["target_code"].unique())
    for target in targets:
        left = set(
            reference.loc[
                reference["target_code"].eq(target)
                & reference["consensus_rank"].le(3),
                "feature_code",
            ]
        )
        right = set(
            alternate.loc[
                alternate["target_code"].eq(target)
                & alternate["consensus_rank"].le(3),
                "feature_code",
            ]
        )
        rows.append(
            {
                "comparison": label,
                "target_code": target,
                "intersection": len(left & right),
                "union": len(left | right),
                "jaccard": len(left & right) / len(left | right),
                "exact_top3": left == right,
            }
        )
    return pd.DataFrame(rows)


def _enrich(
    consensus: pd.DataFrame, candidates: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    return (
        consensus.merge(
            candidates[
                [
                    "feature_code",
                    "compound_name",
                    "chemical_name_short",
                    "unit",
                    "factor_type",
                ]
            ],
            on="feature_code",
            how="left",
            validate="many_to_one",
        )
        .merge(
            targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
            on="target_code",
            how="left",
            validate="many_to_one",
        )
    )


def run_e108(output_dir: Path = RESULT_DIR) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    x, y, style, _ = load_q2()
    candidates = build_candidate_dictionary(x.columns)
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    feature_codes = (
        candidates.loc[candidates["eligible_strict"], "feature_code"]
        .astype(str)
        .tolist()
    )
    feature_positions = {feature: x.columns.get_loc(feature) for feature in feature_codes}
    target_codes = y.columns.astype(str).tolist()
    all_folds = load_frozen_outer_folds(len(x), 10)

    adjusted_rows: list[pd.DataFrame] = []
    for position, split in enumerate(all_folds):
        train, validation = split.train_index, split.validation_index
        x_adjusted = fit_style_residualizer(
            x.iloc[train], x.iloc[validation], style.iloc[train], style.iloc[validation]
        )
        y_adjusted = fit_style_residualizer(
            y.iloc[train], y.iloc[validation], style.iloc[train], style.iloc[validation]
        )
        adjusted_rows.extend(
            _one_fold(
                x_adjusted.train,
                y_adjusted.train,
                x_adjusted.validation,
                y_adjusted.validation,
                feature_codes=feature_codes,
                feature_positions=feature_positions,
                target_codes=target_codes,
                split_id=f"adjusted_{split.split_id}",
                seed=RANDOM_STATE + position * 10_000,
            )
        )
        print(
            json.dumps(
                {"stage": "style_adjusted", "fold": position + 1, "total": 30},
                ensure_ascii=False,
            ),
            flush=True,
        )
    adjusted, adjusted_methods = _aggregate(adjusted_rows)
    reference = pd.read_csv(output_dir / "q2_consensus_rankings_E106.csv")
    adjusted_comparison = _jaccard_by_target(
        reference, adjusted, label="E106_vs_style_adjusted"
    )

    repeat_zero = load_frozen_outer_folds(len(x), 1)
    raw_rows: list[pd.DataFrame] = []
    for position, split in enumerate(repeat_zero):
        raw_rows.extend(
            _one_fold(
                x.iloc[split.train_index].reset_index(drop=True),
                y.iloc[split.train_index].reset_index(drop=True),
                x.iloc[split.validation_index].reset_index(drop=True),
                y.iloc[split.validation_index].reset_index(drop=True),
                feature_codes=feature_codes,
                feature_positions=feature_positions,
                target_codes=target_codes,
                split_id=f"raw_{split.split_id}",
                seed=RANDOM_STATE + 400_000 + position * 10_000,
            )
        )
    raw_reference, raw_methods = _aggregate(raw_rows)
    raw_calibration = _jaccard_by_target(
        reference, raw_reference, label="E106_vs_fixed_parameter_repeat0"
    )

    deletion_consensus = []
    deletion_jaccard = []
    styles = sorted(style.astype(str).unique())
    for style_position, omitted_style in enumerate(styles):
        keep = ~style.astype(str).eq(omitted_style).to_numpy()
        rows: list[pd.DataFrame] = []
        for fold_position, split in enumerate(repeat_zero):
            train = split.train_index[keep[split.train_index]]
            validation = split.validation_index[keep[split.validation_index]]
            rows.extend(
                _one_fold(
                    x.iloc[train].reset_index(drop=True),
                    y.iloc[train].reset_index(drop=True),
                    x.iloc[validation].reset_index(drop=True),
                    y.iloc[validation].reset_index(drop=True),
                    feature_codes=feature_codes,
                    feature_positions=feature_positions,
                    target_codes=target_codes,
                    split_id=f"drop_{style_position:02d}_{fold_position}",
                    seed=RANDOM_STATE
                    + 500_000
                    + style_position * 30_000
                    + fold_position * 10_000,
                )
            )
        consensus, _ = _aggregate(rows)
        consensus.insert(0, "omitted_style", omitted_style)
        deletion_consensus.append(consensus)
        comparison = _jaccard_by_target(
            raw_reference,
            consensus,
            label="repeat0_reference_vs_leave_one_style_out",
        )
        comparison.insert(0, "omitted_style", omitted_style)
        deletion_jaccard.append(comparison)
        print(
            json.dumps(
                {
                    "stage": "leave_one_style_out",
                    "style": style_position + 1,
                    "total": len(styles),
                    "omitted_style": omitted_style,
                    "rows_remaining": int(keep.sum()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    deletion_consensus_frame = pd.concat(deletion_consensus, ignore_index=True)
    deletion_jaccard_frame = pd.concat(deletion_jaccard, ignore_index=True)
    selected_deletion = deletion_consensus_frame.loc[
        deletion_consensus_frame["consensus_rank"].le(3),
        ["omitted_style", "target_code", "feature_code"],
    ]
    survival = (
        selected_deletion.groupby(["target_code", "feature_code"], sort=False)[
            "omitted_style"
        ]
        .nunique()
        .rename("selected_deletions")
        .reset_index()
    )
    reference_pairs = reference.loc[
        reference["consensus_rank"].le(3),
        ["target_code", "feature_code", "consensus_rank"],
    ]
    survival = reference_pairs.merge(
        survival, on=["target_code", "feature_code"], how="left"
    )
    survival["selected_deletions"] = survival["selected_deletions"].fillna(0).astype(int)
    survival["leave_one_style_survival_frequency"] = (
        survival["selected_deletions"] / len(styles)
    )

    bootstrap = pd.read_csv(output_dir / "q2_bootstrap_selection_E107.csv")
    integrated = reference_pairs.merge(
        bootstrap[
            [
                "target_code",
                "feature_code",
                "selection_frequency",
                "median_rank",
                "rank_iqr",
                "direction_consistency_when_selected",
            ]
        ],
        on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
    ).merge(
        adjusted[["target_code", "feature_code", "consensus_rank"]].rename(
            columns={"consensus_rank": "style_adjusted_rank"}
        ),
        on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
    ).merge(
        survival[
            [
                "target_code",
                "feature_code",
                "leave_one_style_survival_frequency",
            ]
        ],
        on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
    )
    integrated["style_adjusted_top3"] = integrated["style_adjusted_rank"].le(3)
    integrated = _enrich(integrated, candidates, targets)

    adjusted_enriched = _enrich(adjusted, candidates, targets)
    adjusted_methods.to_csv(
        output_dir / "q2_style_adjusted_method_rankings_E108.csv.gz",
        index=False,
        compression="gzip",
    )
    adjusted_enriched.to_csv(output_dir / "q2_style_adjusted_rankings_E108.csv", index=False)
    adjusted_enriched.loc[adjusted_enriched["consensus_rank"].le(3)].to_csv(
        output_dir / "q2_style_adjusted_top3_E108.csv", index=False
    )
    anchor_audit = targets.loc[targets["direct_anchor"].notna()].merge(
        adjusted_enriched,
        left_on=["target_code", "direct_anchor"],
        right_on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_style_adjusted"),
    )
    anchor_audit[
        [
            "target_code",
            "target_name_en",
            "direct_anchor",
            "consensus_rank",
            "method_top3_support",
            "method_top10_support",
        ]
    ].to_csv(output_dir / "q2_anchor_recovery_E108.csv", index=False)
    adjusted_comparison.to_csv(
        output_dir / "q2_style_adjustment_jaccard_E108.csv", index=False
    )
    raw_methods.to_csv(
        output_dir / "q2_fixed_parameter_repeat0_method_rankings_E108.csv.gz",
        index=False,
        compression="gzip",
    )
    raw_reference.to_csv(
        output_dir / "q2_fixed_parameter_repeat0_rankings_E108.csv", index=False
    )
    raw_calibration.to_csv(
        output_dir / "q2_fixed_parameter_repeat0_calibration_E108.csv", index=False
    )
    deletion_consensus_frame.to_csv(
        output_dir / "q2_leave_one_style_rankings_E108.csv.gz",
        index=False,
        compression="gzip",
    )
    deletion_jaccard_frame.to_csv(
        output_dir / "q2_leave_one_style_jaccard_E108.csv", index=False
    )
    survival.to_csv(output_dir / "q2_leave_one_style_survival_E108.csv", index=False)
    integrated.to_csv(output_dir / "q2_integrated_stability_E108.csv", index=False)

    manifest = {
        "experiment": EXPERIMENT_ID,
        "status": "complete",
        "style_count": len(styles),
        "style_adjustment": "training-fold style means for both X and Y",
        "style_adjusted_vs_E106": {
            "mean_jaccard": float(adjusted_comparison["jaccard"].mean()),
            "median_jaccard": float(adjusted_comparison["jaccard"].median()),
            "exact_top3_targets": int(adjusted_comparison["exact_top3"].sum()),
        },
        "repeat0_fixed_parameter_vs_E106": {
            "mean_jaccard": float(raw_calibration["jaccard"].mean()),
            "median_jaccard": float(raw_calibration["jaccard"].median()),
            "exact_top3_targets": int(raw_calibration["exact_top3"].sum()),
        },
        "leave_one_style_out": {
            "mean_jaccard": float(deletion_jaccard_frame["jaccard"].mean()),
            "median_jaccard": float(deletion_jaccard_frame["jaccard"].median()),
            "minimum_jaccard": float(deletion_jaccard_frame["jaccard"].min()),
            "mean_reference_candidate_survival": float(
                survival["leave_one_style_survival_frequency"].mean()
            ),
        },
        "gpu_used": False,
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output_dir / "q2_run_manifest_E108.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run_e108()
