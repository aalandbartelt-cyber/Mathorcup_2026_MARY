"""Run 500 style-stratified Q2 bootstraps with OOB nonlinear importance."""

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

from bootstrap_stability import (
    consensus_ranks_from_methods,
    rank_scores_descending,
    stratified_bootstrap_indices,
    update_coclustering,
)
from permutation_importance import (
    assign_quantile_groups,
    connected_modules,
    fit_quantile_edges,
    grouped_permutation_order,
    strongest_neighbor,
)
from q2_contract import RESULT_DIR, build_candidate_dictionary, load_q2
from selection_models import (
    build_quadratic_penalty_rows,
    fit_fold_preprocessor,
    fit_multi_target_with_penalty_rows,
    spearman_target_scores,
)


EXPERIMENT_ID = "E107"
RANDOM_STATE = 20261024
N_ESTIMATORS = 300
MAX_FEATURES = 1.0
MIN_SAMPLES_LEAF = 2
LAMBDA_L1 = 0.03
LAMBDA_L2 = 1.0
N_CONDITIONING_BINS = 4
MODULE_THRESHOLD = 0.70
STABLE_MODULE_THRESHOLD = 0.70


def _suffix(label: str) -> str:
    if not label:
        return ""
    if not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run label contains unsafe characters")
    return f"_{label}"


def _indicator_map(imputer: SimpleImputer, n_original: int) -> dict[int, int]:
    return {
        int(feature): n_original + position
        for position, feature in enumerate(imputer.indicator_.features_)
    }


def _tree_conditional_scores(
    x_bootstrap: pd.DataFrame,
    y_bootstrap: np.ndarray,
    x_oob: pd.DataFrame,
    y_oob: np.ndarray,
    feature_codes: list[str],
    feature_positions: dict[str, int],
    training_correlation: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    train = imputer.fit_transform(x_bootstrap)
    oob = imputer.transform(x_oob)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=None,
        random_state=seed,
        n_jobs=4,
    )
    model.fit(train, y_bootstrap)
    baseline = np.asarray(model.predict(oob), dtype=float)
    baseline_mae = np.mean(np.abs(y_oob - baseline), axis=0)
    indicator_map = _indicator_map(imputer, x_bootstrap.shape[1])
    batch = np.repeat(oob[None, :, :], len(feature_codes), axis=0)
    for feature_index, feature in enumerate(feature_codes):
        neighbor = strongest_neighbor(training_correlation, feature)
        edges = fit_quantile_edges(
            x_bootstrap[neighbor].to_numpy(float), n_bins=N_CONDITIONING_BINS
        )
        groups = assign_quantile_groups(x_oob[neighbor].to_numpy(float), edges)
        order = grouped_permutation_order(
            groups, np.random.default_rng(seed + feature_index + 1)
        )
        original = feature_positions[feature]
        columns = [original]
        if original in indicator_map:
            columns.append(indicator_map[original])
        batch[feature_index][:, columns] = oob[order][:, columns]
    changed = np.asarray(
        model.predict(batch.reshape(-1, oob.shape[1])), dtype=float
    ).reshape(len(feature_codes), len(x_oob), y_oob.shape[1])
    return np.mean(np.abs(y_oob[None, :, :] - changed), axis=1) - baseline_mae


def _bootstrap_jaccard(
    selection: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    pair_samples: int = 10_000,
) -> pd.DataFrame:
    n_bootstraps, _, n_targets = selection.shape
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_bootstraps, size=pair_samples)
    right = rng.integers(0, n_bootstraps - 1, size=pair_samples)
    right = right + (right >= left)
    rows = []
    for target in range(n_targets):
        values = selection[:, :, target]
        reference_values = reference[:, target]
        intersections = np.sum(values & reference_values[None, :], axis=1)
        unions = np.sum(values | reference_values[None, :], axis=1)
        reference_jaccard = intersections / unions
        pair_intersection = np.sum(values[left] & values[right], axis=1)
        pair_union = np.sum(values[left] | values[right], axis=1)
        pair_jaccard = pair_intersection / pair_union
        expected = 9.0 / values.shape[1]
        pair_kuncheva = (pair_intersection - expected) / (3.0 - expected)
        rows.append(
            {
                "target_position": target,
                "reference_jaccard_mean": float(reference_jaccard.mean()),
                "reference_jaccard_q025": float(np.quantile(reference_jaccard, 0.025)),
                "reference_jaccard_median": float(np.median(reference_jaccard)),
                "reference_jaccard_q975": float(np.quantile(reference_jaccard, 0.975)),
                "pairwise_jaccard_mean": float(pair_jaccard.mean()),
                "pairwise_jaccard_q025": float(np.quantile(pair_jaccard, 0.025)),
                "pairwise_jaccard_median": float(np.median(pair_jaccard)),
                "pairwise_jaccard_q975": float(np.quantile(pair_jaccard, 0.975)),
                "pairwise_kuncheva_mean": float(pair_kuncheva.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_e107(
    *,
    n_bootstraps: int,
    run_label: str,
    output_dir: Path = RESULT_DIR,
) -> dict:
    started = time.perf_counter()
    suffix = _suffix(run_label)
    output_dir = Path(output_dir)
    x, y, style, _ = load_q2()
    target_codes = y.columns.astype(str).tolist()
    candidates = build_candidate_dictionary(x.columns)
    feature_codes = (
        candidates.loc[candidates["eligible_strict"], "feature_code"]
        .astype(str)
        .tolist()
    )
    feature_positions = {feature: x.columns.get_loc(feature) for feature in feature_codes}
    strict_positions = np.asarray([feature_positions[feature] for feature in feature_codes])
    styles = style.astype(str).to_numpy()
    target_values = y.to_numpy(float)
    bootstrap_ranks = np.empty(
        (n_bootstraps, len(feature_codes), len(target_codes)), dtype=np.uint16
    )
    oob_sizes = np.empty(n_bootstraps, dtype=np.uint16)
    co_cluster_counts = np.zeros(
        (len(feature_codes), len(feature_codes)), dtype=np.uint16
    )
    positive_counts = np.zeros((len(feature_codes), len(target_codes)), dtype=np.uint16)
    negative_counts = np.zeros_like(positive_counts)
    uncertain_counts = np.zeros_like(positive_counts)

    for bootstrap in range(n_bootstraps):
        fold_started = time.perf_counter()
        seed = RANDOM_STATE + bootstrap * 10_000
        sample, oob = stratified_bootstrap_indices(
            styles, np.random.default_rng(seed)
        )
        if len(oob) < 20:
            raise RuntimeError(f"bootstrap {bootstrap} has only {len(oob)} OOB rows")
        oob_sizes[bootstrap] = len(oob)
        x_boot = x.iloc[sample].reset_index(drop=True)
        x_oob = x.iloc[oob].reset_index(drop=True)
        y_boot_frame = y.iloc[sample].reset_index(drop=True)
        y_boot = y_boot_frame.to_numpy(float)
        y_oob = target_values[oob]
        correlation = x_boot.corr(method="spearman")

        spearman = spearman_target_scores(x_boot, y_boot_frame).loc[
            feature_codes, target_codes
        ].to_numpy(float)
        spearman_ranks = rank_scores_descending(np.abs(spearman), feature_codes)

        processed = fit_fold_preprocessor(x_boot, x_oob)
        zero_graph = np.zeros((processed.x_train.shape[1],) * 2, dtype=float)
        penalty = build_quadratic_penalty_rows(
            zero_graph,
            n_samples=len(sample),
            lambda_l2=LAMBDA_L2,
            lambda_graph=0.0,
        )
        elastic_model = fit_multi_target_with_penalty_rows(
            processed.x_train,
            y_boot,
            penalty,
            lambda_l1=LAMBDA_L1,
        )
        elastic = elastic_model.coef_[:, strict_positions].T
        elastic_ranks = rank_scores_descending(np.abs(elastic), feature_codes)

        tree = _tree_conditional_scores(
            x_boot,
            y_boot,
            x_oob,
            y_oob,
            feature_codes,
            feature_positions,
            correlation,
            seed=seed + 5000,
        )
        tree_ranks = rank_scores_descending(tree, feature_codes)
        consensus = consensus_ranks_from_methods(
            [spearman_ranks, elastic_ranks, tree_ranks], feature_codes
        )
        bootstrap_ranks[bootstrap] = consensus
        selected = consensus <= 3
        signed_elastic = np.sign(elastic)
        signed_spearman = np.sign(spearman)
        positive = selected & (
            ((signed_elastic >= 0) & (signed_spearman > 0))
            | ((signed_elastic > 0) & (signed_spearman >= 0))
        )
        negative = selected & (
            ((signed_elastic <= 0) & (signed_spearman < 0))
            | ((signed_elastic < 0) & (signed_spearman <= 0))
        )
        uncertain = selected & ~(positive | negative)
        positive_counts += positive
        negative_counts += negative
        uncertain_counts += uncertain

        modules = connected_modules(
            correlation,
            eligible_features=feature_codes,
            threshold=MODULE_THRESHOLD,
        )
        update_coclustering(co_cluster_counts, modules, feature_codes)
        if (bootstrap + 1) % 25 == 0 or bootstrap == 0:
            print(
                json.dumps(
                    {
                        "bootstrap": bootstrap + 1,
                        "total": n_bootstraps,
                        "oob_rows": len(oob),
                        "modules": len(modules),
                        "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    selection = bootstrap_ranks <= 3
    selection_frequency = selection.mean(axis=0)
    rank_median = np.median(bootstrap_ranks, axis=0)
    rank_q25 = np.quantile(bootstrap_ranks, 0.25, axis=0)
    rank_q75 = np.quantile(bootstrap_ranks, 0.75, axis=0)
    reference = pd.read_csv(output_dir / "q2_top3_by_target_E106.csv")
    reference_rank = pd.read_csv(output_dir / "q2_consensus_rankings_E106.csv")[[
        "target_code",
        "feature_code",
        "consensus_rank",
    ]]
    reference_selection = np.zeros(
        (len(feature_codes), len(target_codes)), dtype=bool
    )
    feature_lookup = {value: index for index, value in enumerate(feature_codes)}
    target_lookup = {value: index for index, value in enumerate(target_codes)}
    for row in reference.itertuples():
        reference_selection[
            feature_lookup[row.feature_code], target_lookup[row.target_code]
        ] = True

    rows = []
    for feature_position, feature in enumerate(feature_codes):
        for target_position, target in enumerate(target_codes):
            selected_count = int(selection[:, feature_position, target_position].sum())
            rows.append(
                {
                    "target_code": target,
                    "feature_code": feature,
                    "selection_frequency": selection_frequency[
                        feature_position, target_position
                    ],
                    "median_rank": rank_median[feature_position, target_position],
                    "rank_q25": rank_q25[feature_position, target_position],
                    "rank_q75": rank_q75[feature_position, target_position],
                    "rank_iqr": rank_q75[feature_position, target_position]
                    - rank_q25[feature_position, target_position],
                    "positive_direction_count": int(
                        positive_counts[feature_position, target_position]
                    ),
                    "negative_direction_count": int(
                        negative_counts[feature_position, target_position]
                    ),
                    "uncertain_direction_count": int(
                        uncertain_counts[feature_position, target_position]
                    ),
                    "selected_count": selected_count,
                    "direction_consistency_when_selected": (
                        max(
                            positive_counts[feature_position, target_position],
                            negative_counts[feature_position, target_position],
                        )
                        / selected_count
                        if selected_count
                        else np.nan
                    ),
                }
            )
    stability = pd.DataFrame(rows).merge(
        reference_rank,
        on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
    )
    stability = stability.sort_values(
        [
            "target_code",
            "selection_frequency",
            "median_rank",
            "rank_q75",
            "consensus_rank",
            "feature_code",
        ],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    stability["bootstrap_rank"] = stability.groupby("target_code").cumcount() + 1
    stability = stability.merge(
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
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    stability = stability.merge(
        targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
        on="target_code",
        how="left",
        validate="many_to_one",
    )
    top3 = stability.loc[stability["bootstrap_rank"].le(3)].copy()

    jaccard = _bootstrap_jaccard(
        selection, reference_selection, seed=RANDOM_STATE + 91
    )
    jaccard["target_code"] = [target_codes[int(i)] for i in jaccard.target_position]
    jaccard = jaccard.drop(columns="target_position")

    co_frequency = co_cluster_counts.astype(float) / n_bootstraps
    stable_modules = connected_modules(
        pd.DataFrame(co_frequency, index=feature_codes, columns=feature_codes),
        eligible_features=feature_codes,
        threshold=STABLE_MODULE_THRESHOLD,
    )
    module_rows = []
    membership_rows = []
    for module_position, members in enumerate(stable_modules, start=1):
        module_id = f"SM{module_position:03d}"
        positions = [feature_lookup[feature] for feature in members]
        for feature in members:
            membership_rows.append(
                {
                    "stable_module_id": module_id,
                    "module_size": len(members),
                    "feature_code": feature,
                }
            )
        for target_position, target in enumerate(target_codes):
            module_selected = selection[:, positions, target_position].any(axis=1)
            module_rows.append(
                {
                    "stable_module_id": module_id,
                    "module_size": len(members),
                    "module_members": "|".join(members),
                    "target_code": target,
                    "module_selection_frequency": float(module_selected.mean()),
                }
            )
    module_selection = pd.DataFrame(module_rows)
    membership = pd.DataFrame(membership_rows)
    upper = np.triu_indices(len(feature_codes), k=1)
    keep = co_frequency[upper] >= 0.10
    co_edges = pd.DataFrame(
        {
            "feature_a": np.asarray(feature_codes, dtype=object)[upper[0][keep]],
            "feature_b": np.asarray(feature_codes, dtype=object)[upper[1][keep]],
            "coclustering_frequency": co_frequency[upper][keep],
        }
    ).sort_values("coclustering_frequency", ascending=False)

    anchor_audit = targets.loc[targets["direct_anchor"].notna()].merge(
        stability,
        left_on=["target_code", "direct_anchor"],
        right_on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_bootstrap"),
    )
    anchor_audit = anchor_audit[
        [
            "target_code",
            "target_name_en",
            "direct_anchor",
            "bootstrap_rank",
            "selection_frequency",
            "median_rank",
            "rank_iqr",
            "direction_consistency_when_selected",
        ]
    ].sort_values("target_code")

    np.savez_compressed(
        output_dir / f"q2_bootstrap_ranks_E107{suffix}.npz",
        consensus_ranks=bootstrap_ranks,
        feature_codes=np.asarray(feature_codes, dtype=str),
        target_codes=np.asarray(target_codes, dtype=str),
        oob_sizes=oob_sizes,
        co_cluster_counts=co_cluster_counts,
    )
    stability.to_csv(
        output_dir / f"q2_bootstrap_selection_E107{suffix}.csv", index=False
    )
    top3.to_csv(output_dir / f"q2_bootstrap_top3_E107{suffix}.csv", index=False)
    jaccard.to_csv(output_dir / f"q2_bootstrap_jaccard_E107{suffix}.csv", index=False)
    membership.to_csv(
        output_dir / f"q2_stable_module_membership_E107{suffix}.csv", index=False
    )
    module_selection.to_csv(
        output_dir / f"q2_module_selection_E107{suffix}.csv", index=False
    )
    co_edges.to_csv(
        output_dir / f"q2_bootstrap_coclustering_E107{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    anchor_audit.to_csv(
        output_dir / f"q2_anchor_recovery_E107{suffix}.csv", index=False
    )
    threshold_counts = {
        str(threshold): int((top3["selection_frequency"] >= threshold).sum())
        for threshold in (0.60, 0.70, 0.80)
    }
    manifest = {
        "experiment": EXPERIMENT_ID,
        "status": "complete" if n_bootstraps == 500 else "pilot_complete",
        "run_label": run_label,
        "n_bootstraps": n_bootstraps,
        "sampling": "within-style bootstrap; tree importance evaluated on OOB rows",
        "oob_size": {
            "min": int(oob_sizes.min()),
            "median": float(np.median(oob_sizes)),
            "max": int(oob_sizes.max()),
        },
        "fixed_parameters": {
            "elastic_lambda_l1": LAMBDA_L1,
            "elastic_lambda_l2": LAMBDA_L2,
            "tree_n_estimators": N_ESTIMATORS,
            "tree_max_features": MAX_FEATURES,
            "tree_min_samples_leaf": MIN_SAMPLES_LEAF,
        },
        "strict_candidate_count": len(feature_codes),
        "stable_module_count": len(stable_modules),
        "stable_module_threshold": STABLE_MODULE_THRESHOLD,
        "top3_threshold_counts": threshold_counts,
        "median_top3_selection_frequency": float(top3["selection_frequency"].median()),
        "mean_pairwise_jaccard": float(jaccard["pairwise_jaccard_mean"].mean()),
        "gpu_used": False,
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output_dir / f"q2_run_manifest_E107{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstraps", type=int, default=20)
    parser.add_argument("--run-label", default="pilot_B20")
    args = parser.parse_args()
    if args.n_bootstraps < 2:
        raise ValueError("at least two bootstraps are required")
    run_e107(n_bootstraps=args.n_bootstraps, run_label=args.run_label)


if __name__ == "__main__":
    main()
