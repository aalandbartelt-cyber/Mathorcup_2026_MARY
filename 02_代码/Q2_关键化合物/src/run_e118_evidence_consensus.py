"""Fuse three fold-safe Q2 evidence families and resolve the replacement gate."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from downstream_input_gates import audit_e118_inputs
from evidence_consensus import (
    audit_payload_passed,
    fuse_evidence_families,
    reference_half_jaccard,
    replacement_decision,
)
from q2_contract import RESULT_DIR


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "02_代码" / "Q2_关键化合物" / "configs" / "E118_evidence_consensus.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("experiment") != "E118":
        raise ValueError("configuration experiment must be E118")
    if config.get("candidate_scope") != "strict_compounds_218":
        raise ValueError("E118 candidate scope must remain strict_compounds_218")
    if config.get("chemical_evidence_in_ranking") is not False:
        raise ValueError("chemical evidence must remain post-ranking")
    if int(config["required_stability_gains"]) != 2:
        raise ValueError("E118 requires exactly two of three pre-registered stability gains")
    return config


def graph_module_family(result_dir: Path, identities: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(result_dir / "q2_stable_module_membership_E107.csv")
    module_selection = pd.read_csv(result_dir / "q2_module_selection_E107.csv")
    graph = identities.merge(
        membership[["feature_code", "stable_module_id", "module_size"]],
        on="feature_code",
        how="left",
        validate="many_to_one",
    ).merge(
        module_selection[["target_code", "stable_module_id", "module_selection_frequency"]],
        on=["target_code", "stable_module_id"],
        how="left",
        validate="many_to_one",
    )
    if graph[["stable_module_id", "module_size", "module_selection_frequency"]].isna().any().any():
        raise ValueError("E107 graph-module fallback does not cover all strict target-feature pairs")
    graph["graph_module_rank"] = (
        graph.groupby("target_code")["module_selection_frequency"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return graph


def leave_style_frequency(result_dir: Path) -> pd.DataFrame:
    rankings = pd.read_csv(result_dir / "q2_leave_one_style_rankings_E108.csv.gz")
    return (
        rankings.assign(selected=lambda frame: frame["consensus_rank"].le(3))
        .groupby(["target_code", "feature_code"], as_index=False)
        .agg(
            leave_style_survival=("selected", "mean"),
            leave_style_deletions=("selected", "size"),
        )
    )


def stability_summary(
    top3: pd.DataFrame,
    half_selections: pd.DataFrame,
) -> dict[str, float]:
    jaccard = reference_half_jaccard(top3, half_selections)
    return {
        "median_cpss_selection_probability": float(top3["selection_probability"].median()),
        "mean_reference_half_jaccard": float(jaccard.mean()),
        "median_leave_style_survival": float(top3["leave_style_survival"].median()),
        "mean_bootstrap_selection_frequency": float(top3["bootstrap_selection_frequency"].mean()),
    }


def run(output_dir: Path = RESULT_DIR, config_path: Path = CONFIG_PATH) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    config = load_config(config_path)
    gate = audit_e118_inputs(output_dir)
    if gate["status"] != "ready":
        raise RuntimeError("E118 input gate failed: " + "; ".join(gate["problems"]))

    graph_decision = json.loads((output_dir / "q2_graph_decision_E113.json").read_text(encoding="utf-8"))
    if graph_decision.get("promote_partial_graph") is not False:
        raise ValueError("this frozen E118 configuration expects the E107 graph-module fallback")

    alignment = pd.read_csv(output_dir / "q2_cpss_nnls_alignment_E117.csv.gz")
    identities = alignment[["target_code", "feature_code"]].copy()
    partial = pd.read_csv(output_dir / "q2_partial_effect_rankings_E114.csv")
    graph = graph_module_family(output_dir, identities)
    bootstrap = pd.read_csv(output_dir / "q2_bootstrap_selection_E107.csv")
    style = leave_style_frequency(output_dir)
    candidates = pd.read_csv(output_dir / "q2_candidate_dictionary.csv")
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    error = pd.read_csv(output_dir / "q2_error_control_E117.csv")

    family = (
        alignment.merge(
            partial[
                [
                    "target_code",
                    "feature_code",
                    "partial_effect_rank",
                    "top3_frequency",
                    "top10_frequency",
                    "median_effect",
                    "direction",
                    "sign_consistency",
                ]
            ],
            on=["target_code", "feature_code"],
            validate="one_to_one",
        )
        .merge(
            graph[
                [
                    "target_code",
                    "feature_code",
                    "stable_module_id",
                    "module_size",
                    "module_selection_frequency",
                    "graph_module_rank",
                ]
            ],
            on=["target_code", "feature_code"],
            validate="one_to_one",
        )
        .merge(
            bootstrap[["target_code", "feature_code", "selection_frequency"]].rename(
                columns={"selection_frequency": "bootstrap_selection_frequency"}
            ),
            on=["target_code", "feature_code"],
            validate="one_to_one",
        )
        .merge(style, on=["target_code", "feature_code"], validate="one_to_one")
        .merge(
            candidates.loc[candidates["eligible_strict"].eq(True), [
                "feature_code", "compound_name", "chemical_name_short", "unit", "factor_type"
            ]],
            on="feature_code",
            validate="many_to_one",
        )
        .merge(targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
               on="target_code", validate="many_to_one")
        .rename(
            columns={
                "nnls_predictive_rank": "predictive_rank",
                "partial_effect_rank": "partial_rank",
                "graph_module_rank": "module_rank",
            }
        )
    )
    if len(family) != 10_900 or family[["target_code", "feature_code"]].duplicated().any():
        raise ValueError("E118 family table must contain 50x218 unique rows")

    ranking = fuse_evidence_families(
        family,
        rank_columns=["predictive_rank", "partial_rank", "module_rank"],
        rrf_k=float(config["fusion"]["rrf_k"]),
    )
    confirmed_threshold = (
        error.loc[error["is_confirmed_threshold"].eq(True), ["target_code", "threshold"]]
        .rename(columns={"threshold": "cpss_confirmed_threshold"})
    )
    ranking = ranking.merge(confirmed_threshold, on="target_code", how="left", validate="many_to_one")
    ranking["cpss_status"] = np.where(
        ranking["cpss_confirmed_threshold"].notna()
        & ranking["selection_probability"].ge(ranking["cpss_confirmed_threshold"]),
        "confirmed",
        "exploratory",
    )
    ranking["predictive_dependence_positive"] = ranking["nnls_median_delta_mae"].gt(0)
    ranking["family_sources"] = "final_NNLS|fold_local_partial_effect|bootstrap_graph_module"
    top3 = ranking.loc[ranking["consensus_rank"].le(3)].copy()
    if len(top3) != 150 or not top3.groupby("target_code").size().eq(3).all():
        raise ValueError("E118 statistical Top-3 must contain exactly 50x3 rows")

    half_selections = pd.read_csv(output_dir / "q2_cpss_half_selections_E117.csv.gz")
    old_top3 = pd.read_csv(output_dir / "q2_final_top3_statistical_E111.csv")
    old_for_compare = (
        old_top3[["target_code", "feature_code"]]
        .merge(
            ranking[
                [
                    "target_code",
                    "feature_code",
                    "selection_probability",
                    "bootstrap_selection_frequency",
                    "leave_style_survival",
                ]
            ],
            on=["target_code", "feature_code"],
            validate="one_to_one",
        )
    )
    new_metrics = stability_summary(top3, half_selections)
    old_metrics = stability_summary(old_for_compare, half_selections)
    metric_names = list(config["stability_metrics"])
    stability_gains = {
        name: new_metrics[name] > old_metrics[name] + 1e-12 for name in metric_names
    }
    comparison = pd.DataFrame(
        [
            {
                "metric": name,
                "previous_value": old_metrics[name],
                "candidate_value": new_metrics[name],
                "delta": new_metrics[name] - old_metrics[name],
                "improved": stability_gains[name],
            }
            for name in metric_names
        ]
        + [
            {
                "metric": "mean_bootstrap_selection_frequency",
                "previous_value": old_metrics["mean_bootstrap_selection_frequency"],
                "candidate_value": new_metrics["mean_bootstrap_selection_frequency"],
                "delta": new_metrics["mean_bootstrap_selection_frequency"]
                - old_metrics["mean_bootstrap_selection_frequency"],
                "improved": new_metrics["mean_bootstrap_selection_frequency"]
                > old_metrics["mean_bootstrap_selection_frequency"] + 1e-12,
            }
        ]
    )

    prior_anchors = pd.read_csv(output_dir / "q2_anchor_recovery_E111.csv")
    prior_anchors = prior_anchors.loc[
        prior_anchors["consensus_rank"].le(3) & prior_anchors["direct_anchor"].notna(),
        ["target_code", "direct_anchor"],
    ]
    selected_pairs = set(map(tuple, top3[["target_code", "feature_code"]].to_numpy()))
    anchor_pairs = set(map(tuple, prior_anchors[["target_code", "direct_anchor"]].to_numpy()))
    missing_anchors = sorted(anchor_pairs - selected_pairs)
    audits = {
        label: json.loads((output_dir / filename).read_text(encoding="utf-8"))
        for label, filename in {
            "E113": "q2_audit_E113.json",
            "E115": "q2_audit_E115_final_nnls.json",
            "E117": "q2_audit_E117.json",
        }.items()
    }
    leakage_ok = all(audit_payload_passed(payload) for payload in audits.values())
    confirmed_rows = error.loc[error["is_confirmed_threshold"].eq(True)]
    fdp_ok = bool(confirmed_rows["empirical_fdp"].le(float(config["confirmed_fdp_max"])).all())
    e115_manifest = json.loads(
        (output_dir / "q2_run_manifest_E115_final_nnls.json").read_text(encoding="utf-8")
    )
    prediction_ok = (
        e115_manifest.get("status") == "complete"
        and e115_manifest.get("q1_retrained") is False
        and e115_manifest.get("nnls_refit") is False
    )
    decision = replacement_decision(
        complete_rows=len(top3),
        prediction_ok=prediction_ok,
        stability_gains=stability_gains,
        anchors_preserved=not missing_anchors,
        fdp_ok=fdp_ok,
        leakage_ok=leakage_ok,
        required_stability_gains=int(config["required_stability_gains"]),
    )
    decision.update(
        {
            "experiment_id": "E118",
            "comparison_baseline": "previous_robust_statistical_top3",
            "candidate": "three_family_support_aware_rrf",
            "prediction_policy": "ranking_only; frozen final NNLS predictions unchanged",
            "prior_anchor_count": len(anchor_pairs),
            "missing_prior_anchors": [
                {"target_code": target, "feature_code": feature}
                for target, feature in missing_anchors
            ],
            "confirmed_relations_in_candidate_top3": int(top3["cpss_status"].eq("confirmed").sum()),
            "chemical_evidence_used_for_ranking": False,
        }
    )

    paths = {
        "rankings": output_dir / "q2_evidence_family_rankings_E118.csv.gz",
        "top3": output_dir / "q2_final_top3_statistical_E118.csv",
        "comparison": output_dir / "q2_replacement_comparison_E118.csv",
        "decision": output_dir / "q2_replacement_decision_E118.json",
        "manifest": output_dir / "q2_run_manifest_E118.json",
    }
    ranking.to_csv(paths["rankings"], index=False, compression="gzip")
    top3.to_csv(paths["top3"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    paths["decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment_id": "E118",
        "status": "complete",
        "promoted": decision["promoted"],
        "n_candidates": len(ranking),
        "n_top3_rows": len(top3),
        "n_targets": top3["target_code"].nunique(),
        "n_features_strict": ranking["feature_code"].nunique(),
        "families": config["families"],
        "fusion": config["fusion"],
        "graph_decision": graph_decision,
        "chemical_evidence_used_for_ranking": False,
        "input_gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "gpu_used": False,
        "outputs": [str(path) for key, path in paths.items() if key != "manifest"],
    }
    manifest["output_sha256"] = {
        path.name: sha256(path) for key, path in paths.items() if key != "manifest"
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
