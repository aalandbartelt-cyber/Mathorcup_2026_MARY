"""Fail-closed input contracts for Q2 E118 consensus and E119 chemistry.

This module validates upstream artifacts only.  It intentionally contains no
ranking fusion, chemical grading, or file-writing side effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _manifest(result_dir: Path, filename: str) -> tuple[Path, dict]:
    path = Path(result_dir) / filename
    if not path.is_file():
        return path, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def _check_complete(
    result_dir: Path,
    filename: str,
    label: str,
    problems: list[str],
) -> dict:
    _, payload = _manifest(result_dir, filename)
    if payload.get("status") != "complete":
        problems.append(f"{label} manifest must have status=complete")
    return payload


def audit_e118_inputs(result_dir: Path) -> dict:
    """Require all statistical families before consensus is even considered."""

    result_dir = Path(result_dir)
    problems: list[str] = []
    e113 = _check_complete(result_dir, "q2_run_manifest_E113.json", "E113", problems)
    e114 = _check_complete(result_dir, "q2_run_manifest_E114.json", "E114", problems)
    e115 = _check_complete(
        result_dir, "q2_run_manifest_E115_final_nnls.json", "E115 final NNLS", problems
    )
    e117 = _check_complete(result_dir, "q2_run_manifest_E117.json", "E117", problems)
    if e113 and (e113.get("n_outer_folds") != 30 or e113.get("n_features") != 218):
        problems.append("E113 dimensions disagree with the frozen 30-fold/218-field contract")
    if e114 and e114.get("n_outer_folds") != 30:
        problems.append("E114 must contain all 30 outer folds")
    if e115 and (
        e115.get("repeat_ids") != list(range(10))
        or e115.get("n_outer_folds") != 30
        or e115.get("n_features_strict") != 218
        or e115.get("n_targets") != 50
    ):
        problems.append("E115 final NNLS dimensions/repeats are incomplete")
    if e117 and (
        e117.get("formal_run") is not True
        or e117.get("n_features_strict") != 218
        or e117.get("n_targets") != 50
    ):
        problems.append("E117 must be a formal 218-field/50-target run")
    return {
        "stage": "E118_input_gate",
        "status": "ready" if not problems else "blocked",
        "formal_ranking_generated": False,
        "problems": problems,
        "upstream_status": {
            "E113": e113.get("status"),
            "E114": e114.get("status"),
            "E115_final_NNLS": e115.get("status"),
            "E117": e117.get("status"),
        },
    }


def audit_e119_inputs(result_dir: Path) -> dict:
    """Require a promoted, complete E118 statistical table without reranking it."""

    result_dir = Path(result_dir)
    problems: list[str] = []
    manifest = _check_complete(result_dir, "q2_run_manifest_E118.json", "E118", problems)
    decision_path = result_dir / "q2_replacement_decision_E118.json"
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {}
    )
    if decision.get("promoted") is not True:
        problems.append("E118 replacement decision is not promoted")
    table_path = result_dir / "q2_final_top3_statistical_E118.csv"
    if not table_path.is_file():
        problems.append("missing E118 statistical Top-3 table")
        rows = targets = unique_pairs = None
    else:
        table = pd.read_csv(table_path)
        required = {"target_code", "consensus_rank", "feature_code"}
        if not required.issubset(table.columns):
            problems.append("E118 Top-3 table is missing identity/rank columns")
            rows = targets = unique_pairs = None
        else:
            rows = len(table)
            targets = table["target_code"].nunique()
            unique_pairs = table[["target_code", "consensus_rank"]].drop_duplicates().shape[0]
            if rows != 150 or targets != 50 or unique_pairs != 150:
                problems.append("E118 Top-3 table must have 150 unique target-rank rows")
            if sorted(table["consensus_rank"].dropna().unique().tolist()) != [1, 2, 3]:
                problems.append("E118 consensus_rank must be exactly 1, 2, 3")
    return {
        "stage": "E119_input_gate",
        "status": "ready" if not problems else "blocked",
        "chemical_grading_generated": False,
        "reranking_permitted": False,
        "problems": problems,
        "e118_status": manifest.get("status"),
        "e118_promoted": decision.get("promoted"),
        "statistical_rows": rows,
        "statistical_targets": targets,
        "unique_target_rank_pairs": unique_pairs,
    }
