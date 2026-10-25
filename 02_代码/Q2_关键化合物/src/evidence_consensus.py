"""Support-aware fusion and replacement gates for Q2 statistical evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_payload_passed(payload: dict) -> bool:
    """Accept current and legacy machine-audit schemas without weakening checks."""

    if payload.get("audit_status") is not None:
        return payload.get("audit_status") == "pass" and not payload.get("failed_checks", [])
    return (
        payload.get("status") == "complete"
        and payload.get("scope_verified") in {True, "outer_train_fold"}
        and not payload.get("errors", [])
    )


def fuse_evidence_families(
    frame: pd.DataFrame,
    *,
    rank_columns: list[str],
    rrf_k: float = 60.0,
) -> pd.DataFrame:
    required = {"target_code", "feature_code", *rank_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing fusion columns: {sorted(required - set(frame.columns))}")
    if len(rank_columns) < 2 or float(rrf_k) <= 0:
        raise ValueError("at least two rank families and a positive rrf_k are required")
    result = frame.copy()
    ranks = result[rank_columns].apply(pd.to_numeric, errors="coerce")
    if ranks.isna().any().any() or (ranks <= 0).any().any():
        raise ValueError("family ranks must be finite positive values")
    result["family_top3_support"] = ranks.le(3).sum(axis=1).astype(int)
    result["family_top10_support"] = ranks.le(10).sum(axis=1).astype(int)
    result["family_rrf_score"] = (1.0 / (float(rrf_k) + ranks)).sum(axis=1)
    result["family_best_rank"] = ranks.min(axis=1)
    result = result.sort_values(
        [
            "target_code",
            "family_top10_support",
            "family_top3_support",
            "family_rrf_score",
            "family_best_rank",
            "feature_code",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["consensus_rank"] = result.groupby("target_code", sort=False).cumcount().add(1)
    return result


def reference_half_jaccard(top3: pd.DataFrame, halves: pd.DataFrame) -> np.ndarray:
    required_top = {"target_code", "feature_code"}
    required_half = {"pair", "half", "target_code", "feature_code"}
    if not required_top.issubset(top3.columns) or not required_half.issubset(halves.columns):
        raise ValueError("reference/half selection identity columns are incomplete")
    if not top3.groupby("target_code").size().eq(3).all():
        raise ValueError("reference table must contain exactly three fields per target")
    reference = {
        target: set(frame["feature_code"].astype(str))
        for target, frame in top3.groupby("target_code", sort=True)
    }
    values = []
    for (_, _, target), frame in halves.groupby(["pair", "half", "target_code"], sort=True):
        if target not in reference:
            raise ValueError(f"half selection target is absent from reference: {target}")
        selected = set(frame["feature_code"].astype(str))
        union = reference[target] | selected
        values.append(len(reference[target] & selected) / len(union) if union else 1.0)
    return np.asarray(values, dtype=float)


def replacement_decision(
    *,
    complete_rows: int,
    prediction_ok: bool,
    stability_gains: dict[str, bool],
    anchors_preserved: bool,
    fdp_ok: bool,
    leakage_ok: bool,
    required_stability_gains: int = 2,
) -> dict:
    gains = int(sum(bool(value) for value in stability_gains.values()))
    checks = {
        "complete_50x3": int(complete_rows) == 150,
        "prediction_ok": bool(prediction_ok),
        "stability_gain_count_ok": gains >= int(required_stability_gains),
        "anchors_preserved": bool(anchors_preserved),
        "fdp_ok": bool(fdp_ok),
        "leakage_ok": bool(leakage_ok),
    }
    return {
        "promoted": all(checks.values()),
        "required_stability_gains": int(required_stability_gains),
        "observed_stability_gains": gains,
        "stability_gains": {key: bool(value) for key, value in stability_gains.items()},
        "checks": checks,
    }
