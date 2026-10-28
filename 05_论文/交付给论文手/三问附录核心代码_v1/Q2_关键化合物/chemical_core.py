# Source logical name: official_chemical_grading; SHA256: f8f14fd6f30f0cd30fb007fba204109fde6cc6a7d05c0f528e379cf381fa2bcd
from __future__ import annotations
import numpy as np
import pandas as pd


def _oav_summary(feature: str, threshold: float, calibrated: pd.DataFrame) -> dict:
    empty = {
        "oav_n": 0,
        "raw_concentration_median": np.nan,
        "oav_q25": np.nan,
        "oav_median": np.nan,
        "oav_q75": np.nan,
        "oav_max": np.nan,
        "oav_above_1_fraction": np.nan,
        "oav_evidence": "unavailable",
    }
    if feature not in calibrated.columns or not np.isfinite(threshold) or threshold <= 0:
        return empty
    values = pd.to_numeric(calibrated[feature], errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return empty
    oav = values / float(threshold)
    median = float(np.median(oav))
    fraction = float(np.mean(oav >= 1.0))
    if median >= 1.0:
        evidence = "median_above_threshold"
    elif fraction >= 0.25:
        evidence = "subset_above_threshold"
    else:
        evidence = "mostly_below_threshold"
    return {
        "oav_n": int(len(values)),
        "raw_concentration_median": float(np.median(values)),
        "oav_q25": float(np.quantile(oav, 0.25)),
        "oav_median": median,
        "oav_q75": float(np.quantile(oav, 0.75)),
        "oav_max": float(np.max(oav)),
        "oav_above_1_fraction": fraction,
        "oav_evidence": evidence,
    }


def _grade(row: pd.Series) -> str:
    semantic = int(row["semantic_evidence_level"])
    statistical = str(row["statistical_robustness_tier"])
    prediction = str(row["predictability_tier"])
    if (
        semantic == 4
        and statistical in {"R1_strong", "R2_moderate"}
        and prediction != "P3_weak"
    ) or (
        semantic == 3
        and statistical == "R1_strong"
        and prediction != "P3_weak"
    ):
        return "A"
    if (
        semantic >= 2
        and statistical in {"R1_strong", "R2_moderate"}
        and prediction != "P3_weak"
    ) or (semantic >= 3 and statistical in {"R1_strong", "R2_moderate"}):
        return "B"
    if semantic >= 1 or statistical in {"R1_strong", "R2_moderate"}:
        return "C"
    return "D"
