"""Attach OAV, semantic, and literature evidence to the Q2 statistical Top-3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemical_evidence import semantic_support
from q2_contract import ATTACHMENTS, RESULT_DIR


EXPERIMENT_ID = "E112"


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


def _limitations(row: pd.Series) -> str:
    values = []
    if row["predictability_tier"] == "P3_weak":
        values.append("目标OOF预测较弱")
    if row["statistical_robustness_tier"] == "R3_exploratory":
        values.append("四路统计稳定性较弱")
    if row["style_evidence"] == "style_sensitive":
        values.append("风格校正后未进入Top10")
    if row["semantic_evidence_level"] == 0:
        values.append("缺少描述词特异化学证据")
    if row["oav_evidence"] == "mostly_below_threshold":
        values.append("多数样本低于附件阈值")
    return "；".join(values) if values else "未见主要降级项"


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    top3 = pd.read_csv(output_dir / "q2_final_top3_statistical_E111.csv")
    calibrated = pd.read_excel(
        ATTACHMENTS / "8.xlsx", sheet_name="Calibrated chemical dataset"
    )
    rows = []
    for row in top3.itertuples(index=False):
        semantic = semantic_support(
            row.target_code,
            row.feature_code,
            row.chemical_name_short,
            row.origin_paper,
        )
        threshold = float(row.threshold_median) if pd.notna(row.threshold_median) else np.nan
        oav = _oav_summary(row.feature_code, threshold, calibrated)
        rows.append(
            {
                "target_code": row.target_code,
                "feature_code": row.feature_code,
                "semantic_evidence_level": semantic.level,
                "semantic_evidence": semantic.label,
                "semantic_reason": semantic.reason,
                "literature_code": semantic.literature_code,
                **oav,
            }
        )
    evidence = pd.DataFrame(rows)
    final = top3.merge(
        evidence, on=["target_code", "feature_code"], validate="one_to_one"
    )
    final["chemical_evidence_grade"] = final.apply(_grade, axis=1)
    final["evidence_limitations"] = final.apply(_limitations, axis=1)
    final["claim_scope"] = "稳定关键预测关联；不作因果解释"
    final = final.sort_values(["target_code", "consensus_rank"]).reset_index(drop=True)

    audit = final[
        [
            "target_code",
            "target_name_en",
            "consensus_rank",
            "feature_code",
            "chemical_name_short",
            "chemical_evidence_grade",
            "semantic_evidence",
            "semantic_reason",
            "literature_code",
            "oav_evidence",
            "oav_median",
            "oav_above_1_fraction",
            "statistical_robustness_tier",
            "predictability_tier",
            "bootstrap_selection_frequency",
            "style_adjusted_rank",
            "leave_style_top3_frequency",
            "evidence_limitations",
        ]
    ]
    final.to_csv(output_dir / "q2_final_top3_E112.csv", index=False)
    audit.to_csv(output_dir / "q2_chemical_evidence_audit_E112.csv", index=False)

    manifest = {
        "experiment": EXPERIMENT_ID,
        "status": "complete",
        "rows": len(final),
        "targets": int(final["target_code"].nunique()),
        "grade_counts": {
            str(key): int(value)
            for key, value in final["chemical_evidence_grade"].value_counts().items()
        },
        "semantic_evidence_counts": {
            str(key): int(value)
            for key, value in final["semantic_evidence"].value_counts().items()
        },
        "oav_available_rows": int(final["oav_n"].gt(0).sum()),
        "oav_median_above_threshold_rows": int(
            final["oav_evidence"].eq("median_above_threshold").sum()
        ),
        "direct_identity_rows": int(final["semantic_evidence_level"].eq(4).sum()),
        "causal_claims": False,
        "gpu_used": False,
    }
    (output_dir / "q2_run_manifest_E112.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
