"""Q1 data contract and deterministic audit artifacts.

This module never fits a predictive model. It verifies the official tables,
aligns samples by ``beer``, and exports metadata needed by later pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ATTACHMENTS = (
    ROOT
    / "data"
    / "2026初赛"
    / "2026年MathorCup大数据挑战赛-赛道A初赛"
    / "附件"
)
RESULT_DIR = ROOT / "03_结果" / "Q1_感官评分"
DATA_DIR = ROOT / "01_数据"

META_COLUMNS = ["beer", "beer_id", "tasting_category_fine"]
TARGET_GROUPS = (
    (0, 15, "Aroma"),
    (15, 30, "Flavour-core"),
    (30, 45, "Taste-other"),
    (45, 49, "Palate"),
    (49, 50, "Overall"),
)
DIRECT_ANCHORS = {
    "alcohol": "ethanol..v.v.",
    "body": "ethanol..v.v.",
    "bitternes": "bitterness",
    "A_malt_cara": "color.EBC.",
    "co2": "CO2.PSI.",
    "acidity": "pH",
    "lactic": "acids_sum",
    "X4vg": "X4VG",
    "dms": "DMS",
    "diacetyl": "X2.3.butanedione",
    "t2n": "X.E..2.nonenal",
}


def load_q1() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and align Q1 predictors, targets, and style labels by ``beer``."""
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    sensory = pd.read_csv(ATTACHMENTS / "4.csv")

    for name, frame in (("1.csv", chemistry), ("4.csv", sensory)):
        missing = set(META_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing metadata columns: {sorted(missing)}")
        if frame["beer"].isna().any() or not frame["beer"].is_unique:
            raise ValueError(f"{name}: beer must be complete and unique")

    if set(chemistry["beer"]) != set(sensory["beer"]):
        raise ValueError("1.csv and 4.csv do not contain the same beer keys")

    sensory = sensory.set_index("beer").loc[chemistry["beer"]].reset_index()
    if not chemistry["tasting_category_fine"].equals(
        sensory["tasting_category_fine"]
    ):
        raise ValueError("style labels disagree between 1.csv and 4.csv")

    x = chemistry.drop(columns=META_COLUMNS)
    y = sensory.drop(columns=META_COLUMNS)
    style = chemistry["tasting_category_fine"].copy()

    if x.shape != (250, 231):
        raise ValueError(f"unexpected X shape: {x.shape}")
    if y.shape != (250, 50):
        raise ValueError(f"unexpected Y shape: {y.shape}")
    if y.isna().any().any():
        raise ValueError("sensory target matrix contains missing values")
    return x, y, style


def target_dictionary(y: pd.DataFrame) -> pd.DataFrame:
    """Build a target dictionary while correcting the 4.csv/5.csv tail order."""
    attachment5 = pd.read_csv(ATTACHMENTS / "5.csv", nrows=0)
    display_attachment_order = attachment5.columns[1:].tolist()
    # In 5.csv: carbonation, aftertaste, body. In 4.csv: aftertaste, body, co2.
    attachment_position_for_y = list(range(46)) + [47, 48, 46, 49]
    group_by_position = {
        i: group for start, end, group in TARGET_GROUPS for i in range(start, end)
    }
    rows = []
    for y_pos, code in enumerate(y.columns):
        attachment5_pos = attachment_position_for_y[y_pos]
        rows.append(
            {
                "target_code": code,
                "target_name_en": display_attachment_order[attachment5_pos].strip(),
                "target_group": group_by_position[y_pos],
                "attachment4_position": y_pos,
                "attachment5_position": attachment5_pos,
                "direct_anchor": DIRECT_ANCHORS.get(code, ""),
            }
        )
    return pd.DataFrame(rows)


def compound_name_map(x: pd.DataFrame) -> pd.DataFrame:
    """Map the English names in attachment 2 to the field codes in attachment 1.

    Attachment 2 uses exactly the same row/column order as the 231 predictor
    columns in attachment 1. Attachment 8 independently covers every field code
    and is retained as a provenance cross-check; its ``chemical_name`` column
    contains mojibake for 19 Greek-letter names, so the clean display name is
    taken from attachment 2 rather than silently repaired.
    """
    attachment2 = pd.read_csv(ATTACHMENTS / "2.csv")
    thresholds = pd.read_excel(
        ATTACHMENTS / "8.xlsx", sheet_name="Sensory thresholds"
    )
    row_names = attachment2.iloc[:, 0].astype(str).str.strip()
    column_names = pd.Index(attachment2.columns[1:]).astype(str).str.strip()
    if attachment2.shape != (231, 232):
        raise ValueError(f"unexpected 2.csv shape: {attachment2.shape}")
    if not np.array_equal(row_names.to_numpy(), column_names.to_numpy()):
        raise ValueError("2.csv row names and correlation-matrix column names disagree")
    if list(x.columns) == [] or len(x.columns) != len(row_names):
        raise ValueError("1.csv predictor count does not match 2.csv names")
    if thresholds["field_name_R"].duplicated().any():
        raise ValueError("8.xlsx field_name_R must be unique")

    mapping = pd.DataFrame(
        {
            "position_1based": np.arange(1, len(x.columns) + 1),
            "compound_name": row_names.to_numpy(),
            "feature_code": x.columns.to_numpy(),
        }
    ).merge(
        thresholds[
            [
                "field_name_R",
                "field_name_python",
                "chemical_name",
                "chemical_name_short",
                "unit",
            ]
        ],
        left_on="feature_code",
        right_on="field_name_R",
        how="left",
        validate="one_to_one",
    )
    if mapping["field_name_R"].isna().any():
        missing = mapping.loc[mapping["field_name_R"].isna(), "feature_code"].tolist()
        raise ValueError(f"8.xlsx lacks field codes: {missing}")
    mapping["mapping_basis"] = (
        "official positional identity: 1.csv predictor columns = "
        "2.csv row/column names; cross-checked by 8.xlsx field_name_R"
    )
    return mapping


def validate_compound_name_map(mapping: pd.DataFrame, x: pd.DataFrame) -> dict:
    """Validate a compound map against all three official attachment contracts."""
    required = {"position_1based", "compound_name", "feature_code", "field_name_R"}
    missing_columns = required - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"mapping missing columns: {sorted(missing_columns)}")
    if len(mapping) != 231:
        raise ValueError(f"mapping must contain 231 rows, got {len(mapping)}")
    if not mapping["compound_name"].is_unique:
        raise ValueError("compound_name must be unique")
    if not mapping["feature_code"].is_unique:
        raise ValueError("feature_code must be unique")
    if mapping["feature_code"].tolist() != x.columns.tolist():
        raise ValueError("mapping feature-code order disagrees with 1.csv")
    if not mapping["feature_code"].equals(mapping["field_name_R"]):
        raise ValueError("8.xlsx field_name_R cross-check failed")
    return {
        "rows": len(mapping),
        "unique_compound_names": int(mapping["compound_name"].nunique()),
        "unique_feature_codes": int(mapping["feature_code"].nunique()),
        "attachment8_field_codes_matched": int(mapping["field_name_R"].notna().sum()),
        "status": "PASS",
    }


def audit_correlations(x: pd.DataFrame, y: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Verify that attachments 2/5/6 are Spearman correlations."""
    attachment2 = pd.read_csv(ATTACHMENTS / "2.csv")
    attachment5 = pd.read_csv(ATTACHMENTS / "5.csv")
    attachment6 = pd.read_csv(ATTACHMENTS / "6.csv")

    official2 = attachment2.iloc[:, 1:].to_numpy(float)
    calc2 = x.corr(method="spearman").to_numpy()

    attachment_y_order = list(range(46)) + [48, 46, 47, 49]
    y_attachment = y.iloc[:, attachment_y_order]
    official5 = attachment5.iloc[:, 1:].to_numpy(float)
    calc5 = y_attachment.corr(method="spearman").to_numpy()

    cross = np.array(
        [
            [x[x_col].corr(y_attachment[y_col], method="spearman") for y_col in y_attachment]
            for x_col in x
        ]
    )
    official6 = attachment6.iloc[:, 1:].to_numpy(float)
    mapping_rows = []
    mapped = []
    for row_position, official_row in enumerate(official6):
        row_errors = np.nanmean(np.abs(cross - official_row), axis=1)
        best_position = int(np.nanargmin(row_errors))
        mapped.append(cross[best_position])
        mapping_rows.append(
            {
                "attachment6_position": row_position,
                "attachment6_name": str(attachment6.iloc[row_position, 0]).strip(),
                "feature_code": x.columns[best_position],
                "mean_absolute_difference": float(row_errors[best_position]),
            }
        )
    mapped_array = np.asarray(mapped)

    audit = {
        "attachment2_spearman_max_abs_diff": float(np.nanmax(np.abs(official2 - calc2))),
        "attachment5_spearman_max_abs_diff": float(np.nanmax(np.abs(official5 - calc5))),
        "attachment6_spearman_max_abs_diff": float(
            np.nanmax(np.abs(official6 - mapped_array))
        ),
        "attachment6_unique_feature_matches": int(
            pd.DataFrame(mapping_rows)["feature_code"].nunique()
        ),
        "sensory_tail_order_attachment4": ["alcohol", "aftertaste", "body", "co2", "overall"],
        "sensory_tail_order_attachment5_6": [
            "Alcohol",
            "Carbonation",
            "Aftertaste",
            "Body",
            "Overall",
        ],
    }
    return audit, pd.DataFrame(mapping_rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x, y, style = load_q1()
    dictionary = target_dictionary(y)
    name_map = compound_name_map(x)
    map_audit = validate_compound_name_map(name_map, x)
    corr_audit, row_map = audit_correlations(x, y)

    dictionary.to_csv(RESULT_DIR / "q1_target_dictionary.csv", index=False)
    row_map.to_csv(RESULT_DIR / "q1_attachment6_row_map.csv", index=False)
    name_map.to_csv(DATA_DIR / "compound_name_map.csv", index=False)

    audit = {
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_targets": y.shape[1],
        "n_styles": int(style.nunique()),
        "x_missing_cells": int(x.isna().sum().sum()),
        "x_missing_columns": int((x.isna().sum() > 0).sum()),
        "y_missing_cells": int(y.isna().sum().sum()),
        "beer_key": "beer",
        "forbidden_model_fields": ["beer", "beer_id"],
        "compound_name_map": map_audit,
        "correlation_audit": corr_audit,
    }
    (RESULT_DIR / "q1_data_contract.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

