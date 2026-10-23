"""Q1 data contract and deterministic audit artifacts.

This module never fits a predictive model. It verifies the official tables,
aligns samples by ``beer``, and exports metadata needed by later pipelines.
"""

from __future__ import annotations

import hashlib
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


def feature_scale_audit(chemistry: pd.DataFrame) -> pd.DataFrame:
    """Identify fields already stored as log10 by comparison with attachment 8.

    Only positive calibrated values are compared because attachment 8 uses zero
    for some measurements at/below detection limits. A field is labelled
    ``log10`` when ``1.csv == log10(calibrated value)`` matches more positive
    entries than the raw-value identity; otherwise it is labelled ``raw``.
    No values in ``1.csv`` are transformed by this audit.
    """
    calibrated = pd.read_excel(
        ATTACHMENTS / "8.xlsx", sheet_name="Calibrated chemical dataset"
    )
    if not chemistry["beer"].is_unique or not calibrated["beer"].is_unique:
        raise ValueError("beer must be unique before scale audit")
    if set(chemistry["beer"]) != set(calibrated["beer"]):
        raise ValueError("attachment 1 and calibrated attachment 8 beer keys disagree")
    calibrated = calibrated.set_index("beer").loc[chemistry["beer"]]
    feature_columns = [column for column in chemistry.columns if column not in META_COLUMNS]
    shared = [column for column in feature_columns if column in calibrated.columns]
    rows = []
    for column in shared:
        stored = chemistry[column].to_numpy(float)
        raw = calibrated[column].to_numpy(float)
        positive = np.isfinite(stored) & np.isfinite(raw) & (raw > 0)
        if not positive.any():
            raise ValueError(f"no positive calibrated values for {column}")
        raw_match = np.isclose(
            stored[positive], raw[positive], rtol=1e-7, atol=2e-6
        )
        log10_match = np.isclose(
            stored[positive], np.log10(raw[positive]), rtol=1e-7, atol=2e-6
        )
        scale = "log10" if log10_match.mean() > raw_match.mean() else "raw"
        selected_match = log10_match if scale == "log10" else raw_match
        rows.append(
            {
                "feature_code": column,
                "scale_in_1_csv": scale,
                "n_positive_comparisons": int(positive.sum()),
                "positive_value_match_rate": float(selected_match.mean()),
                "raw_identity_match_rate": float(raw_match.mean()),
                "log10_identity_match_rate": float(log10_match.mean()),
                "calibrated_zero_count": int(np.sum(np.isfinite(raw) & (raw == 0))),
                "negative_values_in_1_csv": int(np.sum(stored < 0)),
                "model_action": (
                    "keep supplied log10 values; do not log again"
                    if scale == "log10"
                    else "keep supplied raw values"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 70:
        raise ValueError(f"expected 70 attachment-8 comparable fields, got {len(audit)}")
    if (audit["scale_in_1_csv"] == "log10").sum() != 37:
        raise ValueError("official scale audit no longer identifies 37 log10 fields")
    return audit


def missingness_audit(x: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature missingness without imputing the full dataset."""
    missing = x.isna().sum()
    return pd.DataFrame(
        {
            "feature_code": x.columns,
            "missing_cells": missing.to_numpy(int),
            "missing_rate": (missing / len(x)).to_numpy(float),
            "model_action": np.where(
                missing.to_numpy() > 0,
                "training-fold median + training-fold missing indicator",
                "retain as supplied",
            ),
        }
    )


def preprocessing_steps(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Build the paper-facing cleaning table with explicit shape changes."""
    missing_columns = int((x.isna().sum() > 0).sum())
    rows = [
        {
            "step_id": "P0",
            "stage": "读取",
            "operation": "读取附件1与附件4，核对字段和数值类型",
            "shape_before": "1.csv 250×234; 4.csv 250×53",
            "shape_after": "1.csv 250×234; 4.csv 250×53",
            "rows_before": "250",
            "cols_before": "234/53",
            "rows_after": "250",
            "cols_after": "234/53",
            "fit_scope": "全数据契约核验，不估计模型参数",
            "sample_deletion": "no",
            "reason": "确认实际附件规模，避免按题面约数预设列数",
        },
        {
            "step_id": "P1",
            "stage": "连接",
            "operation": "以beer唯一键对齐附件1与附件4；禁用beer_id",
            "shape_before": "两个250行表",
            "shape_after": "250个一一对应样本",
            "rows_before": "250",
            "cols_before": "—",
            "rows_after": "250",
            "cols_after": "—",
            "fit_scope": "全数据键约束",
            "sample_deletion": "no",
            "reason": "beer_id在附件4中有浮点精度损失，beer完整且唯一",
        },
        {
            "step_id": "P2",
            "stage": "拆分",
            "operation": "移除标识字段，形成231维X和50维连续Y",
            "shape_before": "250×234; 250×53",
            "shape_after": f"250×{x.shape[1]}; 250×{y.shape[1]}",
            "rows_before": "250",
            "cols_before": "234/53",
            "rows_after": "250",
            "cols_after": f"{x.shape[1]}/{y.shape[1]}",
            "fit_scope": "字段白名单",
            "sample_deletion": "no",
            "reason": "beer仅作主键，beer_id不连接/建模，风格仅作分层和基线",
        },
        {
            "step_id": "P3",
            "stage": "尺度",
            "operation": "附件8可核验的70个字段中，37个log10字段保持附件尺度",
            "shape_before": f"250×{x.shape[1]}",
            "shape_after": f"250×{x.shape[1]}",
            "rows_before": "250",
            "cols_before": str(x.shape[1]),
            "rows_after": "250",
            "cols_after": str(x.shape[1]),
            "fit_scope": "由附件8原量纲表核验",
            "sample_deletion": "no",
            "reason": "其余33个可核验字段为原尺度；对数字段负值表示原浓度小于1，不重复取对数",
        },
        {
            "step_id": "P4",
            "stage": "缺失",
            "operation": "训练折中位数填补，并为训练折含缺失的列添加指示变量",
            "shape_before": "n_train×231",
            "shape_after": "n_train×(231+k)",
            "rows_before": "n_train",
            "cols_before": "231",
            "rows_after": "n_train",
            "cols_after": f"231+k (0≤k≤{missing_columns})",
            "fit_scope": "每个训练折单独拟合并应用到对应验证折",
            "sample_deletion": "no",
            "reason": "保留63个含缺失样本并防止全数据统计泄漏",
        },
        {
            "step_id": "P5",
            "stage": "异常值",
            "operation": "保留全部样本；不按全数据分位数删样本或截尾",
            "shape_before": f"250×{x.shape[1]}",
            "shape_after": f"250×{x.shape[1]}",
            "rows_before": "250",
            "cols_before": str(x.shape[1]),
            "rows_after": "250",
            "cols_after": str(x.shape[1]),
            "fit_scope": "主分析；winsorize仅允许在线性模型训练折内做消融",
            "sample_deletion": "no",
            "reason": "极端值可能是真实啤酒化学差异，全数据删样会泄漏并损伤小样本",
        },
        {
            "step_id": "P6",
            "stage": "目标与输出",
            "operation": "在标准化连续值上拟合；折外预测用训练折经验CDF映射到0—6",
            "shape_before": "n_test×50 z预测",
            "shape_after": "n_test×50 z预测 + n_test×50相对等级",
            "rows_before": "n_test",
            "cols_before": "50",
            "rows_after": "n_test",
            "cols_after": "50+50",
            "fit_scope": "每个外层训练折单独构造经验分布",
            "sample_deletion": "no",
            "reason": "附件4不是原始0—6评分，不能做逆标准化",
        },
    ]
    return pd.DataFrame(rows)


def attachment_hashes() -> dict[str, str]:
    """Fingerprint official attachments 1--8 used by the contract."""
    names = ["1.csv", "2.csv", "3.xlsx", "4.csv", "5.csv", "6.csv", "7.csv", "8.xlsx"]
    return {
        name: hashlib.sha256((ATTACHMENTS / name).read_bytes()).hexdigest()
        for name in names
    }


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
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    sensory = pd.read_csv(ATTACHMENTS / "4.csv")
    dictionary = target_dictionary(y)
    name_map = compound_name_map(x)
    map_audit = validate_compound_name_map(name_map, x)
    scale_audit = feature_scale_audit(chemistry)
    missing_audit = missingness_audit(x)
    cleaning_steps = preprocessing_steps(x, y)
    corr_audit, row_map = audit_correlations(x, y)

    dictionary.to_csv(RESULT_DIR / "q1_target_dictionary.csv", index=False)
    row_map.to_csv(RESULT_DIR / "q1_attachment6_row_map.csv", index=False)
    scale_audit.to_csv(RESULT_DIR / "q1_feature_scale_audit.csv", index=False)
    missing_audit.to_csv(RESULT_DIR / "q1_missingness_by_feature.csv", index=False)
    cleaning_steps.to_csv(RESULT_DIR / "q1_preprocessing_steps.csv", index=False)
    name_map.to_csv(DATA_DIR / "compound_name_map.csv", index=False)

    audit = {
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_targets": y.shape[1],
        "n_styles": int(style.nunique()),
        "x_missing_cells": int(x.isna().sum().sum()),
        "x_missing_columns": int((x.isna().sum() > 0).sum()),
        "x_rows_with_missing": int(x.isna().any(axis=1).sum()),
        "y_missing_cells": int(y.isna().sum().sum()),
        "beer_key": "beer",
        "beer_unique_1_csv": int(chemistry["beer"].nunique()),
        "beer_unique_4_csv": int(sensory["beer"].nunique()),
        "beer_id_unique_1_csv": int(chemistry["beer_id"].nunique()),
        "beer_id_unique_4_csv": int(sensory["beer_id"].nunique()),
        "beer_id_exact_matches_after_beer_alignment": int(
            np.sum(
                chemistry["beer_id"].to_numpy()
                == sensory.set_index("beer").loc[chemistry["beer"], "beer_id"].to_numpy()
            )
        ),
        "forbidden_model_fields": ["beer", "beer_id", "tasting_category_fine"],
        "stratification_field": "tasting_category_fine",
        "preprocessing": {
            "log10_fields_kept_without_retransform": int(
                (scale_audit["scale_in_1_csv"] == "log10").sum()
            ),
            "raw_fields_verified_against_attachment8": int(
                (scale_audit["scale_in_1_csv"] == "raw").sum()
            ),
            "missing_imputation": "training-fold median",
            "missing_indicator": "training-fold missing-only columns",
            "sample_deletion": "none",
            "outlier_policy": "retain all samples; optional winsorize only as training-fold linear-model ablation",
            "target_scale": "standardized continuous values from 4.csv",
            "grade_mapping": "training-fold empirical CDF, monotone to [0, 6]",
        },
        "existing_experiment_compatibility": {
            "E001": "compliant: fold-local median + missing indicator; no row deletion or extra log",
            "E002": "compliant: fold-local median + missing indicator + rank-Gaussian model transform; no row deletion or extra log",
            "E003": "compliant: fold-local median + missing indicator; tree model keeps supplied scales; no row deletion",
            "rerun_required": False,
            "reason": "the clarified cleaning policy matches the already executed pipelines and supplied attachment files are unchanged",
        },
        "official_attachment_sha256": attachment_hashes(),
        "compound_name_map": map_audit,
        "correlation_audit": corr_audit,
    }
    (RESULT_DIR / "q1_data_contract.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

