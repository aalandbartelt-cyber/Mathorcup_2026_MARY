"""Q3 schema and measurement audit. No imputation or model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

META = ["beer", "beer_id", "tasting_category_fine"]
DERIVED = {"kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"}
DEFECTS = [
    ("DMS", "DMS", "dms", "ug/L"),
    ("Diacetyl", "X2.3.butanedione", "diacetyl", "mg/L"),
    ("4VG", "X4VG", "X4vg", "mg/L"),
    ("trans-2-nonenal", "X.E..2.nonenal", "t2n", "mg/L"),
]


def align_by_beer(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    for label, frame in (("reference", reference), (name, other)):
        if "beer" not in frame or frame.beer.isna().any() or not frame.beer.is_unique:
            raise ValueError(f"{label}: complete unique beer keys required")
    if set(reference.beer) != set(other.beer):
        raise ValueError(f"{name}: beer key sets differ")
    aligned = other.set_index("beer").loc[reference.beer].reset_index()
    if "tasting_category_fine" in reference and "tasting_category_fine" in aligned:
        left = reference.tasting_category_fine.reset_index(drop=True)
        right = aligned.tasting_category_fine
        if left.isna().any() or right.isna().any() or not left.equals(right):
            raise ValueError(f"{name}: style labels disagree")
    return aligned


def _metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if ("field_name_R" not in frame or frame.field_name_R.isna().any()
            or frame.field_name_R.duplicated().any()):
        raise ValueError("field_name_R metadata must be complete and unique")
    return frame.set_index("field_name_R")


def _unit(value) -> str:
    return "" if pd.isna(value) else str(value).strip().replace("μ", "u").replace("µ", "u")


def _cell_records(chemistry: pd.DataFrame, calibrated: pd.DataFrame,
                  fields: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in fields.itertuples(index=False):
        if not row.reference_available:
            continue
        for beer, style, stored, raw in zip(
            chemistry.beer, chemistry.tasting_category_fine,
            chemistry[row.feature_code], calibrated[row.feature_code],
        ):
            inferred = np.nan
            if row.scale_in_1_csv == "raw":
                inferred = float(stored)
            elif row.scale_in_1_csv == "log10":
                with np.errstate(over="ignore", invalid="ignore"):
                    inferred = float(np.power(10., stored))
            if not np.isfinite(stored):
                status = "missing_stored"
            elif not np.isfinite(raw):
                status = "missing_reference"
            elif raw < 0:
                status = "reference_negative"
            elif raw == 0:
                status = "reference_zero_ambiguous"
            elif row.scale_in_1_csv == "unverified":
                status = "scale_unverified"
            else:
                expected = np.log10(raw) if row.scale_in_1_csv == "log10" and raw > 0 else raw
                if raw < 0 or not np.isfinite(inferred) or inferred < 0 or not np.isclose(
                        stored, expected, rtol=1e-7, atol=2e-6):
                    status = "reference_conflict"
                elif not row.unit:
                    status = "unit_unknown"
                else:
                    status = "observed"
            records.append({"beer": beer, "tasting_category_fine": style,
                            "feature_code": row.feature_code, "stored_value": stored,
                            "reference_value": raw, "converted_value": inferred,
                            "status": status, "unit": row.unit})
    return pd.DataFrame(records, columns=["beer", "tasting_category_fine", "feature_code",
                                         "stored_value", "reference_value", "converted_value",
                                         "status", "unit"])


def feature_contract(chemistry: pd.DataFrame, calibrated: pd.DataFrame,
                     metadata: pd.DataFrame, *, minimum_pairs: int = 3,
                     minimum_match: float = .95) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minimum_pairs < 1 or not .5 < minimum_match <= 1:
        raise ValueError("invalid scale-recognition settings")
    calibrated = align_by_beer(chemistry, calibrated, "calibrated")
    meta = _metadata(metadata)
    features = [c for c in chemistry if c not in META]
    if not set(features).issubset(meta.index):
        raise ValueError("chemical metadata misses feature codes")
    rows = []
    for feature in features:
        stored = chemistry[feature].to_numpy(float)
        if np.isinf(stored).any():
            raise ValueError(f"{feature}: infinite stored measurement")
        available = feature in calibrated
        raw = calibrated[feature].to_numpy(float) if available else np.full(len(stored), np.nan)
        if np.isinf(raw).any():
            raise ValueError(f"{feature}: infinite calibrated measurement")
        positive = np.isfinite(stored) & np.isfinite(raw) & (raw > 0)
        raw_rate = log_rate = np.nan
        scale, scale_status = "unverified", "unverified"
        if positive.sum() >= minimum_pairs:
            raw_rate = float(np.isclose(stored[positive], raw[positive], rtol=1e-7, atol=2e-6).mean())
            log_rate = float(np.isclose(stored[positive], np.log10(raw[positive]), rtol=1e-7, atol=2e-6).mean())
            best = max(raw_rate, log_rate)
            if best >= minimum_match and raw_rate != log_rate:
                scale = "raw" if raw_rate > log_rate else "log10"
                scale_status = "recognized_exact_positive_pairs" if best == 1 else "recognized_with_conflicts"
        rows.append({"feature_code": feature, "scale_in_1_csv": scale,
                     "scale_status": scale_status, "reference_available": available,
                     "n_positive_pairs": int(positive.sum()), "raw_match_rate": raw_rate,
                     "log10_match_rate": log_rate, "positive_match_rate": max(raw_rate, log_rate),
                     "unit": _unit(meta.loc[feature].get("unit", np.nan)),
                     "factor_kind": "derived_aggregate" if feature in DERIVED else "measurement_or_parameter",
                     "stored_missing_count": int(np.isnan(stored).sum()),
                     "calibrated_zero_count": int((raw == 0).sum())})
    fields = pd.DataFrame(rows)
    cells = _cell_records(chemistry, calibrated, fields)
    conflicts = cells.loc[cells.status.eq("reference_conflict")].feature_code.value_counts()
    fields["conflict_count"] = fields.feature_code.map(conflicts).fillna(0).astype(int)
    return fields, cells


def defect_registry(metadata: pd.DataFrame) -> pd.DataFrame:
    meta = _metadata(metadata)
    rows = []
    for defect, code, target, expected_unit in DEFECTS:
        if code not in meta.index or _unit(meta.loc[code].get("unit")) != expected_unit:
            raise ValueError(f"{code}: expected calibrated unit {expected_unit}")
        row = {"defect": defect, "feature_code": code, "sensory_target": target,
               "unit": expected_unit, "evidence_type": "direct_chemical"}
        for statistic in ["mean", "median"]:
            value = meta.loc[code].get(f"threshold_{statistic}", np.nan)
            if pd.notna(value) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{code}: invalid {statistic} threshold")
            row[f"threshold_{statistic}"] = value
        rows.append(row)
    rows.append({"defect": "Stale hops", "feature_code": "", "sensory_target": "stale_hops",
                 "unit": "standardized_sensory", "evidence_type": "sensory_proxy_only",
                 "threshold_mean": np.nan, "threshold_median": np.nan})
    return pd.DataFrame(rows)


def source_defect_table(chemistry: pd.DataFrame, calibrated: pd.DataFrame,
                        fields: pd.DataFrame, source_style: str = "Pils/Lager") -> pd.DataFrame:
    chemistry = chemistry.reset_index(drop=True)
    calibrated = align_by_beer(chemistry, calibrated, "calibrated")
    source = chemistry.loc[chemistry.tasting_category_fine.eq(source_style)]
    if source.empty:
        raise ValueError(f"source style missing: {source_style}")
    rows = _cell_records(source, calibrated.loc[source.index], fields)
    rows = rows.loc[rows.feature_code.isin([r[1] for r in DEFECTS])].copy()
    if len(rows) != len(source) * len(DEFECTS):
        raise ValueError("source defect fields are incomplete")
    rows["source_known"] = rows.status.eq("observed")
    rows["source_concentration"] = rows.converted_value.where(rows.source_known)
    rows["scope"] = "descriptive_source_audit_not_model_fit"
    return rows.reset_index(drop=True)
