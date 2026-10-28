#!/usr/bin/env python3
"""Provisional external validation of the frozen Q1 final model against Kaggle.

Reads the delivered Q1 final-model OOF (mean prediction), aggregates by official
style, and compares 9 pre-declared proxy dimensions with the Kaggle style
profiles. This is read-only: Q1 v5 is not modified and no model is retrained.

Outputs (docs/analysis/2026初赛/Kaggle深度分析/):
  kaggle_q1_model_external_validation.csv
  kaggle_q1_model_style_level.csv
  kaggle_q1_model_external_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026初赛"
ATT = DATA / "2026年MathorCup大数据挑战赛-赛道A初赛" / "附件"
PROC = DATA / "公开数据" / "Kaggle_top_beer_information" / "processed"
Q1 = ROOT / "05_论文" / "Q1_感官评分" / "数据与结果"
OUT = ROOT / "docs" / "analysis" / "2026初赛" / "Kaggle深度分析"

DIMS = [
    ("Body", "body", "mean_prop_Body", "exploratory"),
    ("Alcohol", "alcohol", "mean_prop_Alcohol", "secondary"),
    ("Bitter", "bitternes", "mean_prop_Bitter", "primary"),
    ("Sweet", "sweetness", "mean_prop_Sweet", "primary"),
    ("Sour", "acidity", "mean_prop_Sour", "primary"),
    ("Hoppy", "F_hops_all", "mean_prop_Hoppy", "primary"),
    ("Malty", "F_malt_all", "mean_prop_Malty", "primary"),
    ("Fruits", "F_esters_fruity", "mean_prop_Fruits", "secondary"),
    ("Spices", "spices_proxy", "mean_prop_Spices", "secondary"),
]


def perm_p(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 20261025) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(spearmanr(a, b).statistic)
    return float((sum(abs(spearmanr(a, rng.permutation(b)).statistic) >= observed
                      for _ in range(n)) + 1) / (n + 1))


def loo_range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    vals = []
    for i in range(len(a)):
        keep = np.ones(len(a), dtype=bool)
        keep[i] = False
        vals.append(spearmanr(a[keep], b[keep]).statistic)
    return float(min(vals)), float(max(vals))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(Q1 / "final_model_oof.npz", allow_pickle=True)
    y = pd.read_csv(ATT / "4.csv")
    kg = pd.read_csv(PROC / "kaggle_official_style_profiles_draft.csv", encoding="utf-8-sig")

    target_codes = [str(x) for x in z["target_code"]]
    mean_pred = np.asarray(z["mean_prediction"], dtype=float)
    pred = np.asarray(z["prediction"], dtype=float)
    beer = np.asarray(z["beer"], dtype=int)
    style = np.asarray(z["style"]).astype(str)
    if mean_pred.shape != (len(beer), len(target_codes)):
        raise ValueError("Unexpected mean_prediction shape")

    # Build a beer-level frame, then aggregate both official truth and model OOF by style.
    frame = pd.DataFrame(mean_pred, columns=target_codes)
    frame["spices_proxy"] = (frame["clove"] + frame["coriander"]) / 2.0
    frame.insert(0, "style", style)
    frame.insert(0, "beer", beer)
    y = y.copy()
    y["spices_proxy"] = (y["clove"] + y["coriander"]) / 2
    frame = frame.merge(y[["beer", "tasting_category_fine"]], on="beer", how="left")
    if frame["tasting_category_fine"].isna().any():
        raise ValueError("Beer keys do not align with 4.csv")

    raw = y.groupby("tasting_category_fine").agg(
        n_official=("beer", "size"),
        **{c: (c, "mean") for _, c, _, _ in DIMS},
        overall=("overall", "mean"),
    ).reset_index().rename(columns={"tasting_category_fine": "official_style"})
    model = frame.groupby("tasting_category_fine").agg(
        **{f"model_{c}": (c, "mean") for _, c, _, _ in DIMS},
        model_overall=("overall", "mean"),
    ).reset_index().rename(columns={"tasting_category_fine": "official_style"})
    merged = raw.merge(model, on="official_style", how="inner").merge(
        kg, on="official_style", how="left"
    )
    if merged["n_flavor_usable"].isna().any():
        raise ValueError("Missing Kaggle profile for an official style")

    rows = []
    for name, target, kfield, role in DIMS:
        d = merged[[target, f"model_{target}", kfield, "n_official"]].dropna()
        a_raw = d[target].to_numpy(float)
        a_model = d[f"model_{target}"].to_numpy(float)
        b = d[kfield].to_numpy(float)
        r_raw = float(spearmanr(a_raw, b).statistic)
        r_model = float(spearmanr(a_model, b).statistic)
        lo, hi = loo_range(a_model, b)
        # per-repeat stability of the model's style means
        rep_rhos = []
        for r in range(pred.shape[0]):
            f = pd.DataFrame(pred[r], columns=target_codes)
            f["spices_proxy"] = (f["clove"] + f["coriander"]) / 2.0
            f["style"] = style
            sm = f.groupby("style")[target].mean().reindex(merged["official_style"]).to_numpy(float)
            rep_rhos.append(spearmanr(sm, b).statistic)
        rows.append({
            "dimension": name,
            "official_target": target,
            "kaggle_field": kfield,
            "role": role,
            "n_styles": len(d),
            "rho_raw_4csv": r_raw,
            "rho_q1_model": r_model,
            "delta_model_minus_raw": r_model - r_raw,
            "perm_p_model": perm_p(a_model, b),
            "loo_model_min": lo,
            "loo_model_max": hi,
            "repeat_rho_mean": float(np.mean(rep_rhos)),
            "repeat_rho_sd": float(np.std(rep_rhos, ddof=1)),
        })
    out = pd.DataFrame(rows)

    # Overall rating as a numerical check (official overall vs Kaggle rating).
    d = merged[["overall", "model_overall", "rating_mean"]].dropna()
    overall = {
        "rho_raw_overall_vs_rating": float(spearmanr(d["overall"], d["rating_mean"]).statistic),
        "rho_model_overall_vs_rating": float(spearmanr(d["model_overall"], d["rating_mean"]).statistic),
        "perm_p_model": perm_p(d["model_overall"].to_numpy(float), d["rating_mean"].to_numpy(float)),
        "n_styles": int(len(d)),
    }

    style_cols = ["official_style", "n_official", "n_flavor_usable", "abv_median"]
    for name, target, kfield, _ in DIMS:
        style_cols += [target, f"model_{target}", kfield]
    style_cols += ["overall", "model_overall", "rating_mean"]
    merged[style_cols].to_csv(OUT / "kaggle_q1_model_style_level.csv",
                              index=False, encoding="utf-8-sig")
    out.to_csv(OUT / "kaggle_q1_model_external_validation.csv",
               index=False, encoding="utf-8-sig")
    summary = {
        "model": "Q1 final NNLS (mean OOF)",
        "q1_model_file": str((Q1 / "final_model_oof.npz").relative_to(ROOT)),
        "n_styles": int(len(merged)),
        "dimension_validation": out.to_dict(orient="records"),
        "overall_rating": overall,
        "notes": [
            "Provisional external validation only; Q1 v5 and Q2/Q3 results are unchanged.",
            "Style means are simple means over the 250 official beers; no model refit.",
            "Kaggle profiles are frozen primary_pool mappings; mapping/proxy remain draft until frozen.",
            "Primary dims are Sour/Sweet/Bitter/Malty/Hoppy; Alcohol/Fruits/Spices secondary; Body exploratory.",
            "Q1 model has no ABV prediction; ABV external check is raw-data-only in the earlier audit.",
        ],
    }
    (OUT / "kaggle_q1_model_external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False))
    print("overall", overall)
    print("output", OUT)


if __name__ == "__main__":
    main()
