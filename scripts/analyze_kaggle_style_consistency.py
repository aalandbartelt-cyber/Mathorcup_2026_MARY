#!/usr/bin/env python3
"""Audit official-style sensory profiles against Kaggle style profiles.

This is a RAW-DATA consistency audit: official 4.csv style means vs Kaggle
flavour word-frequency profiles. It does not use Q1/Q2/Q3 model predictions,
does not train anything, and does not modify modelling results.

Outputs (docs/analysis/2026初赛/Kaggle深度分析/):
  kaggle_dimension_consistency.csv
  kaggle_numeric_consistency.csv
  kaggle_style_level_consistency.csv
  kaggle_style_reliability.csv
  kaggle_flavor_spearman.csv
  kaggle_flavor_pca_loadings.csv
  kaggle_analysis_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026初赛"
ATT = DATA / "2026年MathorCup大数据挑战赛-赛道A初赛" / "附件"
PROC = DATA / "公开数据" / "Kaggle_top_beer_information" / "processed"
OUT = ROOT / "docs" / "analysis" / "2026初赛" / "Kaggle深度分析"

FLAVORS = [
    "Astringency", "Body", "Alcohol", "Bitter", "Sweet",
    "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty",
]

# dimension, official primary, official secondary, kaggle profile field
DIMS = [
    ("Body", "body", "", "mean_prop_Body"),
    ("Alcohol", "alcohol", "", "mean_prop_Alcohol"),
    ("Bitter", "bitternes", "", "mean_prop_Bitter"),
    ("Sweet", "sweetness", "", "mean_prop_Sweet"),
    ("Sour", "acidity", "lactic | acetic", "mean_prop_Sour"),
    ("Hoppy", "F_hops_all", "A_hops_all", "mean_prop_Hoppy"),
    ("Malty", "F_malt_all", "A_malt_all", "mean_prop_Malty"),
    ("Fruits", "F_esters_fruity", "A_esters_fruity | orange", "mean_prop_Fruits"),
    ("Spices", "spices_proxy", "clove | coriander", "mean_prop_Spices"),
]


def perm_p(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 20261025) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(spearmanr(a, b).statistic)
    count = 0
    for _ in range(n):
        if abs(spearmanr(a, rng.permutation(b)).statistic) >= observed:
            count += 1
    return (count + 1) / (n + 1)


def loo_range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    values = []
    for i in range(len(a)):
        keep = np.ones(len(a), dtype=bool)
        keep[i] = False
        values.append(spearmanr(a[keep], b[keep]).statistic)
    return float(min(values)), float(max(values))


def add_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spices_proxy"] = (out["clove"] + out["coriander"]) / 2.0
    return out


def official_means() -> pd.DataFrame:
    y = pd.read_csv(ATT / "4.csv")
    y = add_proxy(y)
    cols = ["body", "alcohol", "bitternes", "sweetness", "acidity",
            "F_hops_all", "F_malt_all", "F_esters_fruity", "spices_proxy", "overall"]
    agg = y.groupby("tasting_category_fine").agg(
        n_official=("beer", "size"), **{c: (c, "mean") for c in cols}
    ).reset_index().rename(columns={"tasting_category_fine": "official_style"})

    chem = pd.read_csv(ATT / "1.csv")
    chem["abv_raw"] = 10 ** chem["ethanol..v.v."]
    abv = chem.groupby("tasting_category_fine").agg(
        official_abv_median=("abv_raw", "median")
    ).reset_index().rename(columns={"tasting_category_fine": "official_style"})
    return agg.merge(abv, on="official_style", how="left")


def dimension_consistency(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, primary, secondary, kfield in DIMS:
        d = merged[[primary, kfield, "n_official"]].dropna()
        a = d[primary].to_numpy(float)
        b = d[kfield].to_numpy(float)
        rho = float(spearmanr(a, b).statistic)
        p = perm_p(a, b)
        lo, hi = loo_range(a, b)

        d5 = d[d["n_official"] >= 5]
        a5 = d5[primary].to_numpy(float)
        b5 = d5[kfield].to_numpy(float)
        rho5 = float(spearmanr(a5, b5).statistic)
        p5 = perm_p(a5, b5)
        lo5, hi5 = loo_range(a5, b5)

        if name in {"Sour", "Sweet", "Bitter", "Malty", "Hoppy"} and p < 0.05 and lo > 0:
            role = "primary_external_check"
        elif name in {"Alcohol", "Fruits", "Spices"} and p < 0.05:
            role = "secondary_external_check"
        elif name == "Body":
            role = "exploratory_only"
        else:
            role = "exploratory_only"
        rows.append({
            "dimension": name,
            "official_target_primary": primary,
            "official_target_secondary": secondary,
            "kaggle_profile_field": kfield,
            "n_styles_all": len(d),
            "rho_all": rho,
            "perm_p_all": p,
            "loo_min_all": lo,
            "loo_max_all": hi,
            "n_styles_n5": len(d5),
            "rho_n5": rho5,
            "perm_p_n5": p5,
            "loo_min_n5": lo5,
            "loo_max_n5": hi5,
            "recommended_role": role,
        })
    return pd.DataFrame(rows)


def numeric_consistency(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = [
        ("ABV", "official_abv_median", "abv_median", "official 10**ethanol style median vs Kaggle ABV median"),
        ("Overall rating", "overall", "rating_mean", "official overall style mean vs Kaggle average rating"),
    ]
    for name, oc, kc, note in checks:
        d = merged[[oc, kc, "n_official"]].dropna()
        a = d[oc].to_numpy(float)
        b = d[kc].to_numpy(float)
        lo, hi = loo_range(a, b)
        rows.append({
            "check": name,
            "official_field": oc,
            "kaggle_field": kc,
            "n_styles": len(d),
            "rho": float(spearmanr(a, b).statistic),
            "perm_p": perm_p(a, b),
            "loo_min": lo,
            "loo_max": hi,
            "note": note,
        })
    return pd.DataFrame(rows)


def style_level(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for name, primary, _, kfield in DIMS:
        out[f"official_{name}"] = out[primary]
        out[f"kaggle_{name}"] = out[kfield]
    keep = ["official_style", "n_official", "n_flavor_usable",
            "official_abv_median", "abv_median", "overall", "rating_mean"]
    for name, _, _, _ in DIMS:
        keep += [f"official_{name}", f"kaggle_{name}"]
    return out[keep].sort_values("official_style").reset_index(drop=True)


def style_reliability(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for style, g in clean.groupby("Style", sort=True):
        rows.append({
            "kaggle_style": style,
            "n_total": int(len(g)),
            "n_flavor_usable": int(g["flavor_usable"].sum()),
            "usable_rate": float(g["flavor_usable"].mean()),
            "abv_nonpositive": int((g["abv_flag"] == "nonpositive").sum()),
            "abv_high": int((g["abv_flag"] == "high").sum()),
            "rating_mean": float(g.loc[g["rating_valid"], "Ave Rating"].mean())
            if g["rating_valid"].any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["n_flavor_usable", "n_total"], ascending=[False, False])


def flavor_structure(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    usable = clean[clean["flavor_usable"]]
    props = [f"prop_{f}" for f in FLAVORS]
    corr = usable[props].corr(method="spearman")
    corr.index = [c.removeprefix("prop_") for c in corr.index]
    corr.columns = [c.removeprefix("prop_") for c in corr.columns]

    X = usable[props].to_numpy(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    pca = PCA(n_components=4, random_state=20261025)
    pca.fit(X)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[c.removeprefix("prop_") for c in props],
        columns=["PC1", "PC2", "PC3", "PC4"],
    )
    pairs = []
    for i in range(len(FLAVORS)):
        for j in range(i + 1, len(FLAVORS)):
            pairs.append((FLAVORS[i], FLAVORS[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    summary = {
        "n_usable_rows": int(len(usable)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "top_abs_correlations": [
            {"a": a, "b": b, "rho": r} for a, b, r in pairs[:10]
        ],
    }
    return corr, loadings, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kg = pd.read_csv(PROC / "kaggle_official_style_profiles_draft.csv", encoding="utf-8-sig")
    clean = pd.read_csv(PROC / "kaggle_beer_clean.csv", encoding="utf-8-sig")
    official = official_means()
    merged = official.merge(kg, on="official_style", how="left")

    if merged["n_flavor_usable"].isna().any():
        missing = merged.loc[merged["n_flavor_usable"].isna(), "official_style"].tolist()
        raise ValueError(f"Missing Kaggle official profile for: {missing}")

    dims = dimension_consistency(merged)
    numeric = numeric_consistency(merged)
    styles = style_level(merged)
    reliability = style_reliability(clean)
    corr, loadings, flavor_summary = flavor_structure(clean)

    dims.to_csv(OUT / "kaggle_dimension_consistency.csv", index=False, encoding="utf-8-sig")
    numeric.to_csv(OUT / "kaggle_numeric_consistency.csv", index=False, encoding="utf-8-sig")
    styles.to_csv(OUT / "kaggle_style_level_consistency.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUT / "kaggle_style_reliability.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "kaggle_flavor_spearman.csv", encoding="utf-8-sig")
    loadings.to_csv(OUT / "kaggle_flavor_pca_loadings.csv", encoding="utf-8-sig")

    summary = {
        "n_official_styles": int(len(merged)),
        "n_kaggle_styles": int(clean["Style"].nunique()),
        "dimension_roles": dims.set_index("dimension")["recommended_role"].to_dict(),
        "primary_dimensions": dims.loc[
            dims["recommended_role"].eq("primary_external_check"), "dimension"
        ].tolist(),
        "secondary_dimensions": dims.loc[
            dims["recommended_role"].eq("secondary_external_check"), "dimension"
        ].tolist(),
        "exploratory_dimensions": dims.loc[
            dims["recommended_role"].eq("exploratory_only"), "dimension"
        ].tolist(),
        "numeric_checks": numeric.to_dict(orient="records"),
        "flavor_structure": flavor_summary,
        "notes": [
            "Raw official 4.csv style means vs Kaggle word-frequency profiles; no model predictions.",
            "Small official styles (Faro n=3, Brut n=4) are excluded in the n>=5 sensitivity.",
            "Salty and Astringency remain do_not_use; no clean official target.",
            "Mapping and proxy must be frozen before any model-based external validation.",
        ],
    }
    (OUT / "kaggle_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"styles={len(merged)} dims={len(dims)} primary={summary['primary_dimensions']}")
    print(dims[["dimension", "rho_all", "perm_p_all", "rho_n5", "recommended_role"]].to_string(index=False))
    print(numeric[["check", "rho", "perm_p", "n_styles"]].to_string(index=False))
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
