#!/usr/bin/env python3
"""Second-level Kaggle audit: multivariate alignment, bootstrap CIs, sub-dimension scan.

This is still a raw-data audit. It does not use Q1/Q2/Q3 model predictions and
does not train anything.

Outputs (docs/analysis/2026初赛/Kaggle深度分析/):
  kaggle_multivariate_alignment.json
  kaggle_bootstrap_ci.csv
  kaggle_target_flavor_crosscorr.csv
  kaggle_target_flavor_top_pairs.csv
  kaggle_mapping_shape_audit.csv
  kaggle_deep_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026初赛"
ATT = DATA / "2026年MathorCup大数据挑战赛-赛道A初赛" / "附件"
PROC = DATA / "公开数据" / "Kaggle_top_beer_information" / "processed"
OUT = ROOT / "docs" / "analysis" / "2026初赛" / "Kaggle深度分析"

FLAVORS = ["Astringency", "Body", "Alcohol", "Bitter", "Sweet",
           "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty"]

DIMS = [
    ("Body", "body", "mean_prop_Body"),
    ("Alcohol", "alcohol", "mean_prop_Alcohol"),
    ("Bitter", "bitternes", "mean_prop_Bitter"),
    ("Sweet", "sweetness", "mean_prop_Sweet"),
    ("Sour", "acidity", "mean_prop_Sour"),
    ("Hoppy", "F_hops_all", "mean_prop_Hoppy"),
    ("Malty", "F_malt_all", "mean_prop_Malty"),
    ("Fruits", "F_esters_fruity", "mean_prop_Fruits"),
    ("Spices", "spices_proxy", "mean_prop_Spices"),
]


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0)) / a.std(axis=0, ddof=0)


def mantel(x: np.ndarray, y: np.ndarray, perms: int = 5000, seed: int = 20261025):
    dx, dy = pdist(x), pdist(y)
    r = float(np.corrcoef(dx, dy)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([
        np.corrcoef(dx, pdist(y[rng.permutation(len(y))]))[0, 1]
        for _ in range(perms)
    ])
    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (perms + 1))
    return r, p


def rv_coefficient(x: np.ndarray, y: np.ndarray, perms: int = 5000, seed: int = 20261025):
    def rv(a, b):
        n = len(a)
        sxy = a.T @ b / n
        sxx = a.T @ a / n
        syy = b.T @ b / n
        return float(np.trace(sxy.T @ sxy) /
                     np.sqrt(np.trace(sxx.T @ sxx) * np.trace(syy.T @ syy)))
    value = rv(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([rv(x, y[rng.permutation(len(y))]) for _ in range(perms)])
    p = float((np.sum(null >= value) + 1) / (perms + 1))
    return value, p


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def official_style_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.read_csv(ATT / "4.csv")
    y["spices_proxy"] = (y["clove"] + y["coriander"]) / 2
    targets = [c for c in y.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
    means = y.groupby("tasting_category_fine")[targets].mean().reset_index()
    means = means.rename(columns={"tasting_category_fine": "official_style"})
    n = y.groupby("tasting_category_fine").size().rename("n_official").reset_index()
    n = n.rename(columns={"tasting_category_fine": "official_style"})
    means = means.merge(n, on="official_style")
    return means, y


def official_abv() -> pd.DataFrame:
    chem = pd.read_csv(ATT / "1.csv")
    chem["abv_raw"] = 10 ** chem["ethanol..v.v."]
    out = chem.groupby("tasting_category_fine").agg(
        official_abv_median=("abv_raw", "median")
    ).reset_index().rename(columns={"tasting_category_fine": "official_style"})
    return out


def multivariate(merged: pd.DataFrame) -> dict:
    x = zscore(merged[[a for _, a, _ in DIMS]].to_numpy(float))
    y = zscore(merged[[b for _, _, b in DIMS]].to_numpy(float))
    mantel_r, mantel_p = mantel(x, y)
    rv, rv_p = rv_coefficient(x, y)
    _, _, disparity = procrustes(x, y)
    return {
        "n_styles": int(len(merged)),
        "n_dimensions": len(DIMS),
        "mantel_r": mantel_r,
        "mantel_perm_p": mantel_p,
        "rv": rv,
        "rv_perm_p": rv_p,
        "procrustes_disparity": float(disparity),
        "procrustes_corr": float(1 - disparity),
        "note": "Moderate overall alignment; CCA is omitted because 9 dimensions on 22 styles is overfit-prone.",
    }


def hierarchical_bootstrap(y: pd.DataFrame, merged: pd.DataFrame, abv: pd.DataFrame,
                           reps: int = 1000, seed: int = 20261025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    targets = ["body", "alcohol", "bitternes", "sweetness", "acidity",
               "F_hops_all", "F_malt_all", "F_esters_fruity", "spices_proxy", "overall"]
    y = y.copy()
    y["spices_proxy"] = (y["clove"] + y["coriander"]) / 2
    style_groups = {s: g[targets].to_numpy(float) for s, g in y.groupby("tasting_category_fine")}
    kg_cols = [b for _, _, b in DIMS] + ["rating_mean"]
    kg_values = merged[[*kg_cols]].to_numpy(float)
    styles = merged["official_style"].tolist()

    rho_samples = {c: [] for c in targets}
    for _ in range(reps):
        means = []
        for s in styles:
            g = style_groups[s]
            idx = rng.integers(0, len(g), size=len(g))
            means.append(g[idx].mean(axis=0))
        means = np.asarray(means)
        for j, c in enumerate(targets[:-1]):
            rho_samples[c].append(spearmanr(means[:, j], kg_values[:, j]).statistic)
        rho_samples["overall"].append(spearmanr(means[:, -1], kg_values[:, -1]).statistic)

    # ABV bootstrap (resample official beers within style, recompute median)
    chem = pd.read_csv(ATT / "1.csv")
    chem["abv_raw"] = 10 ** chem["ethanol..v.v."]
    abv_groups = {s: g["abv_raw"].to_numpy(float) for s, g in chem.groupby("tasting_category_fine")}
    kg_abv = merged.set_index("official_style")["abv_median"]
    abv_rho = []
    for _ in range(reps):
        vals = []
        for s in styles:
            g = abv_groups[s]
            vals.append(np.median(g[rng.integers(0, len(g), size=len(g))]))
        abv_rho.append(spearmanr(vals, kg_abv.loc[styles].to_numpy(float)).statistic)

    observed = {}
    obs_means = y.groupby("tasting_category_fine")[targets].mean().loc[styles]
    for j, c in enumerate(targets):
        observed[c] = float(spearmanr(obs_means[c].to_numpy(float),
                                      kg_values[:, j]).statistic)

    rows = []
    for c in targets:
        lo, hi = np.percentile(rho_samples[c], [2.5, 97.5])
        rows.append({"metric": c, "observed_rho": observed[c],
                      "boot_ci_low": float(lo), "boot_ci_high": float(hi),
                      "n_boot": reps})
    lo, hi = np.percentile(abv_rho, [2.5, 97.5])
    rows.append({"metric": "ABV", "observed_rho": float(spearmanr(
        abv.set_index("official_style").loc[styles, "official_abv_median"].to_numpy(float),
        kg_abv.loc[styles].to_numpy(float)).statistic),
        "boot_ci_low": float(lo), "boot_ci_high": float(hi), "n_boot": reps})
    return pd.DataFrame(rows)


def target_flavor_scan(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(ATT / "4.csv")
    original_targets = [c for c in original.columns
                        if c not in ("beer", "beer_id", "tasting_category_fine")]
    targets = [c for c in original_targets if c in merged.columns]
    rows = []
    for t in targets:
        for f in FLAVORS:
            rho, p = spearmanr(merged[t], merged[f"mean_prop_{f}"])
            rows.append({"official_target": t, "kaggle_dimension": f,
                         "rho": float(rho), "p": float(p)})
    df = pd.DataFrame(rows)
    df["q_bh"] = bh_qvalues(df["p"].to_numpy())
    top = df[(df["q_bh"] < 0.05) & (df["rho"].abs() >= 0.6)].copy()
    top = top.sort_values("rho", key=lambda s: s.abs(), ascending=False)
    return df, top


def mapping_shape_audit(merged: pd.DataFrame, profiles: pd.DataFrame,
                        mapping: pd.DataFrame) -> pd.DataFrame:
    official_cols = [a for _, a, _ in DIMS]
    kaggle_cols = [b for _, _, b in DIMS]
    x = zscore(merged[official_cols].to_numpy(float))
    y = zscore(profiles[kaggle_cols].to_numpy(float))
    rows = []
    for i, row in merged.iterrows():
        rho = np.array([spearmanr(x[i], y[j]).statistic for j in range(len(y))])
        order = np.argsort(-rho)
        primary = mapping[(mapping["official_style"].eq(row["official_style"])) &
                          (mapping["primary_pool"].eq(1))]
        p_idx = [profiles.index[profiles["kaggle_style"].eq(k)][0] for k in primary["kaggle_style"]]
        best_p = max(p_idx, key=lambda j: rho[j])
        rank = int(np.where(order == best_p)[0][0] + 1)
        rows.append({
            "official_style": row["official_style"],
            "n_official": int(row["n_official"]),
            "best_primary_kaggle_style": profiles.iloc[best_p]["kaggle_style"],
            "best_primary_shape_rho": float(rho[best_p]),
            "best_primary_rank": rank,
            "top1_kaggle_style": profiles.iloc[order[0]]["kaggle_style"],
            "top1_shape_rho": float(rho[order[0]]),
            "n_kaggle_candidates": int(len(profiles)),
        })
    return pd.DataFrame(rows).sort_values("best_primary_rank")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    means, y = official_style_table()
    abv = official_abv()
    kg = pd.read_csv(PROC / "kaggle_official_style_profiles_draft.csv", encoding="utf-8-sig")
    mapping = pd.read_csv(PROC / "kaggle_official_style_mapping_draft.csv", encoding="utf-8-sig")
    profiles = pd.read_csv(PROC / "kaggle_style_profiles.csv", encoding="utf-8-sig")
    merged = means.merge(kg, on="official_style", how="left").merge(abv, on="official_style", how="left")
    if merged["n_flavor_usable"].isna().any():
        raise ValueError("Missing Kaggle profile for an official style")

    mult = multivariate(merged)
    boot = hierarchical_bootstrap(y, merged, abv)
    cross, top = target_flavor_scan(merged)
    shape = mapping_shape_audit(merged, profiles, mapping)

    (OUT / "kaggle_multivariate_alignment.json").write_text(
        json.dumps(mult, ensure_ascii=False, indent=2), encoding="utf-8")
    boot.to_csv(OUT / "kaggle_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "kaggle_target_flavor_crosscorr.csv", index=False, encoding="utf-8-sig")
    top.to_csv(OUT / "kaggle_target_flavor_top_pairs.csv", index=False, encoding="utf-8-sig")
    shape.to_csv(OUT / "kaggle_mapping_shape_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "multivariate_alignment": mult,
        "bootstrap_ci": boot.to_dict(orient="records"),
        "target_flavor_scan": {
            "n_tests": int(len(cross)),
            "n_q_lt_0_05_abs_rho_ge_0_6": int(len(top)),
            "top_pairs": top.head(20).to_dict(orient="records"),
            "caution": "Style-level confounding inflates many pairs; do not expand primary proxies from this scan alone.",
        },
        "mapping_shape_audit": {
            "n_styles": int(len(shape)),
            "median_best_primary_rank": float(shape["best_primary_rank"].median()),
            "n_rank_le_10": int((shape["best_primary_rank"] <= 10).sum()),
            "n_rank_gt_20": int((shape["best_primary_rank"] > 20).sum()),
            "note": "Shape-only diagnostic; semantic mapping remains primary. Do not silently remap from it.",
        },
    }
    (OUT / "kaggle_deep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(mult, ensure_ascii=False, indent=2))
    print("bootstrap:")
    print(boot.to_string(index=False))
    print("top target-flavour pairs:", len(top))
    print(top.head(15).to_string(index=False))
    print("mapping shape median rank:", shape["best_primary_rank"].median(),
          "rank<=10:", int((shape["best_primary_rank"] <= 10).sum()),
          "rank>20:", int((shape["best_primary_rank"] > 20).sum()))
    print("output:", OUT)


if __name__ == "__main__":
    main()
