#!/usr/bin/env python3
"""Q1 validation add-ons from the frozen final-model OOF.

No model is retrained. Outputs:
  q1_repeat_metrics_summary.csv
  q1_repeat_model_ranks.csv
  q1_calibration_bins.csv
  q1_calibration_summary.csv
  q1_interval_coverage.csv
  q1_target_reliability.csv
  q1_validation_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "05_论文" / "Q1_感官评分" / "数据与结果"
OUT = DATA / "validation_v1"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_MAP = {
    "Aroma": "香气",
    "Flavour": "风味",
    "Flavour-core": "风味",
    "Taste": "味觉",
    "Taste-other": "味觉",
    "Mouthfeel": "口感",
    "Palate": "口感",
    "Overall": "总体",
    "Defect": "其他/缺陷",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeat_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA / "model_metrics_by_repeat.csv")
    metrics = ["r2", "mae", "rmse", "spearman", "qwk_proxy", "acc_pm1_proxy"]
    summary = df.groupby("display_name")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["display_name"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")]
    summary["r2_cv"] = summary["r2_std"] / summary["r2_mean"].abs()
    summary["mae_cv"] = summary["mae_std"] / summary["mae_mean"].abs()

    rows = []
    for metric, higher_better in [("r2", True), ("mae", False), ("spearman", True),
                                  ("qwk_proxy", True), ("acc_pm1_proxy", True)]:
        for repeat, g in df.groupby("repeat"):
            g = g.sort_values(metric, ascending=not higher_better).reset_index(drop=True)
            for rank, row in g.iterrows():
                rows.append({"metric": metric, "repeat": int(repeat),
                             "display_name": row["display_name"], "rank": rank + 1})
    ranks = pd.DataFrame(rows)
    rank_summary = ranks.groupby(["metric", "display_name"])["rank"].agg(
        ["median", "min", "max", "mean", "std"]
    ).reset_index()
    frac = ranks.assign(is1=ranks["rank"].eq(1)).groupby(
        ["metric", "display_name"]
    )["is1"].mean().reset_index(name="rank1_fraction")
    rank_summary = rank_summary.merge(frac, on=["metric", "display_name"], how="left")
    return summary, rank_summary


def calibration(y_true: np.ndarray, mean_pred: np.ndarray, target_codes: list[str],
                meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins_rows = []
    summary_rows = []
    for j, target in enumerate(target_codes):
        pred = mean_pred[:, j]
        truth = y_true[:, j]
        try:
            bin_id = pd.qcut(pred, 5, labels=False, duplicates="drop")
        except ValueError:
            bin_id = np.zeros(len(pred), dtype=int)
        frame = pd.DataFrame({"bin": bin_id, "pred": pred, "truth": truth})
        target_bins = []
        for b, g in frame.groupby("bin"):
            rec = {"target_code": target, "bin": int(b), "n": len(g),
                   "mean_pred": float(g["pred"].mean()),
                   "mean_truth": float(g["truth"].mean()),
                   "abs_gap": float(abs(g["truth"].mean() - g["pred"].mean()))}
            bins_rows.append(rec)
            target_bins.append(rec)
        ece = sum(r["n"] * r["abs_gap"] for r in target_bins) / len(pred)
        mce = max(r["abs_gap"] for r in target_bins)
        summary_rows.append({"target_code": target, "ece": float(ece),
                             "mce": float(mce), "n": int(len(pred))})
    bins = pd.DataFrame(bins_rows).merge(
        meta[["target_code", "target_name_en", "target_group"]], on="target_code", how="left"
    )
    summary = pd.DataFrame(summary_rows).merge(
        meta[["target_code", "target_name_en", "target_group"]], on="target_code", how="left"
    )
    summary["group_plot"] = summary["target_group"].map(GROUP_MAP).fillna("其他/缺陷")
    bins["group_plot"] = bins["target_group"].map(GROUP_MAP).fillna("其他/缺陷")
    return bins, summary


def interval_coverage(y_true: np.ndarray, pred: np.ndarray, mean_pred: np.ndarray,
                      target_codes: list[str], meta: pd.DataFrame) -> pd.DataFrame:
    n_rep, n, p = pred.shape
    rows = []
    levels = [(0.80, 0.10, 0.90), (0.90, 0.05, 0.95), (0.95, 0.025, 0.975)]
    rep_sd = pred.std(axis=0, ddof=1)  # n x p
    for j, target in enumerate(target_codes):
        # Method A: normal intervals from repeat-to-repeat prediction SD.
        for nominal, lo_q, hi_q in levels:
            z = float(norm.ppf(0.5 + nominal / 2))
            low = mean_pred[:, j] - z * rep_sd[:, j]
            high = mean_pred[:, j] + z * rep_sd[:, j]
            inside = (y_true[:, j] >= low) & (y_true[:, j] <= high)
            rows.append({"target_code": target, "method": "repeat_sd",
                         "nominal": nominal,
                         "coverage": float(inside.mean()),
                         "coverage_sd": 0.0,
                         "width": float(np.mean(high - low)),
                         "n_repeats": n_rep})
        # Method B: cross-repeat residual quantiles; this is a calibration check,
        # not an independent conformal guarantee.
        for nominal, lo_q, hi_q in levels:
            covs, widths = [], []
            for r in range(n_rep):
                other = np.delete(np.arange(n_rep), r)
                resid = (y_true[:, j][None, :] - pred[other, :, j]).reshape(-1)
                q_lo, q_hi = np.quantile(resid, [lo_q, hi_q])
                low = pred[r, :, j] + q_lo
                high = pred[r, :, j] + q_hi
                inside = (y_true[:, j] >= low) & (y_true[:, j] <= high)
                covs.append(float(inside.mean()))
                widths.append(float(np.mean(high - low)))
            rows.append({"target_code": target, "method": "cross_repeat_residual",
                         "nominal": nominal,
                         "coverage": float(np.mean(covs)),
                         "coverage_sd": float(np.std(covs, ddof=1)),
                         "width": float(np.mean(widths)),
                         "n_repeats": n_rep})
    out = pd.DataFrame(rows).merge(
        meta[["target_code", "target_name_en", "target_group"]], on="target_code", how="left"
    )
    out["group_plot"] = out["target_group"].map(GROUP_MAP).fillna("其他/缺陷")
    return out


def target_reliability(y_true: np.ndarray, mean_pred: np.ndarray,
                       target_codes: list[str], meta: pd.DataFrame,
                       n_boot: int = 1000, seed: int = 20261026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, p = y_true.shape
    rows = []
    for j, target in enumerate(target_codes):
        truth = y_true[:, j]
        pred = mean_pred[:, j]

        def r2(idx):
            yt, yp = truth[idx], pred[idx]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        obs = r2(np.arange(n))
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot.append(r2(idx))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"target_code": target, "r2": float(obs),
                     "r2_ci_low": float(lo), "r2_ci_high": float(hi),
                     "r2_boot_sd": float(np.std(boot, ddof=1)),
                     "mae": float(np.mean(np.abs(truth - pred)))})
    out = pd.DataFrame(rows).merge(
        meta[["target_code", "target_name_en", "target_group"]], on="target_code", how="left"
    ).sort_values("r2", ascending=False).reset_index(drop=True)
    out["group_plot"] = out["target_group"].map(GROUP_MAP).fillna("其他/缺陷")
    out["reliability_tier"] = pd.qcut(
        out["r2"].rank(method="first"), 3, labels=["较低", "中等", "较高"]
    ).astype(str)
    return out


def main() -> None:
    z = np.load(DATA / "final_model_oof.npz", allow_pickle=True)
    y_true = np.asarray(z["y_true"], dtype=float)
    pred = np.asarray(z["prediction"], dtype=float)
    mean_pred = np.asarray(z["mean_prediction"], dtype=float)
    target_codes = [str(x) for x in z["target_code"]]
    meta = pd.read_csv(DATA / "target_metrics_final.csv")

    rep_summary, rank_summary = repeat_stability()
    cal_bins, cal_summary = calibration(y_true, mean_pred, target_codes, meta)
    coverage = interval_coverage(y_true, pred, mean_pred, target_codes, meta)
    reliability = target_reliability(y_true, mean_pred, target_codes, meta)

    rep_summary.to_csv(OUT / "q1_repeat_metrics_summary.csv", index=False, encoding="utf-8-sig")
    rank_summary.to_csv(OUT / "q1_repeat_model_ranks.csv", index=False, encoding="utf-8-sig")
    cal_bins.to_csv(OUT / "q1_calibration_bins.csv", index=False, encoding="utf-8-sig")
    cal_summary.to_csv(OUT / "q1_calibration_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT / "q1_interval_coverage.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUT / "q1_target_reliability.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "inputs": {
            "final_model_oof.npz": sha256(DATA / "final_model_oof.npz"),
            "model_metrics_by_repeat.csv": sha256(DATA / "model_metrics_by_repeat.csv"),
            "target_metrics_final.csv": sha256(DATA / "target_metrics_final.csv"),
        },
        "outputs": sorted(p.name for p in OUT.glob("*.csv")),
        "notes": [
            "No model retraining; all calculations use frozen OOF predictions.",
            "Calibration uses 5 prediction-quantile bins per target.",
            "Interval coverage has two methods: repeat_sd uses the SD across the 10 repeats; cross_repeat_residual uses residuals from the other 9 repeats. The latter is a calibration check, not an independent conformal guarantee.",
            "Per-target R2 CI uses 1000 beer-level bootstrap resamples.",
        ],
    }
    (OUT / "q1_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("repeat", rep_summary.shape, "ranks", rank_summary.shape)
    print("calibration", cal_bins.shape, cal_summary.shape,
          "mean ECE", cal_summary["ece"].mean())
    print("coverage", coverage.shape)
    print(coverage.groupby("nominal")[["coverage", "width"]].mean().round(3).to_string())
    print("reliability", reliability.shape)
    print(reliability.head(5)[["target_code", "r2", "r2_ci_low", "r2_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
