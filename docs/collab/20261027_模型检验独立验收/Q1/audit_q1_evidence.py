"""Independent read-only recomputation of the transferred Q1 validation evidence.

Writes only alongside this script. Does not train or import the candidate builder.
"""
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "05_论文/Q1_感官评分/数据与结果"
VAL = DATA / "validation_v1"
ABL = DATA / "validation_ablation_v1"
SOURCE = ROOT / ".worktrees/q1-nnls-opt/03_结果/Q1_感官评分"
FIG = ROOT / "04_图表/Q1_感官评分/validation_v1"
AT = Path("/mnt/d/上海大学__/26年妈妈杯数模/审计_模型检验_20261026")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare_frame(path, fresh, keys, cols):
    old = pd.read_csv(path).sort_values(keys).reset_index(drop=True)
    fresh = fresh.sort_values(keys).reset_index(drop=True)
    assert old[keys].equals(fresh[keys]), (str(path), "identity mismatch")
    delta = float(np.max(np.abs(old[cols].to_numpy(float) - fresh[cols].to_numpy(float))))
    return {"rows": len(old), "max_absolute_difference": delta, "match_at_1e_12": delta < 1e-12}


def main():
    z = np.load(DATA / "final_model_oof.npz", allow_pickle=False)
    y = z["y_true"]
    pred = z["prediction"]
    mean = pred.mean(0)
    code = z["target_code"].tolist()
    result = {"scope": "read_only_independent_recomputation_no_training"}
    result["integrity"] = {
        "prediction_shape": list(pred.shape),
        "finite_prediction_cells": int(np.isfinite(pred).sum()),
        "all_prediction_finite": bool(np.isfinite(pred).all()),
        "truth_shape": list(y.shape),
        "all_truth_finite": bool(np.isfinite(y).all()),
        "unique_beer": int(len(np.unique(z["beer"]))),
        "unique_targets": int(len(set(code))),
        "stored_mean_max_difference": float(np.max(np.abs(mean - z["mean_prediction"]))),
    }
    raw = np.load(SOURCE / "q1_oof_predictions_E009.npz", allow_pickle=False)
    result["frozen_source_match"] = {
        k: bool(np.array_equal(z[k], raw[k]))
        for k in ["y_true", "beer", "style", "target_code", "fold_assignment"]
    }
    result["frozen_source_match"]["prediction"] = bool(np.array_equal(pred, raw["pred_nnls_group"]))
    splits = pd.read_csv(DATA / "cross_validation_splits.csv")
    valid_splits = len(splits) == 2500
    for r in range(10):
        frame = splits.loc[splits["repeat"].eq(r)].sort_values("row_position")
        valid_splits &= bool(np.array_equal(frame.row_position.to_numpy(), np.arange(250)))
        valid_splits &= bool(np.array_equal(frame.beer.to_numpy(), z["beer"]))
        valid_splits &= bool(np.array_equal(frame.fold.to_numpy(), z["fold_assignment"][r]))
    result["integrity"]["split_identity_and_once_per_repeat"] = valid_splits
    attachments=ROOT/"data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件"
    truth=pd.read_csv(attachments/"4.csv").set_index("beer").loc[z["beer"],code].to_numpy(float)
    result["integrity"]["attachment4_truth_exact_match"] = bool(np.array_equal(y,truth))

    ss = ((y - y.mean(0)) ** 2).sum(0)
    r2_by_target = 1 - ((pred - y) ** 2).sum(1) / ss
    rep = pd.DataFrame({"repeat": np.arange(10), "r2": r2_by_target.mean(1),
                        "mae": np.abs(pred-y).mean((1,2))})
    saved_rep = pd.read_csv(DATA / "model_metrics_by_repeat.csv")
    saved_final = saved_rep.loc[saved_rep.display_name.eq("感官组收缩NNLS")].sort_values("repeat")
    result["repeat_metrics"] = {
        "r2_mean": float(rep.r2.mean()), "r2_sd": float(rep.r2.std(ddof=1)),
        "mae_mean": float(rep.mae.mean()), "mae_sd": float(rep.mae.std(ddof=1)),
        "saved_r2_max_difference": float(np.max(np.abs(rep.r2.to_numpy()-saved_final.r2.to_numpy()))),
        "saved_mae_max_difference": float(np.max(np.abs(rep.mae.to_numpy()-saved_final.mae.to_numpy()))),
        "mean_oof_r2": float((1-((mean-y)**2).sum(0)/ss).mean()),
        "mean_oof_mae": float(np.abs(mean-y).mean()),
    }
    rep.to_csv(OUT / "independent_repeat_metrics.csv", index=False)

    bins_rows, cal_rows = [], []
    for j, c in enumerate(code):
        # Five equal-count groups, with the original qcut convention for ties.
        group = pd.qcut(mean[:,j], q=5, labels=False, duplicates="drop")
        gaps = []
        for b in np.unique(group):
            mask = group == b
            pbar = mean[mask,j].mean(); ybar = y[mask,j].mean()
            gap = abs(pbar-ybar)
            bins_rows.append({"target_code":c,"bin":int(b),"n":int(mask.sum()),
                              "mean_pred":pbar,"mean_truth":ybar,"abs_gap":gap})
            gaps.append((mask.sum(),gap))
        cal_rows.append({"target_code":c,"ece":sum(n*g for n,g in gaps)/250,
                         "mce":max(g for _,g in gaps),"n":250})
    fresh_bins, fresh_cal = pd.DataFrame(bins_rows), pd.DataFrame(cal_rows)
    result["calibration"] = {"ece_mean":float(fresh_cal.ece.mean()),
        "bins_check":compare_frame(VAL/"q1_calibration_bins.csv",fresh_bins,["target_code","bin"],["n","mean_pred","mean_truth","abs_gap"]),
        "summary_check":compare_frame(VAL/"q1_calibration_summary.csv",fresh_cal,["target_code"],["ece","mce","n"]),
        "unit":"attachment_4_standardized_sensory_scale",
        "interpretation":"same fixed OOF; descriptive bin calibration, no independent validation"}

    cover = []
    sd = np.std(pred,axis=0,ddof=1)
    for j,c in enumerate(code):
        for nominal in [.8,.9,.95]:
            q = (1-nominal)/2
            zvalue = norm.ppf(1-q)
            low=mean[:,j]-zvalue*sd[:,j]; high=mean[:,j]+zvalue*sd[:,j]
            cover.append({"target_code":c,"method":"repeat_sd","nominal":nominal,
                          "coverage":np.mean((y[:,j]>=low)&(y[:,j]<=high)),"coverage_sd":0.,
                          "width":np.mean(high-low),"n_repeats":10})
            actual,widths=[],[]
            for r in range(10):
                others=np.array([i for i in range(10) if i!=r])
                residual=(y[:,j]-pred[others,:,j]).ravel()
                lo,hi=np.quantile(residual,[q,1-q])
                low=pred[r,:,j]+lo;high=pred[r,:,j]+hi
                actual.append(np.mean((y[:,j]>=low)&(y[:,j]<=high)));widths.append(np.mean(high-low))
            cover.append({"target_code":c,"method":"cross_repeat_residual","nominal":nominal,
                          "coverage":np.mean(actual),"coverage_sd":np.std(actual,ddof=1),
                          "width":np.mean(widths),"n_repeats":10})
    fresh_cover=pd.DataFrame(cover)
    result["intervals"]={"csv_check":compare_frame(VAL/"q1_interval_coverage.csv",fresh_cover,["target_code","method","nominal"],["coverage","coverage_sd","width","n_repeats"]),
        "summary":fresh_cover.groupby(["method","nominal"])[["coverage","width"]].mean().reset_index().to_dict("records"),
        "interpretation":"cross_repeat_residual calibrates using the same 250 beer truth labels in other repeats; calibration is not independent"}

    rng=np.random.default_rng(20261026)
    rel=[]
    for j,c in enumerate(code):
        obs=1-np.sum((y[:,j]-mean[:,j])**2)/ss[j]
        values=[]
        for b in range(1000):
            ii=rng.integers(250,size=250);yt=y[ii,j];yp=mean[ii,j]
            values.append(1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2))
        lo,hi=np.percentile(values,[2.5,97.5])
        rel.append({"target_code":c,"r2":obs,"r2_ci_low":lo,"r2_ci_high":hi,
                    "r2_boot_sd":np.std(values,ddof=1),"mae":np.abs(y[:,j]-mean[:,j]).mean()})
    fresh_rel=pd.DataFrame(rel)
    saved_rel=pd.read_csv(VAL/"q1_target_reliability.csv")
    tiers=pd.qcut(saved_rel.r2.rank(method="first"),3,labels=["较低","中等","较高"]).astype(str)
    result["reliability"]={"csv_check":compare_frame(VAL/"q1_target_reliability.csv",fresh_rel,["target_code"],["r2","r2_ci_low","r2_ci_high","r2_boot_sd","mae"]),
        "r2_min":float(fresh_rel.r2.min()),"r2_max":float(fresh_rel.r2.max()),"r2_median":float(fresh_rel.r2.median()),
        "saved_tier_counts":pd.read_csv(VAL/"q1_target_reliability.csv").reliability_tier.value_counts().to_dict(),
        "tiers_match_rank_tertiles":bool((tiers==saved_rel.reliability_tier).all()),
        "interpretation":"R2 and beer bootstrap computed on mean of ten OOF predictions; tiers are relative thirds rather than fixed clinical usefulness thresholds"}

    abl=pd.read_csv(ABL/"q1_ablation_summary.csv"); leave=pd.read_csv(ABL/"q1_leave_style_out.csv")
    result["ablation"]={"values":abl.to_dict("records"),
        "full_repeat0_matches_frozen_r2":bool(np.isclose(abl.iloc[0].r2,rep.iloc[0].r2,atol=1e-14)),
        "full_repeat0_matches_frozen_mae":bool(np.isclose(abl.iloc[0].mae,rep.iloc[0].mae,atol=1e-14)),
        "source_csv_equal":{n: digest(ABL/n)==digest(SOURCE/"validation_ablation_v1"/n) for n in ["q1_ablation_summary.csv","q1_ablation_folds.csv","q1_leave_style_out.csv"]},
        "raw_retrain_predictions_available":False,
        "interpretation":"ablation and leave-style CSV verified against run source; deletion/permutation raw predictions not saved, so those values cannot be independently recomputed without retraining; label permutation is a single global beer-row permutation before CV, not fold-only permutation"}
    result["leave_style"]={"rows":len(leave),"sum_n":int(leave.n.sum()),"n_min":int(leave.n.min()),"n_max":int(leave.n.max()),
        "spearman_min":float(leave.spearman.min()),"spearman_max":float(leave.spearman.max()),"mae_min":float(leave.mae.min()),"mae_max":float(leave.mae.max()),"all_metrics_finite":bool(np.isfinite(leave.select_dtypes('number')).all().all())}

    figs=[]
    for name in ["图5_重复验证稳定性与预测校准","附图S2_消融与负对照","附图S3_逐目标预测可靠性"]:
        p=FIG/f"{name}.png";im=Image.open(p)
        ap=AT/"Q1_图/final_v6"/p.name
        figs.append({"name":name,"size":list(im.size),"dpi":list(im.info.get('dpi',[])),
                     "source_png_sha256":digest(p),"audit_package_png_sha256":digest(ap),"same_as_audit_package":digest(p)==digest(ap)})
    result["figures"]=figs
    result["issues"]=[
        {"severity":"presentation","item":"S2 (b) MAE is a column chart in the repository R script and actual PNG; user requests a point chart; bars currently start at zero, so not a truncated-bar distortion, but redesign is still required."},
        {"severity":"provenance","item":"ablation manifest input_hashes['1.csv'] is null; original retrain OOF/leave-style predictions not retained. Full repeat0 numerically matches frozen OOF, but removed-feature/permutation/leave-style results can only be checked against source CSV/log and implementation."},
        {"severity":"wording","item":"do not call global label permutation fold-only: code shuffles beer rows once before outer CV. Keep wording as single label-permutation negative control."},
        {"severity":"wording","item":"0.033–0.748 and median 0.212 use mean-OOF target R2, whereas 0.2564±0.0070 uses mean of repeated macro R2. Explicitly name each convention."},
    ]
    inputs=[DATA/"final_model_oof.npz",DATA/"model_metrics_by_repeat.csv",DATA/"cross_validation_splits.csv",DATA/"target_metrics_final.csv",
            SOURCE/"q1_oof_predictions_E009.npz",ROOT/"scripts/build_q1_validation.py",ROOT/"scripts/plot_q1_validation.R",ROOT/"scripts/plot_q1_validation_s2.R",
            SOURCE.parent.parent/"02_代码/Q1_感官评分/src/run_q1_validation_ablations.py"]
    inputs += list(VAL.glob("*.csv"))+list(ABL.glob("*"))
    inputs += [attachments/"1.csv", attachments/"4.csv"]
    result["input_sha256"]={str(p.relative_to(ROOT)):digest(p) for p in inputs if p.is_file()}
    (OUT/"Q1_independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["integrity","repeat_metrics","calibration","intervals","reliability","leave_style"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
