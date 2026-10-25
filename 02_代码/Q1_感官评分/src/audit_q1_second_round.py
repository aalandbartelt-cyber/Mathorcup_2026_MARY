"""Audit Q1 second-round candidates against the frozen E003 gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data_contract import RESULT_DIR


HARD_THRESHOLDS = {
    "r2": 0.2359,
    "mae": 0.3422,
    "spearman": 0.4260,
    "qwk_proxy": 0.4026,
    "acc_pm1_proxy": 0.5734,
}
METRICS = ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]


def _prediction_pair(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    if truth.ndim != 2 or baseline.shape != truth.shape or candidate.shape != truth.shape:
        raise ValueError("truth, baseline and candidate must have identical 2D shapes")
    if not np.isfinite(truth).all() or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("truth and predictions must be finite")
    return truth, baseline, candidate


def _row_absolute_rank_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth_rank = rankdata(y_true, axis=0, method="average")
    pred_rank = rankdata(y_pred, axis=0, method="average")
    scale = max(len(y_true) - 1, 1)
    return np.mean(np.abs(truth_rank - pred_rank) / scale, axis=1)


def paired_beer_bootstrap(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    n_bootstraps: int = 20000,
    random_state: int = 20261025,
) -> pd.DataFrame:
    """Bootstrap paired per-beer loss differences; negative favors candidate."""
    truth, baseline, candidate = _prediction_pair(
        y_true, baseline_prediction, candidate_prediction
    )
    if int(n_bootstraps) < 1:
        raise ValueError("n_bootstraps must be positive")
    losses = {
        "mae": (
            np.mean(np.abs(truth - baseline), axis=1),
            np.mean(np.abs(truth - candidate), axis=1),
        ),
        "absolute_rank_error": (
            _row_absolute_rank_error(truth, baseline),
            _row_absolute_rank_error(truth, candidate),
        ),
    }
    rng = np.random.default_rng(int(random_state))
    indices = rng.integers(0, len(truth), size=(int(n_bootstraps), len(truth)))
    rows = []
    for metric, (baseline_loss, candidate_loss) in losses.items():
        per_beer_delta = candidate_loss - baseline_loss
        sampled_delta = per_beer_delta[indices].mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "baseline": float(baseline_loss.mean()),
                "candidate": float(candidate_loss.mean()),
                "delta_candidate_minus_baseline": float(per_beer_delta.mean()),
                "ci_low": float(np.quantile(sampled_delta, 0.025)),
                "ci_high": float(np.quantile(sampled_delta, 0.975)),
                "improvement_probability": float(
                    np.mean(sampled_delta < 0) + 0.5 * np.mean(sampled_delta == 0)
                ),
                "n_bootstraps": int(n_bootstraps),
                "random_state": int(random_state),
            }
        )
    return pd.DataFrame(rows)


def bottom_quartile_target_summary(
    baseline_by_target: pd.DataFrame,
    candidate_by_target: pd.DataFrame,
) -> dict:
    """Compare metrics on targets selected solely by baseline mean R2."""
    required = {"target_code", "r2", "mae"}
    for name, frame in (
        ("baseline", baseline_by_target),
        ("candidate", candidate_by_target),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} target metrics miss {sorted(missing)}")
    baseline_mean = (
        baseline_by_target.groupby("target_code", as_index=False)[["r2", "mae"]]
        .mean()
        .sort_values(["r2", "target_code"], kind="mergesort")
    )
    n_bottom = max(1, int(np.ceil(0.25 * len(baseline_mean))))
    bottom_targets = baseline_mean.head(n_bottom)["target_code"].astype(str).tolist()
    candidate_mean = candidate_by_target.groupby("target_code", as_index=False)[
        ["r2", "mae"]
    ].mean()
    baseline_bottom = baseline_mean.loc[
        baseline_mean["target_code"].astype(str).isin(bottom_targets)
    ]
    candidate_bottom = candidate_mean.loc[
        candidate_mean["target_code"].astype(str).isin(bottom_targets)
    ]
    if len(candidate_bottom) != n_bottom:
        raise RuntimeError("candidate target metrics do not cover the baseline tail")
    return {
        "bottom_targets": bottom_targets,
        "n_bottom_targets": n_bottom,
        "baseline_r2": float(baseline_bottom["r2"].mean()),
        "candidate_r2": float(candidate_bottom["r2"].mean()),
        "delta_r2": float(candidate_bottom["r2"].mean() - baseline_bottom["r2"].mean()),
        "baseline_mae": float(baseline_bottom["mae"].mean()),
        "candidate_mae": float(candidate_bottom["mae"].mean()),
        "delta_mae": float(candidate_bottom["mae"].mean() - baseline_bottom["mae"].mean()),
    }


def style_mae_table(
    y_true: np.ndarray,
    style: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> pd.DataFrame:
    """Return aligned repeat-by-style MAE for two OOF prediction tensors."""
    truth = np.asarray(y_true, dtype=float)
    styles = np.asarray(style, dtype=str)
    baseline = np.asarray(baseline_prediction, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    if baseline.ndim == 2:
        baseline = baseline[None, ...]
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    if truth.ndim != 2 or baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate prediction tensors must align")
    if baseline.shape[1:] != truth.shape or len(styles) != len(truth):
        raise ValueError("prediction tensors, truth and styles must align")
    rows = []
    for repeat in range(baseline.shape[0]):
        for value in sorted(np.unique(styles)):
            mask = styles == value
            baseline_mae = float(np.mean(np.abs(truth[mask] - baseline[repeat, mask])))
            candidate_mae = float(np.mean(np.abs(truth[mask] - candidate[repeat, mask])))
            rows.append(
                {
                    "repeat": repeat,
                    "style": value,
                    "n_beers": int(mask.sum()),
                    "baseline_mae": baseline_mae,
                    "candidate_mae": candidate_mae,
                    "delta_mae": candidate_mae - baseline_mae,
                }
            )
    return pd.DataFrame(rows)


def worst_style_summary(style_metrics: pd.DataFrame) -> dict:
    """Summarise the style with the largest candidate MAE across repeats."""
    required = {"style", "baseline_mae", "candidate_mae", "delta_mae"}
    missing = required - set(style_metrics.columns)
    if missing:
        raise ValueError(f"style metrics miss {sorted(missing)}")
    summary = (
        style_metrics.groupby("style", as_index=False)[
            ["baseline_mae", "candidate_mae", "delta_mae"]
        ]
        .mean()
        .sort_values(["candidate_mae", "style"], ascending=[False, True], kind="mergesort")
    )
    if summary.empty:
        raise ValueError("style metrics are empty")
    worst = summary.iloc[0]
    return {
        "worst_style": str(worst["style"]),
        "baseline_mae": float(worst["baseline_mae"]),
        "candidate_mae": float(worst["candidate_mae"]),
        "delta_mae": float(worst["delta_mae"]),
    }


def select_final_candidate(decisions: Mapping[str, Mapping]) -> str | None:
    """Choose only among candidates passing every frozen promotion gate."""
    eligible = []
    for experiment_id, decision in decisions.items():
        if not bool(decision.get("promoted", False)):
            continue
        means = decision.get("candidate_means", {})
        missing = set(METRICS) - set(means)
        if missing:
            raise ValueError(
                f"{experiment_id} candidate means miss {sorted(missing)}"
            )
        rank = (
            float(means["spearman"]),
            float(means["qwk_proxy"]),
            float(means["acc_pm1_proxy"]),
            float(means["r2"]),
            -float(means["mae"]),
            str(experiment_id),
        )
        eligible.append((rank, str(experiment_id)))
    return max(eligible)[1] if eligible else None


def screening_decision(
    candidate_means: Mapping[str, float], baseline_means: Mapping[str, float]
) -> dict:
    """Apply the pre-registered three-repeat promotion rule."""
    required = set(HARD_THRESHOLDS)
    missing = required - set(candidate_means)
    missing |= required - set(baseline_means)
    if missing:
        raise ValueError(f"missing screening metrics: {sorted(missing)}")
    improved = {
        "r2": float(candidate_means["r2"]) > float(baseline_means["r2"]),
        "mae": float(candidate_means["mae"]) < float(baseline_means["mae"]),
        "spearman": float(candidate_means["spearman"])
        > float(baseline_means["spearman"]),
        "qwk_proxy": float(candidate_means["qwk_proxy"])
        > float(baseline_means["qwk_proxy"]),
        "acc_pm1_proxy": float(candidate_means["acc_pm1_proxy"])
        > float(baseline_means["acc_pm1_proxy"]),
    }
    continuous_tolerance = {
        "r2": float(candidate_means["r2"])
        >= float(baseline_means["r2"]) - 0.002,
        "mae": float(candidate_means["mae"])
        <= float(baseline_means["mae"]) + 0.001,
    }
    promote = (
        sum(improved.values()) >= 4
        and (improved["spearman"] or improved["acc_pm1_proxy"])
        and all(continuous_tolerance.values())
    )
    return {
        "promote_to_full": bool(promote),
        "improved": improved,
        "improved_count": int(sum(improved.values())),
        "continuous_tolerance": continuous_tolerance,
    }


def promotion_decision(
    metric_means: Mapping[str, float],
    repeat_win_counts: Mapping[str, int],
    *,
    minimum_repeat_wins: int = 7,
) -> dict:
    """Apply the pre-registered five strict means and five 7/10 gates."""
    missing = set(HARD_THRESHOLDS) - set(metric_means)
    missing |= set(HARD_THRESHOLDS) - set(repeat_win_counts)
    if missing:
        raise ValueError(f"missing promotion metrics: {sorted(missing)}")
    mean_flags = {
        "r2": float(metric_means["r2"]) > HARD_THRESHOLDS["r2"],
        "mae": float(metric_means["mae"]) < HARD_THRESHOLDS["mae"],
        "spearman": float(metric_means["spearman"])
        > HARD_THRESHOLDS["spearman"],
        "qwk_proxy": float(metric_means["qwk_proxy"])
        > HARD_THRESHOLDS["qwk_proxy"],
        "acc_pm1_proxy": float(metric_means["acc_pm1_proxy"])
        > HARD_THRESHOLDS["acc_pm1_proxy"],
    }
    repeat_flags = {
        name: int(repeat_win_counts[name]) >= int(minimum_repeat_wins)
        for name in HARD_THRESHOLDS
    }
    return {
        "promoted": bool(all(mean_flags.values()) and all(repeat_flags.values())),
        "mean_threshold_flags": mean_flags,
        "repeat_win_flags": repeat_flags,
        "minimum_repeat_wins": int(minimum_repeat_wins),
    }


def _repeat_rows(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    selected = frame.loc[frame["evaluation"].str.startswith(prefix)].copy()
    if selected.empty:
        raise RuntimeError(f"no repeat metrics found for prefix {prefix}")
    selected["repeat"] = selected["evaluation"].str.rsplit("_", n=1).str[-1].astype(int)
    if selected["repeat"].duplicated().any():
        raise RuntimeError(f"duplicated repeat metrics for prefix {prefix}")
    return selected[["repeat", *METRICS]].sort_values("repeat").reset_index(drop=True)


def compare_screening_frames(
    baseline_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    experiment_id: str,
    model: str,
    candidate_prefix: str,
) -> tuple[dict, pd.DataFrame, dict]:
    """Align one screening candidate with the same E003 repeat IDs."""
    baseline_rows = _repeat_rows(baseline_frame, "repeat_")
    candidate_rows = _repeat_rows(candidate_frame, candidate_prefix)
    repeat_ids = candidate_rows["repeat"].tolist()
    baseline = baseline_rows.loc[baseline_rows["repeat"].isin(repeat_ids)].copy()
    if baseline["repeat"].tolist() != repeat_ids:
        raise RuntimeError(f"E003 repeat IDs do not align with {experiment_id}")
    baseline_means = {name: float(baseline[name].mean()) for name in METRICS}
    candidate_means = {
        name: float(candidate_rows[name].mean()) for name in METRICS
    }
    gate = screening_decision(candidate_means, baseline_means)
    decision = {
        "model": model,
        "repeat_ids": repeat_ids,
        "baseline_means": baseline_means,
        "candidate_means": candidate_means,
        **gate,
    }
    summary = {
        "experiment_id": experiment_id,
        "model": model,
        "n_repeats": len(repeat_ids),
        "promote_to_full": gate["promote_to_full"],
        "improved_count": gate["improved_count"],
    }
    for name in METRICS:
        summary[f"baseline_{name}"] = baseline_means[name]
        summary[f"candidate_{name}"] = candidate_means[name]
        summary[f"delta_{name}"] = candidate_means[name] - baseline_means[name]
        summary[f"improved_{name}"] = gate["improved"][name]
    paired = candidate_rows.merge(
        baseline, on="repeat", suffixes=("_candidate", "_baseline")
    )
    paired.insert(0, "model", model)
    paired.insert(0, "experiment_id", experiment_id)
    return summary, paired, decision


def compare_full_frames(
    baseline_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    experiment_id: str,
    model: str,
    candidate_prefix: str,
) -> tuple[dict, pd.DataFrame, dict]:
    """Apply the frozen five-threshold and 7/10 gate to a full candidate."""
    baseline_rows = _repeat_rows(baseline_frame, "repeat_")
    candidate_rows = _repeat_rows(candidate_frame, candidate_prefix)
    repeat_ids = candidate_rows["repeat"].tolist()
    if repeat_ids != list(range(10)):
        raise RuntimeError(
            f"{experiment_id} full audit requires repeats 0--9, got {repeat_ids}"
        )
    baseline = baseline_rows.loc[baseline_rows["repeat"].isin(repeat_ids)].copy()
    if baseline["repeat"].tolist() != repeat_ids:
        raise RuntimeError(f"E003 repeat IDs do not align with {experiment_id}")
    paired = candidate_rows.merge(
        baseline, on="repeat", suffixes=("_candidate", "_baseline")
    )
    paired.insert(0, "model", model)
    paired.insert(0, "experiment_id", experiment_id)
    candidate_means = {
        name: float(candidate_rows[name].mean()) for name in METRICS
    }
    baseline_means = {name: float(baseline[name].mean()) for name in METRICS}
    repeat_win_counts = {
        "r2": int((paired["r2_candidate"] >= paired["r2_baseline"]).sum()),
        "mae": int((paired["mae_candidate"] <= paired["mae_baseline"]).sum()),
        "spearman": int(
            (paired["spearman_candidate"] >= paired["spearman_baseline"]).sum()
        ),
        "qwk_proxy": int(
            (paired["qwk_proxy_candidate"] >= paired["qwk_proxy_baseline"]).sum()
        ),
        "acc_pm1_proxy": int(
            (
                paired["acc_pm1_proxy_candidate"]
                >= paired["acc_pm1_proxy_baseline"]
            ).sum()
        ),
    }
    gate = promotion_decision(candidate_means, repeat_win_counts)
    decision = {
        "model": model,
        "repeat_ids": repeat_ids,
        "baseline_means": baseline_means,
        "candidate_means": candidate_means,
        "repeat_win_counts": repeat_win_counts,
        **gate,
    }
    summary = {
        "experiment_id": experiment_id,
        "model": model,
        "n_repeats": len(repeat_ids),
        "promoted": gate["promoted"],
    }
    for name in METRICS:
        summary[f"threshold_{name}"] = HARD_THRESHOLDS[name]
        summary[f"baseline_{name}"] = baseline_means[name]
        summary[f"candidate_{name}"] = candidate_means[name]
        summary[f"delta_{name}"] = candidate_means[name] - baseline_means[name]
        summary[f"mean_gate_{name}"] = gate["mean_threshold_flags"][name]
        summary[f"repeat_wins_{name}"] = repeat_win_counts[name]
        summary[f"repeat_gate_{name}"] = gate["repeat_win_flags"][name]
    return summary, paired, decision


def audit_e004_screening() -> tuple[pd.DataFrame, dict]:
    """Compare CPU/GPU E004 screening with E003 on repeats 0--2 only."""
    baseline_all = pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv")
    candidates = [
        (
            "E004",
            "ordinal_hgb",
            RESULT_DIR / "q1_extended_metrics_E004_screening.csv",
            "pred_ordinal_hgb:repeat_",
        ),
        (
            "E004_GPU",
            "ordinal_xgb_cuda",
            RESULT_DIR / "q1_extended_metrics_E004_GPU_screening.csv",
            "pred_ordinal_xgb:repeat_",
        ),
    ]
    summary_rows = []
    repeat_frames = []
    decisions = {}
    for experiment_id, model, path, prefix in candidates:
        summary_row, paired, decision = compare_screening_frames(
            baseline_all,
            pd.read_csv(path),
            experiment_id=experiment_id,
            model=model,
            candidate_prefix=prefix,
        )
        decisions[experiment_id] = decision
        summary_rows.append(summary_row)
        repeat_frames.append(paired)

    summary = pd.DataFrame(summary_rows)
    by_repeat = pd.concat(repeat_frames, ignore_index=True)
    summary_path = RESULT_DIR / "q1_screening_comparison_E004.csv"
    repeat_path = RESULT_DIR / "q1_screening_by_repeat_E004.csv"
    decision_path = RESULT_DIR / "q1_screening_decision_E004.json"
    summary.to_csv(summary_path, index=False)
    by_repeat.to_csv(repeat_path, index=False)
    payload = {
        "status": "screening_only",
        "full_promotion_requires": "10 repeats and all five hard gates plus 7/10 wins",
        "decisions": decisions,
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def audit_e005_screening() -> tuple[pd.DataFrame, dict]:
    """Compare E005 continuous-blend screening with E003 on repeats 0--2."""
    baseline_all = pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv")
    summary_row, paired, decision = compare_screening_frames(
        baseline_all,
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E005_screening.csv"),
        experiment_id="E005",
        model="continuous_blend",
        candidate_prefix="pred_continuous_blend:repeat_",
    )
    summary = pd.DataFrame([summary_row])
    summary_path = RESULT_DIR / "q1_screening_comparison_E005.csv"
    repeat_path = RESULT_DIR / "q1_screening_by_repeat_E005.csv"
    decision_path = RESULT_DIR / "q1_screening_decision_E005.json"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(repeat_path, index=False)
    payload = {
        "status": "screening_only",
        "full_promotion_requires": "10 repeats and all five hard gates plus 7/10 wins",
        "decisions": {"E005": decision},
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def audit_e005_full() -> tuple[pd.DataFrame, dict]:
    """Audit E005 full 10-repeat OOF against every frozen gate."""
    summary_row, paired, decision = compare_full_frames(
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv"),
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E005.csv"),
        experiment_id="E005",
        model="continuous_blend",
        candidate_prefix="pred_continuous_blend:repeat_",
    )
    summary = pd.DataFrame([summary_row])
    summary_path = RESULT_DIR / "q1_full_comparison_E005.csv"
    repeat_path = RESULT_DIR / "q1_full_by_repeat_E005.csv"
    decision_path = RESULT_DIR / "q1_full_decision_E005.json"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(repeat_path, index=False)
    payload = {
        "status": "full_gate",
        "hard_thresholds": HARD_THRESHOLDS,
        "decision": decision,
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def audit_e006_screening() -> tuple[pd.DataFrame, dict]:
    """Compare E006 dual-route screening with E003 on repeats 0--2."""
    summary_row, paired, decision = compare_screening_frames(
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv"),
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E006_screening.csv"),
        experiment_id="E006",
        model="dual_route",
        candidate_prefix="pred_dual_route:repeat_",
    )
    summary = pd.DataFrame([summary_row])
    summary_path = RESULT_DIR / "q1_screening_comparison_E006.csv"
    repeat_path = RESULT_DIR / "q1_screening_by_repeat_E006.csv"
    decision_path = RESULT_DIR / "q1_screening_decision_E006.json"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(repeat_path, index=False)
    payload = {
        "status": "screening_only",
        "full_promotion_requires": "10 repeats and all five hard gates plus 7/10 wins",
        "decisions": {"E006": decision},
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def audit_e006_full() -> tuple[pd.DataFrame, dict]:
    """Audit E006 full 10-repeat dual-route OOF against every frozen gate."""
    summary_row, paired, decision = compare_full_frames(
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv"),
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E006.csv"),
        experiment_id="E006",
        model="dual_route",
        candidate_prefix="pred_dual_route:repeat_",
    )
    summary = pd.DataFrame([summary_row])
    summary_path = RESULT_DIR / "q1_full_comparison_E006.csv"
    repeat_path = RESULT_DIR / "q1_full_by_repeat_E006.csv"
    decision_path = RESULT_DIR / "q1_full_decision_E006.json"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(repeat_path, index=False)
    payload = {
        "status": "full_gate",
        "hard_thresholds": HARD_THRESHOLDS,
        "decision": decision,
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def audit_e007_screening() -> tuple[pd.DataFrame, dict]:
    """Compare E007 shrunk-gating screening with E003 on repeats 0--2."""
    summary_row, paired, decision = compare_screening_frames(
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv"),
        pd.read_csv(RESULT_DIR / "q1_extended_metrics_E007_screening.csv"),
        experiment_id="E007",
        model="shrunk_gate",
        candidate_prefix="pred_shrunk_gate:repeat_",
    )
    summary = pd.DataFrame([summary_row])
    summary_path = RESULT_DIR / "q1_screening_comparison_E007.csv"
    repeat_path = RESULT_DIR / "q1_screening_by_repeat_E007.csv"
    decision_path = RESULT_DIR / "q1_screening_decision_E007.json"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(repeat_path, index=False)
    payload = {
        "status": "screening_only",
        "full_promotion_requires": "10 repeats and all five hard gates plus 7/10 wins",
        "decisions": {"E007": decision},
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "e004-screening",
            "e005-screening",
            "e005-full",
            "e006-screening",
            "e006-full",
            "e007-screening",
        ],
        default="e004-screening",
    )
    args = parser.parse_args()
    if args.mode == "e004-screening":
        summary, payload = audit_e004_screening()
    elif args.mode == "e005-screening":
        summary, payload = audit_e005_screening()
    elif args.mode == "e005-full":
        summary, payload = audit_e005_full()
    elif args.mode == "e006-screening":
        summary, payload = audit_e006_screening()
    elif args.mode == "e006-full":
        summary, payload = audit_e006_full()
    else:
        summary, payload = audit_e007_screening()
    print(summary.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
