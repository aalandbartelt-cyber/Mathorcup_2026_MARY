"""Read-only OOF error-complementarity diagnostics; no weight fitting or training."""

from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data_contract import RESULT_DIR
from q1_metrics import matrix_metrics, quantile_levels, target_metrics


def validate_oof_alignment(reference: dict, candidate: dict) -> None:
    for key in ["beer", "target_code", "repeat", "style", "fold_assignment", "y_true"]:
        if key not in reference or key not in candidate:
            raise ValueError(f"OOF alignment metadata missing: {key}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"OOF alignment mismatch: {key}")


def _correlation(first, second) -> float:
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def pairwise_complementarity(y_true, predictions: dict, target_codes) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != len(target_codes) or not np.isfinite(truth).all():
        raise ValueError("truth must be a finite samples-by-targets matrix")
    if len(predictions) < 2:
        raise ValueError("need at least two prediction models")
    matrices = {str(k): np.asarray(v, dtype=float) for k, v in predictions.items()}
    shape = next(iter(matrices.values())).shape
    if (len(shape) != 3 or shape[1:] != truth.shape
            or any(v.shape != shape or not np.isfinite(v).all() for v in matrices.values())):
        raise ValueError("predictions must be aligned, complete repeats-by-samples-by-targets")
    truth_levels = [quantile_levels(truth[:, j], truth[:, j]) for j in range(truth.shape[1])]
    rows = []
    for repeat in range(shape[0]):
        for target, code in enumerate(target_codes):
            observed = truth[:, target]
            bins = {k: quantile_levels(v[repeat, :, target], v[repeat, :, target])
                    for k, v in matrices.items()}
            bad = {k: np.abs(value - truth_levels[target]) > 1 for k, value in bins.items()}
            for first, second in combinations(matrices, 2):
                p, q = matrices[first][repeat, :, target], matrices[second][repeat, :, target]
                rescue_first = bad[first] & ~bad[second]
                rescue_second = bad[second] & ~bad[first]
                rows.append({
                    "repeat": repeat, "target_code": str(code),
                    "first_model": first, "second_model": second,
                    "residual_correlation": _correlation(observed - p, observed - q),
                    "prediction_rank_correlation": _correlation(rankdata(p), rankdata(q)),
                    "first_pm1": float(np.mean(~bad[first])),
                    "second_pm1": float(np.mean(~bad[second])),
                    "first_failure_count": int(bad[first].sum()),
                    "second_failure_count": int(bad[second].sum()),
                    "first_rescued_by_second": float(rescue_first.sum() / max(bad[first].sum(), 1)),
                    "second_rescued_by_first": float(rescue_second.sum() / max(bad[second].sum(), 1)),
                    "exclusive_rescue_mass": float(np.mean(rescue_first | rescue_second)),
                    "oracle_union_pm1_diagnostic": float(np.mean(~(bad[first] & bad[second]))),
                })
    return pd.DataFrame(rows)


def run_diagnostics() -> dict:
    paths = {name: RESULT_DIR / f"q1_oof_predictions_{name}.npz" for name in ["E003", "E005"]}
    archives = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            archives[name] = {k: archive[k].copy() for k in archive.files}
    reference, base = archives["E003"], archives["E005"]
    validate_oof_alignment(reference, base)
    models = {"E003_extra_trees": reference["pred_extra_trees"]}
    for model in ["extra_trees", "hgb", "ridge_quantile", "pls2_quantile"]:
        models[model] = base[f"pred_{model}"]
    truth, codes = reference["y_true"], reference["target_code"]
    pairwise = pairwise_complementarity(truth, models, codes)
    metric_rows, target_rows = [], []
    for name, prediction in models.items():
        for repeat in range(len(prediction)):
            metric_rows.append({"model": name, "repeat": repeat,
                                **matrix_metrics(truth, prediction[repeat])})
            for j, code in enumerate(codes):
                target_rows.append({"model": name, "repeat": repeat, "target_code": str(code),
                                    **target_metrics(truth[:, j], prediction[repeat, :, j])})
    metrics = pd.DataFrame(metric_rows)
    pair_summary = pairwise.groupby(["first_model", "second_model"], sort=False).agg(
        residual_correlation=("residual_correlation", "mean"),
        prediction_rank_correlation=("prediction_rank_correlation", "mean"),
        first_pm1=("first_pm1", "mean"), second_pm1=("second_pm1", "mean"),
        first_rescued_by_second=("first_rescued_by_second", "mean"),
        second_rescued_by_first=("second_rescued_by_first", "mean"),
        exclusive_rescue_mass=("exclusive_rescue_mass", "mean"),
        oracle_union_pm1_diagnostic=("oracle_union_pm1_diagnostic", "mean"),
    ).reset_index()
    output_dir = RESULT_DIR / "diagnostics_nnls"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frame = pd.DataFrame(target_rows)
    dictionary_path = RESULT_DIR / "q1_target_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path)[["target_code", "target_group"]]
    grouped_targets = target_frame.merge(dictionary, on="target_code", validate="many_to_one")
    metric_columns = ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]
    group_metrics = grouped_targets.groupby(["target_group", "model"])[metric_columns].mean().reset_index()
    baseline_targets = target_frame.loc[target_frame["model"].eq("E003_extra_trees")]
    tail = baseline_targets.groupby("target_code")["r2"].mean().sort_values().head(
        max(1, int(np.ceil(len(codes) / 4))))
    tail_definition = tail.rename("baseline_mean_r2").reset_index()
    tail_metrics = target_frame.loc[target_frame["target_code"].isin(tail.index)].groupby(
        "model")[metric_columns].mean().reset_index()
    frames = {"q1_complementarity_pairs.csv": pairwise,
              "q1_complementarity_summary.csv": pair_summary,
              "q1_base_model_metrics_by_repeat.csv": metrics,
              "q1_base_model_metrics_by_target.csv": target_frame,
              "q1_grouped_base_metrics.csv": group_metrics,
              "q1_tail_target_baseline_definition.csv": tail_definition,
              "q1_tail_base_metrics.csv": tail_metrics}
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False)
    manifest = {
        "diagnostic_only": True, "trained_model": False, "fitted_stacking_weights": False,
        "models": list(models), "n_repeats": len(reference["repeat"]),
        "n_targets": len(codes), "n_beers": len(truth),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in paths.items()},
        "target_dictionary_sha256": hashlib.sha256(dictionary_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "complete existing E003/E005 OOF development diagnostics",
        "oracle_warning": "union of correct predictions is an infeasible diagnostic upper bound, not a model score",
        "outputs": [str(output_dir / name) for name in frames],
    }
    (output_dir / "q1_complementarity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_diagnostics(), ensure_ascii=False, indent=2))
