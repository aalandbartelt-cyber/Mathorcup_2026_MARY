"""Fold-local robust sensitivity and baseline comparison for completed E203."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from audit_e202 import training_winsorize
from q3_evaluation import distribution_diagnostics
from run_e200_contract import validate_cv


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果/Q3_风格生成"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    attachment = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    split_path = ROOT / "03_结果/Q1_感官评分/q1_cv_splits.csv"
    manifest_path = RESULT / "q3_run_manifest_E203.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("n_outer_folds") != 30:
        raise ValueError("complete E203 manifest required")
    chemistry = pd.read_csv(attachment)
    splits = pd.read_csv(split_path)
    validate_cv(chemistry, splits, n_repeats=10, n_folds=3)
    features = [column for column in chemistry if column not in ["beer", "beer_id", "tasting_category_fine"]]
    x = chemistry[features].to_numpy(float)
    styles = chemistry.tasting_category_fine.to_numpy(str)
    rows = []
    checkpoint_hashes = {}
    for (repeat, fold), group in splits.groupby(["repeat", "fold"], sort=True):
        repeat, fold = int(repeat), int(fold)
        split_id = f"R{repeat:02d}F{fold}"
        slot = RESULT / "q3_checkpoints_E203" / split_id
        checkpoint_hashes[split_id] = sha(slot / "manifest.json")
        generated = pd.read_csv(slot / "generated.csv.gz")
        valid = np.sort(group.row_position.to_numpy(int))
        train = np.setdiff1d(np.arange(len(x)), valid)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(x[train])
        valid_x = imputer.transform(x[valid])
        lower = np.quantile(train_x, 0.01, axis=0)
        upper = np.quantile(train_x, 0.99, axis=0)
        for (method, style, seed_index), sample in generated.groupby(
            ["method", "style", "seed_index"], sort=True
        ):
            real = valid_x[styles[valid] == style]
            local = train_x[styles[train] == style]
            generated_x = sample[features].to_numpy(float)
            clipped_training, (clipped_real, clipped_generated), scale, _ = training_winsorize(
                train_x, real, generated_x, lower=0.01, upper=0.99
            )
            clipped_local = np.clip(local, lower, upper)
            metrics = distribution_diagnostics(clipped_real, clipped_generated, clipped_local, scale)
            metrics.pop("w1_by_feature")
            diagnostic_scope = metrics.pop("scope")
            rows.append(
                {
                    "evaluation_scope": "outer_validation_fold_train_q01_q99_sensitivity",
                    "diagnostic_scope": diagnostic_scope,
                    "repeat": repeat,
                    "fold": fold,
                    "style": style,
                    "method": method,
                    "seed_index": int(seed_index),
                    "status": "evaluated",
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    if len(table) != 1350:
        raise ValueError("E203 robust audit row count changed")
    table_path = RESULT / "q3_e203_robust_sensitivity.csv"
    table.to_csv(table_path, index=False)
    names = ["mean_w1", "energy_v", "coverage", "near_duplicate_fraction", "spearman_pair_rmse"]
    repeat_table = table.groupby(["method", "style", "repeat"], as_index=False)[names].mean()
    repeat_path = RESULT / "q3_e203_robust_by_repeat.csv"
    repeat_table.to_csv(repeat_path, index=False)
    summary = repeat_table.groupby(["method", "style"])[names].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary_path = RESULT / "q3_e203_robust_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)

    copula = pd.read_csv(RESULT / "q3_e202_robust_by_repeat.csv")
    copula = copula[copula.method == "style_copula_shrunk"].copy()
    comparisons = []
    for method, candidate in repeat_table.groupby("method"):
        merged = candidate.merge(copula, on=["style", "repeat"], suffixes=("_candidate", "_copula"))
        for metric in ["mean_w1", "energy_v", "coverage", "spearman_pair_rmse"]:
            direction = -1.0 if metric == "coverage" else 1.0
            delta = direction * (merged[f"{metric}_candidate"] - merged[f"{metric}_copula"])
            comparisons.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_style_repeats": len(delta),
                    "candidate_minus_copula_oriented_mean": float(delta.mean()),
                    "candidate_better_fraction": float((delta < 0).mean()),
                }
            )
    comparison_path = RESULT / "q3_e203_vs_copula_robust_comparison.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    audit = {
        "status": "complete",
        "scope": "fold_training_q01_q99_sensitivity_and_paired_repeat_comparison",
        "n_rows": len(table),
        "n_outer_folds": 30,
        "input_manifest_sha256": sha(manifest_path),
        "e202_robust_manifest_sha256": sha(RESULT / "q3_e202_audit_manifest.json"),
        "source_sha256": sha(Path(__file__)),
        "checkpoint_manifest_sha256": checkpoint_hashes,
        "output_sha256": {
            path.name: sha(path)
            for path in [table_path, repeat_path, summary_path, comparison_path]
        },
        "notes": [
            "分位边界只由当前外层训练折估计",
            "原始口径仍为主结果，稳健口径仅作异常值敏感性",
            "成对比较单位为风格×重复，五个种子不当作独立样本",
        ],
    }
    audit_path = RESULT / "q3_e203_audit_manifest.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    run()
