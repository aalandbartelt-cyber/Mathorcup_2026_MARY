"""Summarize registered-versus-wide E109 sensitivity without refitting models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q2_contract import RESULT_DIR


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    base_repeat = pd.read_csv(
        output_dir / "q2_model_summary_by_repeat_E101_E104.csv"
    )
    wide_repeat = pd.read_csv(
        output_dir / "q2_model_summary_by_repeat_E101_E104_E109_wide.csv"
    )
    base_top3 = pd.read_csv(output_dir / "q2_top3_by_method_E101_E104.csv")
    wide_top3 = pd.read_csv(
        output_dir / "q2_top3_by_method_E101_E104_E109_wide.csv"
    )
    parameters = pd.read_csv(
        output_dir / "q2_selected_parameters_E101_E104_E109_wide.csv"
    )

    rows = []
    details = {}
    for method, experiment in (
        ("E103_ElasticNet", "E103"),
        ("E104_GraphElasticNet", "E104"),
    ):
        base = base_repeat.loc[base_repeat["method"].eq(method)].set_index("repeat")
        wide = wide_repeat.loc[wide_repeat["method"].eq(method)].set_index("repeat")
        deltas = wide[["macro_mae", "macro_r2", "macro_spearman"]] - base[
            ["macro_mae", "macro_r2", "macro_spearman"]
        ]
        jaccards = []
        exact = 0
        for target in sorted(base_top3["target_code"].unique()):
            left = set(
                base_top3.loc[
                    base_top3["method"].eq(method)
                    & base_top3["target_code"].eq(target),
                    "feature_code",
                ]
            )
            right = set(
                wide_top3.loc[
                    wide_top3["method"].eq(method)
                    & wide_top3["target_code"].eq(target),
                    "feature_code",
                ]
            )
            value = len(left & right) / len(left | right)
            jaccards.append(value)
            exact += int(value == 1.0)
        selected = parameters.loc[parameters["experiment_id"].eq(experiment)]
        beyond_l1 = int((selected["lambda_l1"] < 0.01).sum())
        beyond_l2 = int((selected["lambda_l2"] > 1.0).sum())
        beyond_beta = (
            int((selected["beta"] > 6).sum()) if selected["beta"].notna().any() else 0
        )
        rows.append(
            {
                "method": method,
                "wide_minus_registered_macro_mae_mean": deltas["macro_mae"].mean(),
                "wide_minus_registered_macro_r2_mean": deltas["macro_r2"].mean(),
                "wide_minus_registered_macro_spearman_mean": deltas[
                    "macro_spearman"
                ].mean(),
                "top3_jaccard_mean": float(np.mean(jaccards)),
                "top3_jaccard_median": float(np.median(jaccards)),
                "exact_top3_targets": exact,
                "n_targets": len(jaccards),
                "selected_l1_below_registered_folds": beyond_l1,
                "selected_l2_above_registered_folds": beyond_l2,
                "selected_beta_above_registered_folds": beyond_beta,
            }
        )
        details[method] = {
            "paired_repeat_deltas": deltas.reset_index().to_dict(orient="records"),
            "selected_parameter_counts": {
                column: {
                    str(key): int(value)
                    for key, value in selected[column]
                    .dropna()
                    .value_counts()
                    .sort_index()
                    .items()
                }
                for column in ("lambda_l1", "lambda_l2", "beta", "lambda_graph")
            },
        }

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "q2_e109_boundary_sensitivity_summary.csv", index=False)
    report = {
        "experiment": "E109",
        "comparison": "e109_wide minus registered E101-E104 grid",
        "summary": summary.to_dict(orient="records"),
        "details": details,
        "decision": {
            "E103": "retain registered grid; wide run is identical",
            "E104": "do not promote graph penalty; wider beta does not improve OOF MAE",
        },
    }
    (output_dir / "q2_e109_boundary_sensitivity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
