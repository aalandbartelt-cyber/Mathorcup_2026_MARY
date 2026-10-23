"""Run E100: freeze Q2 data, candidate, and graph contracts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from q2_contract import (
    ATTACHMENTS,
    DATA_DIR,
    Q1_RESULT_DIR,
    RESULT_DIR,
    audit_official_graph,
    build_candidate_dictionary,
    build_node_table,
    build_soft_edge_table,
    graph_threshold_audit,
    load_q2,
)


BETAS = (1, 2, 4, 6)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
RANDOM_STATE = 20261024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_e100(output_dir: Path = RESULT_DIR) -> dict:
    """Generate every registered E100 artifact without fitting a model."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, style, beer = load_q2()
    candidates = build_candidate_dictionary(x.columns)
    correlation = x.corr(method="spearman")
    graph_audit = audit_official_graph(x)
    edges = build_soft_edge_table(correlation, BETAS)
    nodes = build_node_table(candidates, correlation, BETAS)
    threshold_audit = graph_threshold_audit(correlation, THRESHOLDS)

    # Add readable names while retaining field codes as graph identifiers.
    display = candidates.set_index("feature_code")["compound_name"]
    edges.insert(1, "source_name", edges["source"].map(display))
    edges.insert(3, "target_name", edges["target"].map(display))

    target_dictionary_path = Q1_RESULT_DIR / "q1_target_dictionary.csv"
    target_dictionary = pd.read_csv(target_dictionary_path)
    for column in target_dictionary.select_dtypes(include="object"):
        target_dictionary[column] = target_dictionary[column].map(
            lambda value: re.sub(r"\s+", " ", value).strip()
            if isinstance(value, str)
            else value
        )
    if target_dictionary["target_code"].tolist() != y.columns.tolist():
        raise ValueError("Q1 target dictionary order disagrees with attachment 4")

    candidates.to_csv(output_dir / "q2_candidate_dictionary.csv", index=False)
    nodes.to_csv(output_dir / "q2_graph_nodes.csv", index=False)
    edges.to_csv(output_dir / "q2_graph_edges.csv.gz", index=False, compression="gzip")
    threshold_audit.to_csv(
        output_dir / "q2_graph_threshold_audit.csv", index=False
    )
    target_dictionary.to_csv(output_dir / "q2_target_dictionary.csv", index=False)

    strict = candidates["eligible_strict"]
    contract = {
        "experiment_id": "E100",
        "n_samples": int(len(x)),
        "n_features": int(x.shape[1]),
        "n_targets": int(y.shape[1]),
        "n_styles": int(style.nunique()),
        "beer_key": "beer",
        "beer_unique": int(beer.nunique()),
        "forbidden_fields": ["beer", "beer_id", "tasting_category_fine"],
        "x_missing_cells": int(x.isna().sum().sum()),
        "x_rows_with_missing": int(x.isna().any(axis=1).sum()),
        "x_columns_with_missing": int(x.isna().any().sum()),
        "target_scale": "standardized continuous values from 4.csv",
        "graph_definition": "spearman",
        "official_graph_role": "full-data verification and final descriptive graph",
        "validation_graph_role": "recompute from each outer-training fold",
        "soft_graph_betas": list(BETAS),
        "hard_thresholds_for_sensitivity_only": list(THRESHOLDS),
        "official_graph_audit": graph_audit,
        "candidate_scope": {
            "all_factor_count": int(len(candidates)),
            "strict_count": int(strict.sum()),
            "excluded_count": int((~strict).sum()),
            "excluded_feature_codes": candidates.loc[
                ~strict, "feature_code"
            ].tolist(),
            "primary_answer": "strict single compounds",
            "sensitivity_answer": "all measured factors",
        },
        "preprocessing": {
            "imputation": "training-fold median",
            "missing_indicator": "training-fold missing columns",
            "scale_action": "keep supplied scales; do not repeat log transform",
            "sample_deletion": "none",
        },
        "frozen_outer_cv": "03_结果/Q1_感官评分/q1_cv_splits.csv",
        "input_sha256": {
            name: _sha256(ATTACHMENTS / name)
            for name in ("1.csv", "2.csv", "4.csv", "8.xlsx")
        },
        "name_map_sha256": _sha256(DATA_DIR / "compound_name_map.csv"),
        "target_dictionary_sha256": _sha256(target_dictionary_path),
        "outer_cv_sha256": _sha256(Q1_RESULT_DIR / "q1_cv_splits.csv"),
    }
    _json_dump(output_dir / "q2_data_contract.json", contract)

    manifest = {
        "experiment_id": "E100",
        "question": "Q2",
        "status": "complete",
        "supervised_model_fitted": False,
        "random_state": RANDOM_STATE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "artifact_count": 7,
    }
    _json_dump(output_dir / "q2_run_manifest_E100.json", manifest)
    return manifest


def main() -> None:
    manifest = run_e100()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
