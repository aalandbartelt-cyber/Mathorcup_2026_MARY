"""Reproducible integrity audit for the final Q2 NNLS attribution outputs."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果" / "Q2_关键化合物"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    """Convert NumPy scalar audit values into standard JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _historical_csv(path: Path) -> pd.DataFrame:
    relative = path.relative_to(ROOT).as_posix()
    payload = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


def _same_rows(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(left[column], right[column], rtol=1e-12, atol=1e-12):
                return False
        elif not left[column].fillna("").equals(right[column].fillna("")):
            return False
    return True


def main() -> None:
    paths = {
        "module": RESULT / "q2_final_nnls_module_importance_E115.csv.gz",
        "conditional": RESULT / "q2_final_nnls_conditional_importance_E115.csv.gz",
        "deletion": RESULT / "q2_final_nnls_sufficiency_E115.csv",
    }
    manifest_path = RESULT / "q2_run_manifest_E115_final_nnls.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module = pd.read_csv(paths["module"])
    conditional = pd.read_csv(paths["conditional"])
    deletion = pd.read_csv(paths["deletion"])

    fold_columns = ["repeat", "fold"]
    expected_folds = {(repeat, fold) for repeat in range(10) for fold in range(3)}
    observed_folds = {
        tuple(row) for row in module[fold_columns].drop_duplicates().to_numpy().tolist()
    }
    strict_features = set(conditional["feature_code"].astype(str))

    member_checks = []
    for _, fold_frame in module.groupby(fold_columns, sort=True):
        modules = fold_frame.drop_duplicates("module_id")
        members = [
            member
            for packed in modules["module_members"].astype(str)
            for member in packed.split("|")
        ]
        member_checks.append(
            len(members) == 218
            and len(set(members)) == 218
            and set(members) == strict_features
        )

    top3_lists = deletion["top3_features"].astype(str).str.split("|")
    top3_valid = bool(
        top3_lists.map(lambda values: len(values) == 3 and len(set(values)) == 3).all()
        and top3_lists.map(lambda values: set(values).issubset(strict_features)).all()
    )

    metadata_paths = sorted((RESULT / "q2_nnls_cache_E115").glob("R??F?.json"))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    max_validation_error = max(item["validation_max_abs_error"] for item in metadata)

    pilot_module = _historical_csv(paths["module"])
    pilot_conditional = _historical_csv(paths["conditional"])
    pilot_deletion = _historical_csv(paths["deletion"])
    final_repeat0_module = module.loc[module["repeat"] == 0, pilot_module.columns]
    final_repeat0_conditional = conditional.loc[
        conditional["repeat"] == 0, pilot_conditional.columns
    ]
    final_repeat0_deletion = deletion.loc[
        deletion["repeat"] == 0, pilot_deletion.columns
    ]
    deletion_compare = pilot_deletion.merge(
        final_repeat0_deletion,
        on=["repeat", "fold", "target_code"],
        suffixes=("_pilot", "_final"),
        validate="one_to_one",
    )
    changed_top3 = int(
        (deletion_compare["top3_features_pilot"] != deletion_compare["top3_features_final"]).sum()
    )

    actual_hashes = {path.name: _sha256(path) for path in paths.values()}
    hash_match = actual_hashes == manifest["output_sha256"]
    finite_columns = {
        "module": bool(np.isfinite(module[["median_delta_mae", "positive_fraction"]]).all().all()),
        "conditional": bool(
            np.isfinite(conditional[["median_delta_mae", "positive_fraction"]]).all().all()
        ),
        "deletion": bool(
            np.isfinite(
                deletion[
                    ["baseline_mae", "masked_mae", "delta_mae_masked_minus_baseline"]
                ]
            ).all().all()
        ),
    }

    checks = {
        "manifest_complete": manifest.get("status") == "complete",
        "ten_repeats_thirty_folds": manifest.get("repeat_ids") == list(range(10))
        and manifest.get("n_outer_folds") == 30
        and observed_folds == expected_folds,
        "q1_not_retrained": manifest.get("q1_retrained") is False,
        "nnls_not_refit": manifest.get("nnls_refit") is False,
        "no_hyperparameter_search": manifest.get("hyperparameter_search") is False,
        "sufficiency_not_claimed": manifest.get("sufficiency_claimed") is False,
        "module_rows_and_keys": len(module) == 13_900
        and not module.duplicated(fold_columns + ["module_id", "target_code"]).any()
        and module.groupby(fold_columns + ["module_id"])["target_code"].nunique().eq(50).all(),
        "module_partition_exact_218": bool(all(member_checks)),
        "conditional_rows_and_keys": len(conditional) == 327_000
        and not conditional.duplicated(fold_columns + ["feature_code", "target_code"]).any()
        and conditional.groupby(fold_columns)["feature_code"].nunique().eq(218).all()
        and conditional.groupby(fold_columns + ["feature_code"])["target_code"].nunique().eq(50).all(),
        "conditional_scope_valid": conditional["conditional_scope"]
        .eq("observed_only_missing_mask_preserved")
        .all(),
        "deletion_rows_and_keys": len(deletion) == 1_500
        and not deletion.duplicated(fold_columns + ["target_code"]).any()
        and deletion.groupby(fold_columns)["target_code"].nunique().eq(50).all(),
        "deletion_top3_valid": top3_valid,
        "deletion_is_matching_fold_fixed_model": deletion["top3_source"]
        .eq("E114_matching_outer_training_fold_partial_effect")
        .all()
        and deletion["evaluation_scope"]
        .eq("fixed_final_nnls_deletion_mask_not_top3_sufficiency")
        .all(),
        "ten_permutations": module["n_permutations"].eq(10).all()
        and conditional["n_permutations"].eq(10).all(),
        "all_numeric_outputs_finite": all(finite_columns.values()),
        "cache_metadata_complete": len(metadata) == 30,
        "cache_flags_valid": all(
            item["q1_outputs_written"] is False
            and item["nnls_refit"] is False
            and item["hyperparameter_search"] is False
            and item["scope"] == "Q2_reconstruction_of_frozen_outer_training_predictor"
            for item in metadata
        ),
        "single_adapter_signature": len({item["signature"] for item in metadata}) == 1
        and metadata[0]["signature"] == manifest["adapter_signature"],
        "reconstruction_tolerance": max_validation_error <= 1e-10
        and manifest["max_reconstruction_error"] <= 1e-10,
        "manifest_hashes_match": hash_match,
        "pilot_repeat0_module_exact": _same_rows(
            pilot_module,
            final_repeat0_module,
            ["repeat", "fold", "module_id", "target_code"],
        ),
        "pilot_repeat0_conditional_exact": _same_rows(
            pilot_conditional,
            final_repeat0_conditional,
            ["repeat", "fold", "feature_code", "target_code"],
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    audit = {
        "experiment_id": "E115_final_nnls",
        "audit_status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "checks": checks,
        "counts": {
            "module_rows": len(module),
            "conditional_rows": len(conditional),
            "deletion_rows": len(deletion),
            "folds": len(observed_folds),
            "repeats": module["repeat"].nunique(),
            "targets": conditional["target_code"].nunique(),
            "strict_features": conditional["feature_code"].nunique(),
            "cache_metadata": len(metadata),
        },
        "pilot_vs_final": {
            "pilot_folds": int(pilot_module[fold_columns].drop_duplicates().shape[0]),
            "final_folds": len(observed_folds),
            "repeat0_module_exact": checks["pilot_repeat0_module_exact"],
            "repeat0_conditional_exact": checks["pilot_repeat0_conditional_exact"],
            "deletion_rows_compared": len(deletion_compare),
            "deletion_top3_changed_after_fold_safe_repair": changed_top3,
            "note": "The pilot deletion table used globally aggregated Top-3 fields; the final table supersedes it with matching-fold Top-3 fields.",
        },
        "reconstruction": {
            "max_cache_validation_abs_error": max_validation_error,
            "manifest_max_reconstruction_error": manifest["max_reconstruction_error"],
        },
        "descriptive": {
            "module_positive_delta_fraction": float((module["median_delta_mae"] > 0).mean()),
            "module_median_delta_mae": float(module["median_delta_mae"].median()),
            "conditional_positive_delta_fraction": float(
                (conditional["median_delta_mae"] > 0).mean()
            ),
            "conditional_median_delta_mae": float(conditional["median_delta_mae"].median()),
            "deletion_positive_delta_fraction": float(
                (deletion["delta_mae_masked_minus_baseline"] > 0).mean()
            ),
            "deletion_median_delta_mae": float(
                deletion["delta_mae_masked_minus_baseline"].median()
            ),
            "deletion_mean_delta_mae": float(
                deletion["delta_mae_masked_minus_baseline"].mean()
            ),
        },
        "output_sha256": actual_hashes,
    }
    output = RESULT / "q2_audit_E115_final_nnls.json"
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default))
    if failed:
        raise SystemExit(f"E115 audit failed: {failed}")


if __name__ == "__main__":
    main()
