"""Run E115 hierarchical module and conditional-residual attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from hierarchical_importance import (
    aggregate_hierarchical_importance,
    fit_conditional_feature_model,
    joint_permutation,
    residual_permutation,
)
from permutation_importance import connected_modules
from q2_contract import RESULT_DIR, load_q2
from q2_pipeline import load_frozen_outer_folds


EXPERIMENT_ID = "E115"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "E115_hierarchical_importance.yaml"
)
Q1_PARAMETER_FILE = Path("03_结果/Q1_感官评分/q1_selected_parameters_E003.csv")
N_ESTIMATORS = 300
FALLBACK_MODULE_THRESHOLD = 0.70


def _atomic_to_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression="gzip" if gzip else None,
    )
    temporary.replace(path)


def select_fold_top3(
    fold_effects: pd.DataFrame, *, repeat: int, fold: int
) -> dict[str, list[str]]:
    """Read Top-3 chosen only from the matching outer-training fold."""
    required = {"repeat", "fold", "target_code", "feature_code", "fold_rank"}
    missing = required - set(fold_effects.columns)
    if missing:
        raise ValueError(f"E114 fold effects miss {sorted(missing)}")
    selected = fold_effects.loc[
        fold_effects["repeat"].eq(int(repeat))
        & fold_effects["fold"].eq(int(fold))
        & fold_effects["fold_rank"].le(3)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"no E114 Top-3 for repeat={repeat}, fold={fold}")
    result = {}
    for target, group in selected.groupby("target_code", sort=False):
        ordered = group.sort_values(
            ["fold_rank", "feature_code"], kind="mergesort"
        )["feature_code"].astype(str).tolist()
        if len(ordered) != 3 or len(set(ordered)) != 3:
            raise RuntimeError(f"E114 Top-3 is incomplete for {target}")
        result[str(target)] = ordered
    return result


def _load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if str(config.get("experiment")) != EXPERIMENT_ID:
        raise ValueError("E115 config experiment ID is invalid")
    if int(config["module_repeats_per_fold"]) < 1 or int(
        config["field_repeats_per_fold"]
    ) < 1:
        raise ValueError("E115 permutation repeats must be positive")
    return config


def _strict_codes(x: pd.DataFrame) -> list[str]:
    dictionary = pd.read_csv(RESULT_DIR / "q2_candidate_dictionary.csv")
    eligible = dictionary["eligible_strict"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().map({"true": True, "false": False})
    codes = dictionary.loc[eligible, "feature_code"].astype(str).tolist()
    if len(codes) != 218 or set(codes) - set(x.columns.astype(str)):
        raise ValueError("E115 strict candidate dictionary is invalid")
    return codes


def _fold_parameters(repeat: int, fold: int) -> tuple[float, int]:
    table = pd.read_csv(Q1_PARAMETER_FILE)
    selected = table.loc[
        table["repeat"].eq(int(repeat)) & table["fold"].eq(int(fold))
    ]
    if len(selected) != 1:
        raise RuntimeError("E115 cannot resolve unique E105/Q1 fold parameters")
    return (
        float(selected.iloc[0]["selected_max_features"]),
        int(selected.iloc[0]["selected_min_samples_leaf"]),
    )


def _indicator_map(imputer: SimpleImputer, n_original: int) -> dict[int, int]:
    if imputer.indicator_ is None:
        return {}
    return {
        int(feature): n_original + position
        for position, feature in enumerate(imputer.indicator_.features_)
    }


def _processed_columns(
    positions: list[int], indicator_map: dict[int, int]
) -> list[int]:
    return positions + [indicator_map[value] for value in positions if value in indicator_map]


def _modules_for_fold(
    x_train: pd.DataFrame,
    feature_codes: list[str],
    *,
    repeat: int,
    fold: int,
) -> tuple[list[list[str]], str]:
    decision_path = RESULT_DIR / "q2_graph_decision_E113.json"
    membership_path = RESULT_DIR / "q2_module_membership_E113.csv.gz"
    if decision_path.exists() and membership_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if bool(decision.get("promote_partial_graph", False)):
            membership = pd.read_csv(membership_path)
            selected = membership.loc[
                membership["scope"].eq("outer_train_fold")
                & membership["repeat"].eq(int(repeat))
                & membership["fold"].eq(int(fold))
            ]
            modules = [
                group["feature_code"].astype(str).tolist()
                for _, group in selected.groupby("module_id", sort=True)
            ]
            if sorted(value for module in modules for value in module) != sorted(feature_codes):
                raise RuntimeError("E113 fold modules do not cover strict compounds")
            return modules, "E113_fold_local_partial_graph"
    correlation = x_train[feature_codes].corr(method="spearman")
    return (
        connected_modules(
            correlation,
            eligible_features=feature_codes,
            threshold=FALLBACK_MODULE_THRESHOLD,
        ),
        "E107_fold_local_spearman_rule",
    )


def _fit_backbone(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    *,
    max_features: float,
    min_samples_leaf: int,
    seed: int,
) -> tuple[SimpleImputer, ExtraTreesRegressor, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    train = imputer.fit_transform(x_train)
    validation = imputer.transform(x_validation)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=None,
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(train, y_train)
    return imputer, model, train, validation


def _module_importance(
    model: ExtraTreesRegressor,
    validation_processed: np.ndarray,
    y_validation: np.ndarray,
    baseline_prediction: np.ndarray,
    modules: list[list[str]],
    *,
    feature_position: dict[str, int],
    indicator_map: dict[int, int],
    target_codes: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    baseline_mae = np.mean(np.abs(y_validation - baseline_prediction), axis=0)
    rows = []
    for permutation_repeat in range(int(repeats)):
        rng = np.random.default_rng(seed + permutation_repeat)
        for module_position, members in enumerate(modules, start=1):
            positions = [feature_position[value] for value in members]
            columns = _processed_columns(positions, indicator_map)
            changed, _ = joint_permutation(validation_processed, columns, rng)
            prediction = np.asarray(model.predict(changed), dtype=float)
            delta = np.mean(np.abs(y_validation - prediction), axis=0) - baseline_mae
            rows.append(
                pd.DataFrame(
                    {
                        "permutation_repeat": permutation_repeat,
                        "module_id": f"M{module_position:03d}",
                        "module_size": len(members),
                        "module_members": "|".join(members),
                        "target_code": target_codes,
                        "delta_mae": delta,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _module_gate(module_rows: pd.DataFrame, gate: dict) -> pd.DataFrame:
    summary = (
        module_rows.groupby(
            ["module_id", "module_size", "module_members", "target_code"],
            as_index=False,
            sort=False,
        )
        .agg(
            median_delta_mae=("delta_mae", "median"),
            positive_fraction=("delta_mae", lambda value: float((value > 0).mean())),
        )
    )
    summary["selected_module"] = (
        summary["median_delta_mae"].gt(float(gate["median_delta_mae_min"]))
        & summary["positive_fraction"].ge(float(gate["positive_fraction_min"]))
    )
    return summary


def _field_importance(
    model: ExtraTreesRegressor,
    validation_processed: np.ndarray,
    y_validation: np.ndarray,
    baseline_prediction: np.ndarray,
    x_train_strict: np.ndarray,
    x_validation_strict: np.ndarray,
    train_style: np.ndarray,
    validation_style: np.ndarray,
    modules: list[list[str]],
    gate_summary: pd.DataFrame,
    *,
    feature_codes: list[str],
    feature_position: dict[str, int],
    target_codes: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    baseline_mae = np.mean(np.abs(y_validation - baseline_prediction), axis=0)
    strict_position = {value: index for index, value in enumerate(feature_codes)}
    target_position = {value: index for index, value in enumerate(target_codes)}
    evaluated_rows = []
    selected_by_module = {
        module_id: group.loc[group["selected_module"], "target_code"].astype(str).tolist()
        for module_id, group in gate_summary.groupby("module_id", sort=False)
    }
    for module_position, members in enumerate(modules, start=1):
        module_id = f"M{module_position:03d}"
        selected_targets = selected_by_module.get(module_id, [])
        if not selected_targets:
            continue
        member_positions = [strict_position[value] for value in members]
        for feature in members:
            feature_index = strict_position[feature]
            peer_positions = [value for value in member_positions if value != feature_index]
            if peer_positions:
                conditional = fit_conditional_feature_model(
                    x_train_strict[:, peer_positions],
                    x_train_strict[:, feature_index],
                    train_style,
                )
                conditional_mean = conditional.predict(
                    x_validation_strict[:, peer_positions], validation_style
                )
                residual = x_validation_strict[:, feature_index] - conditional_mean
            else:
                conditional_mean = np.zeros(len(x_validation_strict), dtype=float)
                residual = x_validation_strict[:, feature_index].copy()
            for permutation_repeat in range(int(repeats)):
                changed_feature = residual_permutation(
                    conditional_mean,
                    residual,
                    np.random.default_rng(
                        seed + module_position * 100_000 + feature_index * 100 + permutation_repeat
                    ),
                )
                changed = validation_processed.copy()
                changed[:, feature_position[feature]] = changed_feature
                prediction = np.asarray(model.predict(changed), dtype=float)
                delta = np.mean(np.abs(y_validation - prediction), axis=0) - baseline_mae
                for target in selected_targets:
                    evaluated_rows.append(
                        {
                            "module_id": module_id,
                            "target_code": target,
                            "feature_code": feature,
                            "permutation_repeat": permutation_repeat,
                            "delta_mae": float(delta[target_position[target]]),
                            "evaluated": True,
                        }
                    )
    raw = pd.DataFrame(evaluated_rows)
    if raw.empty:
        scored = pd.DataFrame(
            columns=["target_code", "feature_code", "delta_mae", "positive_fraction"]
        )
    else:
        scored = (
            raw.groupby(["target_code", "feature_code"], as_index=False, sort=False)
            .agg(
                delta_mae=("delta_mae", "median"),
                positive_fraction=("delta_mae", lambda value: float((value > 0).mean())),
            )
        )
    complete = pd.MultiIndex.from_product(
        [target_codes, feature_codes], names=["target_code", "feature_code"]
    ).to_frame(index=False)
    complete = complete.merge(
        scored, on=["target_code", "feature_code"], how="left", validate="one_to_one"
    )
    complete["evaluated"] = complete["delta_mae"].notna()
    complete["delta_mae"] = complete["delta_mae"].fillna(0.0)
    complete["positive_fraction"] = complete["positive_fraction"].fillna(0.0)
    complete = complete.sort_values(
        ["target_code", "evaluated", "positive_fraction", "delta_mae", "feature_code"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    complete["fold_rank"] = complete.groupby("target_code").cumcount() + 1
    return complete


def _sufficiency_deletion(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    baseline_prediction: np.ndarray,
    top3: dict[str, list[str]],
    *,
    target_codes: list[str],
    max_features: float,
    min_samples_leaf: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for target_position, target in enumerate(target_codes):
        selected = top3[target]
        remaining = [value for value in x_train.columns.astype(str) if value not in selected]
        predictions = {}
        for label, columns in (("top3_only", selected), ("delete_top3", remaining)):
            imputer = SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )
            train = imputer.fit_transform(x_train[columns])
            validation = imputer.transform(x_validation[columns])
            model = ExtraTreesRegressor(
                n_estimators=N_ESTIMATORS,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                max_depth=None,
                random_state=seed + target_position * 10 + (0 if label == "top3_only" else 1),
                n_jobs=1,
            )
            model.fit(train, y_train[:, target_position])
            predictions[label] = np.asarray(model.predict(validation), dtype=float)
        truth = y_validation[:, target_position]
        full = baseline_prediction[:, target_position]
        rows.append(
            {
                "target_code": target,
                "top3_features": "|".join(selected),
                "top3_source": "E114_outer_training_partial_effect",
                "full_r2": float(r2_score(truth, full)),
                "full_mae": float(mean_absolute_error(truth, full)),
                "top3_r2": float(r2_score(truth, predictions["top3_only"])),
                "top3_mae": float(mean_absolute_error(truth, predictions["top3_only"])),
                "delete_top3_r2": float(r2_score(truth, predictions["delete_top3"])),
                "delete_top3_mae": float(mean_absolute_error(truth, predictions["delete_top3"])),
            }
        )
    return pd.DataFrame(rows)


def _anchor_recovery(rankings: pd.DataFrame) -> pd.DataFrame:
    targets = pd.read_csv(RESULT_DIR / "q2_target_dictionary.csv")
    anchors = targets.loc[targets["direct_anchor"].notna(), [
        "target_code", "target_name_en", "direct_anchor"
    ]]
    return anchors.merge(
        rankings,
        left_on=["target_code", "direct_anchor"],
        right_on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
    )


def run_e115(*, repeat_ids: list[int], output_dir: Path = RESULT_DIR) -> dict:
    started = time.perf_counter()
    repeat_ids = sorted(set(int(value) for value in repeat_ids))
    if not repeat_ids or repeat_ids[0] < 0 or repeat_ids[-1] > 9:
        raise ValueError("repeat_ids must be a non-empty subset of 0..9")
    config = _load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if repeat_ids == list(range(10)) else "_pilot_r" + "-".join(map(str, repeat_ids))
    x, y, style, _ = load_q2()
    feature_codes = _strict_codes(x)
    target_codes = y.columns.astype(str).tolist()
    target_values = y.to_numpy(float)
    feature_position = {value: x.columns.get_loc(value) for value in feature_codes}
    all_folds = load_frozen_outer_folds(len(x), repeats=max(repeat_ids) + 1)
    folds = [value for value in all_folds if value.repeat in repeat_ids]
    e114_effects = pd.read_csv(RESULT_DIR / "q2_fold_partial_effects_E114.csv.gz")
    checkpoint_dir = output_dir / f"q2_checkpoint_E115{suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    module_frames = []
    field_frames = []
    sufficiency_frames = []
    module_sources = set()

    for number, outer in enumerate(folds, start=1):
        prefix = checkpoint_dir / f"repeat_{outer.repeat}_fold_{outer.fold}"
        module_checkpoint = Path(str(prefix) + "_module.csv.gz")
        field_checkpoint = Path(str(prefix) + "_field.csv.gz")
        sufficiency_checkpoint = Path(str(prefix) + "_sufficiency.csv")
        if all(path.exists() for path in (module_checkpoint, field_checkpoint, sufficiency_checkpoint)):
            module_rows = pd.read_csv(module_checkpoint)
            field_rows = pd.read_csv(field_checkpoint)
            sufficiency = pd.read_csv(sufficiency_checkpoint)
            module_sources.update(module_rows["module_source"].astype(str).unique())
            print(f"[{number:02d}/{len(folds)}] {outer.split_id} resumed", flush=True)
        else:
            x_train = x.iloc[outer.train_index]
            x_validation = x.iloc[outer.validation_index]
            y_train = target_values[outer.train_index]
            y_validation = target_values[outer.validation_index]
            max_features, min_samples_leaf = _fold_parameters(outer.repeat, outer.fold)
            seed = int(config["random_state"]) + outer.repeat * 100 + outer.fold
            imputer, model, _, validation_processed = _fit_backbone(
                x_train,
                y_train,
                x_validation,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                seed=20261023 + outer.repeat * 10 + outer.fold,
            )
            baseline_prediction = np.asarray(model.predict(validation_processed), dtype=float)
            indicator_map = _indicator_map(imputer, x.shape[1])
            modules, module_source = _modules_for_fold(
                x_train,
                feature_codes,
                repeat=outer.repeat,
                fold=outer.fold,
            )
            module_sources.add(module_source)
            module_raw = _module_importance(
                model,
                validation_processed,
                y_validation,
                baseline_prediction,
                modules,
                feature_position=feature_position,
                indicator_map=indicator_map,
                target_codes=target_codes,
                repeats=int(config["module_repeats_per_fold"]),
                seed=seed * 10_000,
            )
            gate_summary = _module_gate(module_raw, config["module_gate"])
            gate_summary.insert(0, "module_source", module_source)
            gate_summary.insert(0, "fold", outer.fold)
            gate_summary.insert(0, "repeat", outer.repeat)
            gate_summary.insert(0, "split_id", outer.split_id)
            module_rows = gate_summary

            strict_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            strict_train = strict_imputer.fit_transform(x_train[feature_codes])
            strict_validation = strict_imputer.transform(x_validation[feature_codes])
            field_rows = _field_importance(
                model,
                validation_processed,
                y_validation,
                baseline_prediction,
                strict_train,
                strict_validation,
                style.iloc[outer.train_index].astype(str).to_numpy(),
                style.iloc[outer.validation_index].astype(str).to_numpy(),
                modules,
                gate_summary,
                feature_codes=feature_codes,
                feature_position=feature_position,
                target_codes=target_codes,
                repeats=int(config["field_repeats_per_fold"]),
                seed=seed * 100_000,
            )
            field_rows.insert(0, "fold", outer.fold)
            field_rows.insert(0, "repeat", outer.repeat)
            field_rows.insert(0, "split_id", outer.split_id)
            top3 = select_fold_top3(
                e114_effects, repeat=outer.repeat, fold=outer.fold
            )
            if set(top3) != set(target_codes):
                raise RuntimeError("E114 fold Top-3 does not cover every target")
            sufficiency = _sufficiency_deletion(
                x_train,
                y_train,
                x_validation,
                y_validation,
                baseline_prediction,
                top3,
                target_codes=target_codes,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                seed=seed * 1000,
            )
            sufficiency.insert(0, "fold", outer.fold)
            sufficiency.insert(0, "repeat", outer.repeat)
            sufficiency.insert(0, "split_id", outer.split_id)
            _atomic_to_csv(module_rows, module_checkpoint, gzip=True)
            _atomic_to_csv(field_rows, field_checkpoint, gzip=True)
            _atomic_to_csv(sufficiency, sufficiency_checkpoint)
            print(f"[{number:02d}/{len(folds)}] {outer.split_id} complete", flush=True)
        module_frames.append(module_rows)
        field_frames.append(field_rows)
        sufficiency_frames.append(sufficiency)

    modules = pd.concat(module_frames, ignore_index=True)
    fields = pd.concat(field_frames, ignore_index=True)
    sufficiency = pd.concat(sufficiency_frames, ignore_index=True)
    rankings = aggregate_hierarchical_importance(fields)
    anchors = _anchor_recovery(rankings)
    paths = {
        "modules": output_dir / f"q2_module_importance_folds_E115{suffix}.csv.gz",
        "fields": output_dir / f"q2_conditional_importance_folds_E115{suffix}.csv.gz",
        "rankings": output_dir / f"q2_hierarchical_rankings_E115{suffix}.csv",
        "sufficiency": output_dir / f"q2_top3_sufficiency_deletion_E115{suffix}.csv",
        "anchors": output_dir / f"q2_anchor_recovery_E115{suffix}.csv",
        "manifest": output_dir / f"q2_run_manifest_E115{suffix}.json",
    }
    modules.to_csv(paths["modules"], index=False, compression="gzip")
    fields.to_csv(paths["fields"], index=False, compression="gzip")
    rankings.to_csv(paths["rankings"], index=False)
    sufficiency.to_csv(paths["sufficiency"], index=False)
    anchors.to_csv(paths["anchors"], index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if repeat_ids == list(range(10)) else "pilot_complete",
        "repeat_ids": repeat_ids,
        "n_outer_folds": len(folds),
        "module_sources": sorted(module_sources),
        "module_repeats_per_fold": int(config["module_repeats_per_fold"]),
        "field_repeats_per_fold": int(config["field_repeats_per_fold"]),
        "top3_source_for_sufficiency": "E114_outer_training_partial_effect",
        "random_state": int(config["random_state"]),
        "gpu_used": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "outputs": [str(path) for key, path in paths.items() if key != "manifest"],
        "checkpoint_directory": str(checkpoint_dir),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    print(json.dumps(run_e115(repeat_ids=args.repeats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
