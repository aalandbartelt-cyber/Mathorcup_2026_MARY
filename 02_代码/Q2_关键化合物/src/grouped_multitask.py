"""Training-fold-only grouped ElasticNet for the registered E116 ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from target_partial_effects import aggregate_partial_effect_rankings


@dataclass
class GroupModel:
    name: str
    target_positions: list[int]
    preprocessor: Pipeline
    model: object
    target_mean: np.ndarray
    target_scale: np.ndarray


@dataclass
class GroupedMultitaskFit:
    models: list[GroupModel]
    target_codes: list[str]
    n_features: int
    coefficients: np.ndarray
    selection_rows: list[dict]

    def predict(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError("prediction features disagree with fitted model")
        if np.isinf(values).any():
            raise ValueError("prediction features must not contain infinity")
        result = np.empty((len(values), len(self.target_codes)), dtype=float)
        for group in self.models:
            prediction = np.asarray(group.model.predict(group.preprocessor.transform(values)))
            prediction = prediction.reshape(len(values), len(group.target_positions))
            result[:, group.target_positions] = (
                prediction * group.target_scale + group.target_mean
            )
        return result


def _fit_group(x, y, *, name, positions, alpha, l1_ratio, seed) -> GroupModel:
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transformed = preprocessor.fit_transform(x)
    mean = y.mean(axis=0)
    scale = y.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    standardized = (y - mean) / scale
    cls = MultiTaskElasticNet if y.shape[1] > 1 else ElasticNet
    model = cls(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=10000,
                tol=1e-5, random_state=int(seed), selection="cyclic")
    model.fit(transformed, standardized if y.shape[1] > 1 else standardized[:, 0])
    return GroupModel(name, positions, preprocessor, model, mean, scale)


def fit_grouped_multitask(
    x, y, *, target_codes, groups, alpha_grid, l1_ratio_grid,
    inner_splits=3, random_state=20261025,
) -> GroupedMultitaskFit:
    """Select parameters with fresh preprocessing in each inner training split.

    Coefficients rank original compounds in standardized X/Y units. Missing
    indicators remain nuisance predictors and do not become compound candidates.
    This interface deliberately accepts no external-validation targets.
    """
    features, targets = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    codes = [str(code) for code in target_codes]
    if (features.ndim != 2 or targets.ndim != 2 or not features.shape[1]
            or targets.shape != (len(features), len(codes)) or not codes
            or len(set(codes)) != len(codes)):
        raise ValueError("training features, targets and unique target codes must align")
    if np.isinf(features).any() or not np.isfinite(targets).all():
        raise ValueError("features may contain NaN only; targets must be finite")
    if not 2 <= int(inner_splits) <= len(features) // 2:
        raise ValueError("inner_splits requires at least two rows in each split")
    alphas, ratios = list(alpha_grid), list(l1_ratio_grid)
    if not alphas or not ratios or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("alpha_grid must contain positive finite values")
    if any(not np.isfinite(a) or not 0 < a <= 1 for a in ratios):
        raise ValueError("l1_ratio_grid must be in (0, 1]")
    units, used, names = [], set(), set()
    for group in groups:
        name = str(group["name"])
        members = [str(code) for code in group["targets"]]
        if (not name or name in names or len(members) < 2
                or len(set(members)) != len(members)
                or set(members) - set(codes) or set(members) & used):
            raise ValueError("target groups must be known, disjoint and uniquely named")
        units.append((name, [codes.index(code) for code in members]))
        names.add(name)
        used.update(members)
    units.extend((code, [i]) for i, code in enumerate(codes) if code not in used)
    splits = list(KFold(int(inner_splits), shuffle=True,
                        random_state=int(random_state)).split(features))
    models, selection = [], []
    coefficients = np.zeros((len(codes), features.shape[1]), dtype=float)
    for name, positions in units:
        group_y = targets[:, positions]
        candidates = []
        for alpha in alphas:
            for ratio in ratios:
                fold_loss = []
                for train, validation in splits:
                    fitted = _fit_group(
                        features[train], group_y[train], name=name, positions=positions,
                        alpha=alpha, l1_ratio=ratio, seed=random_state,
                    )
                    predicted = np.asarray(fitted.model.predict(
                        fitted.preprocessor.transform(features[validation])
                    )).reshape(len(validation), len(positions))
                    truth = (group_y[validation] - fitted.target_mean) / fitted.target_scale
                    fold_loss.append(float(np.mean(np.abs(predicted - truth))))
                candidates.append({
                    "alpha": float(alpha), "l1_ratio": float(ratio),
                    "inner_mae_standardized": float(np.mean(fold_loss)),
                    "inner_train_sizes": [len(tr) for tr, _ in splits],
                    "inner_validation_sizes": [len(va) for _, va in splits],
                })
        best = min(candidates, key=lambda row: (
            row["inner_mae_standardized"], -row["alpha"], -row["l1_ratio"]))
        fitted = _fit_group(features, group_y, name=name, positions=positions,
                            alpha=best["alpha"], l1_ratio=best["l1_ratio"],
                            seed=random_state)
        models.append(fitted)
        coef = np.asarray(fitted.model.coef_).reshape(len(positions), -1)
        coefficients[positions] = coef[:, :features.shape[1]]
        selection.append({
            "group_name": name, "target_codes": [codes[i] for i in positions],
            "model": type(fitted.model).__name__, "alpha": best["alpha"],
            "l1_ratio": best["l1_ratio"],
            "inner_mae_standardized": best["inner_mae_standardized"],
            "candidate_scores": candidates,
        })
    return GroupedMultitaskFit(models, codes, features.shape[1], coefficients, selection)


def evaluate_e116_gate(
    *, baseline_r2, candidate_r2, baseline_mae, candidate_mae,
    baseline_tail_r2, candidate_tail_r2, baseline_stability, candidate_stability,
    baseline_anchors, candidate_anchors, r2_drop_max=.003, mae_increase_max=.002,
) -> dict:
    """Keep E116 unpromoted until every pre-registered evidence gate is known."""
    mandatory = [baseline_r2, candidate_r2, baseline_mae, candidate_mae,
                 baseline_tail_r2, candidate_tail_r2]
    if not np.isfinite(mandatory).all():
        raise ValueError("E116 predictive metrics must be finite")
    pending = any(value is None for value in [
        baseline_stability, candidate_stability, baseline_anchors, candidate_anchors])
    if not pending and (not np.isfinite([baseline_stability, candidate_stability]).all()
                        or not 0 <= baseline_stability <= 1
                        or not 0 <= candidate_stability <= 1):
        raise ValueError("stability probabilities must be finite and in [0, 1]")
    flags = {
        "r2_within_tolerance": bool(candidate_r2 >= baseline_r2 - r2_drop_max),
        "mae_within_tolerance": bool(candidate_mae <= baseline_mae + mae_increase_max),
        "tail_r2_improved": bool(candidate_tail_r2 > baseline_tail_r2),
        "grouped_stability_improved": None if pending else bool(
            candidate_stability > baseline_stability),
        "four_direct_anchors_preserved": None if pending else bool(
            len(set(baseline_anchors)) >= 4
            and set(baseline_anchors).issubset(set(candidate_anchors))),
    }
    return {
        "status": "awaiting_stability_and_anchor_audit" if pending else "gate_evaluated",
        "promoted": bool(all(value is True for value in flags.values())),
        "flags": flags,
    }


def aggregate_grouped_rankings(fold_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep complete ranks but never count a zero sparse coefficient as selected."""
    result = aggregate_partial_effect_rankings(fold_rows)
    selected = fold_rows.assign(
        nonzero=fold_rows["abs_effect"].gt(1e-12),
        selected_top3=fold_rows["abs_effect"].gt(1e-12) & fold_rows["fold_rank"].le(3),
        selected_top10=fold_rows["abs_effect"].gt(1e-12) & fold_rows["fold_rank"].le(10),
    ).groupby(["target_code", "feature_code"], sort=False).agg(
        nonzero_frequency=("nonzero", "mean"),
        top3_frequency=("selected_top3", "mean"),
        top10_frequency=("selected_top10", "mean"),
    ).reset_index()
    result = result.drop(columns=["top3_frequency", "top10_frequency"]).merge(
        selected, on=["target_code", "feature_code"], validate="one_to_one")
    result = result.sort_values(
        ["target_code", "top3_frequency", "top10_frequency", "nonzero_frequency",
         "median_fold_rank", "median_abs_effect", "feature_code"],
        ascending=[True, False, False, False, True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    result["partial_effect_rank"] = result.groupby("target_code").cumcount() + 1
    return result
