"""Shrinkage partial-effect estimators and deterministic Q2 rankings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parents[3]
TARGET_DICTIONARY_PATH = ROOT / "03_结果" / "Q2_关键化合物" / "q2_target_dictionary.csv"


def load_target_codes(path: Path = TARGET_DICTIONARY_PATH) -> list[str]:
    """Load the frozen 50 Q2 sensory target codes in official column order."""
    table = pd.read_csv(path)
    if "target_code" not in table or table["target_code"].duplicated().any():
        raise ValueError("target dictionary must contain unique target_code values")
    codes = table["target_code"].astype(str).tolist()
    if len(codes) != 50:
        raise ValueError(f"expected 50 target codes, got {len(codes)}")
    return codes


def load_target_groups(config_path: Path) -> list[dict]:
    """Load non-overlapping registered multi-task groups from E116 YAML."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    groups = config.get("target_groups", [])
    if not isinstance(groups, list) or not groups:
        raise ValueError("E116 target_groups must be a non-empty list")
    known = set(load_target_codes())
    names = set()
    flattened = []
    normalized = []
    for group in groups:
        name = str(group.get("name", "")).strip()
        targets = [str(value) for value in group.get("targets", [])]
        if not name or name in names or len(targets) < 2 or len(set(targets)) != len(targets):
            raise ValueError("each E116 target group needs a unique name and targets")
        unknown = set(targets) - known
        if unknown:
            raise ValueError(f"E116 target group {name} has unknown targets {sorted(unknown)}")
        names.add(name)
        flattened.extend(targets)
        normalized.append({"name": name, "targets": targets})
    if len(flattened) != len(set(flattened)):
        raise ValueError("E116 fitted target groups must not overlap")
    return normalized


def _finite_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    if features.ndim != 2 or features.shape[0] < 3 or features.shape[1] < 1:
        raise ValueError("x must be a non-empty two-dimensional matrix")
    if target.shape != (len(features),):
        raise ValueError("y must be one-dimensional and aligned with x")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("x and y must be finite")
    return features, target


def ledoit_wolf_target_partial(x, y) -> np.ndarray:
    """Estimate feature-target partial correlations from shrinkage precision."""
    features, target = _finite_xy(x, y)
    matrix = np.column_stack([features, target])
    precision = np.asarray(LedoitWolf().fit(matrix).precision_, dtype=float)
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0) or not np.isfinite(precision).all():
        raise RuntimeError("Ledoit-Wolf returned an invalid precision matrix")
    scale = np.sqrt(diagonal)
    effect = -precision[:-1, -1] / (scale[:-1] * scale[-1])
    return np.clip(effect, -1.0, 1.0)


def rank_signed_effects(
    effect: np.ndarray, feature_codes: Sequence[str]
) -> pd.DataFrame:
    """Rank absolute effects with a deterministic feature-code tie break."""
    values = np.asarray(effect, dtype=float)
    codes = np.asarray([str(value) for value in feature_codes], dtype=object)
    if values.shape != (len(codes),) or not len(codes):
        raise ValueError("effect length disagrees with feature codes")
    if len(set(codes.tolist())) != len(codes) or not np.isfinite(values).all():
        raise ValueError("feature codes must be unique and effects finite")
    order = np.lexsort((codes, -np.abs(values)))
    ranked_values = values[order]
    return pd.DataFrame(
        {
            "feature_code": codes[order],
            "effect": ranked_values,
            "abs_effect": np.abs(ranked_values),
            "direction": np.select(
                [ranked_values > 0, ranked_values < 0],
                ["positive", "negative"],
                default="zero",
            ),
            "fold_rank": np.arange(1, len(values) + 1, dtype=int),
        }
    )


def aggregate_partial_effect_rankings(fold_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30 fold rankings without mixing target-specific score scales."""
    required = {
        "repeat",
        "fold",
        "target_code",
        "feature_code",
        "effect",
        "abs_effect",
        "fold_rank",
    }
    missing = required - set(fold_rows.columns)
    if missing:
        raise ValueError(f"fold partial effects miss {sorted(missing)}")
    frame = fold_rows.copy()
    if frame.empty:
        raise ValueError("fold partial effects are empty")

    def sign_consistency(values: pd.Series) -> float:
        array = values.to_numpy(float)
        nonzero = array[np.abs(array) > 1e-12]
        if not len(nonzero):
            return 0.0
        return float(max(np.sum(nonzero > 0), np.sum(nonzero < 0)) / len(nonzero))

    result = (
        frame.groupby(["target_code", "feature_code"], sort=False)
        .agg(
            median_fold_rank=("fold_rank", "median"),
            rank_q25=("fold_rank", lambda value: float(value.quantile(0.25))),
            rank_q75=("fold_rank", lambda value: float(value.quantile(0.75))),
            top3_frequency=("fold_rank", lambda value: float((value <= 3).mean())),
            top10_frequency=("fold_rank", lambda value: float((value <= 10).mean())),
            median_effect=("effect", "median"),
            median_abs_effect=("abs_effect", "median"),
            sign_consistency=("effect", sign_consistency),
            n_outer_folds=("fold_rank", "size"),
        )
        .reset_index()
    )
    result["rank_iqr"] = result["rank_q75"] - result["rank_q25"]
    result["direction"] = np.select(
        [result["median_effect"] > 0, result["median_effect"] < 0],
        ["positive", "negative"],
        default="zero",
    )
    result = result.sort_values(
        [
            "target_code",
            "top3_frequency",
            "top10_frequency",
            "median_fold_rank",
            "median_abs_effect",
            "feature_code",
        ],
        ascending=[True, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["partial_effect_rank"] = result.groupby("target_code").cumcount() + 1
    return result[
        [
            "target_code",
            "feature_code",
            "partial_effect_rank",
            "median_fold_rank",
            "rank_q25",
            "rank_q75",
            "rank_iqr",
            "top3_frequency",
            "top10_frequency",
            "median_effect",
            "median_abs_effect",
            "direction",
            "sign_consistency",
            "n_outer_folds",
        ]
    ]
