"""Q2 data, candidate-scope, and signed Spearman graph contracts.

The functions in this module do not fit a supervised model. They validate the
official attachments, freeze which fields qualify as strict compounds, and
provide fold-local graph construction for later experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ATTACHMENTS = (
    ROOT
    / "data"
    / "2026初赛"
    / "2026年MathorCup大数据挑战赛-赛道A初赛"
    / "附件"
)
DATA_DIR = ROOT / "01_数据"
RESULT_DIR = ROOT / "03_结果" / "Q2_关键化合物"
Q1_RESULT_DIR = ROOT / "03_结果" / "Q1_感官评分"

META_COLUMNS = ["beer", "beer_id", "tasting_category_fine"]

# Fields without a single, chemically defined compound identity. Carbon
# dioxide, ethanol, acids, and other named molecules remain strict candidates.
STRICT_EXCLUSIONS = {
    "color.EBC.": ("physicochemical", "色度是理化指标，不是单体化合物"),
    "betaglucan.mg.L.": ("composition", "β-葡聚糖表示聚合物组成指标"),
    "iron": ("element", "铁是元素而非化合物"),
    "pH": ("physicochemical", "pH 是酸碱度指标，不是单体化合物"),
    "protein.g.L.": ("composition", "蛋白质为混合组成指标"),
    "bitterness": ("physicochemical", "IBU 是综合苦度指标"),
    "GFS": ("composition", "总糖为混合组成指标"),
    "unfermentables": ("composition", "不可发酵物为混合组成指标"),
    "kcalperc": ("derived_aggregate", "热量为派生指标"),
    "esters_sum": ("derived_aggregate", "总酯为派生聚合量"),
    "aroma_hops_sum": ("derived_aggregate", "总酒花香气为派生聚合量"),
    "acids_sum": ("derived_aggregate", "总酸为派生聚合量"),
    "sulfur_sum": ("derived_aggregate", "总硫为派生聚合量"),
}


def load_q2() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and align Q2 predictors, sensory targets, styles, and beer keys."""
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    sensory = pd.read_csv(ATTACHMENTS / "4.csv")
    for name, frame in (("1.csv", chemistry), ("4.csv", sensory)):
        missing = set(META_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing metadata columns: {sorted(missing)}")
        if frame["beer"].isna().any() or not frame["beer"].is_unique:
            raise ValueError(f"{name}: beer must be complete and unique")
    if set(chemistry["beer"]) != set(sensory["beer"]):
        raise ValueError("1.csv and 4.csv do not contain the same beer keys")
    sensory = sensory.set_index("beer").loc[chemistry["beer"]].reset_index()
    if not chemistry["tasting_category_fine"].equals(
        sensory["tasting_category_fine"]
    ):
        raise ValueError("style labels disagree between 1.csv and 4.csv")

    x = chemistry.drop(columns=META_COLUMNS)
    y = sensory.drop(columns=META_COLUMNS)
    style = chemistry["tasting_category_fine"].copy()
    beer = chemistry["beer"].astype(int).copy()
    if x.shape != (250, 231) or y.shape != (250, 50):
        raise ValueError(f"unexpected Q2 shapes X={x.shape}, Y={y.shape}")
    if y.isna().any().any():
        raise ValueError("sensory target matrix contains missing values")
    return x, y, style, beer


def build_candidate_dictionary(feature_columns: Iterable[str]) -> pd.DataFrame:
    """Create the frozen strict-compound and all-factor candidate dictionary."""
    columns = list(feature_columns)
    mapping = pd.read_csv(DATA_DIR / "compound_name_map.csv")
    if mapping["feature_code"].tolist() != columns:
        raise ValueError("compound_name_map order disagrees with Q2 predictors")
    thresholds = pd.read_excel(
        ATTACHMENTS / "8.xlsx", sheet_name="Sensory thresholds"
    )
    annotation_columns = [
        "field_name_R",
        "calibrated",
        "origin_paper",
        "aging",
        "off_flavors",
        "threshold_mean",
        "threshold_median",
    ]
    annotations = thresholds[annotation_columns].copy()
    if annotations["field_name_R"].duplicated().any():
        raise ValueError("attachment 8 field_name_R must be unique")
    candidates = mapping.merge(
        annotations,
        left_on="feature_code",
        right_on="field_name_R",
        how="left",
        validate="one_to_one",
        suffixes=("", "_annotation"),
    )
    if candidates["field_name_R_annotation"].isna().any():
        raise ValueError("attachment 8 annotations do not cover all Q2 predictors")
    candidates = candidates.drop(columns="field_name_R_annotation")

    candidate_types = []
    strict = []
    reasons = []
    for code in candidates["feature_code"]:
        if code in STRICT_EXCLUSIONS:
            factor_type, reason = STRICT_EXCLUSIONS[code]
            candidate_types.append(factor_type)
            strict.append(False)
            reasons.append(reason)
        else:
            candidate_types.append("single_compound")
            strict.append(True)
            reasons.append("")
    candidates["factor_type"] = candidate_types
    candidates["eligible_strict"] = np.asarray(strict, dtype=bool)
    candidates["eligible_all"] = True
    candidates["exclusion_reason"] = reasons
    candidates["has_calibrated_values"] = candidates["calibrated"].notna()
    candidates["has_sensory_threshold"] = candidates[
        ["threshold_mean", "threshold_median"]
    ].notna().any(axis=1)
    candidates["has_aging_annotation"] = candidates["aging"].notna()
    candidates["has_off_flavor_annotation"] = candidates["off_flavors"].notna()
    return candidates


def fold_spearman_graph(
    x: pd.DataFrame, training_indices: Sequence[int]
) -> pd.DataFrame:
    """Compute a pairwise-complete Spearman matrix using training rows only."""
    index = np.asarray(training_indices, dtype=int)
    if index.ndim != 1 or len(index) < 3:
        raise ValueError("training_indices must contain at least three row positions")
    if (index < 0).any() or (index >= len(x)).any():
        raise IndexError("training_indices outside X")
    if len(np.unique(index)) != len(index):
        raise ValueError("training_indices must be unique")
    graph = x.iloc[index].corr(method="spearman")
    if graph.isna().any().any():
        missing = graph.columns[graph.isna().any()].tolist()
        raise ValueError(f"undefined fold Spearman correlations: {missing}")
    return graph


def audit_official_graph(x: pd.DataFrame) -> dict[str, float | int | bool]:
    """Verify the actual correlation definition used by official attachment 2."""
    attachment = pd.read_csv(ATTACHMENTS / "2.csv")
    row_names = attachment.iloc[:, 0].astype(str).str.strip().tolist()
    column_names = pd.Index(attachment.columns[1:]).astype(str).str.strip().tolist()
    official = attachment.iloc[:, 1:].to_numpy(float)
    if attachment.shape != (231, 232):
        raise ValueError(f"unexpected attachment 2 shape: {attachment.shape}")
    if row_names != column_names:
        raise ValueError("attachment 2 row and column names disagree")
    if x.columns.tolist() != pd.read_csv(DATA_DIR / "compound_name_map.csv")[
        "feature_code"
    ].tolist():
        raise ValueError("Q2 feature order disagrees with frozen name map")
    spearman = x.corr(method="spearman").to_numpy()
    pearson = x.corr(method="pearson").to_numpy()
    return {
        "matrix_size": int(official.shape[0]),
        "symmetric": bool(np.allclose(official, official.T, equal_nan=True)),
        "diagonal_is_one": bool(np.allclose(np.diag(official), 1.0)),
        "spearman_max_abs_diff": float(np.nanmax(np.abs(official - spearman))),
        "pearson_max_abs_diff": float(np.nanmax(np.abs(official - pearson))),
        "pearson_mean_abs_diff": float(np.nanmean(np.abs(official - pearson))),
    }


def _validate_correlation_matrix(correlation: pd.DataFrame) -> None:
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation matrix must be square")
    if correlation.index.tolist() != correlation.columns.tolist():
        raise ValueError("correlation row and column labels must agree")
    values = correlation.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("correlation matrix must be finite")
    if not np.allclose(values, values.T, atol=1e-12):
        raise ValueError("correlation matrix must be symmetric")


def build_soft_edge_table(
    correlation: pd.DataFrame, betas: Sequence[int] = (1, 2, 4, 6)
) -> pd.DataFrame:
    """Return every undirected edge with unsigned and signed soft weights."""
    _validate_correlation_matrix(correlation)
    beta_values = tuple(int(beta) for beta in betas)
    if not beta_values or any(beta <= 0 for beta in beta_values):
        raise ValueError("betas must contain positive integers")
    values = correlation.to_numpy(float)
    names = correlation.columns.to_numpy()
    upper_i, upper_j = np.triu_indices(len(names), k=1)
    rho = values[upper_i, upper_j]
    edges = pd.DataFrame(
        {
            "source": names[upper_i],
            "target": names[upper_j],
            "rho": rho,
            "abs_rho": np.abs(rho),
            "sign": np.sign(rho).astype(int),
        }
    )
    for beta in beta_values:
        weight = np.abs(rho) ** beta
        edges[f"weight_beta_{beta}"] = weight
        edges[f"signed_weight_beta_{beta}"] = np.sign(rho) * weight
    return edges


def signed_normalized_laplacian(
    correlation: pd.DataFrame, beta: int
) -> np.ndarray:
    """Build a positive-semidefinite normalized signed graph Laplacian."""
    _validate_correlation_matrix(correlation)
    if int(beta) <= 0:
        raise ValueError("beta must be positive")
    rho = correlation.to_numpy(float).copy()
    np.fill_diagonal(rho, 0.0)
    adjacency = np.sign(rho) * np.abs(rho) ** int(beta)
    degree = np.sum(np.abs(adjacency), axis=1)
    inverse_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalized_adjacency = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    laplacian = np.diag(nonzero.astype(float)) - normalized_adjacency
    return (laplacian + laplacian.T) / 2.0


def graph_threshold_audit(
    correlation: pd.DataFrame,
    thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Describe hard-threshold sensitivity without selecting a main threshold."""
    _validate_correlation_matrix(correlation)
    values = correlation.to_numpy(float).copy()
    np.fill_diagonal(values, 0.0)
    rows = []
    n = len(values)
    possible_edges = n * (n - 1) // 2
    for threshold in thresholds:
        mask = np.triu(np.abs(values) >= float(threshold), k=1)
        i, j = np.where(mask)
        degree = np.bincount(np.r_[i, j], minlength=n)
        selected = values[i, j]
        # Connected components from the thresholded undirected graph.
        remaining = set(range(n))
        components = []
        neighbors = [set() for _ in range(n)]
        for left, right in zip(i, j):
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                new = neighbors[node] & remaining
                remaining.difference_update(new)
                stack.extend(new)
            components.append(size)
        rows.append(
            {
                "abs_rho_threshold": float(threshold),
                "n_edges": int(len(i)),
                "density": float(len(i) / possible_edges),
                "positive_edges": int((selected > 0).sum()),
                "negative_edges": int((selected < 0).sum()),
                "isolated_nodes": int((degree == 0).sum()),
                "n_components": int(len(components)),
                "largest_component": int(max(components)),
            }
        )
    return pd.DataFrame(rows)


def build_node_table(
    candidates: pd.DataFrame, correlation: pd.DataFrame, betas: Sequence[int]
) -> pd.DataFrame:
    """Attach soft graph degree statistics to the candidate dictionary."""
    _validate_correlation_matrix(correlation)
    if candidates["feature_code"].tolist() != correlation.columns.tolist():
        raise ValueError("candidate order disagrees with graph labels")
    nodes = candidates.copy()
    rho = correlation.to_numpy(float).copy()
    np.fill_diagonal(rho, 0.0)
    for beta in betas:
        nodes[f"soft_degree_beta_{int(beta)}"] = np.sum(
            np.abs(rho) ** int(beta), axis=1
        )
    nodes["max_abs_rho"] = np.max(np.abs(rho), axis=1)
    return nodes
