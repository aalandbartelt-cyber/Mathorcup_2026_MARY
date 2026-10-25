"""Fold-local sparse partial-correlation graph primitives for Q2.

Every fitted object in this module is trained on the matrix passed to ``fit``.
Callers are responsible for supplying outer-training rows only; validation rows
are accepted solely by ``transform`` or held-out likelihood evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import QuantileTransformer


def _finite_matrix(values, *, name: str, minimum_rows: int = 1) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < minimum_rows or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


class FoldGaussianizer:
    """Training-only empirical-CDF transform with Gaussian marginal output."""

    def __init__(self, *, random_state: int = 20261025) -> None:
        self.random_state = int(random_state)

    def fit(self, x) -> "FoldGaussianizer":
        values = _finite_matrix(x, name="x", minimum_rows=2)
        self.n_features_in_ = int(values.shape[1])
        self.model_ = QuantileTransformer(
            n_quantiles=min(100, len(values)),
            output_distribution="normal",
            subsample=None,
            random_state=self.random_state,
        ).fit(values)
        self.quantiles_ = self.model_.quantiles_.copy()
        return self

    def transform(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("FoldGaussianizer must be fitted before transform")
        values = _finite_matrix(x, name="x")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("x feature count disagrees with fitted transformer")
        return self.model_.transform(values)

    def fit_transform(self, x) -> np.ndarray:
        return self.fit(x).transform(x)


def precision_to_partial_correlation(precision) -> np.ndarray:
    """Convert a precision matrix into a signed partial-correlation matrix."""
    matrix = np.asarray(precision, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("precision must be a non-empty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("precision must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("precision must be symmetric")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError("precision diagonal must be positive")
    scale = np.sqrt(diagonal)
    partial = -matrix / np.outer(scale, scale)
    np.fill_diagonal(partial, 1.0)
    partial = (partial + partial.T) / 2.0
    return np.clip(partial, -1.0, 1.0)


@dataclass(frozen=True)
class SparseGraphFit:
    """Numerical outputs from one training-fold Graphical Lasso fit."""

    alpha: float
    location: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    partial_correlation: np.ndarray
    n_iter: int


def fit_graphical_lasso(
    x,
    *,
    alpha: float,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> SparseGraphFit:
    """Fit Graphical Lasso to a finite training-fold Gaussianized matrix."""
    values = _finite_matrix(x, name="x", minimum_rows=3)
    if float(alpha) <= 0:
        raise ValueError("alpha must be positive")
    model = GraphicalLasso(
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=float(tol),
        assume_centered=False,
    ).fit(values)
    return SparseGraphFit(
        alpha=float(alpha),
        location=np.asarray(model.location_, dtype=float).copy(),
        covariance=np.asarray(model.covariance_, dtype=float).copy(),
        precision=np.asarray(model.precision_, dtype=float).copy(),
        partial_correlation=precision_to_partial_correlation(model.precision_),
        n_iter=int(model.n_iter_),
    )


def heldout_gaussian_log_likelihood(x, *, mean, precision) -> float:
    """Return mean multivariate Gaussian log-likelihood on held-out rows."""
    values = _finite_matrix(x, name="x")
    center = np.asarray(mean, dtype=float)
    matrix = np.asarray(precision, dtype=float)
    if center.shape != (values.shape[1],):
        raise ValueError("mean length disagrees with x")
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("precision shape disagrees with x")
    if not np.isfinite(center).all() or not np.isfinite(matrix).all():
        raise ValueError("mean and precision must be finite")
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("precision must be positive definite")
    residual = values - center
    quadratic = np.einsum("ij,jk,ik->i", residual, matrix, residual)
    normalizer = values.shape[1] * np.log(2.0 * np.pi)
    return float(np.mean(0.5 * (log_determinant - normalizer - quadratic)))


def _validate_partial_and_codes(
    partial_correlation, feature_codes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    matrix = np.asarray(partial_correlation, dtype=float)
    codes = [str(value) for value in feature_codes]
    if matrix.shape != (len(codes), len(codes)) or not codes:
        raise ValueError("partial-correlation shape disagrees with feature codes")
    if len(set(codes)) != len(codes):
        raise ValueError("feature codes must be unique")
    if not np.isfinite(matrix).all():
        raise ValueError("partial-correlation matrix must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("partial-correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ValueError("partial-correlation diagonal must equal one")
    return matrix, codes


def edge_table(
    partial_correlation,
    feature_codes: Sequence[str],
    *,
    edge_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Return deterministic non-zero upper-triangle partial edges."""
    matrix, codes = _validate_partial_and_codes(partial_correlation, feature_codes)
    if float(edge_threshold) < 0:
        raise ValueError("edge_threshold must be non-negative")
    upper_i, upper_j = np.triu_indices(len(codes), k=1)
    values = matrix[upper_i, upper_j]
    keep = np.abs(values) >= float(edge_threshold)
    selected = values[keep]
    return pd.DataFrame(
        {
            "source": np.asarray(codes, dtype=object)[upper_i[keep]],
            "target": np.asarray(codes, dtype=object)[upper_j[keep]],
            "partial_correlation": selected,
            "abs_partial_correlation": np.abs(selected),
            "sign": np.sign(selected).astype(int),
        }
    )


def louvain_membership(
    partial_correlation,
    feature_codes: Sequence[str],
    *,
    edge_threshold: float = 1e-6,
    seed: int = 20261025,
) -> np.ndarray:
    """Compute deterministic Louvain labels; every isolate is a singleton."""
    matrix, codes = _validate_partial_and_codes(partial_correlation, feature_codes)
    edges = edge_table(matrix, codes, edge_threshold=edge_threshold)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(codes)))
    positions = {code: index for index, code in enumerate(codes)}
    for row in edges.itertuples(index=False):
        graph.add_edge(
            positions[str(row.source)],
            positions[str(row.target)],
            weight=float(row.abs_partial_correlation),
        )
    connected_nodes = [node for node, degree in graph.degree if degree > 0]
    communities: list[set[int]] = []
    if connected_nodes:
        subgraph = graph.subgraph(connected_nodes).copy()
        communities.extend(
            set(group)
            for group in nx.community.louvain_communities(
                subgraph, weight="weight", seed=int(seed)
            )
        )
    communities.extend({node} for node, degree in graph.degree if degree == 0)
    communities.sort(key=lambda group: (min(group), len(group), sorted(group)))
    labels = np.empty(len(codes), dtype=int)
    for label, members in enumerate(communities):
        labels[np.asarray(sorted(members), dtype=int)] = label
    return labels


def coclustering_matrix(label_vectors: Sequence[Sequence[int]]) -> np.ndarray:
    """Average pairwise module co-membership, invariant to label names."""
    labels = [np.asarray(value) for value in label_vectors]
    if not labels:
        raise ValueError("at least one module assignment is required")
    n_features = len(labels[0])
    if n_features < 1 or any(vector.ndim != 1 for vector in labels):
        raise ValueError("module assignments must be non-empty vectors")
    result = np.zeros((n_features, n_features), dtype=float)
    for vector in labels:
        if len(vector) != n_features:
            raise ValueError("module assignments must have equal length")
        result += vector[:, None] == vector[None, :]
    return result / float(len(labels))


def stable_edge_table(edge_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate edge frequency, magnitude and sign over a fixed fold set."""
    frames = list(edge_frames)
    if not frames:
        raise ValueError("at least one fold edge table is required")
    required = {"source", "target", "partial_correlation"}
    rows = []
    for fold, frame in enumerate(frames):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"fold {fold} edge table misses {sorted(missing)}")
        seen: set[tuple[str, str]] = set()
        for row in frame.itertuples(index=False):
            source, target = sorted((str(row.source), str(row.target)))
            if source == target or (source, target) in seen:
                raise ValueError(f"fold {fold} contains duplicated/self edges")
            value = float(row.partial_correlation)
            if not np.isfinite(value) or value == 0.0:
                raise ValueError("selected partial edges must be finite and non-zero")
            seen.add((source, target))
            rows.append(
                {
                    "fold": fold,
                    "source": source,
                    "target": target,
                    "partial_correlation": value,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "n_selected",
                "selection_frequency",
                "mean_partial_correlation",
                "median_abs_partial_correlation",
                "sign_consistency",
            ]
        )
    summary_rows = []
    for (source, target), group in long.groupby(["source", "target"], sort=True):
        values = group["partial_correlation"].to_numpy(float)
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        summary_rows.append(
            {
                "source": source,
                "target": target,
                "n_selected": int(len(values)),
                "selection_frequency": float(len(values) / len(frames)),
                "mean_partial_correlation": float(np.mean(values)),
                "median_abs_partial_correlation": float(np.median(np.abs(values))),
                "sign_consistency": float(max(positive, negative) / len(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def graph_structure_summary(
    partial_correlation,
    feature_codes: Sequence[str],
    *,
    edge_threshold: float = 1e-6,
    membership: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Summarise edge density, isolated nodes, degree and module count."""
    matrix, codes = _validate_partial_and_codes(partial_correlation, feature_codes)
    edges = edge_table(matrix, codes, edge_threshold=edge_threshold)
    positions = {code: index for index, code in enumerate(codes)}
    degree = np.zeros(len(codes), dtype=int)
    for row in edges.itertuples(index=False):
        degree[positions[str(row.source)]] += 1
        degree[positions[str(row.target)]] += 1
    labels = (
        louvain_membership(matrix, codes, edge_threshold=edge_threshold)
        if membership is None
        else np.asarray(membership)
    )
    if labels.shape != (len(codes),):
        raise ValueError("membership length disagrees with feature codes")
    possible = len(codes) * (len(codes) - 1) / 2
    return {
        "n_nodes": int(len(codes)),
        "n_edges": int(len(edges)),
        "edge_density": float(len(edges) / possible) if possible else 0.0,
        "n_isolates": int(np.sum(degree == 0)),
        "mean_degree": float(np.mean(degree)),
        "max_degree": int(np.max(degree)),
        "n_modules": int(len(np.unique(labels))),
    }
