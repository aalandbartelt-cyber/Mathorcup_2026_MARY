"""Shared low-rank Q3 generator with style residual shrinkage and floor states."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


METHODS = ("shared_lowrank", "style_residual_shrunk", "style_residual_floor")


def _psd_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, float)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    psd = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (psd + psd.T) / 2.0, eigenvectors * np.sqrt(eigenvalues)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _calibration_offset(logits: np.ndarray, target: float) -> float:
    target = float(np.clip(target, 1e-8, 1.0 - 1e-8))
    lower, upper = -40.0, 40.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _sigmoid(logits + midpoint).mean() < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


class SharedLowRankStyleGenerator:
    """Factor generator estimated only from one training fold.

    Style means, factor covariance and diagonal residual variance are shrunk
    separately.  The optional floor model predicts an observed floor state
    conditional on the generated latent vector and style; it does not infer a
    concentration below an unknown detection limit.
    """

    def fit(
        self,
        x,
        styles,
        *,
        rank: int,
        covariance_kappa: float = 30.0,
        mean_kappa: float = 15.0,
        floor_specs: dict[int, dict] | None = None,
        random_state: int = 20261025,
    ):
        values = np.asarray(x, float)
        labels = np.asarray(styles, str)
        if values.ndim != 2 or labels.shape != (len(values),) or len(values) < 4:
            raise ValueError("training matrix and one style label per row required")
        if not np.isfinite(values).all():
            raise ValueError("finite training matrix required")
        if not isinstance(rank, (int, np.integer)) or not 1 <= rank < min(values.shape):
            raise ValueError("rank must be below both training rows and feature count")
        if covariance_kappa < 0 or mean_kappa < 0:
            raise ValueError("shrinkage strengths must be nonnegative")

        unique = np.unique(labels)
        self.styles = [str(style) for style in unique]
        self.style_index = {style: index for index, style in enumerate(self.styles)}
        self.n_features = values.shape[1]
        self.rank = int(rank)
        self.covariance_kappa = float(covariance_kappa)
        self.mean_kappa = float(mean_kappa)
        self.global_mean = values.mean(axis=0)
        self.global_scale = np.std(values, axis=0)
        self.global_scale = np.where(self.global_scale > 1e-12, self.global_scale, 1.0)
        standardized = (values - self.global_mean) / self.global_scale

        empirical_centers = {
            style: standardized[labels == style].mean(axis=0) for style in self.styles
        }
        residual = np.vstack(
            [standardized[index] - empirical_centers[str(style)] for index, style in enumerate(labels)]
        )
        self.pca = PCA(n_components=self.rank, svd_solver="full", random_state=random_state).fit(residual)
        latent = self.pca.transform(residual)
        reconstructed = self.pca.inverse_transform(latent)
        residual_error = residual - reconstructed

        shared_covariance, self.shared_factor = _psd_factor(LedoitWolf().fit(latent).covariance_)
        shared_residual_variance = np.maximum(np.mean(residual_error ** 2, axis=0), 1e-8)
        self.shared_covariance = shared_covariance
        self.shared_residual_variance = shared_residual_variance
        self.models: dict[str, dict] = {}
        for style in self.styles:
            mask = labels == style
            count = int(mask.sum())
            mean_weight = count / (count + self.mean_kappa) if self.mean_kappa else 1.0
            covariance_weight = count / (count + self.covariance_kappa) if self.covariance_kappa else 1.0
            style_covariance = (
                LedoitWolf().fit(latent[mask]).covariance_
                if count >= 2
                else shared_covariance
            )
            covariance = covariance_weight * style_covariance + (1.0 - covariance_weight) * shared_covariance
            covariance, factor = _psd_factor(covariance)
            style_residual = (
                np.maximum(np.mean(residual_error[mask] ** 2, axis=0), 1e-8)
                if count >= 2
                else shared_residual_variance
            )
            residual_variance = (
                covariance_weight * style_residual
                + (1.0 - covariance_weight) * shared_residual_variance
            )
            self.models[style] = {
                "rows": count,
                "mean": mean_weight * empirical_centers[style],
                "mean_weight": mean_weight,
                "covariance": covariance,
                "factor": factor,
                "covariance_weight": covariance_weight,
                "residual_variance": np.maximum(residual_variance, 1e-8),
                "lower": values[mask].min(axis=0),
                "upper": values[mask].max(axis=0),
                "style_values": values[mask].copy(),
            }

        self.floor_models: dict[int, dict] = {}
        floor_specs = floor_specs or {}
        dummy = np.zeros((len(values), len(self.styles)))
        dummy[np.arange(len(values)), [self.style_index[str(style)] for style in labels]] = 1.0
        design = np.column_stack([latent, dummy])
        for column, raw_spec in floor_specs.items():
            column = int(column)
            if not 0 <= column < self.n_features:
                raise ValueError("floor field index outside feature matrix")
            spec = dict(raw_spec)
            floor_value = float(spec["value"])
            tolerance = float(spec.get("tolerance", 1e-9))
            prior_strength = float(spec.get("prior_strength", 10.0))
            if tolerance < 0 or prior_strength < 0:
                raise ValueError("floor tolerance and prior strength must be nonnegative")
            state = np.isclose(values[:, column], floor_value, atol=tolerance, rtol=0).astype(int)
            global_rate = float(state.mean())
            classifier = None
            if np.unique(state).size == 2:
                classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state).fit(design, state)
                global_logits = classifier.decision_function(design)
            else:
                global_logits = np.full(len(values), np.log((global_rate + 1e-8) / (1 - global_rate + 1e-8)))
            offsets = {}
            target_rates = {}
            nonfloor_values = {}
            for style in self.styles:
                mask = labels == style
                target = (state[mask].sum() + prior_strength * global_rate) / (mask.sum() + prior_strength)
                logits = global_logits[mask]
                offsets[style] = _calibration_offset(logits, target)
                target_rates[style] = float(target)
                candidates = values[mask & (state == 0), column]
                if not len(candidates):
                    candidates = values[state == 0, column]
                nonfloor_values[style] = np.asarray(candidates, float)
            self.floor_models[column] = {
                "value": floor_value,
                "tolerance": tolerance,
                "classifier": classifier,
                "constant_logit": float(global_logits[0]) if classifier is None else None,
                "offsets": offsets,
                "target_rates": target_rates,
                "nonfloor_values": nonfloor_values,
            }
        return self

    def diagnostics(self, style: str) -> dict:
        style = str(style)
        if style not in self.models:
            raise ValueError("unseen style")
        model = self.models[style]
        return {
            "rank": self.rank,
            "n_training_style": model["rows"],
            "mean_weight": float(model["mean_weight"]),
            "covariance_weight": float(model["covariance_weight"]),
            "latent_min_eigenvalue": float(np.linalg.eigvalsh(model["covariance"]).min()),
            "residual_variance_min": float(model["residual_variance"].min()),
            "explained_variance_ratio": float(self.pca.explained_variance_ratio_.sum()),
        }

    def _floor_probability(self, column: int, style: str, latent: np.ndarray) -> np.ndarray:
        floor_model = self.floor_models[column]
        classifier = floor_model["classifier"]
        if classifier is None:
            logits = np.full(len(latent), floor_model["constant_logit"])
        else:
            dummy = np.zeros((len(latent), len(self.styles)))
            dummy[:, self.style_index[style]] = 1.0
            logits = classifier.decision_function(np.column_stack([latent, dummy]))
        return _sigmoid(logits + floor_model["offsets"][style])

    def generate(self, style, n: int, *, method: str = "style_residual_floor", seed: int = 20261025):
        style = str(style)
        if style not in self.models:
            raise ValueError("unseen style")
        if method not in METHODS:
            raise ValueError("unsupported shared-low-rank method")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("positive sample count required")
        model = self.models[style]
        rng = np.random.default_rng(seed)
        if method == "shared_lowrank":
            latent = rng.normal(size=(n, self.rank)) @ self.shared_factor.T
            residual_scale = np.sqrt(self.shared_residual_variance)
        else:
            latent = rng.normal(size=(n, self.rank)) @ model["factor"].T
            residual_scale = np.sqrt(model["residual_variance"])
        noise = rng.normal(size=(n, self.n_features)) * residual_scale
        standardized = model["mean"] + self.pca.mean_ + latent @ self.pca.components_ + noise
        raw = standardized * self.global_scale + self.global_mean
        outside = (raw < model["lower"]) | (raw > model["upper"])
        values = np.clip(raw, model["lower"], model["upper"])
        floor_fraction: dict[int, float] = {}
        if method == "style_residual_floor":
            for column, floor_model in self.floor_models.items():
                probability = self._floor_probability(column, style, latent)
                state = rng.random(n) < probability
                values[state, column] = floor_model["value"]
                invalid_nonfloor = (~state) & (
                    values[:, column] <= floor_model["value"] + floor_model["tolerance"]
                )
                if invalid_nonfloor.any():
                    candidates = floor_model["nonfloor_values"][style]
                    if not len(candidates):
                        values[invalid_nonfloor, column] = np.nextafter(floor_model["value"], np.inf)
                    else:
                        values[invalid_nonfloor, column] = rng.choice(candidates, invalid_nonfloor.sum(), replace=True)
                floor_fraction[column] = float(state.mean())
        return {
            "values": values,
            "style": style,
            "method": method,
            "n_training_style": model["rows"],
            "pre_clip_cell_fraction": float(outside.mean()),
            "pre_clip_row_fraction": float(outside.any(axis=1).mean()),
            "floor_fraction": floor_fraction,
        }
