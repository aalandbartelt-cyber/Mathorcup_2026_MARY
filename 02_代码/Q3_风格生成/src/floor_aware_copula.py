"""Empirical-Copula generator with explicitly registered floor states.

Only fields supplied in ``floor_specs`` are treated as observed-floor/zero-
inflated variables. Repeated minima in other columns remain ordinary tied
measurements. The floor probability is estimated inside each style with a
small pooled-rate prior; the non-floor value distribution is fitted only on
the observed non-floor values.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf


def _psd_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    psd = (eigenvectors * eigenvalues) @ eigenvectors.T
    psd = (psd + psd.T) / 2.0
    return psd, eigenvectors * np.sqrt(eigenvalues)


def _correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    result = covariance / np.outer(scale, scale)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 1.0)
    result, _ = _psd_factor(result)
    scale = np.sqrt(np.clip(np.diag(result), 1e-12, None))
    result = result / np.outer(scale, scale)
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 1.0)
    return result


def _rank_uniform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    if np.ptp(values) <= 1e-12:
        return np.full(len(values), 0.5)
    ranks = rankdata(values, method="average")
    return np.clip((ranks - 0.5) / len(values), 1e-6, 1.0 - 1e-6)


class FloorAwareStyleCopula:
    """Style-conditional empirical Copula with registered floor states."""

    def fit(self, x, styles, *, floor_specs=None, prior_strength=None):
        values = np.asarray(x, dtype=float)
        labels = np.asarray(styles, dtype=str)
        if values.ndim != 2 or labels.shape != (len(values),) or len(values) < 4:
            raise ValueError("finite training matrix and one style label per row required")
        if not np.isfinite(values).all():
            raise ValueError("finite training matrix required")
        if np.unique(labels).size < 2:
            raise ValueError("at least two styles are required")
        self.n_features = int(values.shape[1])
        self.styles = [str(style) for style in np.unique(labels)]
        self.style_index = {style: i for i, style in enumerate(self.styles)}
        self.floor_specs = {}
        for raw_column, raw_spec in (floor_specs or {}).items():
            if isinstance(raw_column, bool) or not isinstance(raw_column, (int, np.integer)):
                raise ValueError("floor field indices must be integers")
            column = int(raw_column)
            if not 0 <= column < self.n_features:
                raise ValueError("floor field index outside feature matrix")
            spec = dict(raw_spec)
            floor_value = float(spec["value"])
            tolerance = float(spec.get("tolerance", 1e-9))
            strength = float(spec.get("prior_strength", 10.0) if prior_strength is None else prior_strength)
            if not np.isfinite(floor_value) or tolerance < 0 or strength < 0:
                raise ValueError("finite floor value and nonnegative tolerance/prior required")
            self.floor_specs[column] = {
                "value": floor_value,
                "tolerance": tolerance,
                "prior_strength": strength,
                "detection": str(spec.get("detection", "registered")),
            }

        self.models = {}
        for style in self.styles:
            block = values[labels == style]
            if len(block) < 2:
                raise ValueError("each style needs at least two training rows")
            latent = np.empty_like(block)
            floor_models = {}
            for column in range(self.n_features):
                if column not in self.floor_specs:
                    latent[:, column] = norm.ppf(_rank_uniform(block[:, column]))
                    continue
                spec = self.floor_specs[column]
                floor_value = spec["value"]
                tolerance = spec["tolerance"]
                state = np.isclose(block[:, column], floor_value, atol=tolerance, rtol=0)
                global_state = np.isclose(values[:, column], floor_value, atol=tolerance, rtol=0)
                global_rate = float(global_state.mean())
                strength = spec["prior_strength"]
                # A style with an observed all-floor (or all-nonfloor) block
                # is kept at that boundary. Shrinking such a block toward the
                # pooled rate would manufacture a state absent from the style.
                if state.all() or (~state).all():
                    p_floor = float(state.mean())
                else:
                    p_floor = (float(state.sum()) + strength * global_rate) / (len(state) + strength)
                p_floor = float(np.clip(p_floor, 0.0, 1.0))
                nonfloor = block[~state, column].astype(float)
                if len(nonfloor) == 0:
                    nonfloor = values[~global_state, column].astype(float)
                if len(nonfloor):
                    u = np.empty(len(block), dtype=float)
                    u[state] = max(0.5 * p_floor, 1e-6)
                    ranks = _rank_uniform(block[~state, column])
                    u[~state] = p_floor + ranks * max(1.0 - p_floor, 1e-6)
                    latent[:, column] = norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
                else:
                    latent[:, column] = 0.0
                floor_models[column] = {
                    "value": floor_value,
                    "tolerance": tolerance,
                    "prior_strength": strength,
                    "floor_probability": p_floor,
                    "observed_floor_fraction": float(state.mean()),
                    "nonfloor_values": np.asarray(nonfloor, dtype=float).copy(),
                }
            covariance = LedoitWolf().fit(latent).covariance_
            correlation = _correlation(covariance)
            correlation, factor = _psd_factor(correlation)
            self.models[style] = {
                "rows": int(len(block)),
                "style_values": block.copy(),
                "correlation": correlation,
                "factor": factor,
                "floor_models": floor_models,
            }
        return self

    def diagnostics(self, style):
        style = str(style)
        if style not in self.models:
            raise ValueError("unseen style")
        model = self.models[style]
        return {
            "n_training_style": int(model["rows"]),
            "copula_min_eigenvalue": float(np.linalg.eigvalsh(model["correlation"]).min()),
            "registered_floor_fields": int(len(model["floor_models"])),
        }

    def generate(self, style, n, *, seed=20261025):
        style = str(style)
        if style not in self.models:
            raise ValueError("unseen style")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("positive sample count required")
        model = self.models[style]
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(int(n), self.n_features)) @ model["factor"].T
        uniforms = np.clip(norm.cdf(latent), 0.0, 1.0)
        values = np.empty_like(uniforms)
        floor_fraction = {}
        for column in range(self.n_features):
            block = model["style_values"][:, column]
            if column not in model["floor_models"]:
                values[:, column] = np.quantile(np.sort(block), uniforms[:, column], method="linear")
                continue
            floor_model = model["floor_models"][column]
            p_floor = float(floor_model["floor_probability"])
            state = uniforms[:, column] < p_floor
            values[state, column] = floor_model["value"]
            nonfloor = floor_model["nonfloor_values"]
            if (~state).any():
                if len(nonfloor):
                    q = np.clip((uniforms[~state, column] - p_floor) / max(1.0 - p_floor, 1e-12), 0.0, 1.0)
                    values[~state, column] = np.quantile(np.sort(nonfloor), q, method="linear")
                    invalid = values[~state, column] <= floor_model["value"] + floor_model["tolerance"]
                    if invalid.any():
                        valid_nonfloor = nonfloor[nonfloor > floor_model["value"] + floor_model["tolerance"]]
                        fallback = valid_nonfloor.min() if len(valid_nonfloor) else np.nextafter(floor_model["value"], np.inf)
                        values[np.flatnonzero(~state)[invalid], column] = fallback
                else:
                    values[~state, column] = np.nextafter(floor_model["value"], np.inf)
            floor_fraction[column] = float(state.mean())
        return {
            "values": values,
            "style": style,
            "method": "floor_copula",
            "n_training_style": int(model["rows"]),
            "pre_clip_cell_fraction": 0.0,
            "pre_clip_row_fraction": 0.0,
            "floor_fraction": floor_fraction,
        }

    def state_dict(self):
        return {
            "n_features": int(self.n_features),
            "styles": list(self.styles),
            "floor_specs": deepcopy(self.floor_specs),
            "models": deepcopy(self.models),
        }

    @classmethod
    def from_state(cls, state):
        required = {"n_features", "styles", "floor_specs", "models"}
        if set(state) != required:
            raise ValueError("incomplete floor-Copula state")
        model = cls()
        model.n_features = int(state["n_features"])
        model.styles = [str(style) for style in state["styles"]]
        model.style_index = {style: i for i, style in enumerate(model.styles)}
        model.floor_specs = deepcopy(state["floor_specs"])
        model.models = deepcopy(state["models"])
        return model
