"""Deterministic local two-anchor proposals in the stored chemical feature space."""

from itertools import combinations

import numpy as np


def nearest_pairs(pool, source, scale, sample_ids, *, k=24):
    values = np.asarray(pool, float)
    source = np.asarray(source, float)
    scale = np.asarray(scale, float)
    sample_ids = np.asarray(sample_ids)
    if (values.ndim != 2 or source.shape != (values.shape[1],)
        or scale.shape != source.shape or sample_ids.shape != (len(values),)
        or not np.isfinite(values).all() or not np.isfinite(source).all()
        or not np.isfinite(scale).all() or (scale <= 0).any()
        or not isinstance(k, int) or not 2 <= k <= len(values)
        or len(set(sample_ids.tolist())) != len(values)):
        raise ValueError("finite aligned pool, source, positive scale and unique IDs required")
    distance = np.mean(((values - source) / scale) ** 2, axis=1)
    nearest = np.lexsort((sample_ids, distance))[:k]
    index_pairs = np.asarray(list(combinations(nearest, 2)), dtype=int)
    return sample_ids[index_pairs], (values[index_pairs[:, 0]] + values[index_pairs[:, 1]]) / 2


def fixed_paths(source, anchors, lambdas):
    source = np.asarray(source, float)
    anchors = np.asarray(anchors, float)
    lambdas = np.asarray(lambdas, float)
    if (source.ndim != 1 or anchors.ndim != 2 or anchors.shape[1] != len(source)
        or len(anchors) == 0 or lambdas.ndim != 1 or len(lambdas) == 0
        or not np.isfinite(source).all() or not np.isfinite(anchors).all()
        or not np.isfinite(lambdas).all() or (lambdas <= 0).any() or (lambdas > 1).any()):
        raise ValueError("finite aligned source/anchors and nonzero lambdas in (0,1] required")
    anchor_index = np.repeat(np.arange(len(anchors)), len(lambdas))
    path_lambdas = np.tile(lambdas, len(anchors))
    paths = (1 - path_lambdas[:, None]) * source + path_lambdas[:, None] * anchors[anchor_index]
    return paths, anchor_index, path_lambdas
