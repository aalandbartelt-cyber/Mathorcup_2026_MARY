"""Stored-scale conversions, direct constraints and candidate geometry."""
from __future__ import annotations
import numpy as np


def concentrations_from_stored(values, scales):
    values = np.asarray(values, float)
    scales = np.asarray(scales, str)
    if values.ndim != 1 or scales.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("finite aligned stored values and scales required")
    if not np.isin(scales, ["raw", "log10"]).all():
        raise ValueError("scale must be raw or log10")
    result = values.copy()
    log_mask = scales == "log10"
    result[log_mask] = np.power(10.0, values[log_mask])
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("stored values cannot be represented as nonnegative concentrations")
    return result


def stored_from_concentrations(values, scales):
    values = np.asarray(values, float)
    scales = np.asarray(scales, str)
    if values.ndim != 1 or scales.shape != values.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("finite aligned nonnegative concentrations and scales required")
    if not np.isin(scales, ["raw", "log10"]).all():
        raise ValueError("scale must be raw or log10")
    result = values.copy()
    log_mask = scales == "log10"
    if (values[log_mask] <= 0).any():
        raise ValueError("log10 fields require positive concentrations")
    result[log_mask] = np.log10(values[log_mask])
    return result


def direct_limits(source, source_known, threshold, scenario):
    source = np.asarray(source, float)
    known = np.asarray(source_known)
    threshold = np.asarray(threshold, float)
    if scenario not in {"C1", "C2", "C3"}:
        raise ValueError("unknown constraint scenario")
    if source.ndim != 1 or known.shape != source.shape or threshold.shape != source.shape or known.dtype.kind != "b":
        raise ValueError("aligned direct-constraint vectors required")
    limits = np.full(source.shape, np.nan)
    valid_source = known & np.isfinite(source)
    if scenario == "C1":
        limits[valid_source] = source[valid_source]
    else:
        valid = valid_source & np.isfinite(threshold)
        if scenario == "C2":
            limits[valid] = np.maximum(source[valid], threshold[valid])
        else:
            limits[valid] = np.minimum(source[valid], threshold[valid])
    return limits


def project_direct_constraints(candidate_stored, source, source_known, threshold, scales, scenario):
    candidate_stored = np.asarray(candidate_stored, float)
    candidate = concentrations_from_stored(candidate_stored, scales)
    limits = direct_limits(source, source_known, threshold, scenario)
    projected = np.where(np.isfinite(limits), np.minimum(candidate, limits), candidate)
    return stored_from_concentrations(projected, scales), limits


def empirical_quantile_anchor(source, source_reference, target_reference):
    source = np.asarray(source, float)
    source_reference = np.asarray(source_reference, float)
    target_reference = np.asarray(target_reference, float)
    if (
        source.ndim != 1
        or source_reference.ndim != 2
        or target_reference.ndim != 2
        or source_reference.shape[1] != source.size
        or target_reference.shape[1] != source.size
        or min(len(source_reference), len(target_reference)) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(source_reference).all()
        or not np.isfinite(target_reference).all()
    ):
        raise ValueError("finite aligned source and style reference matrices required")
    anchor = np.empty_like(source)
    for column, value in enumerate(source):
        reference = source_reference[:, column]
        percentile = float(np.mean(reference < value) + 0.5 * np.mean(reference == value))
        anchor[column] = np.quantile(target_reference[:, column], percentile, method="linear")
    return anchor


def pareto_mask(target_distance, source_change):
    target_distance = np.asarray(target_distance, float)
    source_change = np.asarray(source_change, float)
    if target_distance.ndim != 1 or source_change.shape != target_distance.shape:
        raise ValueError("aligned one-dimensional objective arrays required")
    finite = np.isfinite(target_distance) & np.isfinite(source_change)
    result = np.zeros(len(target_distance), dtype=bool)
    for index in np.flatnonzero(finite):
        dominated = finite & (target_distance <= target_distance[index]) & (source_change <= source_change[index])
        strict = (target_distance < target_distance[index]) | (source_change < source_change[index])
        if not np.any(dominated & strict):
            result[index] = True
    return result
