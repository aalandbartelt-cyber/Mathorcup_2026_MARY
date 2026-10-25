"""Explicit concentration constraints; unknown measurements never certify a pass."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_concentrations(source, candidate, source_known, scenario: str, *,
                         threshold=None, atol: float = 1e-9, rtol: float = 1e-7) -> pd.DataFrame:
    if scenario not in {"C1", "C2", "C3"} or atol < 0 or rtol < 0:
        raise ValueError("unknown scenario or invalid numerical tolerance")
    source, candidate = np.asarray(source, float), np.asarray(candidate, float)
    known = np.asarray(source_known)
    if known.dtype.kind != "b":
        raise ValueError("source_known must contain boolean flags")
    tau = np.asarray(np.nan if threshold is None else threshold, float)
    source, candidate, known, tau = [np.atleast_1d(v) for v in
                                   np.broadcast_arrays(source, candidate, known, tau)]
    if source.ndim != 1:
        raise ValueError("constraints require one-dimensional aligned arrays")
    if np.isinf(source).any() or (source[np.isfinite(source)] < 0).any():
        raise ValueError("source must be nonnegative concentration or missing")
    if scenario != "C1" and (np.isinf(tau).any() or (tau[np.isfinite(tau)] <= 0).any()):
        raise ValueError("threshold must be positive or missing")
    limits = source if scenario == "C1" else (
        np.maximum(source, tau) if scenario == "C2" else np.minimum(source, tau))
    rows = []
    for before, after, observed, limit, value in zip(source, candidate, known, limits, tau):
        passed, excess, tolerance = pd.NA, np.nan, np.nan
        if not np.isfinite(after) or after < 0:
            status, passed = "invalid_candidate", False
        elif not observed or not np.isfinite(before):
            status = "unknown_source"
        elif scenario != "C1" and not np.isfinite(value):
            status = "unknown_threshold"
        else:
            tolerance = atol + rtol * abs(before)
            excess = float(max(after - limit, 0.))
            passed = bool(after <= limit + tolerance)
            status = "pass" if passed else "violation"
        rows.append({"scenario": scenario, "source": before, "candidate": after,
                     "limit": limit, "threshold": value, "status": status,
                     "violation_magnitude": excess, "numerical_tolerance": tolerance,
                     "passed": passed})
    frame = pd.DataFrame(rows)
    frame["passed"] = pd.array(frame["passed"], dtype="boolean")
    return frame


def project_upper_bounds(candidate, upper) -> np.ndarray:
    candidate, upper = np.broadcast_arrays(np.asarray(candidate, float), np.asarray(upper, float))
    if (not np.isfinite(candidate).all() or (candidate < 0).any()
            or np.isinf(upper).any() or (upper[np.isfinite(upper)] < 0).any()):
        raise ValueError("projection requires finite nonnegative candidates and valid bounds")
    return np.where(np.isfinite(upper), np.minimum(candidate, upper), candidate)


def summarize_migration(audit: pd.DataFrame, *, target_gain: bool,
                        proxy_nonworsening: bool | None) -> dict:
    if audit.empty or not {"status", "passed"}.issubset(audit):
        raise ValueError("nonempty direct-constraint audit required")
    passed = audit.passed.astype("boolean")
    n_passed, n_failed = int(passed.fillna(False).sum()), int((passed == False).fillna(False).sum())
    n_unknown = int(passed.isna().sum())
    if n_failed:
        status = "rejected_direct_constraint"
    elif not target_gain:
        status = "no_target_gain"
    elif n_unknown:
        status = "needs_direct_verification"
    elif proxy_nonworsening is None:
        status = "needs_proxy_verification"
    elif not proxy_nonworsening:
        status = "rejected_proxy"
    else:
        status = "eligible_statistical_candidate"
    return {"n_checks": len(audit), "n_passed": n_passed, "n_failed": n_failed,
            "n_unknown": n_unknown,
            "decidable_pass_rate": n_passed / (n_passed + n_failed) if n_passed + n_failed else None,
            "status": status, "eligible": status == "eligible_statistical_candidate",
            "physical_safety_certified": False}
