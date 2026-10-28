"""Direct measurement and model-proxy evidence remain distinct.

Unknown measurements never certify a direct-constraint result, and statistical
candidate status does not certify brewing feasibility or physical safety.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def audit_concentrations(source, candidate, source_known, scenario: str, *,
                         threshold=None, atol: float = 1e-9, rtol: float = 1e-7) -> pd.DataFrame:
    if (scenario not in {"C1", "C2", "C3"} or not np.isfinite([atol, rtol]).all()
            or atol < 0 or rtol < 0):
        raise ValueError("unknown scenario or invalid numerical tolerance")
    source, candidate = np.atleast_1d(np.asarray(source, float)), np.atleast_1d(np.asarray(candidate, float))
    known = np.atleast_1d(np.asarray(source_known))
    if known.dtype.kind != "b":
        raise ValueError("source_known must contain boolean flags")
    if source.ndim != 1 or source.size == 0 or source.shape != candidate.shape or source.shape != known.shape:
        raise ValueError("nonempty aligned source/candidate/known vectors required")
    tau = np.asarray(np.nan if threshold is None else threshold, float)
    tau = np.broadcast_to(tau, source.shape)
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
    if (not isinstance(target_gain, (bool, np.bool_)) or
            (proxy_nonworsening is not None and not isinstance(proxy_nonworsening, (bool, np.bool_)))):
        raise ValueError("migration/proxy decisions must be explicit boolean flags or unknown")
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


def proxy_screen(primary_deltas, audit_deltas, primary_reliable, audit_reliable, *, tolerance=1e-8):
    p=np.asarray(primary_deltas,float);a=np.asarray(audit_deltas,float)
    if p.shape!=(10,) or a.shape!=(10,) or not np.isfinite([*p,*a,tolerance]).all() or tolerance<0:
        raise ValueError('Ten finite paired-repeat changes and a numerical tolerance required')
    pm,am=float(p.mean()),float(a.mean());pq,aq=float(np.quantile(p,.9)),float(np.quantile(a,.9))
    if not primary_reliable or not audit_reliable:status='proxy_unreliable'
    elif (pm>tolerance)!=(am>tolerance):status='proxy_disagreement'
    elif pm>tolerance or am>tolerance:status='predicted_worsening'
    elif pq>tolerance or aq>tolerance:status='proxy_variation_uncertain'
    else:status='predicted_nonworsening'
    return {'proxy_status':status,'primary_mean_delta':pm,'audit_mean_delta':am,
            'primary_repeat_q90_delta':pq,'audit_repeat_q90_delta':aq,
            'physical_safety_certified':False}


def candidate_status(*, available, n_failed, n_unknown, target_gain, proxy_status, support_fraction):
    if not available:return 'rejected_no_structural_candidate'
    if n_failed>0:return 'rejected_direct_constraint'
    if not target_gain:return 'no_target_gain'
    if n_unknown>0:return 'needs_direct_verification'
    if support_fraction<.9:return 'outside_predictor_support'
    if proxy_status!='predicted_nonworsening':return proxy_status
    return 'eligible_statistical_candidate'
