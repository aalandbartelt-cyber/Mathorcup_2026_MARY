"""沿目标路径搜索并施加已知直接缺陷不升约束的核心片段。"""
import numpy as np


def constrained_path(source, endpoint, known_caps, lambdas):
    for lam in lambdas:
        candidate = (1.0 - lam) * source + lam * endpoint
        adjusted = candidate.copy()
        for col, cap in known_caps.items():
            adjusted[col] = min(adjusted[col], cap)
        yield lam, adjusted


def normalized_rms(source, candidate, scale):
    delta = (candidate - source) / np.maximum(scale, 1e-12)
    return float(np.sqrt(np.mean(delta * delta)))
