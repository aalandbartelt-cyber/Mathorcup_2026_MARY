"""风格条件经验边缘与收缩相关结构的核心片段。"""
import numpy as np
from scipy.stats import norm


def latent_rank_transform(x):
    n, p = x.shape
    z = np.zeros_like(x, dtype=float)
    for j in range(p):
        order = np.argsort(np.argsort(x[:, j]))
        u = (order + 0.5) / n
        z[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    return z


def shrink_correlation(z, shrinkage=0.5):
    cov = np.cov(z, rowvar=False)
    target = np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    shrunk = (1 - shrinkage) * cov + shrinkage * target
    scale = np.sqrt(np.clip(np.diag(shrunk), 1e-12, None))
    return shrunk / np.outer(scale, scale)


def sample_style(correlation, empirical_inverse, size, rng):
    z = rng.multivariate_normal(np.zeros(correlation.shape[0]), correlation, size=size)
    u = norm.cdf(z)
    return np.column_stack([empirical_inverse[:, j](u[:, j]) for j in range(u.shape[1])])
