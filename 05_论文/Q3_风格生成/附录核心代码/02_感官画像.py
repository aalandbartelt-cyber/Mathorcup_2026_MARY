"""调用冻结感官模型生成相对画像的核心片段。"""
import numpy as np


def sensory_profile(spectra, predictor, train_empirical_cdf):
    standardized = predictor.predict(spectra)
    grades = np.zeros_like(standardized)
    for j in range(standardized.shape[1]):
        grades[:, j] = 6.0 * train_empirical_cdf[j](standardized[:, j])
    return standardized, np.clip(grades, 0.0, 6.0)
