"""Shared preprocessing utilities (RMT denoising, covariance shrinkage, …).

These helpers are optimizer-agnostic so any classical / quantum / hybrid
algorithm can opt in to the same numerically stable preprocessing pipeline.
"""

from core.preprocessing.covariance_estimation import ledoit_wolf_covariance
from core.preprocessing.rmt import (
    marchenko_pastur_threshold,
    denoise_correlation,
    denoise_covariance,
    maybe_denoise_covariance,
    should_apply_rmt,
)

__all__ = [
    "ledoit_wolf_covariance",
    "marchenko_pastur_threshold",
    "denoise_correlation",
    "denoise_covariance",
    "maybe_denoise_covariance",
    "should_apply_rmt",
]
