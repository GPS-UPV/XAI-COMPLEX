"""Empirical scheduling difficulty; no training or filesystem work on import."""

from .instances import summarize_instance, normalize_instance
from .labels import SolverTargetCalibrator, status_to_complexity
from .batch import estimate_features

__version__ = "0.4.1"
__all__ = ["normalize_instance", "summarize_instance",
           "SolverTargetCalibrator", "status_to_complexity", "ComplexityEstimator", "estimate_features"]


def __getattr__(name):
    if name == "ComplexityEstimator":
        from .estimator import ComplexityEstimator
        return ComplexityEstimator
    raise AttributeError(name)
