"""CEJSP: JSP Complexity Estimator, with compatible model persistence."""
from xai_complex import normalize_instance, summarize_instance, estimate_features
from xai_complex import SolverTargetCalibrator, status_to_complexity, __version__
from .io import load_instance
from xai_complex.categories import CATEGORIES

__all__ = ["normalize_instance", "summarize_instance", "estimate_features",
           "SolverTargetCalibrator", "status_to_complexity", "ComplexityEstimator",
           "JobShopComplexityClassifier", "load_instance", "CATEGORIES"]


def __getattr__(name):
    if name == "JobShopComplexityClassifier":
        from .classifier import JobShopComplexityClassifier
        return JobShopComplexityClassifier
    if name == "ComplexityEstimator":
        from xai_complex import ComplexityEstimator
        return ComplexityEstimator
    raise AttributeError(name)
