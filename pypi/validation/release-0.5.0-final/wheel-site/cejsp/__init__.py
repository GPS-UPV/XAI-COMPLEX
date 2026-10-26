"""CEJSP: fixed-reference difficulty classification for job shop scheduling."""
from xai_complex import __version__
from xai_complex.categories import CATEGORIES
from .api import CEJSP,classify,classify_many
from .io import load_instance

__all__=['CEJSP','classify','classify_many','CATEGORIES','load_instance','__version__']


def __getattr__(name):
    # Keep earlier research scripts importable; the supported inference API above
    # never accepts a custom model. Legacy training APIs remain research tools.
    if name=='JobShopComplexityClassifier':
        from .classifier import JobShopComplexityClassifier
        return JobShopComplexityClassifier
    if name=='ComplexityEstimator':
        from xai_complex import ComplexityEstimator
        return ComplexityEstimator
    if name in ('normalize_instance','summarize_instance','estimate_features','SolverTargetCalibrator','status_to_complexity'):
        import xai_complex
        return getattr(xai_complex,name)
    raise AttributeError(name)
