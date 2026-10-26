"""Fixed-reference inference for existing job shop scheduling instances."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
import json
from importlib.metadata import version


@lru_cache(maxsize=1)
def _reference():
    import joblib
    from xai_complex.classifier import JobShopComplexityClassifier
    resource=files('cejsp').joinpath('data')
    metadata=json.loads(resource.joinpath('reference.json').read_text(encoding='utf-8'))
    payload=resource.joinpath('reference.joblib').read_bytes()
    if sha256(payload).hexdigest()!=metadata['sha256']:
        raise RuntimeError('The bundled CEJSP reference failed its integrity check. Reinstall CEJSP.')
    for package,required in metadata['dependencies'].items():
        if version(package)!=required:
            raise RuntimeError(f'The bundled reference requires {package}=={required}. Reinstall CEJSP with its declared dependencies.')
    model=joblib.load(BytesIO(payload))
    if not isinstance(model,JobShopComplexityClassifier) or model.model_version_!=metadata['model_version']:
        raise RuntimeError('Invalid bundled CEJSP reference. Reinstall CEJSP.')
    return model


def CEJSP(instance) -> str:
    """Return toy, easy, medium, hard or open using the bundled fixed reference.

    Accept a JSON/DZN/Taillard file path, a supported instance mapping, or an
    IGJSP instance object. No model path, fitting or configuration is required.
    """
    return classify(instance)['category']


def classify(instance) -> dict:
    """Return the class, uncalibrated votes, applicability and model provenance.

    This function uses the same fixed reference as CEJSP. The returned mapping
    is independent of the cached model and can be modified by the caller.
    """
    return _reference().classify(instance)


def classify_many(instances, *, workers: int = 1) -> list[dict]:
    """Classify a batch in input order, optionally using independent worker threads.

    Workers change execution concurrency only. They do not change the reference
    or its feature extraction settings. An invalid input raises its normal error.
    """
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer')
    if workers==1:
        return [classify(instance) for instance in instances]
    _reference()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(classify,instances))
