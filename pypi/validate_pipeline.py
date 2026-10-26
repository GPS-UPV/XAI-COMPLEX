"""Run graph, batch and persistent inference checks on existing benchmark files."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time


SCORE_ATOL = 1e-9


def compare_predictions(individual, together, *, rtol=0.0, atol=SCORE_ATOL):
    if len(individual) != len(together):
        raise AssertionError("Prediction counts differ")
    largest_difference = 0.0
    for index, (one, batch) in enumerate(zip(individual, together)):
        a, b = one["complexity_0_1"], batch["complexity_0_1"]
        difference = abs(a - b)
        largest_difference = max(largest_difference, difference)
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"Score differs at row {index}: individual={a!r}, batch={b!r}, "
                f"absolute_difference={difference!r}, rtol={rtol}, atol={atol}"
            )
        for field in ("category", "signal"):
            if one[field] != batch[field]:
                raise AssertionError(f"{field} differs at row {index}: {one[field]!r} != {batch[field]!r}")
    return largest_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pipeline-validation.json"))
    args = parser.parse_args()
    from cejsp import ComplexityEstimator, estimate_features
    from cejsp.graphs import extract_instance_features
    import numpy as np

    names = ["ft06", "ft10", "ft20", "orb01", "orb02", "orb03", "orb04", "orb05", "orb06", "orb07", "orb08", "orb09", "orb10", "la01", "la02", "la03", "la04", "la05", "la06", "la07"]
    started = time.perf_counter()
    features = [extract_instance_features(json.loads((args.folder / f"{name}.json").read_text(encoding="utf-8-sig"))) for name in names]
    batch = estimate_features(features)
    assert len(batch) == len(names)
    model = ComplexityEstimator(n_estimators=30).fit(features)
    individual = [model.predict([row])[0] for row in features]
    together = model.predict(features)
    # La igualdad numérica admite redondeo; categorías y señales deben coincidir.
    try:
        maximum_difference = compare_predictions(individual, together)
    except AssertionError as exc:
        diagnostic = {"status": "failed", "error": str(exc),
                      "predictions": [{"instance": name, "individual": one, "batch": many}
                                      for name, one, many in zip(names, individual, together)]}
        args.output.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        raise
    assert np.isfinite(batch["complexity_0_1"]).all()
    with tempfile.TemporaryDirectory() as temp:
        path = model.save(Path(temp) / "reference.joblib")
        persistence_difference = compare_predictions(model.load(path, trusted=True).predict(features), together)
    result = {"passed": len(names), "instances": names, "seconds": time.perf_counter() - started,
              "checks": ["real JSON", "original graph features", "batch estimation", "individual/batch invariance", "persistence"],
              "performance_comparison": False,
              "max_individual_batch_difference": maximum_difference,
              "max_persistence_difference": persistence_difference,
              "score_tolerance": {"rtol": 0.0, "atol": SCORE_ATOL}}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
