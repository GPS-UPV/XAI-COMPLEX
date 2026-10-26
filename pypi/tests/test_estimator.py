import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "scipy", "joblib"))


@unittest.skipUnless(AVAILABLE, "Requires numpy, scikit-learn, scipy and joblib")
class EstimatorTests(unittest.TestCase):
    def setUp(self):
        from xai_complex import ComplexityEstimator
        self.cls = ComplexityEstimator
        self.rows = [{"a": float(i), "b": float((i * 7) % 11)} for i in range(35)]
        self.model = self.cls(n_estimators=20).fit(self.rows)

    def test_batch_invariance(self):
        query = {"a": 16., "b": 4.}
        self.assertEqual(self.model.predict([query])[0], self.model.predict([query, self.rows[-1]])[0])

    def test_roundtrip_and_trust(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.model.save(Path(temp) / "model.joblib")
            with self.assertRaises(ValueError):
                self.cls.load(path)
            self.assertEqual(self.model.predict(self.rows), self.cls.load(path, trusted=True).predict(self.rows))

    def test_invalid_schema_and_sizes(self):
        for rows in ([], [{"a": 1.}], [{"a": 1.}] * 3):
            with self.assertRaises(ValueError):
                self.cls().fit(rows)
        for rows in ([{"a": 1}], [{"a": float("inf"), "b": 0}]):
            with self.assertRaises(ValueError):
                self.model.predict(rows)

    def test_supervised(self):
        model = self.cls(n_estimators=20).fit(self.rows, [i / 34 for i in range(35)])
        self.assertEqual(model.predict([self.rows[0]])[0]["signal"], "solver_calibrated_blend")

    def test_grouped_evaluation(self):
        from xai_complex.estimator import evaluate_supervised
        labels = [{"solveTime": i + 1, "maxTime": 100, "status": "FEASIBLE"} for i in range(35)]
        results = evaluate_supervised(self.rows, labels, groups=[i // 5 for i in range(35)], n_splits=3)
        self.assertEqual(len(results["oof"]), 35)
        self.assertTrue(results["grouped"])


if __name__ == "__main__":
    unittest.main()
