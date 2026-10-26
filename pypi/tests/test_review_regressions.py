import importlib.util
import math
import unittest

from xai_complex import SolverTargetCalibrator


class WeightRegressionTests(unittest.TestCase):
    def test_numeric_strings_and_overflow(self):
        model = SolverTargetCalibrator({"q": "1"}).fit([{"status": "OPTIMAL"}])
        self.assertEqual(model.transform([{"status": "FEASIBLE"}]), [.5])
        for weights in ({"q": True}, {"q": 1e308, "rt": 1e308}):
            with self.assertRaises(ValueError):
                SolverTargetCalibrator(weights)


AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "sklearn", "scipy", "joblib"))


@unittest.skipUnless(AVAILABLE, "Requires scientific dependencies")
class EvaluationRegressionTests(unittest.TestCase):
    def test_full_identifiers_required(self):
        import pandas as pd
        from xai_complex.calibration import supervised_calibration
        features = pd.DataFrame({"a": [1., 2., 3.]}, index=["2-2-0.pt", "2-2-1.pt", "2-2-2.pt"])
        labels = pd.DataFrame({"status": ["OPTIMAL"]}, index=["2_2-0-3-cp-sat"])
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            supervised_calibration(features, None, labels)

    def test_groups_and_training_calibration(self):
        from xai_complex.estimator import evaluate_supervised
        features = [{"a": float(i), "b": float(i % 3)} for i in range(24)]
        labels = [{"solveTime": i + 1., "base_instance_id": i // 4} for i in range(24)]
        result = evaluate_supervised(features, labels, n_splits=3, n_estimators=5)
        self.assertTrue(result["grouped"])
        for fold in result["folds"]:
            train, test = fold["train"], fold["test"]
            self.assertFalse({i // 4 for i in train} & {i // 4 for i in test})
            calibrator = SolverTargetCalibrator().fit([labels[i] for i in train])
            self.assertEqual(calibrator.transform([labels[i] for i in test]), [result["targets"][i] for i in test])

    def test_constant_batch_rejected(self):
        from xai_complex import estimate_features
        with self.assertRaisesRegex(ValueError, "constant"):
            estimate_features([{"a": 1.}] * 3)

    def test_missing_groups_rejected(self):
        from xai_complex.estimator import evaluate_supervised
        features = [{"a": float(i)} for i in range(12)]
        labels = [{"solveTime": i + 1} for i in range(12)]
        with self.assertRaises(ValueError):
            evaluate_supervised(features, labels, groups=[None] * 12, n_splits=3)


if __name__ == "__main__":
    unittest.main()
