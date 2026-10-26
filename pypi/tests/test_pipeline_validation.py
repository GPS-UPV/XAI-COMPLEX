import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("pipeline_validation", Path(__file__).resolve().parents[1] / "validate_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PredictionComparisonTests(unittest.TestCase):
    def prediction(self, score, category="medium"):
        return [{"complexity_0_1": score, "category": category, "signal": "structural_atypicality"}]

    def test_roundoff_is_accepted_and_reported(self):
        difference = pipeline.compare_predictions(self.prediction(.3), self.prediction(.30000000000000004))
        self.assertGreater(difference, 0)

    def test_material_difference_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "absolute_difference"):
            pipeline.compare_predictions(self.prediction(.3), self.prediction(.301))

    def test_observed_benchmark_difference(self):
        pipeline.compare_predictions(self.prediction(.7395004563919388), self.prediction(.7395004562761607))
        pipeline.compare_predictions(self.prediction(0.), self.prediction(4.631124372878048e-10))
        with self.assertRaises(AssertionError):
            pipeline.compare_predictions(self.prediction(0.), self.prediction(2e-9))

    def test_category_change_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "category differs"):
            pipeline.compare_predictions(self.prediction(.3), self.prediction(.3, "hard"))


if __name__ == "__main__":
    unittest.main()
