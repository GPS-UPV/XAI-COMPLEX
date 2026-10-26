"""Public name and compatibility regression checks."""
import unittest

import cejsp
import xai_complex


class PublicNameTests(unittest.TestCase):
    def test_same_api_and_schema(self):
        self.assertIs(cejsp.normalize_instance, xai_complex.normalize_instance)
        self.assertIs(cejsp.SolverTargetCalibrator, xai_complex.SolverTargetCalibrator)
        self.assertIs(cejsp.estimate_features, xai_complex.estimate_features)
        self.assertEqual(cejsp.__version__, xai_complex.__version__)
