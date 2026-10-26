import importlib.util
import unittest

from test_contracts import fixture

AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "networkx", "torch", "torch_geometric", "tqdm"))


@unittest.skipUnless(AVAILABLE, "Requires the graphs optional dependencies")
class GraphIntegrationTests(unittest.TestCase):
    def test_sampled_betweenness_is_repeatable(self):
        from xai_complex.graphs import extract_instance_features
        options = {"approximate_betweenness": True, "betweenness_samples": 1}
        first = extract_instance_features(fixture(), **options)
        self.assertEqual(first, extract_instance_features(fixture(), **options))
        self.assertEqual(set(first), set(extract_instance_features(fixture())))
        with self.assertRaises(ValueError):
            extract_instance_features(fixture(), betweenness_samples=0)

    def test_existing_instance_features(self):
        from xai_complex.graphs import extract_instance_features
        result = extract_instance_features(fixture(dates=1))
        self.assertTrue(result)
        self.assertTrue(any("betweenness" in k for k in result))

    def test_explicit_projection_and_flexible_rejection(self):
        from xai_complex.graphs import extract_instance_features
        with self.assertRaises(ValueError):
            extract_instance_features(fixture(speeds=3))
        self.assertTrue(extract_instance_features(fixture(speeds=3), speed_index=2))
        with self.assertRaises(ValueError):
            extract_instance_features(fixture(flexible=True))
