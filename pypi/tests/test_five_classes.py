import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from xai_complex.categories import CATEGORIES, validate_protocol


def protocol():
    return {"id": "synthetic-software-test-only", "class_definitions": {c: f"Synthetic {c} label for testing" for c in CATEGORIES}}


class ProtocolTests(unittest.TestCase):
    def test_five_named_categories_required(self):
        self.assertEqual(CATEGORIES, ("toy", "easy", "medium", "hard", "open"))
        with self.assertRaises(ValueError):
            validate_protocol({"id": "x", "class_definitions": {"toy": "small"}})


AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "joblib"))


@unittest.skipUnless(AVAILABLE, "Requires classifier dependencies")
class FiveClassTests(unittest.TestCase):
    def test_file_classification_agrees_across_formats(self):
        from cejsp import JobShopComplexityClassifier
        from cejsp.io import parse_dzn
        from test_instance_files import DZN, TAILLARD
        instances = []
        for i in range(20):
            text = DZN.replace("[2,3,5,7]", json.dumps([v * (i + 1) for v in (2, 3, 5, 7)]))
            instances.append(parse_dzn(text))
        model = JobShopComplexityClassifier(feature_backend="summary", n_estimators=10).fit_instances(
            instances, [CATEGORIES[i // 4] for i in range(20)], protocol=protocol(), model_version="synthetic-files-test")
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("a.json", "a.dzn", "a.txt")]
            for path, text in zip(paths, (json.dumps(instances[0]), DZN, TAILLARD)):
                path.write_text(text, encoding="utf-8")
            result = [model.classify(path) for path in paths]
            self.assertEqual(len({r["category"] for r in result}), 1)
            self.assertEqual(result[0]["class_probabilities"], result[1]["class_probabilities"])
            self.assertEqual(result[0]["class_probabilities"], result[2]["class_probabilities"])

    def test_labels_invariance_persistence_and_new_version(self):
        from cejsp import JobShopComplexityClassifier
        rows = [{"duration": float(i), "jobs": float(i % 3 + 1)} for i in range(20)]
        labels = [CATEGORIES[i // 4] for i in range(20)]
        model = JobShopComplexityClassifier(feature_backend="summary", n_estimators=10).fit(rows, labels, protocol=protocol(), model_version="test-v1")
        self.assertEqual(model.predict(rows[:1])[0], model.predict(rows)[0])
        self.assertEqual(set(model.predict(rows)[0]["class_probabilities"]), set(CATEGORIES))
        with tempfile.TemporaryDirectory() as temp:
            path = model.save(Path(temp) / "v1.joblib")
            with self.assertRaises(ValueError):
                model.load(path)
            self.assertEqual(model.predict(rows), model.load(path, trusted=True).predict(rows))
            new = JobShopComplexityClassifier(feature_backend="summary", n_estimators=10).fit(rows, labels, protocol=protocol(), model_version="test-v2")
            new.save(Path(temp) / "v2.joblib")
            self.assertTrue(path.exists())

    def test_missing_class_cannot_be_invented(self):
        from cejsp import JobShopComplexityClassifier
        with self.assertRaises(ValueError):
            JobShopComplexityClassifier(feature_backend="summary").fit([{"a": 1}] * 10, ["toy"] * 10, protocol=protocol(), model_version="invalid")
