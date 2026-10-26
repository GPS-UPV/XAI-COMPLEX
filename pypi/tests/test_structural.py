import copy
import tempfile
from pathlib import Path
import unittest

from test_contracts import fixture
from test_five_classes import protocol
from xai_complex.categories import CATEGORIES
from xai_complex.io import load_instance
from xai_complex.structural import structural_features
from xai_complex.classifier import JobShopComplexityClassifier


class StructuralTests(unittest.TestCase):
    def test_machine_and_job_relabeling_does_not_change_features(self):
        value=load_instance(fixture())
        expected=structural_features(value)
        changed=copy.deepcopy(value)
        changed['jobs'].reverse()
        for job in changed['jobs']:
            for operation in job['operations']:
                operation['machine']=1-operation['machine']
        actual=structural_features(changed)
        for name in expected:
            self.assertAlmostEqual(expected[name],actual[name],places=12)

    def test_speed_projection_and_unsupported_constraints(self):
        with self.assertRaises(ValueError):
            structural_features(fixture(speeds=3))
        self.assertTrue(structural_features(fixture(speeds=3),speed_index=2))
        with self.assertRaises(ValueError):
            structural_features(fixture(dates=1))
        with self.assertRaises(ValueError):
            structural_features(fixture(flexible=True))

    def test_actual_model_persistence_across_candidates(self):
        rows=[]
        for i in range(20):
            value=fixture()
            for job in value['timeEnergy']:
                for operation in job['operations'].values():
                    operation['speed-scaling'][0]['procTime']*=i+1
            rows.append(structural_features(value))
        labels=[CATEGORIES[i//4] for i in range(20)]
        for kind in ('random_forest','extra_trees','hist_gradient','rbf_svm','ensemble'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                model=JobShopComplexityClassifier(feature_backend='structural',estimator_kind=kind,n_estimators=5).fit(rows,labels,protocol=protocol(),model_version='test')
                path=Path(temp)/'model.joblib'
                model.save(path)
                restored=model.load(path,trusted=True)
                self.assertEqual(model.predict(rows[:2]),restored.predict(rows[:2]))
                self.assertEqual(model.predict(rows[:1])[0],model.predict(rows[:2])[0])

    def test_invalid_training_weights_are_rejected(self):
        features=[{'load':float(i)} for i in range(10)]
        labels=[label for label in CATEGORIES for _ in range(2)]
        for weights in ([1.], [1.]*9+[float('nan')], [1.]*9+[-1.]):
            with self.assertRaises(ValueError):
                JobShopComplexityClassifier(feature_backend='structural').fit(features,labels,protocol=protocol(),model_version='test',sample_weight=weights)
