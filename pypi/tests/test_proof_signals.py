"""Regression checks for proof signals and ordered class probabilities."""
import itertools
import unittest
import numpy as np
from xai_complex.proof_features import proof_probe
from xai_complex.cumulative import CumulativeDifficultyClassifier
from xai_complex.categories import CATEGORIES


class ProofSignalTests(unittest.TestCase):
    def test_proof_matches_enumerated_two_job_optima(self):
        # With two machines and two jobs enumerate machine orders, independently
        # propagate the resulting DAG, and discard cyclic orientations.
        for routes in itertools.product(((0, 1), (1, 0)), repeat=2):
            p = np.array([[2, 3], [4, 1]])
            r = np.array(routes)
            best = float('inf')
            for orders in itertools.product((0, 1), repeat=2):
                edges = [(0, 1), (2, 3)]
                for machine, reverse in enumerate(orders):
                    nodes = [j*2 + int(np.flatnonzero(r[j] == machine)[0]) for j in range(2)]
                    if reverse:
                        nodes.reverse()
                    edges.append(tuple(nodes))
                finish = np.zeros(4)
                remaining = set(range(4))
                while remaining:
                    ready = [v for v in remaining if all(u not in remaining for u,w in edges if w == v)]
                    if not ready:
                        break
                    for v in ready:
                        finish[v] = max((finish[u] for u,w in edges if w == v), default=0) + p.ravel()[v]
                        remaining.remove(v)
                if not remaining:
                    best = min(best, finish.max())
            result = proof_probe(r, p, 5., float(p.sum()), .1)
            self.assertEqual(result['proved'], 1.)
            recovered = p.sum() * (1-result['incumbent_gain'])
            self.assertAlmostEqual(recovered, best)
            self.assertEqual(result['gap'], 0.)
            self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_proof_is_repeatable_and_zero_safe(self):
        route = np.array([[0, 1], [1, 0]])
        p = np.array([[2, 3], [4, 1]])
        a = proof_probe(route, p, 5., 10., .1, strong=True)
        self.assertEqual(a, proof_probe(route, p, 5., 10., .1, strong=True))
        zero = proof_probe(np.array([[0]]), np.array([[0]]), 0., 0., .1)
        self.assertEqual(zero['gap'], 0.)
        self.assertEqual(zero['proved'], 1.)
        self.assertTrue(all(np.isfinite(v) for v in zero.values()))
        with self.assertRaises(ValueError):
            proof_probe(route, p+.5, 5., 10., .1)

    def test_crossing_cumulative_probabilities_are_valid(self):
        result = CumulativeDifficultyClassifier.ordered_probabilities([[.2,.8,.1,.4], [1.,.8,.5,0.]])
        self.assertTrue((result >= 0).all())
        np.testing.assert_allclose(result.sum(axis=1), 1.)
        np.testing.assert_allclose(result[1], [0., .2, .3, .5, 0.])

    def test_cumulative_model_persistence_and_class_order(self):
        import joblib
        import tempfile
        from pathlib import Path
        X = np.arange(30.)[:,None]
        y = np.array([CATEGORIES[i//6] for i in range(30)])
        model = CumulativeDifficultyClassifier(n_estimators=10).fit(X, y)
        expected = model.predict_proba(X[:4])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.joblib'
            joblib.dump(model,path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(expected, loaded.predict_proba(X[:4]))
            self.assertEqual(list(loaded.classes_), sorted(CATEGORIES))
            np.testing.assert_allclose(expected.sum(axis=1), 1.)

class BottleneckBoundTests(unittest.TestCase):
    def test_machine_relaxation_bound_does_not_exceed_enumerated_optimum(self):
        from xai_complex.bottleneck import machine_relaxation
        r=np.array([0,2,1]);p=np.array([2,3,1]);q=np.array([3,0,4])
        exact=float('inf')
        for order in itertools.permutations(range(3)):
            time=0;objective=0
            for i in order:
                time=max(time,r[i])+p[i];objective=max(objective,time+q[i])
            exact=min(exact,objective)
        result=machine_relaxation(r,p,q,.1)
        self.assertLessEqual(result['lower_bound'],exact)
        self.assertEqual(result['proved'],1.)
        self.assertAlmostEqual(result['lower_bound'],exact)

    def test_bottleneck_zero_workload_is_finite(self):
        from xai_complex.bottleneck import relaxation_features
        from xai_complex.io import parse_taillard
        values=relaxation_features(parse_taillard('1 1\n0 0\n'))
        self.assertTrue(all(np.isfinite(v) for v in values.values()))
        self.assertEqual(values['bottleneck_bound_gain'],0.)
        self.assertEqual(values['bottleneck_dispatch_gap'],0.)

class PublicProofBackendTests(unittest.TestCase):
    def test_proof_backend_and_cumulative_model_round_trip(self):
        import tempfile
        from pathlib import Path
        from xai_complex.classifier import JobShopComplexityClassifier
        from xai_complex.io import parse_taillard
        protocol={'id':'test-proof-v1','class_definitions':{c:c for c in CATEGORIES}}
        model=JobShopComplexityClassifier(feature_backend='proof',estimator_kind='cumulative',probe_budget=.02,n_estimators=5)
        instance=parse_taillard('2 2\n0 2 1 3\n1 4 0 1\n')
        row=model._features(instance)
        self.assertIn('proof_proved',row)
        self.assertIn('disjunctive_proved',row)
        rows=[dict(row, duration_mean=float(i+1)) for i in range(15)]
        labels=[CATEGORIES[i//3] for i in range(15)]
        model.fit(rows,labels,protocol=protocol,model_version='test-proof')
        self.assertIn('ortools',model.dependencies_)
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'reference.joblib'
            model.save(target)
            loaded=model.load(target,trusted=True)
            self.assertEqual(model.classify(instance),loaded.classify(instance))

class SubsetVotingTests(unittest.TestCase):
    def test_views_and_class_alignment_survive_public_model_persistence(self):
        import tempfile
        from pathlib import Path
        from xai_complex.classifier import JobShopComplexityClassifier
        rows=[{'first':float(i),'second':float(i%4)} for i in range(25)]
        labels=[CATEGORIES[i//5] for i in range(25)]
        members=[{'indices':[0],'config':{'kind':'et','fraction':1.,'balanced':False}},
                 {'indices':[1],'config':{'kind':'hist'}}]
        protocol={'id':'ensemble-test','class_definitions':{c:c for c in CATEGORIES}}
        model=JobShopComplexityClassifier(feature_backend='proof',estimator_kind='subset_voting',estimator_params={'members':members})
        model.fit(rows,labels,protocol=protocol,model_version='ensemble-test',sample_weight=np.ones(25))
        prediction=model.predict(rows[:2])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.joblib';model.save(path)
            loaded=model.load(path,trusted=True)
            self.assertEqual(prediction,loaded.predict(rows[:2]))
        for result in prediction:
            self.assertAlmostEqual(sum(result['class_probabilities'].values()),1.)
