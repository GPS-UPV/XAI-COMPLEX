import itertools
import unittest
import numpy as np
from xai_complex.io import parse_taillard
from xai_complex.search_features import preemptive_bound,search_features
from xai_complex.classifier import JobShopComplexityClassifier
from xai_complex.categories import CATEGORIES
from test_five_classes import protocol


class SearchFeatureTests(unittest.TestCase):
    def test_preemptive_bound_never_exceeds_enumerated_nonpreemptive_optimum(self):
        rng=np.random.default_rng(19)
        for _ in range(20):
            r=rng.integers(0,8,4);p=rng.integers(0,6,4);q=rng.integers(0,8,4)
            best=float('inf')
            for order in itertools.permutations(range(4)):
                t=0;cost=0
                for i in order:
                    t=max(t,r[i])+p[i];cost=max(cost,t+q[i])
                best=min(best,cost)
            bound=preemptive_bound(r,p,q)
            self.assertLessEqual(bound,best+1e-10)
            self.assertGreaterEqual(bound,float(np.max(r+p+q))-1e-10)

    def test_probe_proves_known_optimum_and_is_reproducible(self):
        instance=parse_taillard('2 2\n0 2 1 3\n1 4 0 1\n')
        a=search_features(instance,probe_budget=.02)
        self.assertEqual(a,search_features(instance,probe_budget=.02))
        self.assertEqual(a['probe_proved_optimal'],1.)
        self.assertEqual(a['probe_gap'],0.)
        self.assertTrue(all(np.isfinite(v) for v in a.values()))
        self.assertGreaterEqual(a['dispatch_best_gap'],0.)

    def test_probe_rejects_fractional_durations_instead_of_rounding(self):
        instance=parse_taillard('1 1\n0 1\n')
        instance['jobs'][0]['operations'][0]['speeds'][0]['processing_time']=1.25
        self.assertTrue(search_features(instance))
        with self.assertRaises(ValueError):
            search_features(instance,probe_budget=.01)

    def test_decision_adjustment_and_worker_validation(self):
        with self.assertRaises(ValueError):
            JobShopComplexityClassifier(n_jobs=0)
        with self.assertRaises(ValueError):
            JobShopComplexityClassifier(class_decision_weights={'easy':2})
        weights={c:1. for c in CATEGORIES};weights['easy']=2.
        rows=[{'x':float(i)} for i in range(20)];labels=[CATEGORIES[i//4] for i in range(20)]
        model=JobShopComplexityClassifier(feature_backend='search',estimator_kind='extra_trees',n_estimators=5,n_jobs=2,class_decision_weights=weights).fit(rows,labels,protocol=protocol(),model_version='test')
        result=model.predict(rows[:1])[0]
        self.assertAlmostEqual(sum(result['class_probabilities'].values()),1.)

    def test_ordinal_and_easy_specialist_outputs_are_persistable(self):
        import tempfile
        from pathlib import Path
        rows=[{'x':float(i),'other':float(i%3)} for i in range(25)]
        labels=[CATEGORIES[i//5] for i in range(25)]
        for kind,params in (('ordinal_forest',{}),('easy_specialist',{'specialist_kind':'forest'}),('easy_specialist',{'specialist_kind':'svm'}),('catboost',{'depth':2}),('selected_easy',{'k':1})):
            with self.subTest(kind=kind,params=params),tempfile.TemporaryDirectory() as temp:
                model=JobShopComplexityClassifier(feature_backend='search',estimator_kind=kind,estimator_params=params,n_estimators=3).fit(rows,labels,protocol=protocol(),model_version='test')
                model.save(Path(temp)/'reference.joblib')
                loaded=model.load(Path(temp)/'reference.joblib',trusted=True)
                self.assertEqual(model.predict(rows[:2]),loaded.predict(rows[:2]))
                self.assertTrue(all(np.isfinite(v) for r in loaded.predict(rows[:2]) for v in r['class_probabilities'].values()))

    def test_easy_selection_is_not_refitted_during_prediction(self):
        from xai_complex.feature_selector import SelectedEasyClassifier
        X=np.array([[float(i),float(i%3)] for i in range(25)]);y=np.array([CATEGORIES[i//5] for i in range(25)])
        model=SelectedEasyClassifier(k=1,n_estimators=3).fit(X,y)
        selected=model.selected_.copy();scores=model.selection_scores_.copy()
        model.predict_proba(np.full((3,2),1e6))
        np.testing.assert_array_equal(model.selected_,selected)
        np.testing.assert_array_equal(model.selection_scores_,scores)

    def test_critical_blocks_match_known_two_job_schedule(self):
        from xai_complex.landscape import landscape_features
        instance=parse_taillard('2 2\n0 2 1 3\n1 4 0 1\n')
        result=landscape_features(instance)
        self.assertAlmostEqual(result['critical_operation_fraction'],.5)
        self.assertAlmostEqual(result['critical_machine_fraction'],.5)
        self.assertAlmostEqual(result['critical_block_max_relative'],1.)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_trajectory_retains_an_early_optimal_proof(self):
        from xai_complex.landscape import landscape_features
        instance=parse_taillard('2 2\n0 2 1 3\n1 4 0 1\n')
        result=landscape_features(instance,probe_budget=.5)
        self.assertEqual(result['probe_proved_optimal'],1.)
        self.assertEqual(result['late_probe_proved_optimal'],1.)
        self.assertAlmostEqual(result['trajectory_gap_reduction'],0.)

    def test_zero_workload_has_zero_gaps(self):
        from xai_complex.landscape import landscape_features
        instance=parse_taillard('1 1\n0 0\n')
        result=landscape_features(instance,probe_budget=.02)
        self.assertEqual(result['dispatch_best_gap'],0.)
        self.assertEqual(result['probe_gap'],0.)
        self.assertEqual(result['late_probe_gap'],0.)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
