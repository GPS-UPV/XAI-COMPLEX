import copy
import itertools
import math
import unittest

from xai_complex import normalize_instance, summarize_instance, SolverTargetCalibrator, status_to_complexity


def fixture(speeds=1, dates=0, flexible=False):
    data = {"nbJobs": [0, 1], "nbMchs": [0, 1], "speed": speeds, "timeEnergy": []}
    for j in range(2):
        ops = {}
        for o, m in enumerate((j, 1 - j)):
            op = {"speed-scaling": [{"procTime": 10 + j + o + s, "energyCons": 5 + s} for s in range(speeds)]}
            if dates == 2:
                op.update({"release-date": o * 20, "due-date": o * 20 + 18})
            ops[str(m)] = op
        job = {"jobId": j, "operations": ops}
        if dates == 1:
            job.update({"release-date": 0, "due-date": 100})
        if flexible:
            job["available_machines"] = [1, 1]
        data["timeEnergy"].append(job)
    return data


class InstanceTests(unittest.TestCase):
    def test_mode_contracts(self):
        for speeds, dates, flexible in itertools.product((1, 3), (0, 1, 2), (False, True)):
            with self.subTest(speeds=speeds, dates=dates, flexible=flexible):
                src = fixture(speeds, dates, flexible)
                before = copy.deepcopy(src)
                inst = normalize_instance(src)
                self.assertEqual(src, before)
                self.assertEqual(len(inst["jobs"][0]["operations"][0]["speeds"]), speeds)
                self.assertEqual(inst["instance_type"], "FJSP" if flexible else "JSP")
                self.assertTrue(all(math.isfinite(v) for v in summarize_instance(inst).values()))

    def test_ids_and_order(self):
        a = normalize_instance(fixture())
        self.assertEqual(a["instance_id"], normalize_instance(fixture())["instance_id"])
        self.assertEqual([o["machine"] for o in a["jobs"][1]["operations"]], [1, 0])
        self.assertNotEqual(a["instance_id"], normalize_instance(fixture(), metadata={"distribution": "normal"})["instance_id"])

    def test_all_speeds_affect_summary(self):
        a, b = fixture(3), fixture(3)
        b["timeEnergy"][0]["operations"]["0"]["speed-scaling"][2]["procTime"] = 1000
        self.assertNotEqual(summarize_instance(a)["processing_max"], summarize_instance(b)["processing_max"])

    def test_array_input_and_job_dates(self):
        data = {"numJobs": 2, "numMchs": 2, "speed": 2, "ProcessingTime": [[[2, 3], [4, 5]], [[5, 6], [7, 8]]],
                "EnergyConsumption": [[[1, 2], [3, 4]], [[2, 3], [4, 5]]], "Orden": [[0, 1], [1, 0]],
                "rddd": 1, "ReleaseDate": [0, 2], "DueDate": [20, 30]}
        inst = normalize_instance(data)
        self.assertEqual(inst["jobs"][1]["release-date"], 2)
        self.assertEqual(inst["jobs"][0]["operations"][0]["speeds"][1]["processing_time"], 3)

    def test_invalid_inputs(self):
        for field, value in (("procTime", -1), ("procTime", float("inf")), ("energyCons", -1)):
            src = fixture()
            src["timeEnergy"][0]["operations"]["0"]["speed-scaling"][0][field] = value
            with self.assertRaises(ValueError):
                normalize_instance(src)
        bad = fixture(flexible=True)
        bad["timeEnergy"][0]["available_machines"] = [0, 0]
        with self.assertRaises(ValueError):
            normalize_instance(bad)
        bad = fixture(dates=1)
        bad["timeEnergy"][0]["due-date"] = -1
        with self.assertRaises(ValueError):
            normalize_instance(bad)

    def test_zero_duration_operation(self):
        src = fixture()
        src["timeEnergy"][0]["operations"]["0"]["speed-scaling"][0] = {"procTime": 0, "energyCons": 0}
        inst = normalize_instance(src)
        self.assertEqual(inst["jobs"][0]["operations"][0]["speeds"][0]["processing_time"], 0)


class LabelTests(unittest.TestCase):
    def test_statuses(self):
        self.assertEqual(status_to_complexity("INFEASIBLE"), 1.)
        self.assertEqual(status_to_complexity("feasible"), .5)
        self.assertEqual(status_to_complexity("optimal"), 0.)
        self.assertTrue(math.isnan(status_to_complexity("made_up")))

    def test_missing_targets_raise(self):
        with self.assertRaises(ValueError):
            SolverTargetCalibrator().fit([{}, {}])

    def test_weights(self):
        for weights in ({}, {"q": -1}, {"q": 0}, {"q": math.inf}, {"typo": 1}):
            with self.assertRaises(ValueError):
                SolverTargetCalibrator(weights)

    def test_quality_not_normalized_by_batch(self):
        model = SolverTargetCalibrator({"q": 1}).fit([{"status": "OPTIMAL"}, {"status": "TIMEOUT"}])
        self.assertEqual(model.transform([{"status": "FEASIBLE"}]), [.5])
        self.assertEqual(model.transform([{"status": "FEASIBLE"}, {"status": "TIMEOUT"}])[0], .5)
        self.assertEqual(model.transform([{"quality_tag": None, "status": "INFEASIBLE"}]), [1.])

    def test_missing_components_renormalized(self):
        model = SolverTargetCalibrator().fit([{"status": "OPTIMAL", "solveTime": 0}, {"status": "TIMEOUT", "solveTime": 100}])
        self.assertEqual(model.transform([{"status": "FEASIBLE"}]), [.5])
        with self.assertRaises(ValueError):
            model.transform([{}])

    def test_runtime_validation_and_frozen_calibration(self):
        model = SolverTargetCalibrator({"rt": 1}).fit([{"solveTime": 0}, {"solveTime": 100}])
        self.assertEqual(model.transform([{"solveTime": 50}]), [.5])
        with self.assertRaises(ValueError):
            model.transform([{"solveTime": 1, "maxTime": 0}])


if __name__ == "__main__":
    unittest.main()
