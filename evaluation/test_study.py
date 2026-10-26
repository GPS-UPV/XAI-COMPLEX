import unittest

from run_study import validate_solution, normalized_input, feature_records


class FeatureSchemaTests(unittest.TestCase):
    def test_train_and_test_retain_named_columns(self):
        columns = ["n_jobs", "processing_mean"]
        train = feature_records([[2., 7.], [3., 9.]], columns)
        test = feature_records([[4., 11.]], columns)
        self.assertEqual(train[0], {"n_jobs": 2., "processing_mean": 7.})
        self.assertEqual(list(train[0]), list(test[0]))
        with self.assertRaises(ValueError):
            feature_records([[2., 7.]], [0, 1])


class SolverSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.instance = (1, 2, [[2, 3]], [[1, 1]], [[1, 0]])
        self.solution = {"solution": {"start_time": [[3, 0]], "SpeedScaling": [[1, 1]], "objective": 7}}

    def test_rank_means_machine_position(self):
        self.assertEqual(validate_solution(self.solution, self.instance), "feasible_schedule_verified")
        data = normalized_input({"instance": self.instance})
        self.assertEqual(list(data["timeEnergy"][0]["operations"]), ["1", "0"])
        self.assertEqual(data["timeEnergy"][0]["operations"]["1"]["speed-scaling"][0]["procTime"], 3)

    def test_wrong_route_is_rejected(self):
        self.solution["solution"]["start_time"] = [[0, 2]]
        with self.assertRaisesRegex(ValueError, "precedence"):
            validate_solution(self.solution, self.instance)

    def test_objective_is_verified(self):
        self.solution["solution"]["objective"] = 6
        with self.assertRaisesRegex(ValueError, "Objective"):
            validate_solution(self.solution, self.instance)

    def test_missing_schedule_is_not_a_feasibility_pass(self):
        self.assertEqual(validate_solution({"solution": None}, self.instance), "no_saved_schedule")


if __name__ == "__main__":
    unittest.main()
