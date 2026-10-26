import tempfile
from pathlib import Path
import unittest

from cejsp.io import load_instance, parse_dzn, parse_taillard


DZN = """JOBS=1..2; MACHINES=1..2; SPEED=1;
time=array3d(JOBS,MACHINES,1..SPEED,[2,3,5,7]);
energy=array3d(JOBS,MACHINES,1..SPEED,[0,0,0,0]);
precedence=array2d(JOBS,MACHINES,[1,0,0,1]);"""
TAILLARD = "2 2\n1 3 0 2\n0 5 1 7\n"


class FileReaderTests(unittest.TestCase):
    def test_jsplib_native_json_does_not_import_difficulty(self):
        source = {"jobs": 2, "machines": 2, "difficulty": "open", "instance": "ignored-name",
                  "operations": [{"operation": 0, "job": 0, "step": 0, "option": 0, "machine": 1, "duration": 0},
                                 {"operation": 1, "job": 0, "step": 1, "option": 0, "machine": 0, "duration": 5},
                                 {"operation": 2, "job": 1, "step": 0, "option": 0, "machine": 0, "duration": 2},
                                 {"operation": 3, "job": 1, "step": 1, "option": 0, "machine": 1, "duration": 3}],
                  "precedences": [{"before": 0, "after": 1, "label": 0}, {"before": 2, "after": 3, "label": 1}]}
        result = load_instance(source)
        self.assertNotIn("difficulty", result)
        self.assertNotIn("instance", result)
        self.assertEqual(result["jobs"][0]["operations"][0]["machine"], 1)
        source["operations"][1]["option"] = 1
        with self.assertRaises(ValueError):
            load_instance(source)

    def test_route_and_times_agree_across_formats(self):
        a, b = parse_dzn(DZN), parse_taillard(TAILLARD)
        self.assertEqual(a["jobs"], b["jobs"])
        self.assertEqual(a["jobs"][0]["operations"][0]["machine"], 1)
        self.assertEqual(a["jobs"][0]["operations"][0]["speeds"][0]["processing_time"], 3)

    def test_zero_duration_and_one_based_ids(self):
        value = parse_taillard("1 2\n2 0 1 5\n")
        self.assertEqual(value["jobs"][0]["operations"][0]["speeds"][0]["processing_time"], 0)

    def test_unknown_constraints_and_extra_rows_rejected(self):
        with self.assertRaises(ValueError):
            parse_dzn(DZN + "release=[1,2];")
        with self.assertRaises(ValueError):
            parse_taillard(TAILLARD + "0 1 1 2\n")

    def test_path_without_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example"
            path.write_text(TAILLARD, encoding="utf-8")
            self.assertEqual(load_instance(path)["n_machines"], 2)

    def test_dzn_keeps_all_speeds(self):
        data = DZN.replace("SPEED=1", "SPEED=2").replace("[2,3,5,7]", "[2,3,4,5,6,7,8,9]").replace("[0,0,0,0]", "[0,0,0,0,0,0,0,0]")
        self.assertEqual(parse_dzn(data)["speed"], 2)

    def test_original_times_and_machines_layout(self):
        text = "number of jobs, machines, seed, upper bound, lower bound:\n2 2 123 15 12\nTimes:\n3 2\n5 7\nMachines:\n2 1\n1 2\n"
        self.assertEqual(parse_taillard(text)["jobs"], parse_taillard(TAILLARD)["jobs"])
