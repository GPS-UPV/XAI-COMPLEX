"""Behavioral checks for the bundled, fixed-reference public inference API."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cejsp import CEJSP,classify,classify_many,CATEGORIES
from cejsp.io import parse_taillard
from cejsp import api


class FixedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.instance=parse_taillard('2 2\n0 2 1 3\n1 4 0 1\n')

    def test_one_argument_label_matches_detailed_report(self):
        label=CEJSP(self.instance)
        self.assertIsInstance(label,str)
        self.assertIn(label,CATEGORIES)
        self.assertEqual(label,classify(self.instance)['category'])
        for function in (CEJSP,classify,classify_many):
            with self.assertRaises(TypeError):function(self.instance,model='custom.joblib')

    def test_equivalent_formats_and_ordered_parallel_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);js=root/'instance.json';txt=root/'instance.txt';dzn=root/'instance.dzn'
            js.write_text(json.dumps(self.instance));txt.write_text('2 2\n0 2 1 3\n1 4 0 1\n')
            dzn.write_text('JOBS=1..2; MACHINES=1..2; SPEED=1; time=array3d(JOBS,MACHINES,1..SPEED,[2,3,1,4]); energy=array3d(JOBS,MACHINES,1..SPEED,[0,0,0,0]); precedence=array2d(JOBS,MACHINES,[0,1,1,0]);')
            paths=[js,dzn,txt];reports=[classify(p) for p in paths]
            self.assertEqual(reports,classify_many(paths,workers=2))
            # Content identifiers may include format-specific energy metadata;
            # every classification field must remain equal.
            predictions=[{k:v for k,v in r.items() if k!='instance_id'} for r in reports]
            self.assertEqual(predictions[0],predictions[1]);self.assertEqual(predictions[0],predictions[2])
        with self.assertRaises(ValueError):classify_many([],workers=0)

    def test_mutating_report_does_not_change_reference(self):
        expected=classify(self.instance)
        report=classify(self.instance);report['class_probabilities']['toy']=-999
        self.assertEqual(expected,classify(self.instance))

    def test_corrupted_reference_is_rejected_before_unpickling(self):
        metadata=api.files('cejsp').joinpath('data/reference.json').read_text()
        api._reference.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'data').mkdir()
                (root/'data/reference.json').write_text(metadata)
                (root/'data/reference.joblib').write_bytes(b'not-a-model')
                with patch.object(api,'files',return_value=root),self.assertRaisesRegex(RuntimeError,'integrity'):
                    CEJSP(self.instance)
        finally:api._reference.cache_clear()

    def test_bundled_reference_preserves_retained_model_predictions(self):
        root=Path(__file__).resolve().parents[2]
        model_path=root/'models/jsplib-easy-v7-trajectory/reference-accuracy.joblib'
        if not model_path.exists():
            self.fail('Run the project regression suite from the source repository')
        from xai_complex.classifier import JobShopComplexityClassifier
        original=JobShopComplexityClassifier.load(model_path,trusted=True)
        # The retained, previously validated reference is the regression oracle.
        path=root/'TaillardInstancesJSON/ft06.json'
        self.assertEqual(original.classify(path),classify(path))

    def test_public_cli_has_no_model_argument(self):
        from cejsp.cli import main
        from contextlib import redirect_stdout,redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.txt';path.write_text('2 2\n0 2 1 3\n1 4 0 1\n')
            output=io.StringIO()
            with redirect_stdout(output):main(['classify',str(path)])
            self.assertEqual(output.getvalue().strip(),CEJSP(path))
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['classify',str(path),'another.joblib'])
