import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import xai_complex


class SourceRegressionTests(unittest.TestCase):
    def test_actual_research_status_function(self):
        source = Path(xai_complex.__file__).parent / "research.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_status_to_complexity")
        namespace = {"np": SimpleNamespace(nan=float("nan"))}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(namespace[fn.name]("INFEASIBLE"), 1.)
        self.assertEqual(namespace[fn.name]("FEASIBLE"), .5)

    def test_package_has_no_generator(self):
        root = Path(xai_complex.__file__).parent
        for source in root.glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("IGJSP"), str(source))
                if isinstance(node, ast.FunctionDef):
                    self.assertNotEqual(node.name, "generate_instances")

    def test_all_modules_compile(self):
        root = Path(xai_complex.__file__).parent
        for source in root.glob("*.py"):
            compile(source.read_text(encoding="utf-8"), str(source), "exec")

    def test_no_global_warning_suppression(self):
        root = Path(xai_complex.__file__).parent
        for source in root.glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    self.assertNotEqual(ast.unparse(node.value.func), "warnings.filterwarnings")
