"""Fail publication validation if tests fail or are skipped."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print("Release validation blocked: skipped tests", file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
