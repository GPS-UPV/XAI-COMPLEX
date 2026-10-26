"""Validate existing benchmark JSON without generating instances."""
import argparse
import json
from pathlib import Path

from cejsp import normalize_instance, summarize_instance

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--output", type=Path, default=Path("real-input-validation.json"))
args = parser.parse_args()
files = sorted(args.folder.glob("*.json"))
passed, errors = [], []
for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        instance = normalize_instance(data)
        summarize_instance(instance)
        passed.append({"file": path.name, "jobs": len(instance["jobs"]), "machines": instance["n_machines"], "speeds": instance["speed"]})
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append({"file": path.name, "error": str(exc)})
result = {"scope": "Input normalization and descriptive summary only; not graph or estimator execution",
          "passed": len(passed), "failed": len(errors), "instances": passed, "errors": errors}
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"passed": len(passed), "failed": len(errors)}))
raise SystemExit(1 if errors or not files else 0)
