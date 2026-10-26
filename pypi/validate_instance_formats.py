"""Validate readers against existing DZN and paired JSON/Taillard benchmarks."""
import argparse
import json
from pathlib import Path

from cejsp.io import load_instance


def route_times(instance):
    return [[(op["machine"], [s["processing_time"] for s in op["speeds"]]) for op in job["operations"]] for job in instance["jobs"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    passed, dzn, errors = 0, 0, []
    for path in sorted((args.root / "TaillardInstancesJSON").glob("*.json")):
        try:
            a, b = load_instance(path), load_instance(args.root / "TaillardInstances" / path.stem)
            if route_times(a) != route_times(b):
                raise ValueError("Routing or processing times differ across formats")
            passed += 1
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
    for path in sorted((args.root / "instances").glob("*.dzn")):
        try:
            load_instance(path)
            dzn += 1
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
    result = {"json_taillard_pairs_passed": passed, "dzn_passed": dzn, "errors": errors,
              "scope": "Readers and format equivalence, not classifier accuracy"}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(1 if errors or not passed or not dzn else 0)


if __name__ == "__main__":
    main()
