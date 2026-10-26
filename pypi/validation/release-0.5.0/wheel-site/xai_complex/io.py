"""Readers for existing IGJSP JSON/DZN and machine-duration benchmark files."""
import json
import math
from pathlib import Path
import re
from hashlib import sha256

from .instances import normalize_instance


def parse_jsplib(data):
    jobs, machines = data.get("jobs"), data.get("machines")
    if type(jobs) is not int or type(machines) is not int or min(jobs, machines) < 1:
        raise ValueError("Invalid JSPLIB dimensions")
    rows = [[] for _ in range(jobs)]
    ids = set()
    for operation in data["operations"]:
        j, m, step, oid = (operation[k] for k in ("job", "machine", "step", "operation"))
        duration = operation["duration"]
        if any(type(v) is not int for v in (j, m, step, oid)) or not 0 <= j < jobs or not 0 <= m < machines or step < 0 or oid in ids:
            raise ValueError("Invalid or repeated JSPLIB operation")
        if operation.get("option", 0) != 0:
            raise ValueError("Flexible operation options are not fixed-route JSP")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid JSPLIB duration")
        ids.add(oid)
        rows[j].append((step, oid, m, float(duration)))
    expected = set()
    result_jobs = []
    for j, row in enumerate(rows):
        row.sort()
        if [r[0] for r in row] != list(range(len(row))) or not row:
            raise ValueError("Each JSPLIB job needs consecutive operation steps")
        expected.update((a[1], b[1]) for a, b in zip(row, row[1:]))
        result_jobs.append({"job_id": j, "available_machines": None, "operations": [
            {"machine": m, "speeds": [{"processing_time": p, "energy": 0.0}]} for _, _, m, p in row]})
    if "precedences" in data:
        actual = {(p["before"], p["after"]) for p in data["precedences"]}
        # JSPLIB label identifies the job; before/after encode the constraint.
        if actual != expected:
            raise ValueError("Unsupported JSPLIB precedences")
    for key in ("release_dates", "due_dates", "setup", "setups"):
        if key in data:
            raise ValueError(f"Unsupported JSPLIB constraint: {key}")
    result = {"schema": "xai-complex.instance.v1", "instance_type": "JSP", "n_machines": machines,
              "speed": 1, "jobs": result_jobs, "metadata": {"energy": "not_provided"}}
    result["instance_id"] = sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def parse_dzn(text):
    text = re.sub(r"%[^\n]*", "", text)
    declarations = {}
    for statement in text.split(";"):
        if not statement.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*", statement, re.S)
        if match is None or match[1] in declarations:
            raise ValueError("Only unique literal DZN assignments are supported")
        declarations[match[1]] = match[2]
    unknown = set(declarations) - {"JOBS", "MACHINES", "SPEED", "time", "energy", "precedence"}
    if unknown:
        raise ValueError(f"Unsupported DZN fields; constraints must not be discarded: {sorted(unknown)}")
    def count(key):
        match = re.fullmatch(r"1\s*\.\.\s*([1-9]\d*)", declarations[key])
        if match is None:
            raise ValueError(f"Expected {key} = 1..N")
        return int(match[1])
    jobs, machines = count("JOBS"), count("MACHINES")
    if not re.fullmatch(r"[1-9]\d*", declarations["SPEED"]):
        raise ValueError("SPEED must be a positive integer")
    speeds = int(declarations["SPEED"])
    def array(key, dimensions, size):
        pattern = r"array" + str(len(dimensions)) + r"d\(\s*" + r"\s*,\s*".join(dimensions) + r"\s*,\s*(\[[^]]*\])\s*\)"
        match = re.fullmatch(pattern, declarations[key], re.S)
        if match is None:
            raise ValueError(f"Unsupported array syntax for {key}")
        values = json.loads(match[1])
        if len(values) != size or any(type(v) not in (int, float) for v in values):
            raise ValueError(f"Invalid values or dimensions for {key}")
        return values
    dimensions = ["JOBS", "MACHINES", r"1\s*\.\.\s*SPEED"]
    p = array("time", dimensions, jobs * machines * speeds)
    e = array("energy", dimensions, jobs * machines * speeds)
    ranks = array("precedence", ["JOBS", "MACHINES"], jobs * machines)
    records = []
    for j in range(jobs):
        rank = ranks[j * machines:(j + 1) * machines]
        if any(type(v) is not int for v in rank) or sorted(rank) not in (list(range(machines)), list(range(1, machines + 1))):
            raise ValueError("Precedence must provide one rank per machine")
        operations = {}
        for m in sorted(range(machines), key=lambda m: rank[m]):
            start = (j * machines + m) * speeds
            operations[str(m)] = {"speed-scaling": [{"procTime": p[start + s], "energyCons": e[start + s]} for s in range(speeds)]}
        records.append({"jobId": j, "operations": operations})
    return normalize_instance({"nbJobs": list(range(jobs)), "nbMchs": list(range(machines)), "speed": speeds, "timeEnergy": records})


def parse_taillard(text):
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    time_markers = [i for i, line in enumerate(lines) if re.fullmatch(r"(?:processing\s+)?times\s*:?", line, re.I)]
    if time_markers:
        time_index = time_markers[0]
        machine_markers = [i for i, line in enumerate(lines) if re.fullmatch(r"machines\s*:?", line, re.I)]
        headers = [line for line in lines[:time_index] if re.fullmatch(r"\d+(?:\s+\d+)+", line)]
        if len(time_markers) != 1 or len(machine_markers) != 1 or len(headers) != 1:
            raise ValueError("Expected one Taillard Times/Machines instance")
        header = list(map(int, headers[0].split()))
        jobs, machines = header[:2]
        mi = machine_markers[0]
        times = lines[time_index + 1:mi]
        routes = lines[mi + 1:]
        if jobs < 1 or machines < 1 or len(times) != jobs or len(routes) != jobs:
            raise ValueError("Taillard matrix dimensions differ")
        pairs = []
        for route, durations in zip(routes, times):
            r, p = route.split(), durations.split()
            if len(r) != machines or len(p) != machines:
                raise ValueError("Invalid Taillard matrix row")
            pairs.append(" ".join(value for pair in zip(r, p) for value in pair))
        return parse_taillard(f"{jobs} {machines}\n" + "\n".join(pairs))
    if not lines or not re.fullmatch(r"[1-9]\d*\s+[1-9]\d*", lines[0]):
        raise ValueError("Expected jobs machines header in machine-duration pair format")
    jobs, machines = map(int, lines[0].split())
    if len(lines) != jobs + 1:
        raise ValueError("Expected exactly one operation row per job")
    records = []
    for j, line in enumerate(lines[1:]):
        values = [int(value) for value in line.split()]
        if len(values) != machines * 2:
            raise ValueError("Each job needs machine-duration pairs")
        route, times = values[::2], values[1::2]
        if sorted(route) == list(range(1, machines + 1)):
            route = [m - 1 for m in route]
        if sorted(route) != list(range(machines)):
            raise ValueError("Machine route must be a zero- or one-based permutation")
        records.append({"jobId": j, "operations": {str(m): {"speed-scaling": [{"procTime": p, "energyCons": 0}]} for m, p in zip(route, times)}})
    return normalize_instance({"nbJobs": list(range(jobs)), "nbMchs": list(range(machines)), "speed": 1, "timeEnergy": records}, metadata={"energy": "not_provided"})


def load_instance(source, *, format=None):
    if not isinstance(source, (str, Path)):
        if isinstance(source, dict) and source.get("schema") == "xai-complex.instance.v1":
            return source
        if isinstance(source, dict) and isinstance(source.get("operations"), list) and "jobs" in source and "machines" in source:
            return parse_jsplib(source)
        return normalize_instance(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8-sig")
    kind = format or ("json" if path.suffix.lower() == ".json" else "dzn" if path.suffix.lower() == ".dzn" else "taillard")
    if kind == "json":
        return load_instance(json.loads(text))
    if kind == "dzn":
        return parse_dzn(text)
    if kind == "taillard":
        return parse_taillard(text)
    raise ValueError("format must be json, dzn or taillard")
