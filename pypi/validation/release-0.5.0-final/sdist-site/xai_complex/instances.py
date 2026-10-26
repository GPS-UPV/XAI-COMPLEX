"""Lossless adapter for IGJSP arrays and its public JSON representation."""

from hashlib import sha256
import json
import math
import numbers
import statistics


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _levels(times, energies):
    times, energies = _plain(times), _plain(energies)
    times = times if isinstance(times, (list, tuple)) else [times]
    energies = energies if isinstance(energies, (list, tuple)) else [energies]
    if not times or len(times) != len(energies):
        raise ValueError("Processing time and energy speed counts differ")
    out = []
    for p, e in zip(times, energies):
        p, e = _number(p, "processing_time"), _number(e, "energy")
        if p < 0 or e < 0:
            raise ValueError("Processing time and energy must be nonnegative")
        out.append({"processing_time": p, "energy": e})
    return out


def _dates(record):
    out = {}
    for key in ("release-date", "due-date"):
        if key in record:
            out[key] = _number(record[key], key)
            if out[key] < 0:
                raise ValueError("Dates must be nonnegative")
    if len(out) == 2 and out["due-date"] < out["release-date"]:
        raise ValueError("Due date precedes release date")
    return out


def _array_date(data, keys, j, o, mode):
    for key in keys:
        values = _plain(data.get(key))
        if values is None or values == []:
            continue
        row = _plain(values[j])
        if isinstance(row, (tuple, list)):
            return row[0 if mode == 1 else o]
        return row
    return None


def normalize_instance(instance, *, instance_type=None, metadata=None):
    """Preserve routing, speed levels and dates; never assume flexible machines."""
    data = instance if isinstance(instance, dict) else vars(instance)
    metadata = dict(metadata or {})
    if data.get("schema") == "xai-complex.instance.v1":
        raise ValueError("Input must be an IGJSP object or JSON, not an already normalized instance")
    flexible = any(k in data for k in ("AllowedMachines", "allowed_machines", "available_machines", "AVJM", "avail_machines", "machine_mask"))
    flexible |= any("available_machines" in j for j in data.get("timeEnergy", []))
    kind = instance_type or data.get("instance_type") or ("FJSP" if flexible else "JSP")
    if kind not in ("JSP", "FJSP"):
        raise ValueError("instance_type must be JSP or FJSP")
    jobs = []
    if "timeEnergy" in data:
        machine_ids = _plain(data["nbMchs"])
        declared_jobs = _plain(data["nbJobs"])
        if not isinstance(machine_ids, list) or machine_ids != list(range(len(machine_ids))):
            raise ValueError("Machine IDs must be zero-based consecutive integers")
        m = len(machine_ids)
        for rec in data["timeEnergy"]:
            ops = []
            for machine, op in rec["operations"].items():
                levels = op["speed-scaling"]
                speeds = _levels([s["procTime"] for s in levels], [s["energyCons"] for s in levels])
                ops.append({"machine": int(machine), "speeds": speeds, **_dates(op)})
            jobs.append({"job_id": rec["jobId"], "operations": ops,
                         "available_machines": rec.get("available_machines"), **_dates(rec)})
        if declared_jobs != list(range(len(jobs))):
            raise ValueError("Declared jobs do not match records")
    else:
        n = _integer(data["numJobs"], "numJobs", 1)
        m = _integer(data["numMchs"], "numMchs", 1)
        p, e, order = (_plain(data[k]) for k in ("ProcessingTime", "EnergyConsumption", "Orden"))
        if any(len(a) != n for a in (p, e, order)):
            raise ValueError("Array job counts differ")
        availability = next((_plain(data[k]) for k in ("AllowedMachines", "allowed_machines", "available_machines", "AVJM") if k in data), None)
        if availability is not None and len(availability) != n:
            raise ValueError("Machine availability must have one row per job")
        mode = metadata.get("dates", data.get("rddd", data.get("ReleaseDateDueDate", 0)))
        for j in range(n):
            if not order[j] or len(p[j]) != len(order[j]) or len(e[j]) != len(order[j]):
                raise ValueError("Array operation counts differ")
            rec = {"job_id": j, "operations": [], "available_machines": availability[j] if availability is not None else None}
            for o, machine in enumerate(order[j]):
                op = {"machine": machine, "speeds": _levels(p[j][o], e[j][o])}
                dates = {}
                if mode in (1, 2):
                    for key, aliases in (("release-date", ("ReleaseDate", "Release", "R")), ("due-date", ("DueDate", "Due", "D"))):
                        v = _array_date(data, aliases, j, o, mode)
                        if v is not None:
                            dates[key] = v
                if mode == 1:
                    rec.update(_dates(dates))
                elif mode == 2:
                    op.update(_dates(dates))
                rec["operations"].append(op)
            jobs.append(rec)
    _integer(m, "machines", 1)
    if not jobs:
        raise ValueError("Instance has no jobs")
    speeds = _integer(data.get("speed", len(jobs[0]["operations"][0]["speeds"])), "speed", 1)
    for j, job in enumerate(jobs):
        if _integer(job["job_id"], "job_id") != j or not job["operations"]:
            raise ValueError("Job IDs must be consecutive and jobs nonempty")
        for op in job["operations"]:
            machine = _integer(op["machine"], "machine")
            if machine >= m or len(op["speeds"]) != speeds:
                raise ValueError("Machine index or speed count inconsistent")
        allowed = _plain(job["available_machines"])
        if kind == "FJSP":
            if allowed is None or len(allowed) != m or any(v not in (0, 1) for v in allowed) or not any(allowed):
                raise ValueError("FJSP requires an explicit nonempty binary availability vector per job")
            job["available_machines"] = [int(v) for v in allowed]
        elif allowed is not None:
            raise ValueError("JSP cannot carry flexible availability; use instance_type='FJSP'")
    result = {"schema": "xai-complex.instance.v1", "instance_type": kind,
              "n_machines": m, "speed": speeds, "jobs": jobs, "metadata": metadata}
    payload = json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":"))
    result["instance_id"] = sha256(payload.encode()).hexdigest()
    return result


def summarize_instance(instance):
    """Compact v1 features include every speed and preserve flexible availability."""
    inst = instance if isinstance(instance, dict) and instance.get("schema") == "xai-complex.instance.v1" else normalize_instance(instance)
    ops = [o for j in inst["jobs"] for o in j["operations"]]
    p = [s["processing_time"] for o in ops for s in o["speeds"]]
    e = [s["energy"] for o in ops for s in o["speeds"]]
    windows = [r["due-date"] - r["release-date"] for j in inst["jobs"] for r in [j, *j["operations"]] if "due-date" in r and "release-date" in r]
    n, m = len(inst["jobs"]), inst["n_machines"]
    loads = [sum(min(s["processing_time"] for s in o["speeds"]) for o in ops if o["machine"] == k) for k in range(m)]
    features = {"n_jobs": n, "n_machines": m, "n_operations": len(ops), "speed": inst["speed"],
                "is_flexible": float(inst["instance_type"] == "FJSP"), "window_count": len(windows)}
    for name, values in (("processing", p), ("energy", e), ("nominal_load", loads), ("window", windows)):
        features.update({f"{name}_min": min(values) if values else 0.0,
                         f"{name}_max": max(values) if values else 0.0,
                         f"{name}_mean": statistics.fmean(values) if values else 0.0,
                         f"{name}_std": statistics.pstdev(values) if values else 0.0})
    features["availability_fraction"] = statistics.fmean(sum(j["available_machines"]) / m for j in inst["jobs"]) if inst["instance_type"] == "FJSP" else 1 / m
    return {k: float(v) for k, v in features.items()}


