"""Reproducible CEJSP study on existing DZN inputs and saved solver runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time


def read_dzn(path):
    text = path.read_text(encoding="utf-8-sig")
    def scalar(pattern):
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Missing DZN declaration: {pattern}")
        return int(match[1])
    jobs = scalar(r"JOBS\s*=\s*1\.\.(\d+)")
    machines = scalar(r"MACHINES\s*=\s*1\.\.(\d+)")
    speeds = scalar(r"SPEED\s*=\s*(\d+)")
    if speeds != 1:
        raise ValueError("This study requires a single speed")
    def vector(key):
        match = re.search(r"\b" + key + r"\s*=\s*array[23]d\([^;]*?\[([^]]*)\]", text)
        if match is None:
            raise ValueError(f"Missing DZN array: {key}")
        values = json.loads("[" + match[1] + "]")
        if len(values) != jobs * machines or any(type(v) is not int for v in values):
            raise ValueError(f"Invalid array: {key}")
        return [values[j * machines:(j + 1) * machines] for j in range(jobs)]
    durations, energy, ranks = vector("time"), vector("energy"), vector("precedence")
    if any(v < 0 for row in durations + energy for v in row):
        raise ValueError("Negative duration or energy")
    if any(sorted(row) != list(range(machines)) for row in ranks):
        raise ValueError("Precedence ranks are not permutations")
    return jobs, machines, durations, energy, ranks


def validate_solution(data, instance):
    jobs, machines, durations, energy, ranks = instance
    solution = data.get("solution")
    if not isinstance(solution, dict):
        return "no_saved_schedule"
    starts, speeds = solution.get("start_time"), solution.get("SpeedScaling")
    if starts is None or speeds is None or len(starts) != jobs or len(speeds) != jobs:
        raise ValueError("Missing schedule arrays")
    for j in range(jobs):
        if len(starts[j]) != machines or len(speeds[j]) != machines:
            raise ValueError("Wrong schedule dimensions")
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) or s < 0 for s in starts[j]):
            raise ValueError("Invalid start time")
        if any(v != 1 for v in speeds[j]):
            raise ValueError("Unexpected speed selection")
        order = sorted(range(machines), key=lambda m: ranks[j][m])
        for a, b in zip(order, order[1:]):
            if starts[j][b] < starts[j][a] + durations[j][a]:
                raise ValueError("Job precedence violation")
    for m in range(machines):
        order = sorted(range(jobs), key=lambda j: starts[j][m])
        for a, b in zip(order, order[1:]):
            if starts[b][m] < starts[a][m] + durations[a][m]:
                raise ValueError("Machine overlap")
    makespan = max(starts[j][m] + durations[j][m] for j in range(jobs) for m in range(machines))
    expected = makespan + sum(map(sum, energy))
    if solution.get("objective") != expected:
        raise ValueError("Objective differs from makespan plus energy")
    return "feasible_schedule_verified"


def describe(instance):
    jobs, machines, p, energy, ranks = instance
    values = [v for row in p for v in row]
    loads = [sum(p[j][m] for j in range(jobs)) for m in range(machines)]
    return {"n_jobs": jobs, "n_machines": machines, "n_operations": jobs * machines,
            "processing_mean": statistics.fmean(values), "processing_std": statistics.pstdev(values),
            "processing_min": min(values), "processing_max": max(values),
            "machine_load_max": max(loads), "machine_load_std": statistics.pstdev(loads),
            "job_load_max": max(map(sum, p))}


def load_records(root):
    records, rejected = [], []
    for path in sorted((root / "instances").glob("*.dzn")):
        if not re.fullmatch(r"\d+_\d+-0-1", path.stem):
            continue
        solution_path = root / "solutions" / (path.stem + "-cp-sat.json")
        if not solution_path.exists():
            rejected.append({"id": path.stem, "reason": "missing_solver_run"})
            continue
        try:
            instance = read_dzn(path)
            data = json.loads(solution_path.read_text(encoding="utf-8-sig"))
            schedule_status = validate_solution(data, instance)
            stats = data.get("statistics", {})
            runtime, limit = float(stats["solveTime"]), float(stats["maxTime"])
            if not all(map(math.isfinite, (runtime, limit))) or runtime < 0 or limit <= 0:
                raise ValueError("Invalid runtime or limit")
            labels = dict(stats, status=data.get("status", "UNKNOWN"))
            objective, bound = stats.get("objective"), stats.get("objectiveBound")
            if objective is not None and bound is not None:
                labels["gap_rel"] = max(0, (objective - bound) / max(abs(objective), 1e-12))
            base = hashlib.sha256(json.dumps([instance[0], instance[1], instance[2], instance[4]], separators=(",", ":")).encode()).hexdigest()
            records.append({"id": path.stem, "base": base, "instance": instance, "labels": labels,
                            "schedule_check": schedule_status, "descriptors": describe(instance),
                            "dzn_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "solver_sha256": hashlib.sha256(solution_path.read_bytes()).hexdigest()})
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            rejected.append({"id": path.stem, "reason": str(exc)})
    return records, rejected


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_records(matrix, columns):
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("Feature columns must be named")
    records = []
    for row in matrix:
        if len(row) != len(columns):
            raise ValueError("Feature values and names differ in length")
        records.append(dict(zip(columns, row)))
    return records


def normalized_input(record):
    jobs, machines, p, energy, ranks = record["instance"]
    return {"nbJobs": list(range(jobs)), "nbMchs": list(range(machines)), "speed": 1,
            "timeEnergy": [{"jobId": j, "operations": {
                str(m): {"speed-scaling": [{"procTime": p[j][m], "energyCons": energy[j][m]}]}
                for m in sorted(range(machines), key=lambda m: ranks[j][m])}}
                for j in range(jobs)]}


def metrics(y, prediction):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rho = float(spearmanr(y, prediction).statistic) if np.ptp(prediction) > 0 and np.ptp(y) > 0 else None
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)), "spearman": rho}


def numpy_baselines(records, out, seeds):
    import numpy as np
    x = np.array([[r["instance"][0], r["instance"][1], r["instance"][0] * r["instance"][1]] for r in records], dtype=float)
    x = np.log1p(x)
    raw = np.array([r["labels"]["solveTime"] / r["labels"]["maxTime"] for r in records])
    y = np.clip(raw, 0, 1)
    groups = np.array([r["base"] for r in records])
    unique = np.unique(groups)
    rows = []
    for seed in seeds:
        chunks = np.array_split(np.random.default_rng(seed).permutation(unique), 5)
        for fold, chunk in enumerate(chunks):
            test = np.flatnonzero(np.isin(groups, chunk))
            train = np.flatnonzero(~np.isin(groups, chunk))
            center, scale = x[train].mean(axis=0), x[train].std(axis=0)
            scale[scale == 0] = 1
            train_x, test_x = (x[train] - center) / scale, (x[test] - center) / scale
            mean_y = y[train].mean()
            coef = np.linalg.solve(train_x.T @ train_x + np.eye(x.shape[1]), train_x.T @ (y[train] - mean_y))
            for name, predicted in (("median", np.full(len(test), np.median(y[train]))),
                                    ("ridge_log_size", test_x @ coef + mean_y)):
                for i, value in zip(test, predicted):
                    rows.append({"id": records[i]["id"], "seed": seed, "fold": fold, "model": name, "actual": float(y[i]), "predicted": float(value)})
    write_csv(out / "size_baseline_predictions.csv", rows)
    summaries = []
    for name in ("median", "ridge_log_size"):
        selected = [r for r in rows if r["model"] == name]
        actual = np.array([r["actual"] for r in selected])
        predicted = np.array([r["predicted"] for r in selected])
        summaries.append({"model": name, "mae": float(abs(actual - predicted).mean()),
                          "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                          "r2": float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))})
    write_csv(out / "size_baseline_metrics.csv", summaries)
    status_counts = {}
    for r in records:
        key = r["labels"]["status"]
        status_counts[key] = status_counts.get(key, 0) + 1
    summary = {"n": len(records), "base_instances": len(unique), "seeds": seeds, "folds": 5,
               "target": "clip(solveTime/maxTime,0,1)", "runtime_fraction_quantiles": np.quantile(raw, [0, .25, .5, .75, 1]).tolist(),
               "runtime_over_limit_count": int(np.sum(raw > 1)), "statuses": status_counts,
               "metrics": summaries, "conclusion_scope": "Baselines only; no CEJSP accuracy or speedup claim"}
    (out / "size_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def full_study(records, out, args):
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import KFold, cross_validate, cross_val_predict
    from cejsp import ComplexityEstimator, SolverTargetCalibrator
    from cejsp.graphs import extract_instance_features
    from threadpoolctl import threadpool_limits

    subset = [r for r in records if max(r["instance"][:2]) <= args.max_size]
    if len(subset) < 40:
        raise ValueError("Need at least 40 verified records in the graph cohort")
    began = time.perf_counter()
    features = []
    for i, row in enumerate(subset):
        features.append(extract_instance_features(normalized_input(row)))
        if (i + 1) % 20 == 0:
            print(f"Graph features: {i + 1}/{len(subset)}", flush=True)
    extraction_seconds = time.perf_counter() - began
    frame = pd.DataFrame(features)
    cols = [c for c in frame if "energy" not in c.lower() and c not in ("num_nodes", "graph_density", "seed")
            and "betweenness_c_" not in c
            and not (("deg" in c or "clustering" in c) and "_c_" not in c and "_d_" not in c)]
    x = frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    size = np.array([[r["instance"][0], r["instance"][1], r["instance"][0] * r["instance"][1]] for r in subset], dtype=float)
    groups = np.array([r["base"] for r in subset])
    unique = np.unique(groups)
    labels = [r["labels"] for r in subset]
    raw = np.array([r["labels"]["solveTime"] / r["labels"]["maxTime"] for r in subset])
    predictions, timings, splits_saved = [], [], []
    splits = []
    for seed in args.seeds:
        shuffled = np.random.default_rng(seed).permutation(unique)
        for fold, test_groups in enumerate(np.array_split(shuffled, 5)):
            test = np.flatnonzero(np.isin(groups, test_groups))
            train = np.flatnonzero(~np.isin(groups, test_groups))
            splits.append((f"grouped_{seed}_{fold}", seed, train, test))
    threshold = float(np.quantile(size[:, 2], .8))
    splits.append(("size_extrapolation", args.seeds[0], np.flatnonzero(size[:, 2] <= threshold), np.flatnonzero(size[:, 2] > threshold)))
    with threadpool_limits(limits=1):
        for split_name, seed, train, test in splits:
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Base instance leakage")
            splits_saved.append({"split": split_name, "train": [subset[i]["id"] for i in train], "test": [subset[i]["id"] for i in test]})
            keep = np.isfinite(x[train]).any(axis=0)
            train_x, test_x = x[train][:, keep], x[test][:, keep]
            retained_columns = [column for column, retained in zip(cols, keep) if retained]
            for protocol in ("runtime_fraction", "weighted_without_model_size"):
                if protocol == "runtime_fraction":
                    y = np.clip(raw, 0, 1)
                else:
                    calibrator = SolverTargetCalibrator({"rt": .35, "gap": .20, "fail": .15, "prop": .10, "q": .10}).fit([labels[i] for i in train])
                    y = np.array(calibrator.transform(labels))
                def forest():
                    return make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), RandomForestRegressor(n_estimators=args.trees, max_features="sqrt", n_jobs=1, random_state=seed))
                models = [("median", DummyRegressor(strategy="median"), size), ("size_rf", forest(), size), ("graph_rf", forest(), None)]
                for name, model, matrix in models:
                    a, b = (train_x, test_x) if matrix is None else (matrix[train], matrix[test])
                    start = time.perf_counter()
                    model.fit(a, y[train])
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    pred = model.predict(b)
                    timings.append({"split": split_name, "target": protocol, "model": name, "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - start, "test_n": len(test)})
                    for i, value in zip(test, pred):
                        predictions.append({"id": subset[i]["id"], "split": split_name, "target": protocol, "model": name, "actual": float(y[i]), "predicted": float(value)})
                start = time.perf_counter()
                model = ComplexityEstimator(random_state=seed, n_estimators=args.trees).fit(feature_records(train_x, retained_columns), y[train])
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                pred = model.predict(feature_records(test_x, retained_columns))
                timings.append({"split": split_name, "target": protocol, "model": "cejsp_inductive", "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - start, "test_n": len(test)})
                for i, value in zip(test, pred):
                    predictions.append({"id": subset[i]["id"], "split": split_name, "target": protocol, "model": "cejsp_inductive", "actual": float(y[i]), "predicted": value["complexity_0_1"]})
            print(f"Finished {split_name}", flush=True)
        table = pd.DataFrame(predictions)
        summaries = []
        for (target, model, split), data in table.groupby(["target", "model", "split"]):
            summaries.append(dict(target=target, model=model, split=split, **metrics(data.actual, data.predicted)))
        table.to_csv(out / "predictions.csv", index=False)
        write_csv(out / "metrics.csv", summaries)
        write_csv(out / "timings.csv", timings)
        (out / "splits.json").write_text(json.dumps(splits_saved, indent=2), encoding="utf-8")
        intervals = []
        grouped = table[table.split.str.startswith("grouped_")].copy()
        grouped["absolute_error"] = abs(grouped.actual - grouped.predicted)
        for target, data in grouped.groupby("target"):
            errors = data.groupby(["id", "model"]).absolute_error.mean().unstack()
            for model in ("graph_rf", "cejsp_inductive"):
                delta = (errors["size_rf"] - errors[model]).to_numpy()
                rng = np.random.default_rng(2026)
                boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
                intervals.append({"target": target, "model": model, "contrast": "size_rf_MAE_minus_model_MAE", "mean_improvement": float(delta.mean()), "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)), "unit": "base_instance_after_averaging_repeats"})
        write_csv(out / "paired_intervals.csv", intervals)

        y = np.clip(raw, 0, 1)
        cv = list(KFold(5, shuffle=True, random_state=42).split(x))
        bench_x = SimpleImputer(strategy="median").fit_transform(x)
        def rf():
            return RandomForestRegressor(n_estimators=args.trees, random_state=42, max_features="sqrt", n_jobs=1)
        def duplicate():
            cross_validate(rf(), bench_x, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
            return cross_val_predict(rf(), bench_x, y, cv=cv, n_jobs=1)
        def reuse():
            from sklearn.metrics import mean_absolute_error
            pred = np.empty(len(y))
            for train, test in cv:
                model = rf().fit(bench_x[train], y[train])
                pred[test] = model.predict(bench_x[test])
                mean_absolute_error(y[test], pred[test])
            return pred
        duplicate()
        reuse()
        bench = []
        for repeat in range(args.repeats):
            result = {}
            for name, operation in (("duplicate", duplicate), ("reuse", reuse))[::1 if repeat % 2 == 0 else -1]:
                start = time.perf_counter()
                pred = operation()
                result[name] = time.perf_counter() - start
                result[name + "_prediction"] = pred
            difference = float(np.max(abs(result["duplicate_prediction"] - result["reuse_prediction"])))
            if difference > 1e-12:
                raise AssertionError("Performance comparison changed predictions")
            bench.append({"repeat": repeat, "duplicate_seconds": result["duplicate"], "reuse_seconds": result["reuse"], "speedup": result["duplicate"] / result["reuse"], "max_prediction_difference": difference})
        write_csv(out / "controlled_performance.csv", bench)
    import importlib.metadata
    report = {"cohort_n": len(subset), "max_size": args.max_size, "graph_extraction_seconds": extraction_seconds,
              "seeds": args.seeds, "trees": args.trees, "size_extrapolation_ops_threshold": threshold,
              "runtime_over_limit_count": int(np.sum(raw > 1)), "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "torch")},
              "controlled_performance_median_speedup": statistics.median(r["speedup"] for r in bench),
              "limitations": ["single saved solver protocol and hardware history", "single speed JSP type 0 only", "size-bounded cohort", "runtime target is capped at budget; censoring is not solved", "weighted target is a constructed proxy, not independent ground truth", "performance is a controlled duplicate-fit ablation, not original end-to-end speedup"]}
    (out / "study.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--max-size", type=int, default=15)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), args.root / "getSolutions.py", args.root / "Minizinc/Models/RD/JSP0.mzn",
               args.root / "pypi/src/xai_complex/estimator.py", args.root / "pypi/src/xai_complex/labels.py",
               args.root / "pypi/src/xai_complex/_graph_features.py"]
    (args.out / "source_hashes.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2), encoding="utf-8")
    records, rejected = load_records(args.root)
    audit = {"accepted": len(records), "rejected": len(rejected), "schedule_checks": {}, "rejection_reasons": {},
             "source": "DZN content and raw solver JSON, exact stem match; no seed inference", "scope": "JSP type 0, speed 1", "time_unit": "milliseconds, per getSolutions.py conversion"}
    for r in records:
        key = r["schedule_check"]
        audit["schedule_checks"][key] = audit["schedule_checks"].get(key, 0) + 1
    for r in rejected:
        key = r["reason"]
        audit["rejection_reasons"][key] = audit["rejection_reasons"].get(key, 0) + 1
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_csv(args.out / "rejected.csv", rejected)
    write_csv(args.out / "manifest.csv", [{k: r[k] for k in ("id", "base", "dzn_sha256", "solver_sha256", "schedule_check")} for r in records])
    print(json.dumps(audit), flush=True)
    if args.baselines_only:
        numpy_baselines(records, args.out, args.seeds)
        return
    if not args.audit_only:
        full_study(records, args.out, args)


if __name__ == "__main__":
    main()
