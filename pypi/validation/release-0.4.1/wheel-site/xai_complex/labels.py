"""Training-only calibration of the solver-dependent weighted target."""

import math
import statistics

WEIGHTS = {"rt": .35, "gap": .20, "fail": .15, "prop": .10, "msize": .10, "q": .10}


def status_to_complexity(status):
    key = str(status).strip().upper()
    return {"OPTIMAL": 0., "OPTIMAL_SOLUTION": 0., "FEASIBLE": .5, "SATISFIED": .5,
            "TIMEOUT": 1., "TIME_LIMIT": 1., "UNKNOWN": .8, "FAIL": 1.,
            "FAILED": 1., "INFEASIBLE": 1., "UNSATISFIABLE": 1.}.get(key, math.nan)


def _finite(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _components(row):
    out = {}
    solve, limit = _finite(row.get("solveTime")), _finite(row.get("maxTime"))
    if solve is not None:
        if solve < 0 or ("maxTime" in row and (limit is None or limit <= 0)):
            raise ValueError("Invalid solver runtime or time limit")
        out["rt"] = solve / limit if limit is not None else solve
    for dest, source in (("gap", "gap_rel"), ("fail", "failures"), ("prop", "propagations")):
        value = _finite(row.get(source))
        if value is not None:
            if value < 0:
                raise ValueError(f"Negative solver statistic: {source}")
            out[dest] = value if dest == "gap" else math.log1p(value)
    values = [_finite(row.get(k)) for k in ("flatIntConstraints", "flatBoolConstraints", "flatIntVars", "flatBoolVars")]
    values = [v for v in values if v is not None]
    if values:
        if any(v < 0 for v in values):
            raise ValueError("Negative model size")
        out["msize"] = math.log1p(sum(values))
    status = status_to_complexity(row.get("quality_tag", ""))
    if not math.isfinite(status):
        status = status_to_complexity(row.get("status", ""))
    if math.isfinite(status):
        out["q"] = status
    return out


def _quantile(values, q):
    a = sorted(values)
    position = (len(a) - 1) * q
    low = int(position)
    return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (position - low)


class SolverTargetCalibrator:
    """Fit quantiles on training records; quality states retain absolute meaning."""

    def __init__(self, weights=None):
        self.weights = dict(WEIGHTS if weights is None else weights)
        if not self.weights or set(self.weights) - set(WEIGHTS):
            raise ValueError("Unknown or empty component weights")
        if any(isinstance(v, bool) or _finite(v) is None or float(v) < 0 for v in self.weights.values()):
            raise ValueError("Weights must be finite, nonnegative and have a positive sum")
        self.weights = {k: float(v) for k, v in self.weights.items()}
        if not math.isfinite(sum(self.weights.values())) or sum(self.weights.values()) <= 0:
            raise ValueError("Weights must have a finite positive sum")

    def fit(self, rows):
        records = [_components(row) for row in rows]
        self.bounds_ = {}
        for key, weight in self.weights.items():
            vals = [r[key] for r in records if key in r]
            if vals and weight:
                self.bounds_[key] = (0., 1.) if key == "q" else (_quantile(vals, .05), _quantile(vals, .95))
        if not self.bounds_:
            raise ValueError("No usable solver signals for calibration")
        return self

    def transform(self, rows):
        if not hasattr(self, "bounds_"):
            raise ValueError("Fit the target calibrator before transforming")
        result = []
        for row in rows:
            components = _components(row)
            terms, weights = [], []
            for key, (lo, hi) in self.bounds_.items():
                if key not in components:
                    continue
                value = max(0., min(1., (components[key] - lo) / (hi - lo))) if hi > lo else 0.
                terms.append(self.weights[key] * value)
                weights.append(self.weights[key])
            if not weights:
                raise ValueError("Record has no calibrated solver signals")
            result.append(sum(terms) / sum(weights))
        return result

    def fit_transform(self, rows):
        rows = list(rows)
        return self.fit(rows).transform(rows)
