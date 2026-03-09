import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------
# Helpers
# ---------
OBJ_REGEX = re.compile(r'Objective function:\s*([0-9]+(?:\.[0-9]+)?)\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*=\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
STARTS_REGEX = re.compile(r'Start\s*Times:\s*\[([^\]]*)\]', re.IGNORECASE)
SPEEDS_REGEX = re.compile(r'Speed\s*Scaling[:;]?\s*\[([^\]]*)\]', re.IGNORECASE)

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def flatten_list_of_lists(x) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list) and all(isinstance(r, list) for r in x):
        out = []
        for r in x:
            out.extend(r)
        return out
    if isinstance(x, list):
        return x
    return []

def gini(arr: np.ndarray) -> float:
    """Gini para no-negativos. Devuelve nan si suma == 0."""
    if arr.size == 0:
        return np.nan
    x = np.sort(np.asarray(arr, dtype=float))
    if np.any(x < 0):
        # No estricto, pero si hay negativos devolvemos nan
        return np.nan
    s = x.sum()
    if s == 0:
        return np.nan
    n = x.size
    # 2 * sum(i*x_i) / (n*sum(x)) - (n+1)/n
    return (2.0 * np.sum((np.arange(1, n + 1) * x))) / (n * s) - (n + 1.0) / n

def parse_output_item(text: str) -> Dict[str, Any]:
    """Intenta parsear _output_item para objective breakdown, start times y speed scaling."""
    if not text or not isinstance(text, str):
        return {}
    out: Dict[str, Any] = {}
    m = OBJ_REGEX.search(text)
    if m:
        mk, en, obj = m.groups()
        out["parsed_makespan"] = float(mk)
        out["parsed_energy"] = float(en)
        out["parsed_objective"] = float(obj)
    m2 = STARTS_REGEX.search(text)
    if m2:
        # ejemplo " [59, 0, 0, 59]"
        raw = m2.group(1)
        try:
            vals = [float(t.strip()) for t in raw.split(",") if t.strip() != ""]
            out["parsed_start_times_flat"] = vals
        except Exception:
            pass
    m3 = SPEEDS_REGEX.search(text)
    if m3:
        raw = m3.group(1)
        try:
            vals = [float(t.strip()) for t in raw.split(",") if t.strip() != ""]
            out["parsed_speeds_flat"] = vals
        except Exception:
            pass
    return out

def classify_quality(status: str, is_optimal: bool, timelimit_hit: bool, failures: float) -> str:
    st = (status or "").upper()
    if is_optimal or "OPTIMAL" in st:
        return "optimal"
    if timelimit_hit:
        return "timeout"
    if "FEASIBLE" in st or "SAT" in st:  # por si hay "FEASIBLE_SOLUTION" u otras variantes
        return "feasible"
    if failures and failures > 0 and ("INFEASIBLE" in st or "FAIL" in st):
        return "fail"
    if "INFEASIBLE" in st:
        return "infeasible"
    return "unknown"

def ratio(a: float, b: float) -> float:
    try:
        if b == 0:
            return math.inf if a > 0 else 0.0
        return a / b
    except Exception:
        return np.nan


# -------------------------
# Feature extraction core
# -------------------------
def extract_features_from_solution(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Campos base
    status = safe_get(data, "status")
    sol = safe_get(data, "solution", default={}) or {}
    stats = safe_get(data, "statistics", default={}) or {}

    # Objetivo y desgloses
    objective = safe_get(sol, "objective")
    makespan = safe_get(stats, "makespan")
    energy = safe_get(stats, "energy")
    objective_bound = safe_get(stats, "objectiveBound")
    obj_from_stats = safe_get(stats, "objective")

    # _output_item parsing
    output_item = safe_get(sol, "_output_item", default="") or ""
    parsed = parse_output_item(output_item)

    # Start times & speeds (estructura matriz -> flatten)
    start_time_ll = safe_get(sol, "start_time")
    start_time_flat = flatten_list_of_lists(start_time_ll)
    speeds_ll = safe_get(sol, "SpeedScaling")
    speeds_flat = flatten_list_of_lists(speeds_ll)

    # Dimensiones (heurística): si start_time es matriz JxM, intentamos inferir n_jobs y n_machines
    n_jobs = len(start_time_ll) if isinstance(start_time_ll, list) else np.nan
    n_machines = None
    if isinstance(start_time_ll, list) and len(start_time_ll) > 0 and isinstance(start_time_ll[0], list):
        # Asumimos rectangular; si no, tomamos la moda de longitudes
        lens = [len(r) for r in start_time_ll if isinstance(r, list)]
        if len(lens) > 0:
            try:
                n_machines = int(pd.Series(lens).mode().iloc[0])
            except Exception:
                n_machines = lens[0]
    if n_machines is None:
        n_machines = np.nan

    # Estadísticas básicas de start_time
    st_arr = np.asarray(start_time_flat, dtype=float) if len(start_time_flat) else np.array([], dtype=float)
    st_min = float(np.min(st_arr)) if st_arr.size else np.nan
    st_max = float(np.max(st_arr)) if st_arr.size else np.nan
    st_mean = float(np.mean(st_arr)) if st_arr.size else np.nan
    st_std = float(np.std(st_arr)) if st_arr.size else np.nan
    st_gini = float(gini(st_arr)) if st_arr.size else np.nan
    st_zeros = int(np.sum(st_arr == 0)) if st_arr.size else 0
    st_zero_ratio = float(st_zeros / st_arr.size) if st_arr.size else np.nan

    # Speed scaling stats
    sp_arr = np.asarray(speeds_flat, dtype=float) if len(speeds_flat) else np.array([], dtype=float)
    sp_unique = int(len(np.unique(sp_arr))) if sp_arr.size else 0
    sp_mean = float(np.mean(sp_arr)) if sp_arr.size else np.nan
    sp_all_one = bool(sp_arr.size and np.all(sp_arr == 1))

    # Consistencias/derivadas
    # Preferimos objective declarado en "solution", si no, el de "statistics"
    obj = objective if objective is not None else obj_from_stats
    # Gap al bound
    gap_abs = None
    gap_rel = None
    if obj is not None and objective_bound is not None:
        gap_abs = float(abs(float(obj) - float(objective_bound)))
        denom = max(1e-12, abs(float(obj)))
        gap_rel = float(gap_abs / denom)

    is_optimal = False
    if status and "OPTIMAL" in status.upper():
        is_optimal = True
    elif (obj is not None) and (objective_bound is not None):
        # si obj == bound, lo consideramos optimal
        try:
            is_optimal = abs(float(obj) - float(objective_bound)) < 1e-9
        except Exception:
            pass

    # Timeout vs maxTime
    solve_time = safe_get(stats, "solveTime")
    flat_time = safe_get(stats, "flatTime")
    total_time = safe_get(stats, "time")
    max_time = safe_get(stats, "maxTime")
    timelimit_hit = False
    try:
        if (max_time is not None) and (total_time is not None):
            timelimit_hit = float(total_time) >= float(max_time) - 1e-9
    except Exception:
        pass

    # Más contadores del solver
    n_solutions = safe_get(stats, "nSolutions")
    failures = safe_get(stats, "failures")
    propagations = safe_get(stats, "propagations")
    method = safe_get(stats, "method")

    flat_bool_vars = safe_get(stats, "flatBoolVars")
    flat_int_vars = safe_get(stats, "flatIntVars")
    flat_bool_cons = safe_get(stats, "flatBoolConstraints")
    flat_int_cons = safe_get(stats, "flatIntConstraints")
    half_reif = safe_get(stats, "evaluatedHalfReifiedConstraints")
    bool_vars_runtime = safe_get(stats, "boolVariables")  # a veces distinto de flatBoolVars
    paths = safe_get(stats, "paths")

    # Consistencia objetivo con makespan+energy (si ambos presentes)
    mk_en_ok = None
    if (makespan is not None) and (energy is not None) and (obj is not None):
        try:
            mk_en_ok = abs(float(makespan) + float(energy) - float(obj)) < 1e-9
        except Exception:
            mk_en_ok = None

    # Contrastado con _output_item
    parsed_obj_ok = None
    if "parsed_objective" in parsed and (obj is not None):
        try:
            parsed_obj_ok = abs(float(parsed["parsed_objective"]) - float(obj)) < 1e-6
        except Exception:
            parsed_obj_ok = None

    # Tamaños/coherencias
    start_time_len = len(start_time_ll) if isinstance(start_time_ll, list) else 0
    start_time_flat_len = len(start_time_flat)
    speeds_len = len(speeds_ll) if isinstance(speeds_ll, list) else 0
    speeds_flat_len = len(speeds_flat)

    # Etiqueta de calidad
    qtag = classify_quality(status, is_optimal, timelimit_hit, failures or 0)

    # Nombre/ID de instancia (del archivo)
    fname = os.path.basename(file_path)
    instance_id = os.path.splitext(fname)[0]

    # Checker
    checker_txt = safe_get(sol, "_checker", default="")
    checker_len = len(checker_txt or "")

    # Build record
    rec: Dict[str, Any] = {
        "file": fname,
        "instance_id": instance_id,

        # Estado / calidad
        "status": status,
        "quality_tag": qtag,
        "is_optimal": bool(is_optimal),
        "timelimit_hit": bool(timelimit_hit),

        # Objetivos y bounds
        "objective": obj if obj is not None else np.nan,
        "objective_from_stats": obj_from_stats if obj_from_stats is not None else np.nan,
        "objective_bound": objective_bound if objective_bound is not None else np.nan,
        "gap_abs": gap_abs if gap_abs is not None else np.nan,
        "gap_rel": gap_rel if gap_rel is not None else np.nan,

        # Desglose tiempo/energía (statistics)
        "makespan": makespan if makespan is not None else np.nan,
        "energy": energy if energy is not None else np.nan,
        "mk_plus_en_eq_objective": mk_en_ok,

        # Parsing _output_item (para trazabilidad)
        "parsed_makespan": parsed.get("parsed_makespan", np.nan),
        "parsed_energy": parsed.get("parsed_energy", np.nan),
        "parsed_objective": parsed.get("parsed_objective", np.nan),
        "parsed_obj_matches_objective": parsed_obj_ok,

        # Start times & SpeedScaling
        "start_time_len": start_time_len,
        "start_time_flat_len": start_time_flat_len,
        "n_jobs_inferred": n_jobs if not (isinstance(n_jobs, float) and math.isnan(n_jobs)) else np.nan,
        "n_machines_inferred": n_machines if not (isinstance(n_machines, float) and math.isnan(n_machines)) else np.nan,

        "start_time_min": st_min,
        "start_time_max": st_max,
        "start_time_mean": st_mean,
        "start_time_std": st_std,
        "start_time_gini": st_gini,
        "start_time_num_zero": st_zeros,
        "start_time_zero_ratio": st_zero_ratio,

        "speeds_len": speeds_len,
        "speeds_flat_len": speeds_flat_len,
        "speeds_unique": sp_unique,
        "speeds_mean": sp_mean,
        "speeds_all_one": sp_all_one,

        # Stats del solver
        "solveTime": solve_time if solve_time is not None else np.nan,
        "flatTime": flat_time if flat_time is not None else np.nan,
        "totalTime": total_time if total_time is not None else np.nan,
        "maxTime": max_time if max_time is not None else np.nan,
        "method": method,

        "nSolutions": n_solutions if n_solutions is not None else np.nan,
        "failures": failures if failures is not None else np.nan,
        "propagations": propagations if propagations is not None else np.nan,
        "paths": paths if paths is not None else np.nan,

        "flatBoolVars": flat_bool_vars if flat_bool_vars is not None else np.nan,
        "flatIntVars": flat_int_vars if flat_int_vars is not None else np.nan,
        "flatBoolConstraints": flat_bool_cons if flat_bool_cons is not None else np.nan,
        "flatIntConstraints": flat_int_cons if flat_int_cons is not None else np.nan,
        "evaluatedHalfReifiedConstraints": half_reif if half_reif is not None else np.nan,
        "boolVariables_runtime": bool_vars_runtime if bool_vars_runtime is not None else np.nan,

        # Metadatos
        "checker_len": checker_len,
    }

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions-dir", type=str, default="./solutions",
                    help="Directorio con los JSON de soluciones.")
    ap.add_argument("--out-prefix", type=str, default="solution_features",
                    help="Prefijo base para CSV/JSON/reporte (sin extensión).")
    args = ap.parse_args()

    sol_dir = args.solutions_dir
    out_csv = os.path.join(sol_dir, f"{args.out_prefix}.csv")
    out_json = os.path.join(sol_dir, f"{args.out_prefix}.json")
    out_report = os.path.join(sol_dir, f"{args.out_prefix}_report.txt")

    files = [os.path.join(sol_dir, f) for f in os.listdir(sol_dir)
             if f.lower().endswith(".json") and os.path.isfile(os.path.join(sol_dir, f))]
    files.sort()

    rows: List[Dict[str, Any]] = []
    debug: Dict[str, str] = {}
    ok = 0
    err = 0

    for fp in files:
        try:
            rec = extract_features_from_solution(fp)
            rows.append(rec)
            ok += 1
        except Exception as e:
            err += 1
            debug[os.path.basename(fp)] = str(e)

    if len(rows) == 0:
        print("[WARN] No se han generado filas. ¿Hay JSON válidos en", sol_dir, "?")

    df = pd.DataFrame(rows)
    # Orden de columnas principal (prioriza lo más útil para tus cruces)
    preferred = [
        "file","instance_id","status","quality_tag","is_optimal","timelimit_hit",
        "objective","objective_bound","gap_abs","gap_rel",
        "makespan","energy","mk_plus_en_eq_objective",
        "n_jobs_inferred","n_machines_inferred",
        "start_time_len","start_time_flat_len",
        "start_time_min","start_time_max","start_time_mean","start_time_std","start_time_gini",
        "start_time_num_zero","start_time_zero_ratio",
        "speeds_len","speeds_flat_len","speeds_unique","speeds_mean","speeds_all_one",
        "solveTime","flatTime","totalTime","maxTime","method",
        "nSolutions","failures","propagations","paths",
        "flatBoolVars","flatIntVars","flatBoolConstraints","flatIntConstraints",
        "evaluatedHalfReifiedConstraints","boolVariables_runtime",
        "parsed_makespan","parsed_energy","parsed_objective","parsed_obj_matches_objective",
        "checker_len",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # Export
    os.makedirs(sol_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # Reporte resumido
    by_status = df["status"].fillna("NA").astype(str).str.upper().value_counts().to_dict() if len(df) else {}
    by_qtag = df["quality_tag"].fillna("NA").astype(str).value_counts().to_dict() if len(df) else {}

    lines = []
    lines.append("=== Solution Features Report ===")
    lines.append(f"Solutions dir: {sol_dir}")
    lines.append(f"Processed OK: {ok} | Errors: {err} | Total rows: {len(df)}")
    lines.append("")
    lines.append("Status counts:")
    for k, v in by_status.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Quality tag counts:")
    for k, v in by_qtag.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    if err > 0:
        lines.append("Errors:")
        for k, v in debug.items():
            lines.append(f"  * {k}: {v}")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] CSV  -> {out_csv}")
    print(f"[OK] JSON -> {out_json}")
    print(f"[OK] TXT  -> {out_report}")


if __name__ == "__main__":
    main()
