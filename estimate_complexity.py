#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimador de complejidad [0,1] para instancias JSP a partir de features (grafo + generador).
- Entrada preferida: ./graphs/features.json  (dict: {instance_id: {feature_name: value, ...}})
- Fallback opcional: ./graphs/features.csv   (si no existe features.json)
- Opcional: ./graphs/labels.csv    (para calibración supervisada con rendimiento CP)
- Salidas:
    ./graphs/complexity_scores.csv
    ./graphs/complexity_scores.json
    ./graphs/complexity_report.txt
"""

import json, os, math, warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(edgeitems=3, suppress=True)

# --------------------------- utilidades -----------------------------

def _safe_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

def _to_float(v):
    try:
        if v is None:
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        if isinstance(v, str) and v.strip().lower() in {"nan", "none", "null", ""}:
            return np.nan
        return float(v)
    except Exception:
        return np.nan

def _load_features_json(path_json="./graphs/features.json", path_csv="./graphs/features.csv") -> pd.DataFrame:
    """Carga features de JSON (formato dict) o, si no existe, de CSV."""
    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = []
        for inst_id, rec in raw.items():
            if isinstance(rec, dict) and "features" in rec and isinstance(rec["features"], dict):
                rec = rec["features"]
            numeric = {k: _to_float(v) for k, v in rec.items()}
            numeric["instance_id"] = inst_id
            rows.append(numeric)
        df = pd.DataFrame(rows).set_index("instance_id")
    elif os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        if "file" in df.columns and "instance_id" not in df.columns:
            df = df.rename(columns={"file": "instance_id"})
        if "instance_id" in df.columns:
            df = df.set_index("instance_id")
    else:
        raise FileNotFoundError("No se encontró ni features.json ni features.csv en ./graphs")
    # columnas totalmente vacías fuera
    df = df.dropna(axis=1, how="all")
    return df

def _load_labels_csv(path="./graphs/labels.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    lab = pd.read_csv(path)
    if "instance_id" not in lab.columns:
        for cand in ["file", "name", "fname"]:
            if cand in lab.columns:
                lab = lab.rename(columns={cand: "instance_id"})
                break
    if "instance_id" not in lab.columns:
        return None
    return lab.set_index("instance_id")

def _clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Ajustes para las nuevas features del generador + ingeniería ligera."""
    df = df.copy()

    # 1) Sentinelas del generador: -1 -> NaN
    neg_as_nan_cols = ["window_min", "window_mean", "window_max", "overlap"]
    for c in neg_as_nan_cols:
        if c in df.columns:
            df.loc[df[c].astype(float) < 0, c] = np.nan

    # 2) Consistencia naming por si acaso (alias comunes)
    ren = {
        "n_machs": "n_machines",
        "makespan_max_min": "makespan_range",
        "energy_max_min": "energy_sum_range",
        "max_min_makespan": "makespan_range",
        "max_min_energy": "energy_sum_range",
    }
    for a,b in ren.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]

    # 3) Ingeniería simple:
    #    - tamaño de instancia (escala log para no dominar)
    if "n_jobs" in df.columns:
        df["size_jobs_log"] = np.log1p(df["n_jobs"].clip(lower=0))
    if "n_machines" in df.columns:
        df["size_machs_log"] = np.log1p(df["n_machines"].clip(lower=0))
    if all(c in df.columns for c in ["n_jobs", "n_machines"]):
        df["size_ops_log"] = np.log1p((df["n_jobs"] * df["n_machines"]).clip(lower=0))

    #    - rango de p y e si hay extremos
    if "p_value_max" in df.columns and "p_value_min" in df.columns:
        df["p_value_range"] = (df["p_value_max"] - df["p_value_min"]).astype(float)
    if "e_value_max" in df.columns and "e_value_min" in df.columns:
        df["e_value_range"] = (df["e_value_max"] - df["e_value_min"]).astype(float)

    # 4) Quitar columnas claramente no informativas o textuales
    drop_like = {"seed", "gen_features_raw"}  # gen_features_raw es JSON string
    keep = [c for c in df.columns if c not in drop_like]
    df = df[keep]

    return df

def _build_matrix(df: pd.DataFrame):
    # Solo numéricas
    X = df.select_dtypes(include=[np.number]).copy()
    imputer = SimpleImputer(strategy="median")
    scaler  = RobustScaler()
    X_imp = imputer.fit_transform(X.values)
    X_scl = scaler.fit_transform(X_imp)
    return X, X_scl, imputer, scaler

# ------------------ complejidad no supervisada ----------------------

def unsupervised_complexity(X_scl: np.ndarray, random_state=42) -> Dict[str, np.ndarray]:
    n = X_scl.shape[0]

    iforest = IsolationForest(n_estimators=400, random_state=random_state, n_jobs=-1)
    iforest.fit(X_scl)
    s_if = _safe_minmax(-iforest.score_samples(X_scl))

    lof = LocalOutlierFactor(n_neighbors=min(20, max(5, int(math.sqrt(n)))), contamination="auto")
    lof.fit(X_scl)
    s_lof = _safe_minmax(-lof.negative_outlier_factor_)

    mcd = MinCovDet(random_state=random_state)
    mcd.fit(X_scl)
    s_md = _safe_minmax(np.sqrt(np.maximum(mcd.mahalanobis(X_scl), 0.0)))

    k = min(20, max(5, int(math.sqrt(n))))
    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    knn.fit(X_scl)
    dists, _ = knn.kneighbors(X_scl)
    s_knn = _safe_minmax(dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists.mean(axis=1))

    s_ens = _safe_minmax(np.vstack([s_if, s_lof, s_md, s_knn]).mean(axis=0))
    return {"iforest": s_if, "lof": s_lof, "mahalanobis": s_md, "knn_density": s_knn, "complexity_unsup": s_ens}

# -------------------- prior de complejidad (nuevo) ------------------

def _col_norm(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return None
    return _safe_minmax(df[col].values.astype(float))

def prior_complexity_from_generator(df: pd.DataFrame) -> np.ndarray:
    """
    Construye un prior [0,1] a partir de métricas del generador si existen:
    - makespan_range ↑ ⇒ más complejo
    - energy_sum_range ↑ ⇒ más complejo
    - ventanas: menor window_mean ⇒ más complejo  (invertimos)
    - overlap ↑ ⇒ (asumimos) más complejo
    - p/e ranges ↑ ⇒ más complejo
    - tamaño (ops) ↑ ⇒ más complejo (suavizado en log)
    Se hace media sobre los componentes disponibles.
    """
    comps: List[np.ndarray] = []

    for c in ["makespan_range", "energy_sum_range", "p_value_range", "e_value_range"]:
        v = _col_norm(df, c)
        if v is not None:
            comps.append(v)

    # ventanas: invertimos window_mean (más pequeña → mayor complejidad)
    if "window_mean" in df.columns:
        w = df["window_mean"].astype(float).values
        w[w < 0] = np.nan  # limpiar sentinelas si quedara alguno
        if np.isfinite(w).sum() > 0:
            w = _safe_minmax(w)
            comps.append(1.0 - w)  # invertir

    # overlap directo
    ov = _col_norm(df, "overlap")
    if ov is not None:
        comps.append(ov)

    # tamaño en log (normalizamos por columna)
    for c in ["size_ops_log", "size_jobs_log", "size_machs_log"]:
        v = _col_norm(df, c)
        if v is not None:
            comps.append(v)

    if not comps:
        return None

    comps_arr = np.vstack(comps)
    prior = np.nanmean(comps_arr, axis=0)
    prior = _safe_minmax(prior)
    return prior

# --------------- calibración supervisada (igual que antes) ----------

def _status_to_complexity(status: str) -> float:
    if not isinstance(status, str):
        return np.nan
    s = status.strip().upper()
    if "OPTIMAL" in s: return 0.0
    if "FEAS" in s:    return 0.5
    if "TIME" in s:    return 1.0
    if "FAIL" in s or "INFEAS" in s: return 1.0
    return np.nan

def supervised_calibration(df_feats: pd.DataFrame,
                           X_scl: np.ndarray,
                           labels: pd.DataFrame) -> Dict[str, Any]:
    y_candidates = {}

    if labels is None:
        return {"used": False}

    if "runtime_ms" in labels.columns:
        rt = labels["runtime_ms"].astype(float).replace([np.inf, -np.inf], np.nan)
        q_low, q_high = rt.quantile([0.05, 0.95])
        y_candidates["from_runtime"] = _safe_minmax(rt.clip(q_low, q_high).values)

    for col in labels.columns:
        if col.lower() in {"status", "solver_status", "cp_status"}:
            y_candidates["from_status"] = labels[col].map(_status_to_complexity).values

    y = None; y_name = None
    if "from_runtime" in y_candidates and np.isfinite(y_candidates["from_runtime"]).sum() > 0:
        y, y_name = y_candidates["from_runtime"], "runtime_ms"
    elif "from_status" in y_candidates and np.isfinite(y_candidates["from_status"]).sum() > 0:
        y, y_name = y_candidates["from_status"], "status"
    else:
        return {"used": False, "reason": "No usable labels"}

    y_series = pd.Series(y, index=labels.index).reindex(df_feats.index)
    mask = np.isfinite(y_series.values)
    if mask.sum() < 20:
        return {"used": False, "reason": "Too few labels"}

    Xs = X_scl[mask]; ys = y_series.values[mask]
    rf = RandomForestRegressor(n_estimators=600, max_features="sqrt", random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2 = cross_val_score(rf, Xs, ys, scoring=make_scorer(r2_score), cv=cv).mean()
    rf.fit(Xs, ys)
    y_pred_full = np.clip(rf.predict(X_scl), 0.0, 1.0)

    return {"used": True, "model": "RandomForestRegressor", "target_from": y_name,
            "cv_r2_mean": float(r2), "complexity_sup_pred": y_pred_full}

# -------------------------------- main --------------------------------

def main():
    os.makedirs("./graphs", exist_ok=True)
    feats_json = "./graphs/features.json"
    feats_csv  = "./graphs/features.csv"
    out_csv = "./graphs/complexity_scores.csv"
    out_json = "./graphs/complexity_scores.json"
    out_txt  = "./graphs/complexity_report.txt"

    df_raw = _load_features_json(feats_json, feats_csv)
    df = _clean_and_engineer(df_raw)

    # 1) Prior desde el generador (si procede)
    prior = prior_complexity_from_generator(df)

    # 2) Ensamble no supervisado sobre TODAS las numéricas (grafo + generador)
    X_raw, X_scl, imputer, scaler = _build_matrix(df)
    unsup = unsupervised_complexity(X_scl)

    # 3) Calibración supervisada (si hay labels.csv)
    labels = _load_labels_csv("./graphs/labels.csv")
    sup = supervised_calibration(df, X_scl, labels)

    # 4) Blending de señales
    complexity = unsup["complexity_unsup"]
    blend = {"unsup": 1.0, "prior": 0.0, "sup": 0.0, "sup_target": None}

    if prior is not None:
        complexity = 0.7 * complexity + 0.3 * prior
        complexity = _safe_minmax(complexity)
        blend["unsup"], blend["prior"] = 0.7, 0.3

    if sup.get("used", False):
        # si tenemos supervised, mezclamos con el blend actual
        if prior is not None:
            # 50% unsup_prior + 50% supervised
            complexity = 0.5 * complexity + 0.5 * sup["complexity_sup_pred"]
            blend["unsup"], blend["prior"], blend["sup"] = 0.35, 0.15, 0.5
        else:
            # 60% unsup + 40% supervised
            complexity = 0.6 * complexity + 0.4 * sup["complexity_sup_pred"]
            blend["unsup"], blend["sup"] = 0.6, 0.4
        complexity = _safe_minmax(complexity)
        blend["sup_target"] = sup.get("target_from")

    # 5) Categorías por cuantiles
    q1, q2 = np.quantile(complexity, [0.33, 0.66])
    cats = np.where(complexity <= q1, "easy",
            np.where(complexity <= q2, "medium", "hard"))

    out = pd.DataFrame({
        "instance_id": df.index,
        "complexity_0_1": complexity,
        "category": cats,
        "s_iforest": unsup["iforest"],
        "s_lof": unsup["lof"],
        "s_mahalanobis": unsup["mahalanobis"],
        "s_knn": unsup["knn_density"]
    }).set_index("instance_id")

    if prior is not None:
        out["complexity_prior_0_1"] = prior

    if sup.get("used", False):
        out["complexity_supervised_0_1"] = sup["complexity_sup_pred"]
        out["cv_r2_mean_supervised"] = sup["cv_r2_mean"]

    # Info del blend (misma para todas las filas, pero útil en el CSV)
    out["blend_w_unsup"] = blend["unsup"]
    out["blend_w_prior"] = blend["prior"]
    out["blend_w_sup"]   = blend["sup"]
    if blend["sup_target"] is not None:
        out["sup_target"] = blend["sup_target"]

    # Guardar
    out.to_csv(out_csv, index=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out["complexity_0_1"].to_dict(), f, ensure_ascii=False, indent=2)

    # Report
    lines = []
    lines.append("=== COMPLEXITY ESTIMATOR REPORT ===")
    lines.append(f"Instances: {len(out)}")
    src = "unsupervised"
    if prior is not None and not sup.get("used", False):
        src = "blend(unsup + prior)"
    if sup.get("used", False) and prior is None:
        src = f"blend(unsup + sup[{sup.get('target_from')}])"
    if sup.get("used", False) and prior is not None:
        src = f"blend(unsup + prior + sup[{sup.get('target_from')}])"
    lines.append(f"Signal source: {src}")
    lines.append(f"Mean complexity: {out['complexity_0_1'].mean():.3f}")
    lines.append(f"Quantiles: q10={out['complexity_0_1'].quantile(0.10):.3f} "
                 f"q50={out['complexity_0_1'].quantile(0.50):.3f} "
                 f"q90={out['complexity_0_1'].quantile(0.90):.3f}")
    lines.append(f"Counts by category: {out['category'].value_counts().to_dict()}")
    if prior is not None:
        lines.append("Prior: YES (from generator metrics)")
    else:
        lines.append("Prior: NO")
    if sup.get("used", False):
        lines.append(f"Supervised: YES ({sup.get('model')}) target={sup.get('target_from')}  cv_r2≈{sup.get('cv_r2_mean'):.3f}")
    else:
        lines.append("Supervised: NO")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {out_csv}\nSaved: {out_json}\nSaved: {out_txt}")

if __name__ == "__main__":
    main()
