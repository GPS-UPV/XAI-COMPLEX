import json, os, math, warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(edgeitems=3, suppress=True)

# --------------------------- utilidades -----------------------------

def normalize_id_from_solutions(sol_id: str) -> str:
    m = re.match(r"^(\d+)_(\d+)-(\d+)-", str(sol_id))
    if not m:
        return np.nan
    j, mchs, seed = m.groups()
    return f"{j}-{mchs}-{seed}.pt"

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

def _load_labels_csv(path: str) -> pd.DataFrame:
    """Carga labels (por defecto, CSV con 'instance_id'). Agrupa duplicados por media (numéricas)."""
    if not os.path.exists(path):
        return None

    lab = pd.read_csv(path)

    # Asegurar instance_id
    if "instance_id" not in lab.columns:
        for cand in ["file", "name", "fname"]:
            if cand in lab.columns:
                lab = lab.rename(columns={cand: "instance_id"})
                break
    if "instance_id" not in lab.columns:
        return None

    # Agrupar duplicados (muy común si hay varias soluciones por instancia)
    if lab["instance_id"].duplicated().any():
        g = lab.groupby("instance_id", sort=False)

        num = g.mean(numeric_only=True)

        cat_cols = [c for c in lab.columns if c != "instance_id" and c not in num.columns]
        if cat_cols:
            cat = g[cat_cols].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
            lab2 = pd.concat([num, cat], axis=1)
        else:
            lab2 = num

        lab = lab2.reset_index()

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


def _clip_quantiles_arr(x: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    """Clip robusto por cuantiles (evita que outliers dominen la normalización)."""
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() == 0:
        return x
    lo, hi = np.nanquantile(x, [q_low, q_high])
    return np.clip(x, lo, hi)

def build_weighted_y_0_1(labels: pd.DataFrame,
                         weights: Dict[str, float] | None = None,
                         return_weights: bool = False,
                         add_columns: bool = False,
                         prefix: str = "ycomp_") -> (
                             pd.Series
                             | tuple[pd.Series, Dict[str, float]]
                             | tuple[pd.Series, pd.DataFrame]
                             | tuple[pd.Series, Dict[str, float], pd.DataFrame]
                         ):
    """Label supervisada y∈[0,1] ("solver hardness") como suma ponderada de un subconjunto representativo.

    Componentes (cada uno normalizado a [0,1]):
      - rt: ratio solveTime/maxTime (o solveTime si no hay maxTime)
      - gap: gap_rel
      - fail: log1p(failures)
      - prop: log1p(propagations)
      - msize: log1p(flat*Vars + flat*Constraints)
      - q: quality_tag/status (optimal<feasible<timeout/fail)

    Si add_columns=True, devuelve además un DataFrame con columnas extra:
      - {prefix}raw_*   (sin normalizar)
      - {prefix}t_*     (transformadas, e.g. log1p)
      - {prefix}c_*     (clipped por cuantiles donde aplica)
      - {prefix}n_*     (normalizadas 0..1)
      - {prefix}term_*  (término ponderado w_i * n_i)
      - {prefix}sum_terms y {prefix}y  (suma de términos y y final)
    """
    d = labels.copy()
    n = len(d)

    def _num(col: str):
        if col not in d.columns:
            return None
        return pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)

    solve = _num("solveTime")
    max_t = _num("maxTime")
    gap_rel = _num("gap_rel")
    failures = _num("failures")
    propag = _num("propagations")

    # runtime ratio (sin normalizar)
    if solve is None:
        rt_ratio = np.full(n, np.nan, dtype=float)
    else:
        if max_t is not None:
            denom = np.where(np.isfinite(max_t) & (max_t > 0), max_t, np.nan)
            rt_ratio = solve / denom
        else:
            rt_ratio = solve.copy()

    # model size aggregate (sin normalizar)
    parts = []
    for c in ["flatIntConstraints", "flatBoolConstraints", "flatIntVars", "flatBoolVars"]:
        arr = _num(c)
        if arr is not None:
            parts.append(np.nan_to_num(arr, nan=0.0))
    model_size = np.sum(np.vstack(parts), axis=0) if parts else None

    # quality raw (0..1, sin normalizar adicional)
    q_raw = np.full(n, np.nan, dtype=float)
    if "quality_tag" in d.columns:
        qt = d["quality_tag"].astype(str).str.lower()
        q_raw = qt.map({
            "optimal": 0.0,
            "feasible": 0.5,
            "timeout": 1.0,
            "unknown": 0.8,
            "fail": 1.0,
            "infeasible": 1.0
        }).to_numpy(dtype=float)

    # fallback to status
    if np.isfinite(q_raw).sum() == 0 and "status" in d.columns:
        q_raw = d["status"].map(_status_to_complexity).to_numpy(dtype=float)

    # ------------------- TRANSFORMACIONES (sin normalizar) -------------------
    gap_raw = gap_rel if gap_rel is not None else np.full(n, np.nan, dtype=float)

    fail_raw = failures if failures is not None else np.full(n, np.nan, dtype=float)
    prop_raw = propag if propag is not None else np.full(n, np.nan, dtype=float)
    msize_raw = model_size if model_size is not None else np.full(n, np.nan, dtype=float)

    fail_log = np.log1p(np.clip(fail_raw, 0, None)) if failures is not None else np.full(n, np.nan, dtype=float)
    prop_log = np.log1p(np.clip(prop_raw, 0, None)) if propag is not None else np.full(n, np.nan, dtype=float)
    msize_log = np.log1p(np.clip(msize_raw, 0, None)) if model_size is not None else np.full(n, np.nan, dtype=float)

    # ------------------- CLIPPING + NORMALIZACIÓN -------------------
    rt_clip = _clip_quantiles_arr(rt_ratio)
    rt_norm = _safe_minmax(rt_clip)

    if gap_rel is None:
        gap_clip = np.full(n, np.nan, dtype=float)
        gap_norm = np.zeros(n, dtype=float)
    else:
        gap_clip = _clip_quantiles_arr(gap_raw)
        gap_norm = _safe_minmax(gap_clip)

    if failures is None:
        fail_clip = np.full(n, np.nan, dtype=float)
        fail_norm = np.zeros(n, dtype=float)
    else:
        fail_clip = _clip_quantiles_arr(fail_log)
        fail_norm = _safe_minmax(fail_clip)

    if propag is None:
        prop_clip = np.full(n, np.nan, dtype=float)
        prop_norm = np.zeros(n, dtype=float)
    else:
        prop_clip = _clip_quantiles_arr(prop_log)
        prop_norm = _safe_minmax(prop_clip)

    if model_size is None:
        msize_clip = np.full(n, np.nan, dtype=float)
        msize_norm = np.zeros(n, dtype=float)
    else:
        msize_clip = _clip_quantiles_arr(msize_log)
        msize_norm = _safe_minmax(msize_clip)

    if np.isfinite(q_raw).sum() == 0:
        q_filled = np.zeros(n, dtype=float)
        q_norm = np.zeros(n, dtype=float)
    else:
        q_filled = q_raw.copy()
        fill = np.nanmedian(q_filled) if np.isfinite(q_filled).sum() else 0.5
        q_filled = np.where(np.isfinite(q_filled), q_filled, fill)
        q_norm = _safe_minmax(q_filled)

    comp_norm = {
        "rt": rt_norm,
        "gap": gap_norm,
        "fail": fail_norm,
        "prop": prop_norm,
        "msize": msize_norm,
        "q": q_norm
    }

    default_w = {"rt": 0.35, "gap": 0.20, "fail": 0.15, "prop": 0.10, "msize": 0.10, "q": 0.10}
    if weights is None:
        weights = default_w

    # keep keys available + normaliza pesos
    w = {k: float(v) for k, v in weights.items() if k in comp_norm and v is not None}
    s = sum(w.values())
    if s <= 0:
        w = default_w.copy()
        s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    # términos ponderados
    order = ["rt", "gap", "fail", "prop", "msize", "q"]
    terms = {}
    for k in order:
        wk = w.get(k, 0.0)
        terms[k] = wk * comp_norm[k]

    sum_terms = np.zeros(n, dtype=float)
    for k in order:
        sum_terms += terms[k]

    y = np.clip(sum_terms, 0.0, 1.0)
    y_series = pd.Series(y, index=d.index, name="y_weighted_0_1")

    # ------------------- Columnas debug (opcional) -------------------
    if add_columns:
        # raw (sin normalizar)
        d[f"{prefix}raw_rt_ratio"] = rt_ratio
        d[f"{prefix}raw_gap_rel"] = gap_raw
        d[f"{prefix}raw_failures"] = fail_raw
        d[f"{prefix}raw_propagations"] = prop_raw
        d[f"{prefix}raw_model_size"] = msize_raw
        d[f"{prefix}raw_quality"] = q_raw
        

        # # transformadas (sin normalizar)
        # d[f"{prefix}t_fail_log1p"] = fail_log
        # d[f"{prefix}t_prop_log1p"] = prop_log
        # d[f"{prefix}t_msize_log1p"] = msize_log

        # clipped
        d[f"{prefix}c_rt"] = rt_clip
        d[f"{prefix}c_gap"] = gap_clip
        d[f"{prefix}c_fail"] = fail_clip
        d[f"{prefix}c_prop"] = prop_clip
        d[f"{prefix}c_msize"] = msize_clip
        d[f"{prefix}c_q"] = q_filled

        # normalizadas
        d[f"{prefix}n_rt"] = rt_norm
        d[f"{prefix}n_gap"] = gap_norm
        d[f"{prefix}n_fail"] = fail_norm
        d[f"{prefix}n_prop"] = prop_norm
        d[f"{prefix}n_msize"] = msize_norm
        d[f"{prefix}n_q"] = q_norm

        # pesos y términos
        for k in order:
            d[f"{prefix}w_{k}"] = w.get(k, 0.0)
            d[f"{prefix}term_{k}"] = terms[k]

        d[f"{prefix}sum_terms"] = sum_terms
        d[f"{prefix}raw_sum_terms"] = rt_ratio+gap_raw+fail_raw+prop_raw+msize_raw+q_raw
        d[f"{prefix}n_sum_terms"] = rt_norm+gap_norm+fail_norm+prop_norm+msize_norm+q_norm
        d[f"{prefix}c_sum_terms"] = rt_clip+gap_clip+fail_clip+prop_clip+msize_clip+q_filled
        d[f"{prefix}y"] = y_series.values

    if add_columns and return_weights:
        return y_series, w, d
    if add_columns:
        return y_series, d
    if return_weights:
        return y_series, w
    return y_series


def weighted_y_formula_str(weights: Dict[str, float] | None = None,
                           decimals: int = 3,
                           use_unicode: bool = True) -> str:
    """Devuelve un string legible con la 'fórmula' usada para y (hardness) en build_weighted_y_0_1().

    Nota: cada término usa normalización a [0,1] con:
      - qclip(x)  := clip por cuantiles [0.05, 0.95]
      - norm01(x) := min-max a [0,1] (con manejo robusto de constantes/NaNs)
      - log1p() en señales de conteos (failures/propagations/model_size)

    Componentes:
      rt    = norm01(qclip(solveTime/maxTime))   (o solveTime si no hay maxTime)
      gap   = norm01(qclip(gap_rel))
      fail  = norm01(qclip(log1p(failures)))
      prop  = norm01(qclip(log1p(propagations)))
      msize = norm01(qclip(log1p(flatIntConstraints+flatBoolConstraints+flatIntVars+flatBoolVars)))
      q     = norm01(quality_tag/status)   (optimal<feasible<timeout/fail)

    Los pesos se normalizan para sumar 1.
    """
    default_w = {"rt": 0.35, "gap": 0.20, "fail": 0.15, "prop": 0.10, "msize": 0.10, "q": 0.10}
    if weights is None:
        weights = default_w

    # normalizar pesos
    w = {k: float(v) for k, v in weights.items() if k in default_w and v is not None}
    s = sum(w.values())
    if s <= 0:
        w = default_w.copy()
        s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    dot = "·" if use_unicode else "*"
    fmt = f"{{:.{decimals}f}}"

    term = {
        "rt":    "norm01(qclip(solveTime/maxTime))",
        "gap":   "norm01(qclip(gap_rel))",
        "fail":  "norm01(qclip(log1p(failures)))",
        "prop":  "norm01(qclip(log1p(propagations)))",
        "msize": "norm01(qclip(log1p(flatIntConstraints+flatBoolConstraints+flatIntVars+flatBoolVars)))",
        "q":     "norm01(quality_tag/status)"
    }

    order = ["rt", "gap", "fail", "prop", "msize", "q"]
    parts = [f"{fmt.format(w[k])}{dot}{term[k]}" for k in order if k in w and w[k] > 0]
    return "y = " + " + ".join(parts)

def print_weighted_y_formula(weights: Dict[str, float] | None = None,
                             decimals: int = 3) -> None:
    """Imprime por pantalla la fórmula de y usada por build_weighted_y_0_1()."""
    print(weighted_y_formula_str(weights=weights, decimals=decimals))
def supervised_calibration(
    df_feats: pd.DataFrame,
    X_scl: np.ndarray,
    labels: pd.DataFrame,
    dump_weighted_components_path: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:

    if labels is None or len(labels) == 0:
        return {"used": False, "reason": "No labels"}

    y_candidates: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}

    if "y" in labels.columns:
        y0 = pd.to_numeric(labels["y"], errors="coerce").to_numpy(dtype=float)
        y_candidates["from_y"] = np.clip(_safe_minmax(_clip_quantiles_arr(y0)), 0.0, 1.0)

    try:
        if dump_weighted_components_path:
            y_w, w_used, labels_dbg = build_weighted_y_0_1(
                labels,
                return_weights=True,
                add_columns=True,
                prefix="y_"
            )
            os.makedirs(os.path.dirname(dump_weighted_components_path) or ".", exist_ok=True)
            labels_dbg.to_csv(dump_weighted_components_path, index=True)
        else:
            y_w, w_used = build_weighted_y_0_1(labels, return_weights=True)

        if np.isfinite(y_w.values).sum() > 0:
            y_candidates["from_weighted"] = y_w.to_numpy(dtype=float)
            meta["weighted_components"] = list(w_used.keys())
            meta["weighted_weights"] = w_used
            if dump_weighted_components_path:
                meta["weighted_components_dump"] = dump_weighted_components_path
    except Exception as e:
        meta["weighted_error"] = str(e)

    if "runtime_ms" in labels.columns:
        rt = pd.to_numeric(labels["runtime_ms"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if rt.notna().sum() > 0:
            q_low, q_high = rt.quantile([0.05, 0.95])
            y_candidates["from_runtime_ms"] = _safe_minmax(rt.clip(q_low, q_high).values)

    for col in labels.columns:
        if col.lower() in {"status", "solver_status", "cp_status"}:
            y_candidates["from_status"] = labels[col].map(_status_to_complexity).to_numpy(dtype=float)

    y = None
    y_name = None
    for k in ["from_weighted", "from_y", "from_runtime_ms", "from_status"]:
        if k in y_candidates and np.isfinite(y_candidates[k]).sum() > 0:
            y = y_candidates[k]
            y_name = k
            break

    if y is None:
        return {"used": False, "reason": "No usable labels", **meta}

    def _normalize_id_to_pt(raw_id: str) -> str | None:
        s = str(raw_id)
        if re.match(r"^\d+-\d+-\d+\.pt$", s):
            return s
        m = re.match(r"^(\d+)_(\d+)-(\d+)-", s)
        if m:
            j, mchs, seed = m.groups()
            return f"{j}-{mchs}-{seed}.pt"
        m2 = re.match(r"^(\d+)-(\d+)-(\d+)-", s)
        if m2:
            j, mchs, seed = m2.groups()
            return f"{j}-{mchs}-{seed}.pt"
        return None

    if "instance_id" in labels.columns:
        raw_ids = labels["instance_id"]
    elif "instance" in labels.columns:
        raw_ids = labels["instance"]
    elif "inst_id" in labels.columns:
        raw_ids = labels["inst_id"]
    else:
        raw_ids = pd.Series(labels.index, index=labels.index)

    idx_norm = raw_ids.apply(_normalize_id_to_pt).astype("object")
    y_series = pd.Series(y, index=idx_norm)
    y_series = y_series[~y_series.index.isna()]

    if y_series.index.duplicated().any():
        y_series = y_series.groupby(level=0).mean()
        meta["labels_aggregated"] = "mean_over_duplicates"

    y_series = y_series.reindex(df_feats.index)
    mask = np.isfinite(y_series.values)

    meta["n_labels_raw"] = int(len(labels))
    meta["n_labels_normalized"] = int(pd.Series(idx_norm).notna().sum())
    meta["n_labels_aligned"] = int(mask.sum())
    meta["target_from"] = y_name

    if mask.sum() < max(20, n_splits * 3):
        return {"used": False, "reason": "Too few labels after alignment", **meta}

    Xs = X_scl[mask]
    ys = y_series.values[mask].astype(float)

    rf = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "r2": make_scorer(r2_score),
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    cv_res = cross_validate(rf, Xs, ys, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False)
    oof = cross_val_predict(rf, Xs, ys, cv=cv, n_jobs=-1)
    oof = np.clip(oof, 0.0, 1.0)

    spearman = spearmanr(ys, oof, nan_policy="omit")
    oof_r2 = r2_score(ys, oof)
    oof_mae = mean_absolute_error(ys, oof)
    oof_rmse = root_mean_squared_error(ys, oof)

    rf.fit(Xs, ys)
    y_pred_full = np.clip(rf.predict(X_scl), 0.0, 1.0)

    feat_imp = pd.Series(rf.feature_importances_, index=df_feats.select_dtypes(include=[np.number]).columns).sort_values(ascending=False)

    return {
        "used": True,
        "model": "RandomForestRegressor",
        "target_from": y_name,
        "cv_r2_mean": float(np.mean(cv_res["test_r2"])),
        "cv_r2_std": float(np.std(cv_res["test_r2"], ddof=1)),
        "cv_mae_mean": float(-np.mean(cv_res["test_mae"])),
        "cv_mae_std": float(np.std(-cv_res["test_mae"], ddof=1)),
        "cv_rmse_mean": float(-np.mean(cv_res["test_rmse"])),
        "cv_rmse_std": float(np.std(-cv_res["test_rmse"], ddof=1)),
        "oof_r2": float(oof_r2),
        "oof_mae": float(oof_mae),
        "oof_rmse": float(oof_rmse),
        "oof_spearman_rho": float(spearman.statistic) if spearman.statistic == spearman.statistic else np.nan,
        "oof_spearman_p": float(spearman.pvalue) if spearman.pvalue == spearman.pvalue else np.nan,
        "complexity_sup_pred": y_pred_full,
        "oof_pred_aligned": pd.Series(oof, index=df_feats.index[mask], name="oof_pred"),
        "y_true_aligned": pd.Series(ys, index=df_feats.index[mask], name="y_true"),
        "feature_importances": feat_imp,
        **meta
    }
# -------------------------------- main --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Estimate JSP complexity and supervised metrics.")
    parser.add_argument("--graphs-dir", default="./graphs")
    parser.add_argument("--solutions-dir", default="./solutions")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dump-all-features", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.graphs_dir, exist_ok=True)
    feats_json = os.path.join(args.graphs_dir, "features.json")
    feats_csv  = os.path.join(args.graphs_dir, "features.csv")
    out_csv = os.path.join(args.graphs_dir, "complexity_scores_W.csv")
    out_json = os.path.join(args.graphs_dir, "complexity_scores_W.json")
    out_txt  = os.path.join(args.graphs_dir, "complexity_report_W.txt")
    out_metrics_json = os.path.join(args.graphs_dir, "supervised_metrics.json")
    out_oof_csv = os.path.join(args.graphs_dir, "supervised_oof_predictions.csv")
    out_featimp_csv = os.path.join(args.graphs_dir, "supervised_feature_importances.csv")

    df_raw = _load_features_json(feats_json, feats_csv)
    df = _clean_and_engineer(df_raw)
    
    df = df.drop(columns=[c for c in df.columns if "energy" in c.strip().lower() or "num_nodes" == c])

    # 1) Prior desde el generador (si procede)
    prior = prior_complexity_from_generator(df)

    # 2) Ensamble no supervisado sobre TODAS las numéricas (grafo + generador)
    X_raw, X_scl, imputer, scaler = _build_matrix(df)
    unsup = unsupervised_complexity(X_scl)

    # 3) Calibración supervisada (si hay labels.csv)
    labels = _load_labels_csv(os.path.join(args.graphs_dir, "labels.csv"))
    if labels is None:
        labels = _load_labels_csv(os.path.join(args.solutions_dir, "solution_features.csv"))  # fallback
    
    # Dump opcional (separado) con TODAS las columnas de la fórmula de y (raw / log / clipped / norm / términos / suma).
    sup = supervised_calibration(
        df,
        X_scl,
        labels,
        dump_weighted_components_path=os.path.join(args.graphs_dir, "weighted_y_components.csv"),
        n_splits=args.cv_splits,
        random_state=args.random_state,
    )

    if args.dump_all_features:
        labels_for_dump = labels.copy() if labels is not None else pd.DataFrame(index=df.index)
        if labels_for_dump is None or len(labels_for_dump) == 0:
            labels_for_dump = pd.DataFrame(index=df.index)
        df_all = df.join(labels_for_dump, how="left")
        if sup.get("used", False):
            df_all["complexity_supervised_0_1"] = sup["complexity_sup_pred"]
        df_all.to_csv(os.path.join(args.graphs_dir, "all_features.csv"))

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
    job_machines = np.array(df.index.to_series().apply(lambda x: list(map(int, x.split("-")[:2]))).to_list())

    out = pd.DataFrame({
        "instance_id": df.index,
        "jobs": job_machines[:,0],
        "machines": job_machines[:,1],
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
        out["cv_r2_std_supervised"] = sup["cv_r2_std"]
        out["cv_mae_mean_supervised"] = sup["cv_mae_mean"]
        out["cv_mae_std_supervised"] = sup["cv_mae_std"]
        out["cv_rmse_mean_supervised"] = sup["cv_rmse_mean"]
        out["cv_rmse_std_supervised"] = sup["cv_rmse_std"]
        out["oof_r2_supervised"] = sup["oof_r2"]
        out["oof_mae_supervised"] = sup["oof_mae"]
        out["oof_rmse_supervised"] = sup["oof_rmse"]
        out["oof_spearman_rho_supervised"] = sup["oof_spearman_rho"]

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

    if sup.get("used", False):
        pd.concat([sup["y_true_aligned"], sup["oof_pred_aligned"]], axis=1).to_csv(out_oof_csv, index=True)
        sup["feature_importances"].rename("importance").to_csv(out_featimp_csv, index=True)
        metrics_payload = {
            "target_from": sup.get("target_from"),
            "cv_r2_mean": sup.get("cv_r2_mean"),
            "cv_r2_std": sup.get("cv_r2_std"),
            "cv_mae_mean": sup.get("cv_mae_mean"),
            "cv_mae_std": sup.get("cv_mae_std"),
            "cv_rmse_mean": sup.get("cv_rmse_mean"),
            "cv_rmse_std": sup.get("cv_rmse_std"),
            "oof_r2": sup.get("oof_r2"),
            "oof_mae": sup.get("oof_mae"),
            "oof_rmse": sup.get("oof_rmse"),
            "oof_spearman_rho": sup.get("oof_spearman_rho"),
            "oof_spearman_p": sup.get("oof_spearman_p"),
            "n_labels_aligned": sup.get("n_labels_aligned"),
            "weighted_weights": sup.get("weighted_weights"),
        }
        with open(out_metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

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
        lines.append(
            f"Supervised: YES ({sup.get('model')}) target={sup.get('target_from')} "
            f"cv_r2={sup.get('cv_r2_mean'):.3f}±{sup.get('cv_r2_std'):.3f} "
            f"cv_mae={sup.get('cv_mae_mean'):.3f}±{sup.get('cv_mae_std'):.3f} "
            f"cv_rmse={sup.get('cv_rmse_mean'):.3f}±{sup.get('cv_rmse_std'):.3f} "
            f"oof_spearman={sup.get('oof_spearman_rho'):.3f}"
        )
    else:
        lines.append("Supervised: NO")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {out_csv}\nSaved: {out_json}\nSaved: {out_txt}")

if __name__ == "__main__":
    main()
