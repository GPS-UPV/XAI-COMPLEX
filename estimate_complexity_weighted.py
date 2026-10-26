import json, os, math, warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import argparse




def normalize_id_from_solutions(sol_id: str) -> str:
    raise ValueError("Proporcione un mapeo explícito con configuración y semilla completas")

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

    df = df.dropna(axis=1, how="all")
    return df

def _load_labels_csv(path: str):
    if not os.path.exists(path):
        return None
    lab = pd.read_csv(path)
    if "instance_id" not in lab:
        for alias in ("file", "name", "fname"):
            if alias in lab:
                lab = lab.rename(columns={alias: "instance_id"})
                break
    if "instance_id" not in lab or lab["instance_id"].isna().any():
        raise ValueError("Las etiquetas requieren identificadores completos")
    if lab["instance_id"].duplicated().any():
        raise ValueError("Agregue las repeticiones mediante un protocolo de solver explícito")
    return lab.set_index("instance_id")

def _clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Ajustes para las nuevas features del generador + ingeniería ligera."""
    df = df.copy()


    neg_as_nan_cols = ["window_min", "window_mean", "window_max", "overlap"]
    for c in neg_as_nan_cols:
        if c in df.columns:
            df.loc[df[c].astype(float) < 0, c] = np.nan


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


    #    - tamaño de instancia (escala log para no dominar)
    if "n_jobs" in df.columns:
        df["size_jobs_log"] = np.log1p(df["n_jobs"].clip(lower=0))
    if "n_machines" in df.columns:
        df["size_machs_log"] = np.log1p(df["n_machines"].clip(lower=0))
    if all(c in df.columns for c in ["n_jobs", "n_machines"]):
        df["size_ops_log"] = np.log1p((df["n_jobs"] * df["n_machines"]).clip(lower=0))


    if "p_value_max" in df.columns and "p_value_min" in df.columns:
        df["p_value_range"] = (df["p_value_max"] - df["p_value_min"]).astype(float)
    if "e_value_max" in df.columns and "e_value_min" in df.columns:
        df["e_value_range"] = (df["e_value_max"] - df["e_value_min"]).astype(float)


    drop_like = {"seed", "gen_features_raw"}
    keep = [c for c in df.columns if c not in drop_like]
    df = df[keep]

    return df

def _build_matrix(df: pd.DataFrame):

    X = df.select_dtypes(include=[np.number]).copy().dropna(axis=1, how="all")
    if X.empty or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("No hay características numéricas válidas o existen infinitos")
    imputer = SimpleImputer(strategy="median")
    scaler  = RobustScaler()
    X_imp = imputer.fit_transform(X.values)
    X_scl = scaler.fit_transform(X_imp)
    return X, X_scl, imputer, scaler



def unsupervised_complexity(X_scl: np.ndarray, random_state=42) -> Dict[str, np.ndarray]:
    n = X_scl.shape[0]
    if n < 3:
        raise ValueError("Se necesitan al menos tres instancias de referencia")

    iforest = IsolationForest(n_estimators=400, random_state=random_state, n_jobs=1)
    iforest.fit(X_scl)
    s_if = _safe_minmax(-iforest.score_samples(X_scl))

    lof = LocalOutlierFactor(n_neighbors=min(n - 1, 20, max(2, int(math.sqrt(n)))), contamination="auto")
    lof.fit(X_scl)
    s_lof = _safe_minmax(-lof.negative_outlier_factor_)

    mcd = MinCovDet(random_state=random_state)
    mcd.fit(X_scl)
    s_md = _safe_minmax(np.sqrt(np.maximum(mcd.mahalanobis(X_scl), 0.0)))

    k = min(n, 20, max(2, int(math.sqrt(n))))
    knn = NearestNeighbors(n_neighbors=k, n_jobs=1)
    knn.fit(X_scl)
    dists, _ = knn.kneighbors(X_scl)
    s_knn = _safe_minmax(dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists.mean(axis=1))

    s_ens = _safe_minmax(np.vstack([s_if, s_lof, s_md, s_knn]).mean(axis=0))
    return {"iforest": s_if, "lof": s_lof, "mahalanobis": s_md, "knn_density": s_knn, "complexity_unsup": s_ens}



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
        w[w < 0] = np.nan
        if np.isfinite(w).sum() > 0:
            w = _safe_minmax(w)
            comps.append(1.0 - w)


    ov = _col_norm(df, "overlap")
    if ov is not None:
        comps.append(ov)


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



def _status_to_complexity(status: str) -> float:
    if not isinstance(status, str):
        return np.nan
    s = status.strip().upper()
    if "OPTIMAL" in s: return 0.0
    if "INFEAS" in s or "UNSAT" in s: return 1.0
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

def build_weighted_y_0_1(labels, weights=None, return_weights=False, add_columns=False, prefix="ycomp_"):
    """Calibra la dificultad empírica y renormaliza por señales disponibles."""
    from xai_complex.labels import SolverTargetCalibrator
    calibrator = SolverTargetCalibrator(weights)
    targets = calibrator.fit_transform(labels.to_dict(orient="records"))
    y = pd.Series(targets, index=labels.index, name="y_weighted_0_1")
    active = {k: float(calibrator.weights[k]) for k in calibrator.bounds_}
    total = sum(active.values())
    active = {k: v / total for k, v in active.items()}
    if add_columns:
        debug = labels.copy()
        debug[f"{prefix}y"] = y
        debug[f"{prefix}sum_terms"] = y
        if return_weights:
            return y, active, debug
        return y, debug
    return (y, active) if return_weights else y


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
        "q":     "quality(quality_tag/status)"
    }

    order = ["rt", "gap", "fail", "prop", "msize", "q"]
    parts = [f"{fmt.format(w[k])}{dot}{term[k]}" for k in order if k in w and w[k] > 0]
    return "y = " + " + ".join(parts)

def print_weighted_y_formula(weights: Dict[str, float] | None = None,
                             decimals: int = 3) -> None:
    """Imprime por pantalla la fórmula de y usada por build_weighted_y_0_1()."""
    print(weighted_y_formula_str(weights=weights, decimals=decimals))
from xai_complex.calibration import supervised_calibration


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


    prior = prior_complexity_from_generator(df)


    X_raw, X_scl, imputer, scaler = _build_matrix(df)
    unsup = unsupervised_complexity(X_scl)


    labels = _load_labels_csv(os.path.join(args.graphs_dir, "labels.csv"))
    if labels is None:
        labels = _load_labels_csv(os.path.join(args.solutions_dir, "solution_features.csv"))


    sup = supervised_calibration(
        df,
        X_scl,
        labels,
        dump_weighted_components_path=os.path.join(args.graphs_dir, "weighted_y_components.csv"),
        n_splits=args.cv_splits,
        random_state=args.random_state,
    )

    if args.dump_all_features:
        id1 = list(labels.index) if labels is not None else list(df.index)
        labels_for_dump = labels.copy() if labels is not None else pd.DataFrame(index=df.index)
        labels_for_dump.index = id1
        if labels_for_dump is None or len(labels_for_dump) == 0:
            labels_for_dump = pd.DataFrame(index=df.index)
        df_all = df.join(labels_for_dump, how="left")
        if sup.get("used", False):
            df_all["complexity_supervised_0_1"] = sup["complexity_sup_pred"]
        df_all.to_csv(os.path.join(args.graphs_dir, "all_features.csv"))


    complexity = unsup["complexity_unsup"]
    blend = {"unsup": 1.0, "prior": 0.0, "sup": 0.0, "sup_target": None}

    if prior is not None:
        complexity = 0.7 * complexity + 0.3 * prior
        complexity = _safe_minmax(complexity)
        blend["unsup"], blend["prior"] = 0.7, 0.3

    if sup.get("used", False):

        if prior is not None:

            complexity = 0.5 * complexity + 0.5 * sup["complexity_sup_pred"]
            blend["unsup"], blend["prior"], blend["sup"] = 0.35, 0.15, 0.5
        else:

            complexity = 0.6 * complexity + 0.4 * sup["complexity_sup_pred"]
            blend["unsup"], blend["sup"] = 0.6, 0.4
        complexity = _safe_minmax(complexity)
        blend["sup_target"] = sup.get("target_from")


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


    out["blend_w_unsup"] = blend["unsup"]
    out["blend_w_prior"] = blend["prior"]
    out["blend_w_sup"]   = blend["sup"]
    if blend["sup_target"] is not None:
        out["sup_target"] = blend["sup_target"]


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
