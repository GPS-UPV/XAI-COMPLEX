import json
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr


FEATURES_JSON = "./graphs/features.json"
SCORES_CSV    = "./graphs/complexity_scores_W.csv"
OUT_DIR       = "./figures"

YCOL_PREF     = [
    "complexity_supervised_0_1",
    "complexity_sup_pred",
    "complexity_sup",
    "sup_pred",
    "complexity_0_1",
]

def load_features(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index")
    return df


def pick_ycol(df: pd.DataFrame) -> str:
    for c in YCOL_PREF:
        if c in df.columns:
            return c
    raise ValueError(
        f"No encuentro ninguna columna target en el CSV. Columnas disponibles (muestra): {list(df.columns)[:30]}..."
    )


def coerce_features_to_numeric(df_feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df_num = df_feats.copy()

    def _to_num(v):
        if isinstance(v, (dict, list, tuple, set)):
            return np.nan
        if isinstance(v, str):
            s = v.strip()
            if s in {"{}", "[]", "", "None", "nan", "NaN"}:
                return np.nan
        return v

    for c in df_num.columns:
        if df_num[c].dtype == "object":
            df_num[c] = df_num[c].map(_to_num)
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    df_num = df_num.replace([np.inf, -np.inf], np.nan)

    dropped_all_nan = df_num.columns[df_num.isna().all()].tolist()
    if dropped_all_nan:
        df_num = df_num.drop(columns=dropped_all_nan)

    dropped_constant = df_num.columns[df_num.nunique(dropna=True) <= 1].tolist()
    if dropped_constant:
        df_num = df_num.drop(columns=dropped_constant)

    return df_num, dropped_all_nan, dropped_constant


def main():
    
    start_time = time.time()
    
    os.makedirs(OUT_DIR, exist_ok=True)

    df_feats = load_features(FEATURES_JSON)
    df_feats = df_feats.drop(columns=[c for c in df_feats.columns if "energy" in c.strip().lower() or "num_nodes" == c])
    scores = pd.read_csv(SCORES_CSV, index_col=0)
    df_all_feats = pd.read_csv("./graphs/all_features.csv", index_col=0)
    
    drop_columns = ["seed", "speed", "rddd", "instance_name", "sup_pred_complexity"]
    feat_mask = [c for c in df_feats.columns if "_d_" not in c or "_c_" not in c]

    # --- Target ---
    ycol = pick_ycol(scores)
    y = scores.reindex(df_feats.index)[ycol]
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(y.values)
    
    df_all = df_all_feats.reindex(df_feats.index)
    df_all = df_all.drop(columns=[c for c in df_all.columns if "energy" in c.strip().lower() or c in drop_columns])
    
    df_num, dropped_all_nan, dropped_constant = coerce_features_to_numeric(df_feats[feat_mask])
    
    lines = []
    lines.append("=== SHAP SUPERVISED REPORT ===\n")

    print(f"[target] {ycol} | aligned labels: {int(mask.sum())}/{len(mask)}")
    lines.append(f"[target] {ycol} | aligned labels: {int(mask.sum())}/{len(mask)}")
    print(f"[features] raw: {df_feats.shape} -> numeric: {df_num.shape}")
    lines.append(f"[features] raw: {df_feats.shape} -> numeric: {df_num.shape}")
    print(f"[features] dropped all-NaN cols: {len(dropped_all_nan)}")
    lines.append(f"[features] dropped all-NaN cols: {len(dropped_all_nan)}")
    print(f"[features] dropped constant cols: {len(dropped_constant)}")
    lines.append(f"[features] dropped constant cols: {len(dropped_constant)}\n")

    if mask.sum() < 20:
        raise RuntimeError(f"Demasiadas pocas labels alineadas tras limpieza: {int(mask.sum())}")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_num.values.astype(float))

    feature_names = df_num.columns.tolist()
    
    feature_map = {
        "n_jobs": r"$|J|$",
        "n_machines": r"$|M|$",
        "node_x_col0_min": r"$E_d^{min}$",
        "node_x_col0_max": r"$E_d^{max}$",
        "node_x_col0_mean": r"$E_d^{mean}$",
        "node_x_col0_median": r"$E_d^{median}$",
        "node_x_col0_std": r"$E_d^{std}$",
        "node_x_col0_range": r"$E_d^{range}$",
        "node_x_col0_q1": r"$E_d^{q1}$",
        "node_x_col0_q3": r"$E_d^{q3}$",
        "node_x_col0_gini": r"$E_c^{gini}$",
        "node_x_col1_min": r"$E_c^{min}$",
        "node_x_col1_max": r"$E_c^{max}$",
        "node_x_col1_mean": r"$E_c^{mean}$",
        "node_x_col1_median": r"$E_c^{median}$",
        "node_x_col1_std": r"$E_c^{std}$",
        "node_x_col1_range": r"$E_c^{range}$",
        "node_x_col1_q1": r"$E_c^{q1}$",
        "node_x_col1_q3": r"$E_c^{q3}$",
        "node_x_col1_gini": r"$E_c^{gini}$",
        "node_x_overall_mean": r"$E^{mean}$",
        "node_x_overall_std": r"$E^{std}$",
        "num_conjunctive_edges": r"$|E_c|$",
        "num_disjunctive_edges": r"$|E_d|$",
        "num_nodes_total": r"$|V|$",
        "num_edges_total": r"$|E|$",
        "disj_graph_density": r"$dens(G^d)$",
        "deg_d_min": r"$\deg_d^{min}$",
        "deg_d_max": r"$\deg_d^{max}$",
        "deg_d_mean": r"$\deg_d^{mean}$",
        "deg_d_median": r"$\deg_d^{median}$",
        "deg_d_std": r"$\deg_d^{std}$",
        "deg_d_range": r"$\deg_d^{range}$",
        "deg_d_q1": r"$\deg_d^{q1}$",
        "deg_d_q3": r"$\deg_d^{q3}$",
        "deg_d_gini": r"$\deg_d^{gini}$",
        "clustering_min": r"$\mathcal{C}_{min}$",
        "clustering_max": r"$\mathcal{C}_{max}$",
        "clustering_mean": r"$\mathcal{C}_{mean}$",
        "clustering_median": r"$\mathcal{C}_{median}$",
        "clustering_std": r"$\mathcal{C}_{std}$",
        "clustering_range": r"$\mathcal{C}_{range}$",
        "clustering_q1": r"$\mathcal{C}_{q1}$",
        "clustering_q3": r"$\mathcal{C}_{q3}$",
        "clustering_gini": r"$\mathcal{C}_{gini}$",
        "conj_graph_density": r"$dens(G^c)$",
        "deg_c_min": r"$\deg_c^{min}$",
        "deg_c_max": r"$\deg_c^{max}$",
        "deg_c_mean": r"$\deg_c^{mean}$",
        "deg_c_median": r"$\deg_c^{median}$",
        "deg_c_std": r"$\deg_c^{std}$",
        "deg_c_range": r"$\deg_c^{range}$",
        "deg_c_q1": r"$\deg_c^{q1}$",
        "deg_c_q3": r"$\deg_c^{q3}$",
        "deg_c_gini": r"$\deg_c^{gini}$",
        "betweenness_min": r"$\mathcal{B}_{min}$",
        "betweenness_max": r"$\mathcal{B}_{max}$",
        "betweenness_mean": r"$\mathcal{B}_{mean}$",
        "betweenness_median": r"$\mathcal{B}_{median}$",
        "betweenness_std": r"$\mathcal{B}_{std}$",
        "betweenness_range": r"$\mathcal{B}_{range}$",
        "betweenness_q1": r"$\mathcal{B}_{q1}$",
        "betweenness_q3": r"$\mathcal{B}_{q3}$",
        "betweenness_gini": r"$\mathcal{B}_{gini}$",
        "makespan_min": r"$C_{min}$",
        "makespan_range": r"$C_{range}$",
        "operation_cost_mean": r"$\mathcal{O}_cost^{mean}$",
        "operation_cost_std": r"$\mathcal{O}_cost^{std}$",
        "makespan_lb_job_sum": r"$\underline{C}_{\max}^{\,\mathrm{J}}$",
        "makespan_lb_machine_sum": r"$\underline{C}_{\max}^{\,\mathrm{M}}$",
        "makespan_lb_meanload": r"$\underline{C}_{\max}^{\,\mathrm{load}}$",
        "num_op_nodes": r"$|\mathcal{O}|$",
    }
    
    Xs = X[mask]
    Xs_df = pd.DataFrame(Xs, index=df_num.index[mask], columns=feature_names)
    ys = y.values.astype(float)[mask]
    
    rf_start_time = time.time()
    
    hours, rest = divmod(rf_start_time, 3600)
    _, hours = divmod(hours, 24)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"RANDOM FOREST REGRESSOR FITTING STARTED AT: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")

    rf = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    r2 = cross_val_score(rf, Xs, ys, scoring=make_scorer(r2_score), cv=cv).mean()
    print(f"[model] RandomForestRegressor | CV R2 mean: {r2:.4f}")
    lines.append(f"[model] RandomForestRegressor | CV R2 mean: {r2:.4f}\n")
    
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
    
    rf_end_time = time.time()
    
    hours, rest = divmod(rf_end_time, 3600)
    _, hours = divmod(hours, 24)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"RANDOM FOREST REGRESSOR FITTING ENDED AT: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
    
    rf_time = rf_end_time - rf_start_time
        
    y_pred_full = np.clip(rf.predict(Xs), 0.0, 1.0)

    feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    metrics = {
        "used": True,
        "model": "RandomForestRegressor",
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
    }
    
    lines.append(
        f"Supervised: YES ({metrics.get('model')}) "
        f"cv_r2={metrics.get('cv_r2_mean'):.3f}±{metrics.get('cv_r2_std'):.3f} "
        f"cv_mae={metrics.get('cv_mae_mean'):.3f}±{metrics.get('cv_mae_std'):.3f} "
        f"cv_rmse={metrics.get('cv_rmse_mean'):.3f}±{metrics.get('cv_rmse_std'):.3f} "
        f"oof_spearman={metrics.get('oof_spearman_rho'):.3f}\n"
    )
    
    print(f"Supervised: YES ({metrics.get('model')}) ")
    print(f"cv_r2={metrics.get('cv_r2_mean'):.3f}±{metrics.get('cv_r2_std'):.3f} ")
    print(f"cv_mae={metrics.get('cv_mae_mean'):.3f}±{metrics.get('cv_mae_std'):.3f} ")
    print(f"cv_rmse={metrics.get('cv_rmse_mean'):.3f}±{metrics.get('cv_rmse_std'):.3f} ")
    print(f"oof_spearman={metrics.get('oof_spearman_rho'):.3f}")
    
    df_taillard = load_features("./TaillardInstancesGRAPHS/features.json")
    df_taillard_tags = pd.read_csv("./all_taillard_with_groundtruth_and_est_tags.csv")
    df_pred_t = df_taillard[feature_names]
    
    df_taillard_tags.index = df_pred_t.index
    
    df_taillard_tags = df_taillard_tags.drop(columns=[c for c in df_taillard_tags if "shap" in c or c in df_taillard.columns or "speed" in c or c in drop_columns])
    
    df_all_taillard = df_taillard.join(df_taillard_tags)
                
    Xt = imputer.fit_transform(df_pred_t.values.astype(float))
    Xt_df = pd.DataFrame(Xt, index=df_pred_t.index, columns=feature_names)
    yt = rf.predict(Xt)
    
    # --- SHAP ---
    
    shap_start_time = time.time()
        
    hours, rest = divmod(shap_start_time, 3600)
    _, hours = divmod(hours, 24)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"TREE EXPLAINER STARTED AT: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs)
    
    shap1_end_time = time.time()
        
    hours, rest = divmod(shap1_end_time, 3600)
    _, hours = divmod(hours, 24)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"EXPLANATION OF Xs OBTAINED AT: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
    
    shap1_time = shap1_end_time - shap_start_time
    
    shap_values_taillard = explainer.shap_values(Xt)
    
    shap2_end_time = time.time()
        
    hours, rest = divmod(shap2_end_time, 3600)
    _, hours = divmod(hours, 24)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"EXPLANATION OF Xt OBTAINED AT: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
    
    shap2_time = shap2_end_time - shap1_end_time
    
    shap_df = pd.DataFrame(shap_values, index=Xs_df.index, columns=feature_names)
    shap_df.to_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}.csv"))
    
    df_all_and_shap = df_all.join(shap_df.add_suffix('_shap'))
    df_all_and_shap.to_csv('all_features_and_shap.csv')

    shap_df_taillard = pd.DataFrame(shap_values_taillard, index=Xt_df.index, columns=feature_names)
    shap_df_taillard.to_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}_taillard.csv"))
    
    df_all_taillard['complexity_supervised_0_1'] = yt
    df_all_taillard = df_all_taillard.join(shap_df_taillard.add_suffix('_shap'))
    df_all_taillard.to_csv('all_taillard_and_shap.csv')
    
    # shap_df = pd.read_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}.csv"))
    # shap_df.index= Xs_df.index
    # shap_df = shap_df[feature_names]
    
    # shap_df_taillard = pd.read_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}_taillard.csv"))
    # shap_df_taillard.index= Xt_df.index
    # shap_df_taillard = shap_df_taillard[feature_names]

    imp = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, f"shap_importance_{ycol}.csv"), header=["mean_abs_shap"])

    imp_taillard = shap_df_taillard.abs().mean(axis=0).sort_values(ascending=False)
    imp_taillard.to_csv(os.path.join(OUT_DIR, f"shap_importance_{ycol}_taillard.csv"), header=["mean_abs_shap"])

    optimal_mask, feasible_mask, timeout_mask = [], [], []
    
    for i in df_all.index:
        if "OPTIMAL" in df_all.loc[i, "quality_tag"].strip().upper():
            optimal_mask.append(i)
        elif "FEASIBLE" in df_all.loc[i, "quality_tag"].strip().upper():
            feasible_mask.append(i)
        elif "TIMEOUT" in df_all.loc[i, "quality_tag"].strip().upper():
            timeout_mask.append(i)
            
    easy_jsplib_mask, medium_jsplib_mask, hard_jsplib_mask = set(), set(), set()
    easy_predict_mask, medium_predict_mask, hard_predict_mask = set(), set(), set() 
            
    for i in df_all_taillard.index:
        if "EASY" in df_all_taillard.loc[i, "jsplib_tag"].strip().upper():
            easy_jsplib_mask.add(i)
        elif "MEDIUM" in df_all_taillard.loc[i, "jsplib_tag"].strip().upper():
            medium_jsplib_mask.add(i)
        elif "HARD" in df_all_taillard.loc[i, "jsplib_tag"].strip().upper():
            hard_jsplib_mask.add(i)
        
        if "EASY" in df_all_taillard.loc[i, "predict_tag"].strip().upper():
            easy_predict_mask.add(i)
        elif "MEDIUM" in df_all_taillard.loc[i, "predict_tag"].strip().upper():
            medium_predict_mask.add(i)
        elif "HARD" in df_all_taillard.loc[i, "predict_tag"].strip().upper():
            hard_predict_mask.add(i)       
            
    autores = {}
    
    for i in df_all_taillard.index:
        a = ''.join(filter(str.isalpha, i[:-3]))
        if a not in autores:
            autores[a] = set()
        autores[a].add(i)
    
    Xs_df_optimal, Xs_df_feasible, Xs_df_timeout = Xs_df.loc[optimal_mask], Xs_df.loc[feasible_mask], Xs_df.loc[timeout_mask]
    
    shap_values_optimal, shap_values_feasible, shap_values_timeout = shap_df.loc[optimal_mask].values, shap_df.loc[feasible_mask].values, shap_df.loc[timeout_mask].values
    
    xmin, xmax = shap_df.min(axis=None), shap_df.max(axis=None)
    
    plt.rcParams['text.usetex'] = True
    plt.ticklabel_format(useMathText=False)
        
    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values_optimal, Xs_df_optimal, show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24, fontweight="bold")
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_optimal.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_feasible, Xs_df_feasible, show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_feasible.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_timeout, Xs_df_timeout, show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_timeout.png"), dpi=400)
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values_optimal, Xs_df_optimal, plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_optimal.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_feasible, Xs_df_feasible, plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_feasible.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_timeout, Xs_df_timeout, plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_timeout.png"), dpi=400)
    plt.close()
    
    for a in autores.keys():
        easy_predict = list(autores[a].intersection(easy_predict_mask))
        easy_jsplib = list(autores[a].intersection(easy_jsplib_mask))
        
        medium_predict = list(autores[a].intersection(medium_predict_mask))
        medium_jsplib = list(autores[a].intersection(medium_jsplib_mask))
        
        hard_predict = list(autores[a].intersection(hard_predict_mask))
        hard_jsplib = list(autores[a].intersection(hard_jsplib_mask))
        
        autor = list(autores[a])
        
        shap.summary_plot(shap_df_taillard.loc[autor].values, Xt_df.loc[autor], show=False, max_display=20)
        ax = plt.gca()
        ax.xaxis.label.set_visible(False)
        new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
        ax.set_yticklabels(new_labels, fontsize=24)
        ax.xaxis.set_tick_params(labelsize=24)
        plt.xlim(xmin, xmax)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_{a}.png"), dpi=400)
        plt.close()
        
        plt.figure()
        shap.summary_plot(shap_df_taillard.loc[autor].values, Xt_df.loc[autor], plot_type="bar", show=False, max_display=20)
        ax = plt.gca()
        ax.xaxis.label.set_visible(False)
        new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
        ax.set_yticklabels(new_labels, fontsize=24)
        ax.xaxis.set_tick_params(labelsize=24)
        plt.xlim(0, 0.03)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_{a}.png"), dpi=400)
        plt.close()
        
        if len(easy_predict) != 0: 
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[easy_predict].values, Xt_df.loc[easy_predict], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_easy_predict_{a}.png"), dpi=400)
            plt.close()
            
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[easy_predict].values, Xt_df.loc[easy_predict], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_easy_predict_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay easy_predict_{a}")
            lines.append(f"No hay easy_predict_{a}")
        
        if len(medium_predict) != 0:
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[medium_predict].values, Xt_df.loc[medium_predict], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_medium_predict_{a}.png"), dpi=400)
            plt.close()
            
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[medium_predict].values, Xt_df.loc[medium_predict], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_medium_predict_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay medium_predict_{a}")
            lines.append(f"No hay medium_predict_{a}")
        
        if len(hard_predict) != 0:    
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[hard_predict].values, Xt_df.loc[hard_predict], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_hard_predict_{a}.png"), dpi=400)
            plt.close()
            
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[hard_predict].values, Xt_df.loc[hard_predict], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_hard_predict_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay hard_predict_{a}")
            lines.append(f"No hay hard_predict_{a}")
            
        if len(easy_jsplib) != 0:
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[easy_jsplib].values, Xt_df.loc[easy_jsplib], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_easy_jsplib_{a}.png"), dpi=400)
            plt.close()
            
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[easy_jsplib].values, Xt_df.loc[easy_jsplib], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_easy_jsplib_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay easy_jsplib_{a}")
            lines.append(f"No hay easy_jsplib_{a}")
        
        if len(medium_jsplib) != 0:
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[medium_jsplib].values, Xt_df.loc[medium_jsplib], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_medium_jsplib_{a}.png"), dpi=400)
            plt.close()
            
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[medium_jsplib].values, Xt_df.loc[medium_jsplib], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_medium_jsplib_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay medium_jsplib_{a}")
            lines.append(f"No hay medium_jsplib_{a}")
            
        if len(hard_jsplib) != 0:
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[hard_jsplib].values, Xt_df.loc[hard_jsplib], show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_hard_jsplib_{a}.png"), dpi=400)
            plt.close()
        
            plt.figure()
            shap.summary_plot(shap_df_taillard.loc[hard_jsplib].values, Xt_df.loc[hard_jsplib], plot_type="bar", show=False, max_display=20)
            ax = plt.gca()
            ax.xaxis.label.set_visible(False)
            new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
            ax.set_yticklabels(new_labels, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=24)
            plt.xlim(0, 0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_hard_jsplib_{a}.png"), dpi=400)
            plt.close()
        else:
            print(f"No hay hard_jsplib_{a}")
            lines.append(f"No hay hard_jsplib_{a}")
            
    easy_jsplib_mask, medium_jsplib_mask, hard_jsplib_mask, easy_predict_mask, medium_predict_mask, hard_predict_mask = list(easy_jsplib_mask), list(medium_jsplib_mask), list(hard_jsplib_mask), list(easy_predict_mask), list(medium_predict_mask), list(hard_predict_mask)
            
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[easy_predict_mask].values, Xt_df.loc[easy_predict_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_easy_predict.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[easy_predict_mask].values, Xt_df.loc[easy_predict_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_easy_predict.png"), dpi=400)
    plt.close()


    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[medium_predict_mask].values, Xt_df.loc[medium_predict_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_medium_predict.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[medium_predict_mask].values, Xt_df.loc[medium_predict_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_medium_predict.png"), dpi=400)
    plt.close()


    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[hard_predict_mask].values, Xt_df.loc[hard_predict_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_hard_predict.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[hard_predict_mask].values, Xt_df.loc[hard_predict_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_hard_predict.png"), dpi=400)
    plt.close()
    
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[easy_jsplib_mask].values, Xt_df.loc[easy_jsplib_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_easy_jsplib.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[easy_jsplib_mask].values, Xt_df.loc[easy_jsplib_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_easy_jsplib.png"), dpi=400)
    plt.close()


    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[medium_jsplib_mask].values, Xt_df.loc[medium_jsplib_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_medium_jsplib.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[medium_jsplib_mask].values, Xt_df.loc[medium_jsplib_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_medium_jsplib.png"), dpi=400)
    plt.close()
    
    
    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[hard_jsplib_mask].values, Xt_df.loc[hard_jsplib_mask], show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_hard_jsplib.png"), dpi=400)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_df_taillard.loc[hard_jsplib_mask].values, Xt_df.loc[hard_jsplib_mask], plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.xaxis.label.set_visible(False)
    new_labels = [feature_map.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_hard_jsplib.png"), dpi=400)
    plt.close()
    
    hours, rest = divmod(rf_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"\nRANDOM FOREST FITTING TIME: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    hours, rest = divmod(shap1_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"\nSHAP VALUES EXPLAINER FOR Xs TIME: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    hours, rest = divmod(shap2_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"\nSHAP VALUES EXPLAINER FOR Xt TIME: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    total_execution_time = time.time() - start_time
    
    hours, rest = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"\nTOTAL EXECUTION TIME: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    with open("shap_supervised_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("OK: SHAP guardado en", OUT_DIR)


if __name__ == "__main__":
    main()