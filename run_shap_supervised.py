import json
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score


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
    os.makedirs(OUT_DIR, exist_ok=True)

    df_feats = load_features(FEATURES_JSON)
    df_feats = df_feats.drop(columns=[c for c in df_feats.columns if "energy" in c.strip().lower() or "num_nodes" == c])
    scores = pd.read_csv(SCORES_CSV, index_col=0)
    df_all_feats = pd.read_csv("all_features.csv", index_col=0)

    # --- Target ---
    ycol = pick_ycol(scores)
    y = scores.reindex(df_feats.index)[ycol]
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(y.values)
    
    df_all = df_all_feats.reindex(df_feats.index)

    df_num, dropped_all_nan, dropped_constant = coerce_features_to_numeric(df_feats)

    y = scores.reindex(df_num.index)[ycol]
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(y.values)

    print(f"[target] {ycol} | aligned labels: {int(mask.sum())}/{len(mask)}")
    print(f"[features] raw: {df_feats.shape} -> numeric: {df_num.shape}")
    print(f"[features] dropped all-NaN cols: {len(dropped_all_nan)}")
    print(f"[features] dropped constant cols: {len(dropped_constant)}")

    if mask.sum() < 20:
        raise RuntimeError(f"Demasiadas pocas labels alineadas tras limpieza: {int(mask.sum())}")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_num.values.astype(float))

    feature_names = df_num.columns.tolist()

    ys = y.values.astype(float)

    rf = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2 = cross_val_score(rf, X[mask], ys[mask], scoring=make_scorer(r2_score), cv=cv).mean()
    print(f"[model] RandomForestRegressor | CV R2 mean: {r2:.4f}")

    rf.fit(X[mask], ys[mask])
    
    df_taillard = load_features("./TaillardInstancesGRAPHS/features.json")
    df_pred_t = df_taillard[feature_names]
    
    Xt = imputer.fit_transform(df_pred_t.values.astype(float))
    Xt_df = pd.DataFrame(Xt, index=df_pred_t.index, columns=feature_names)
    yt = rf.predict(Xt)
    df_taillard['sup_pred_complexity'] = yt
        
    # --- SHAP ---
    Xs = X[mask]
    Xs_df = pd.DataFrame(Xs, index=df_num.index[mask], columns=feature_names)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs)
    shap_values_taillard = explainer.shap_values(Xt)

    shap_df = pd.DataFrame(shap_values, index=Xs_df.index, columns=feature_names)
    shap_df.to_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}.csv"))
    
    df_all_and_shap = df_all.join(shap_df.add_suffix('_shap'))
    df_all_and_shap.to_csv('all_features_and_shap.csv')

    shap_df_taillard = pd.DataFrame(shap_values_taillard, index=Xt_df.index, columns=feature_names)
    shap_df_taillard.to_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}_taillard.csv"))
    
    df_taillard['complexity_supervised_0_1'] = yt
    df_all_taillard = df_taillard.join(shap_df_taillard.add_suffix('_shap'))
    df_all_taillard.to_csv('all_taillard.csv')
    
    #shap_df = pd.read_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}.csv"))
    #shap_df.index= Xs_df.index
    #shap_df = shap_df.drop(columns="Unnamed: 0")

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
    
    Xs_df_optimal, Xs_df_feasible, Xs_df_timeout = Xs_df.loc[optimal_mask], Xs_df.loc[feasible_mask], Xs_df.loc[timeout_mask]
    
    shap_values_optimal, shap_values_feasible, shap_values_timeout = shap_df.loc[optimal_mask].values, shap_df.loc[feasible_mask].values, shap_df.loc[timeout_mask].values

    xmin, xmax = shap_df.min(axis=None), shap_df.max(axis=None)
        
    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values_optimal, Xs_df_optimal, show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.gca().yaxis
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_optimal.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_feasible, Xs_df_feasible, show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_feasible.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_timeout, Xs_df_timeout, show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_timeout.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_taillard, Xt_df, show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}_taillard.png"), dpi=400)
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values_optimal, Xs_df_optimal, plot_type="bar", show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_optimal.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_feasible, Xs_df_feasible, plot_type="bar", show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_feasible.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_timeout, Xs_df_timeout, plot_type="bar", show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_timeout.png"), dpi=400)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_taillard, Xt_df, plot_type="bar", show=False, max_display=20)
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.set_tick_params(labelsize=18)
    plt.gca().xaxis.set_tick_params(labelsize=18)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}_taillard.png"), dpi=400)
    plt.close()

    print("OK: SHAP guardado en", OUT_DIR)


if __name__ == "__main__":
    main()