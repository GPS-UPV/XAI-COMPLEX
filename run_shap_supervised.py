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
    # Formato típico: { "10-10-0.pt": {feat: val, ...}, ... }
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
    scores = pd.read_csv(SCORES_CSV, index_col=0)

    # --- Target ---
    ycol = pick_ycol(scores)
    y = scores.reindex(df_feats.index)[ycol]
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(y.values)

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

    # --- SHAP ---
    Xs = X[mask]
    Xs_df = pd.DataFrame(Xs, index=df_num.index[mask], columns=feature_names)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs)

    shap_df = pd.DataFrame(shap_values, index=Xs_df.index, columns=feature_names)
    shap_df.to_csv(os.path.join(OUT_DIR, f"shap_values_{ycol}.csv"))

    imp = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, f"shap_importance_{ycol}.csv"), header=["mean_abs_shap"])

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, Xs_df, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{ycol}.png"), dpi=220)
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, Xs_df, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{ycol}.png"), dpi=220)
    plt.close()

    print("OK: SHAP guardado en", OUT_DIR)


if __name__ == "__main__":
    main()