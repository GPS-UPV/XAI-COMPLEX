#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evalúa la calidad de las categorías (easy/medium/hard) usando las características de las soluciones,
en especial el tiempo de resolución, y analiza las diferencias entre grupos.

Entradas:
  - ./graphs/complexity_scores.csv
  - ./solutions/solution_features.csv

Salidas:
  - ./analysis/category_evaluation_report.txt
  - ./analysis/category_group_stats.csv
  - ./analysis/group_differences.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, f_oneway, spearmanr

# =========================================================
# Funciones auxiliares
# =========================================================

def normalize_id_from_solutions(sol_id: str) -> str:
    """Convierte '18_36-0-1-cp-sat' -> '18-36-0.pt'."""
    m = re.match(r"^(\d+)_(\d+)-(\d+)-", str(sol_id))
    if not m:
        return np.nan
    j, mchs, seed = m.groups()
    return f"{j}-{mchs}-{seed}.pt"

def safe_stat_test(groups):
    """Aplica ANOVA o Kruskal según disponibilidad."""
    valid_groups = [g for g in groups if len(g) >= 3]
    if len(valid_groups) < 2:
        return np.nan, np.nan
    try:
        f, p = f_oneway(*valid_groups)
    except Exception:
        f, p = kruskal(*valid_groups)
    return f, p


# =========================================================
# Configuración
# =========================================================
COMPLEXITY_FILE = "./graphs/complexity_scores.csv"
SOLUTIONS_FILE  = "./solutions/solution_features.csv"
OUT_DIR = "./analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# Carga y merge
# =========================================================
comp = pd.read_csv(COMPLEXITY_FILE)
sol  = pd.read_csv(SOLUTIONS_FILE)

# Normalización IDs
comp["instance_id_norm"] = comp["instance_id"].astype(str)
sol["instance_id_norm"]  = sol["instance_id"].apply(normalize_id_from_solutions)
sol = sol[sol["instance_id_norm"].notna()]

df = pd.merge(comp, sol, on="instance_id_norm", how="inner", suffixes=("_c", "_s"))

if df.empty:
    raise RuntimeError("No se pudieron emparejar las instancias; revisa los IDs.")

df["category"] = df["category"].str.lower().str.strip()
cat_map = {"easy": 0, "medium": 1, "hard": 2}
df["cat_num"] = df["category"].map(cat_map)

# =========================================================
# Variables clave para evaluar
# =========================================================
key_time_cols = [c for c in df.columns if any(x in c.lower() for x in ["solve", "time", "flattime", "totaltime"])]

# Filtramos las más relevantes y presentes
key_time_cols = [c for c in key_time_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
if not key_time_cols:
    raise ValueError("No se encontraron columnas de tiempo en las features.")

# =========================================================
# Análisis global de tiempos
# =========================================================
summary = []
for col in key_time_cols:
    stats = df.groupby("category")[col].agg(["mean", "median", "std", "min", "max", "count"])
    easy, med, hard = [df.loc[df["category"]==g, col].dropna() for g in ["easy","medium","hard"]]
    fval, pval = safe_stat_test([easy, med, hard])
    rho, p_rho = spearmanr(df["cat_num"], df[col], nan_policy="omit")
    diff_order = (stats.loc["easy","median"] < stats.loc["medium","median"] < stats.loc["hard","median"])
    summary.append({
        "feature": col,
        "easy_med": stats.loc["easy","median"] if "easy" in stats.index else np.nan,
        "medium_med": stats.loc["medium","median"] if "medium" in stats.index else np.nan,
        "hard_med": stats.loc["hard","median"] if "hard" in stats.index else np.nan,
        "anova_p": pval,
        "spearman_r": rho,
        "increasing_order": diff_order
    })

summary_df = pd.DataFrame(summary).sort_values("spearman_r", ascending=False)
summary_df.to_csv(os.path.join(OUT_DIR, "category_group_stats.csv"), index=False)

# =========================================================
# Análisis de otras diferencias entre grupos
# =========================================================
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["cat_num"]]
res = []
for col in num_cols:
    if df[col].nunique() <= 3:
        continue
    easy, med, hard = [df.loc[df["category"]==g, col].dropna() for g in ["easy","medium","hard"]]
    fval, pval = safe_stat_test([easy, med, hard])
    rho, _ = spearmanr(df["cat_num"], df[col], nan_policy="omit")
    if not np.isnan(pval) and pval < 0.05:
        res.append({"feature": col, "anova_p": pval, "spearman_r": rho})
res = pd.DataFrame(res).sort_values("anova_p").reset_index(drop=True)

# =========================================================
# Gráficos comparativos de tiempos
# =========================================================
plt.figure(figsize=(10,6))
melted = df.melt(id_vars=["category"], value_vars=key_time_cols, var_name="Metric", value_name="Time")
sns.boxplot(data=melted, x="category", y="Time", hue="Metric")
plt.title("Comparativa de tiempos por categoría")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "group_differences.png"), dpi=200)
plt.close()

# =========================================================
# Informe textual
# =========================================================
lines = []
lines.append("=== Evaluación de las categorías (easy / medium / hard) ===\n")
lines.append(f"Total instancias analizadas: {len(df)}")
lines.append(f"Columnas de tiempo analizadas: {', '.join(key_time_cols)}\n")

for _, row in summary_df.iterrows():
    lines.append(
        f"- {row['feature']}: ρ={row['spearman_r']:.3f}, p={row['anova_p']:.2e}, "
        f"medianas (E/M/H) = {row['easy_med']:.2f} / {row['medium_med']:.2f} / {row['hard_med']:.2f}, "
        f"orden creciente={row['increasing_order']}"
    )

lines.append("\n=== Análisis de diferencias significativas ===")
if res.empty:
    lines.append("No se encontraron diferencias estadísticamente significativas entre grupos (p<0.05).")
else:
    lines.append("Principales features que diferencian grupos:")
    for _, r in res.head(15).iterrows():
        lines.append(f"  {r['feature']:<35} p={r['anova_p']:.2e}  ρ={r['spearman_r']:.3f}")

# Evaluación general
mean_rho = summary_df["spearman_r"].mean()
good_order = summary_df["increasing_order"].mean()
lines.append("\n=== Valoración global ===")
if mean_rho > 0.4 and good_order > 0.8:
    lines.append("Las categorías están muy bien estructuradas: tiempos y métricas crecen consistentemente con la dificultad.")
elif mean_rho > 0.25 and good_order > 0.5:
    lines.append("Las categorías son razonables pero podrían ajustarse: algunos solapamientos entre grupos.")
else:
    lines.append("Las categorías parecen arbitrarias o mal calibradas: poca relación entre tiempo y nivel asignado.")

lines.append(f"\nCorrelación media tiempo-dificultad: ρ={mean_rho:.3f}")
lines.append(f"Proporción de métricas con orden creciente E<M<H: {good_order*100:.1f}%")

# Guardar
out_txt = os.path.join(OUT_DIR, "category_evaluation_report.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("[OK] Guardado:")
print(" -", out_txt)
print(" -", os.path.join(OUT_DIR, "category_group_stats.csv"))
print(" -", os.path.join(OUT_DIR, "group_differences.png"))
