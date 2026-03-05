#!/usr/bin/env python3
"""Generate evaluation figures.

Input:
  - all_features_and_shap.csv

Output (saved under ./figures):
  - score_by_status_boxplot.png
  - score_vs_time_scatter.png
  - shap_cumulative_mass_by_status.png

Notes
-----
Figure styling is intentionally kept close to the minimalist SHAP-summary look:
no top/right spines, subtle dotted horizontal grid, and compact typography.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "all_features_and_shap.csv"
FIGS = ROOT / "figures"
STATUS_ORDER = ["optimal", "feasible", "timeout"]


def derive_status3(row: pd.Series) -> str:
    if row["status"] == "OPTIMAL_SOLUTION":
        return "optimal"
    if bool(row["timelimit_hit"]):
        return "timeout"
    return "feasible"


def minmax01(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def _style_ax(ax: plt.Axes, *, y_grid: bool = True, x_grid: bool = False) -> None:
    """Apply a clean, SHAP-like look."""
    ax.set_facecolor("white")

    # Minimal spines (SHAP summary style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Clean ticks
    ax.tick_params(axis="y", length=0)

    # Subtle dotted grid (mostly horizontal)
    ax.grid(False)
    if y_grid:
        ax.grid(True, axis="y", linestyle=":", linewidth=0.9, alpha=0.35)
    if x_grid:
        ax.grid(True, axis="x", linestyle=":", linewidth=0.9, alpha=0.20)

    ax.set_axisbelow(True)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    df["C"] = minmax01(df["supervised_complexity"])
    df["status3"] = df.apply(derive_status3, axis=1)

    # --------------------
    # 1) Distribution by solver outcome (horizontal boxplot)
    # --------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    data = [df.loc[df["status3"] == s, "C"].to_numpy() for s in STATUS_ORDER]
    ax.boxplot(
        data,
        vert=False,
        labels=STATUS_ORDER,
        showfliers=False,
        widths=0.6,
        medianprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    ax.set_xlabel("C(x) (0–1)")
    ax.set_ylabel("Solver status")
    ax.set_xlim(0, 1)
    _style_ax(ax, y_grid=True, x_grid=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGS / "score_by_status_boxplot.png", dpi=300)
    plt.close(fig)

    # --------------------
    # 2) Alignment with solver effort: C(x) vs solve time (log-x)
    # --------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.scatter(df["solveTime"].to_numpy(), df["C"].to_numpy(), s=9, alpha=0.45, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("Solve time (ms)")
    ax.set_ylabel("C(x) (0–1)")
    _style_ax(ax, y_grid=True, x_grid=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGS / "score_vs_time_scatter.png", dpi=300)
    plt.close(fig)

    # --------------------
    # 3) SHAP concentration: cumulative mass of |SHAP| for top-k features
    # --------------------
    shap_cols = [c for c in df.columns if c.endswith("_shap")]
    kmax = 25

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    xs = np.arange(1, kmax + 1)

    for s in STATUS_ORDER:
        mean_abs = df.loc[df["status3"] == s, shap_cols].abs().mean().sort_values(ascending=False)
        cum = (mean_abs.cumsum() / (mean_abs.sum() + 1e-12)).iloc[:kmax].to_numpy()
        ax.plot(xs, cum, marker="o", markersize=3, linewidth=1.1, label=s)

    mean_abs_all = df[shap_cols].abs().mean().sort_values(ascending=False)
    cum_all = (mean_abs_all.cumsum() / (mean_abs_all.sum() + 1e-12)).iloc[:kmax].to_numpy()
    ax.plot(xs, cum_all, marker="o", markersize=3, linewidth=1.1, label="all")

    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, kmax)
    ax.set_xlabel("k (number of features)")
    ax.set_ylabel("Cumulative mass of |SHAP|")
    ax.legend(frameon=False, ncol=2)
    _style_ax(ax, y_grid=True, x_grid=False)

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGS / "shap_cumulative_mass_by_status.png", dpi=300)
    plt.close(fig)

    print("OK: figures saved under", FIGS)


if __name__ == "__main__":
    main()
