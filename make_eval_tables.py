from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "all_features_and_shap.csv"
OUT = ROOT / "tables"
STATUS_ORDER = ["optimal", "feasible", "timeout"]


def derive_status3(row: pd.Series) -> str:
    if row["status"] == "OPTIMAL_SOLUTION":
        return "optimal"
    if bool(row["timelimit_hit"]):
        return "timeout"
    return "feasible"


def minmax01(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
):
    rho = stats.spearmanr(x, y).statistic
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)

    X = rx[idx]
    Y = ry[idx]
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    cov = (Xc * Yc).mean(axis=1)
    stdx = Xc.std(axis=1)
    stdy = Yc.std(axis=1)
    boot = cov / (stdx * stdy + 1e-12)

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(rho), float(lo), float(hi)


def cliffs_delta_from_u(u: float, nx: int, ny: int) -> float:
    return (2.0 * u) / (nx * ny) - 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    df["C"] = minmax01(df["supervised_complexity"])
    df["status3"] = df.apply(derive_status3, axis=1)

    # --------------------
    # outcome_score
    # --------------------
    lines = []
    lines.append(r"\begin{tabular}{l r c c r}")
    lines.append(r"\toprule")
    lines.append(r"Outcome & $N$ & $\mathrm{mean}(C)$ & $\mathrm{std}(C)$ & median time (ms)\\")
    lines.append(r"\midrule")
    for s in STATUS_ORDER:
        g = df[df["status3"] == s]
        lines.append(
            f"{s} & {len(g)} & {g['C'].mean():.3f} & {g['C'].std(ddof=0):.3f} & {int(round(g['solveTime'].median()))}\\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "outcome_score.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --------------------
    # score_corr_ci
    # --------------------
    metrics = [
        ("wall time (ms)", "solveTime"),
        ("propagations", "propagations"),
        ("failures", "failures"),
        ("relative gap", "gap_rel"),
        ("absolute gap", "gap_abs"),
    ]

    x = df["C"].to_numpy()
    lines = []
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"Solver-effort indicator & Spearman $\rho$ & 95\% CI (bootstrap)\\")
    lines.append(r"\midrule")
    for label, col in metrics:
        rho, lo, hi = bootstrap_spearman_ci(x, df[col].to_numpy(), n_resamples=2000, seed=0)
        lines.append(f"{label} & {rho:.3f} & [{lo:.3f}, {hi:.3f}]\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "score_corr_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --------------------
    # cat_separability
    # --------------------
    q1, q2 = df["C"].quantile([1 / 3, 2 / 3])

    def cat(c: float) -> str:
        if c <= q1:
            return "easy"
        if c <= q2:
            return "medium"
        return "hard"

    df["category"] = df["C"].apply(cat)

    lines = []
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Category & $N$ & median time (ms) & median propagations & median failures\\")
    lines.append(r"\midrule")
    for c in ["easy", "medium", "hard"]:
        g = df[df["category"] == c]
        lines.append(
            f"\\texttt{{{c}}} & {len(g)} & {int(round(g['solveTime'].median()))} & {int(round(g['propagations'].median()))} & {int(round(g['failures'].median()))}\\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "cat_separability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --------------------
    # score_pairwise_tests (MWU + Cliff's delta)
    # --------------------
    pairs = [("optimal", "feasible"), ("feasible", "timeout"), ("optimal", "timeout")]

    lines = []
    lines.append(r"\begin{tabular}{l l r r l}")
    lines.append(r"\toprule")
    lines.append(r"Group A & Group B & $p$ (MWU) & $\delta$ (Cliff) & Magnitude\\")
    lines.append(r"\midrule")

    for a, b in pairs:
        xa = df.loc[df["status3"] == a, "C"].to_numpy()
        xb = df.loc[df["status3"] == b, "C"].to_numpy()
        mwu = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        delta = cliffs_delta_from_u(mwu.statistic, len(xa), len(xb))

        ad = abs(delta)
        if ad < 0.147:
            mag = "negligible"
        elif ad < 0.33:
            mag = "small"
        elif ad < 0.474:
            mag = "medium"
        else:
            mag = "large"

        lines.append(f"{a} & {b} & {mwu.pvalue:.2e} & {delta:.3f} & {mag}\\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (OUT / "score_pairwise_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK: tables written to", OUT)


if __name__ == "__main__":
    main()
