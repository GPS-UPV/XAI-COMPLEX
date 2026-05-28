#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit estadístico de instancias JSP en formato tipo Taillard / OR-Library.

Analiza automáticamente todas las familias de un directorio o ZIP, por ejemplo:
  - abz*, ft*, la*, orb*, swv*, ta*, yn*, etc.

Qué analiza:
  1) Tiempos de proceso p_ij de cada operación.
  2) Órdenes de máquina como permutaciones, no como variables continuas.
     Para ellos se exportan frecuencias de máquina por posición.

Uso típico:
  python fit_jsp_distributions_by_family.py --input ./test --out fit_results --plots
  python fit_jsp_distributions_by_family.py --input instances.zip --out fit_results --plots

Salidas principales:
  - summary_by_file.csv
  - summary_by_family.csv                 # familias por prefijo: ta, swv, yn, la, ...
  - summary_by_size.csv                   # familias por tamaño: 10x10, 20x15, ...
  - fit_candidates_by_file.csv
  - fit_candidates_by_family.csv
  - fit_candidates_by_size.csv
  - global_fit.csv
  - global_summary.json
  - machine_position_summary.csv
  - parsing_errors.csv                    # solo si algún fichero no se puede leer
  - README_results.txt
  - plots/*.png                           # si se usa --plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# Lectura de instancias
# -----------------------------------------------------------------------------


def clean_useful_lines(path: Path) -> List[str]:
    """Lee líneas no vacías y no comentadas."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def load_jsp_file(path: Path) -> dict:
    """
    Carga una instancia JSP en formato:

        n_jobs n_machines
        m_1 p_1 m_2 p_2 ... m_m p_m
        ...

    Acepta máquinas 0-based y 1-based. Internamente las deja 0-based.
    """
    lines = clean_useful_lines(path)
    if not lines:
        raise ValueError("fichero vacío o solo con comentarios")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("la primera línea útil debe contener al menos dos enteros")

    try:
        num_jobs, num_mchs = int(header[0]), int(header[1])
    except Exception as e:
        raise ValueError(f"no se han podido leer num_jobs/num_mchs: {e}") from e

    if num_jobs <= 0 or num_mchs <= 0:
        raise ValueError(f"dimensiones no válidas: {num_jobs}x{num_mchs}")

    if len(lines) < num_jobs + 1:
        raise ValueError(
            f"se esperaban {num_jobs} líneas de jobs, pero solo hay {len(lines) - 1}"
        )

    order = np.empty((num_jobs, num_mchs), dtype=np.int64)
    proc_by_order = np.empty((num_jobs, num_mchs), dtype=np.int64)

    for j in range(num_jobs):
        vals = np.fromstring(lines[j + 1], sep=" ", dtype=np.int64)
        if vals.size != 2 * num_mchs:
            raise ValueError(
                f"línea del job {j} con {vals.size} valores; se esperaban {2 * num_mchs}"
            )

        machines = vals[0::2].copy()
        times = vals[1::2].copy()

        # Acepta índices 1-based: 1..m -> 0..m-1
        if machines.min() >= 1 and machines.max() <= num_mchs:
            machines -= 1

        if machines.min() < 0 or machines.max() >= num_mchs:
            raise ValueError(f"job {j} contiene máquinas fuera de rango")

        if np.unique(machines).size != num_mchs:
            raise ValueError(f"job {j} no contiene una permutación válida de máquinas")

        if np.any(times < 0):
            raise ValueError(f"job {j} contiene tiempos negativos")

        order[j] = machines
        proc_by_order[j] = times

    return {
        "file": path.name,
        "numJobs": num_jobs,
        "numMchs": num_mchs,
        "family": infer_family(path.name),
        "Orden": order,
        "proc_by_order": proc_by_order,
    }


def infer_family(filename: str) -> str:
    """Extrae el prefijo alfabético inicial: ta01 -> ta, swv01 -> swv, yn1 -> yn."""
    stem = Path(filename).stem.lower()
    m = re.match(r"([a-zA-Z]+)", stem)
    return m.group(1).lower() if m else "unknown"


def natural_sort_key(path: Path):
    """Orden natural por prefijo y número: ta2 antes que ta10."""
    name = path.name.lower()
    pieces = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in pieces]


def iter_instance_files(input_path: Path):
    """
    Devuelve ficheros candidatos desde:
      - un directorio
      - un ZIP
      - un fichero individual

    No filtra por prefijo; intenta procesar todos los ficheros no ocultos.
    Los errores de parseo se guardan luego en parsing_errors.csv.
    """
    input_path = input_path.resolve()

    if input_path.is_dir():
        files = sorted(
            [p for p in input_path.rglob("*") if p.is_file() and not p.name.startswith(".")],
            key=natural_sort_key,
        )
        return files, None

    if input_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp.name)
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(tmp_path)
        files = sorted(
            [p for p in tmp_path.rglob("*") if p.is_file() and not p.name.startswith(".")],
            key=natural_sort_key,
        )
        return files, tmp

    if input_path.is_file():
        return [input_path], None

    raise FileNotFoundError(f"No existe la ruta: {input_path}")


# -----------------------------------------------------------------------------
# Estadísticos y ajuste de distribuciones
# -----------------------------------------------------------------------------


def aic(loglik: float, n_params: int) -> float:
    return 2 * n_params - 2 * loglik


def bic(loglik: float, n_params: int, n: int) -> float:
    return math.log(n) * n_params - 2 * loglik


def summarize_vector(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "cv": float(np.std(x, ddof=1) / np.mean(x)) if x.size > 1 and np.mean(x) != 0 else np.nan,
        "skewness": float(stats.skew(x, bias=False)) if x.size > 2 else np.nan,
        "kurtosis_excess": float(stats.kurtosis(x, bias=False)) if x.size > 3 else np.nan,
        "unique_values": int(np.unique(x).size),
    }


def chisq_uniform_integer_binned(
    x: np.ndarray, low: int, high: int, nbins: int = 10
) -> Tuple[float, float]:
    """
    Test chi-cuadrado frente a U{low,...,high}, agrupado en bins.
    Evita esperados muy pequeños cuando se tienen muchos valores posibles.
    """
    x = np.asarray(x, dtype=int)
    if high < low:
        return np.nan, np.nan

    support = np.arange(low, high + 1)
    bins = min(nbins, len(support))
    if bins < 2:
        return np.nan, np.nan

    # Edges enteros aproximados para cubrir [low, high].
    edges = np.linspace(low, high + 1, bins + 1)
    obs = np.histogram(x, bins=edges)[0]

    exp = []
    for a, b in zip(edges[:-1], edges[1:]):
        support_count = np.sum((support >= a) & (support < b))
        exp.append(len(x) * support_count / len(support))

    exp = np.asarray(exp, dtype=float)
    valid = exp > 0
    if valid.sum() < 2:
        return np.nan, np.nan

    stat, pvalue = stats.chisquare(obs[valid], exp[valid])
    return float(stat), float(pvalue)


def fit_discrete_processing_times(x: np.ndarray) -> pd.DataFrame:
    """Ajustes discretos comparables por log-verosimilitud discreta."""
    x = np.asarray(x, dtype=int)
    n = len(x)
    rows = []

    # Uniformes discretas fijas habituales en benchmarks JSP.
    # Taillard suele ser U{1,...,99}, pero otras familias pueden contener 0 o 100.
    fixed_supports = [(1, 99), (0, 99), (1, 100), (0, 100)]
    for low, high in fixed_supports:
        name = f"discrete_uniform_{low}_{high}"
        if np.all((x >= low) & (x <= high)):
            loglik = -n * math.log(high - low + 1)
            chi2, p_chi2 = chisq_uniform_integer_binned(x, low=low, high=high, nbins=10)
        else:
            loglik = -math.inf
            chi2, p_chi2 = np.nan, np.nan

        rows.append(
            {
                "model_type": "discrete",
                "distribution": name,
                "params": json.dumps({"low": low, "high": high, "fixed_support": True}),
                "n_params": 0,
                "loglik": loglik,
                "aic": aic(loglik, 0) if math.isfinite(loglik) else math.inf,
                "bic": bic(loglik, 0, n) if math.isfinite(loglik) else math.inf,
                "gof_test": f"chi2_uniform_{low}_{high}_bins10",
                "gof_stat": chi2,
                "gof_pvalue": p_chi2,
            }
        )

    # Uniforme discreta con soporte observado. Puede sobreajustar en instancias pequeñas.
    low_fit, high_fit = int(np.min(x)), int(np.max(x))
    k = high_fit - low_fit + 1
    loglik = -n * math.log(k) if k > 0 else -math.inf
    chi2, p_chi2 = chisq_uniform_integer_binned(x, low=low_fit, high=high_fit, nbins=10)
    rows.append(
        {
            "model_type": "discrete",
            "distribution": "discrete_uniform_minmax",
            "params": json.dumps({"low": low_fit, "high": high_fit}),
            "n_params": 2,
            "loglik": loglik,
            "aic": aic(loglik, 2) if math.isfinite(loglik) else math.inf,
            "bic": bic(loglik, 2, n) if math.isfinite(loglik) else math.inf,
            "gof_test": "chi2_uniform_minmax_bins10",
            "gof_stat": chi2,
            "gof_pvalue": p_chi2,
        }
    )

    # Poisson MLE. Normalmente mala para Taillard, pero útil como contraste.
    mu = float(np.mean(x))
    loglik = float(np.sum(stats.poisson.logpmf(x, mu)))
    rows.append(
        {
            "model_type": "discrete",
            "distribution": "poisson",
            "params": json.dumps({"mu": mu}),
            "n_params": 1,
            "loglik": loglik,
            "aic": aic(loglik, 1),
            "bic": bic(loglik, 1, n),
            "gof_test": "",
            "gof_stat": np.nan,
            "gof_pvalue": np.nan,
        }
    )

    return pd.DataFrame(rows).sort_values("aic", ascending=True)


def fit_continuous_exploratory(x: np.ndarray) -> pd.DataFrame:
    """
    Ajustes continuos exploratorios.

    No mezclar sus AIC con los discretos como si fueran el mismo tipo de modelo.
    Tus tiempos son enteros; la conclusión principal debe venir de modelos discretos.
    """
    x = np.asarray(x, dtype=float)
    candidates = [
        ("norm", stats.norm, 2),
        ("uniform", stats.uniform, 2),
        ("expon", stats.expon, 2),
        ("gamma", stats.gamma, 3),
        ("lognorm", stats.lognorm, 3),
        ("weibull_min", stats.weibull_min, 3),
    ]

    rows = []
    for name, dist, n_params in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(x)
                logpdf = dist.logpdf(x, *params)
                if not np.all(np.isfinite(logpdf)):
                    raise ValueError("logpdf no finita")
                loglik = float(np.sum(logpdf))
                ks_stat, ks_p = stats.kstest(x, name, args=params)
            rows.append(
                {
                    "model_type": "continuous_exploratory",
                    "distribution": name,
                    "params": json.dumps([float(v) for v in params]),
                    "n_params": n_params,
                    "loglik": loglik,
                    "aic": aic(loglik, n_params),
                    "bic": bic(loglik, n_params, len(x)),
                    "gof_test": "KS_continuous",
                    "gof_stat": float(ks_stat),
                    "gof_pvalue": float(ks_p),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "model_type": "continuous_exploratory",
                    "distribution": name,
                    "params": json.dumps({"error": str(e)}),
                    "n_params": n_params,
                    "loglik": -math.inf,
                    "aic": math.inf,
                    "bic": math.inf,
                    "gof_test": "KS_continuous",
                    "gof_stat": np.nan,
                    "gof_pvalue": np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values("aic", ascending=True)


def fit_all_models(x: np.ndarray) -> pd.DataFrame:
    return pd.concat(
        [fit_discrete_processing_times(x), fit_continuous_exploratory(x)],
        ignore_index=True,
    )


def best_names(fits: pd.DataFrame) -> dict:
    disc = fits[fits["model_type"] == "discrete"].sort_values("aic")
    cont = fits[fits["model_type"] == "continuous_exploratory"].sort_values("aic")
    out = {
        "best_discrete": disc.iloc[0]["distribution"] if len(disc) else "",
        "best_discrete_params": disc.iloc[0]["params"] if len(disc) else "",
        "best_discrete_aic": float(disc.iloc[0]["aic"]) if len(disc) else np.nan,
        "best_discrete_gof_pvalue": float(disc.iloc[0]["gof_pvalue"]) if len(disc) and pd.notna(disc.iloc[0]["gof_pvalue"]) else np.nan,
        "best_continuous_exploratory": cont.iloc[0]["distribution"] if len(cont) else "",
        "best_continuous_aic": float(cont.iloc[0]["aic"]) if len(cont) else np.nan,
    }

    u = disc[disc["distribution"] == "discrete_uniform_1_99"]
    if len(u):
        out["uniform_1_99_pvalue_bins10"] = float(u.iloc[0]["gof_pvalue"]) if pd.notna(u.iloc[0]["gof_pvalue"]) else np.nan
        out["uniform_1_99_chi2_bins10"] = float(u.iloc[0]["gof_stat"]) if pd.notna(u.iloc[0]["gof_stat"]) else np.nan
    else:
        out["uniform_1_99_pvalue_bins10"] = np.nan
        out["uniform_1_99_chi2_bins10"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Análisis completo
# -----------------------------------------------------------------------------


def add_scope_columns(fits: pd.DataFrame, scope_type: str, scope_value: str, extra: dict) -> pd.DataFrame:
    fits = fits.copy()
    fits.insert(0, "scope_type", scope_type)
    fits.insert(1, "scope_value", scope_value)
    for i, (k, v) in enumerate(extra.items(), start=2):
        fits.insert(i, k, v)
    return fits


def analyse_dataset(input_path: Path, out_dir: Path, make_plots: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    files, tmp = iter_instance_files(input_path)
    if not files:
        raise ValueError("No se han encontrado ficheros candidatos.")

    parsed_instances = []
    errors = []

    for path in files:
        try:
            parsed_instances.append(load_jsp_file(path))
        except Exception as e:
            errors.append({"file": path.name, "path": str(path), "error": str(e)})

    if not parsed_instances:
        if tmp is not None:
            tmp.cleanup()
        raise ValueError("No se pudo parsear ninguna instancia JSP.")

    all_rows = []
    all_fits = []
    family_chunks: Dict[str, List[np.ndarray]] = {}
    size_chunks: Dict[str, List[np.ndarray]] = {}
    all_times = []
    machine_position_rows = []

    for inst in parsed_instances:
        file = inst["file"]
        family = inst["family"]
        jobs = int(inst["numJobs"])
        machines = int(inst["numMchs"])
        size_family = f"{jobs}x{machines}"
        times = inst["proc_by_order"].ravel()
        order = inst["Orden"]

        all_times.append(times)
        family_chunks.setdefault(family, []).append(times)
        size_chunks.setdefault(size_family, []).append(times)

        fits = fit_all_models(times)
        all_fits.append(
            add_scope_columns(
                fits,
                scope_type="file",
                scope_value=file,
                extra={"benchmark_family": family, "jobs": jobs, "machines": machines},
            )
        )

        all_rows.append(
            {
                "file": file,
                "benchmark_family": family,
                "size_family": size_family,
                "jobs": jobs,
                "machines": machines,
                **summarize_vector(times),
                **best_names(fits),
            }
        )

        for pos in range(machines):
            counts = np.bincount(order[:, pos], minlength=machines)
            for m, c in enumerate(counts):
                machine_position_rows.append(
                    {
                        "file": file,
                        "benchmark_family": family,
                        "size_family": size_family,
                        "jobs": jobs,
                        "machines": machines,
                        "position": pos,
                        "machine": m,
                        "count": int(c),
                        "relative_frequency": float(c / jobs),
                    }
                )

    summary_by_file = pd.DataFrame(all_rows).sort_values(
        ["benchmark_family", "jobs", "machines", "file"], ascending=True
    )
    fit_candidates_by_file = pd.concat(all_fits, ignore_index=True)
    machine_position_summary = pd.DataFrame(machine_position_rows)

    # Por familia de benchmark: ta, swv, yn, la, ...
    family_summary_rows = []
    family_fit_rows = []
    for family in sorted(family_chunks):
        x = np.concatenate(family_chunks[family])
        fits = fit_all_models(x)
        n_files = int((summary_by_file["benchmark_family"] == family).sum())
        sizes = ",".join(sorted(summary_by_file.loc[summary_by_file["benchmark_family"] == family, "size_family"].unique()))
        family_summary_rows.append(
            {
                "benchmark_family": family,
                "n_files": n_files,
                "sizes": sizes,
                **summarize_vector(x),
                **best_names(fits),
            }
        )
        family_fit_rows.append(
            add_scope_columns(
                fits,
                scope_type="benchmark_family",
                scope_value=family,
                extra={"n_files": n_files, "sizes": sizes},
            )
        )

    summary_by_family = pd.DataFrame(family_summary_rows).sort_values("benchmark_family")
    fit_candidates_by_family = pd.concat(family_fit_rows, ignore_index=True)

    # Por tamaño: 10x10, 20x15, ...
    size_summary_rows = []
    size_fit_rows = []
    for size_family in sorted(size_chunks, key=lambda s: tuple(int(v) for v in s.split("x"))):
        x = np.concatenate(size_chunks[size_family])
        fits = fit_all_models(x)
        sub = summary_by_file[summary_by_file["size_family"] == size_family]
        n_files = int(len(sub))
        families = ",".join(sorted(sub["benchmark_family"].unique()))
        jobs, machines = [int(v) for v in size_family.split("x")]
        size_summary_rows.append(
            {
                "size_family": size_family,
                "jobs": jobs,
                "machines": machines,
                "n_files": n_files,
                "benchmark_families": families,
                **summarize_vector(x),
                **best_names(fits),
            }
        )
        size_fit_rows.append(
            add_scope_columns(
                fits,
                scope_type="size_family",
                scope_value=size_family,
                extra={"jobs": jobs, "machines": machines, "n_files": n_files, "benchmark_families": families},
            )
        )

    summary_by_size = pd.DataFrame(size_summary_rows).sort_values(["jobs", "machines"])
    fit_candidates_by_size = pd.concat(size_fit_rows, ignore_index=True)

    # Global: todas las operaciones de todas las familias.
    x_global = np.concatenate(all_times)
    global_fits = fit_all_models(x_global)
    global_fit = add_scope_columns(
        global_fits,
        scope_type="global",
        scope_value="all_instances",
        extra={"n_files": len(parsed_instances)},
    )
    global_summary = {
        "scope": "global_processing_times_all_instances",
        "n_files": len(parsed_instances),
        "families": sorted(summary_by_family["benchmark_family"].tolist()),
        **summarize_vector(x_global),
        **best_names(global_fits),
    }

    # Exportación.
    summary_by_file.to_csv(out_dir / "summary_by_file.csv", index=False)
    summary_by_family.to_csv(out_dir / "summary_by_family.csv", index=False)
    summary_by_size.to_csv(out_dir / "summary_by_size.csv", index=False)
    fit_candidates_by_file.to_csv(out_dir / "fit_candidates_by_file.csv", index=False)
    fit_candidates_by_family.to_csv(out_dir / "fit_candidates_by_family.csv", index=False)
    fit_candidates_by_size.to_csv(out_dir / "fit_candidates_by_size.csv", index=False)
    global_fit.to_csv(out_dir / "global_fit.csv", index=False)
    machine_position_summary.to_csv(out_dir / "machine_position_summary.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "parsing_errors.csv", index=False)

    with open(out_dir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    readme = f"""Resultados generados para: {input_path}

Instancias leídas correctamente: {len(parsed_instances)}
Ficheros no parseados: {len(errors)}
Familias detectadas: {', '.join(global_summary['families'])}

Interpretación rápida:
- Variable principal analizada: tiempos de proceso p_ij.
- Los tiempos son enteros, por lo que la comparación principal está en la familia discreta.
- Se generan resultados por fichero, por familia de benchmark, por tamaño y globales.
- La distribución global discreta más plausible según AIC es: {global_summary['best_discrete']}.
- Media global: {global_summary['mean']:.4f}
- Desviación típica global: {global_summary['std']:.4f}
- p-value global frente a U{{1,...,99}} con chi-cuadrado agrupado: {global_summary['uniform_1_99_pvalue_bins10']:.6g}

Archivos clave:
- summary_by_family.csv: estadísticos y mejor distribución para ta, swv, yn, la, etc.
- fit_candidates_by_family.csv: ranking de distribuciones candidatas por familia.
- summary_by_size.csv: estadísticos por tamaño de instancia, por ejemplo 10x10, 20x15.
- global_fit.csv: fitting de todos los tiempos juntos.
- machine_position_summary.csv: frecuencias de máquinas por posición de operación.

Notas:
- Los ajustes continuos son exploratorios y no deben compararse directamente con los AIC discretos.
- Los órdenes de máquina son permutaciones, no variables continuas. Para ellos no tiene sentido ajustar normal/gamma/weibull.
- Si una instancia pequeña da discrete_uniform_minmax en vez de discrete_uniform_1_99, puede deberse simplemente a que no aparecen los extremos 1 o 99 en esa muestra.
"""
    (out_dir / "README_results.txt").write_text(readme, encoding="utf-8")

    if make_plots:
        make_all_plots(out_dir, x_global, family_chunks, size_chunks)

    if tmp is not None:
        tmp.cleanup()

    return summary_by_file, summary_by_family, summary_by_size, global_summary


# -----------------------------------------------------------------------------
# Gráficas
# -----------------------------------------------------------------------------


def plot_histogram(x: np.ndarray, title: str, output_path: Path):
    import matplotlib.pyplot as plt

    x = np.asarray(x)
    low, high = int(np.min(x)), int(np.max(x))
    bins = np.arange(low, high + 2) - 0.5
    plt.figure(figsize=(8, 4.5))
    plt.hist(x, bins=bins, density=True, alpha=0.8)
    if low <= 1 and high >= 99:
        plt.axhline(1 / 99, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Tiempo de proceso")
    plt.ylabel("Densidad empírica")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_all_plots(out_dir: Path, x_global: np.ndarray, family_chunks: Dict[str, List[np.ndarray]], size_chunks: Dict[str, List[np.ndarray]]):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plot_histogram(
        x_global,
        "Distribución global de tiempos de proceso",
        plot_dir / "global_processing_times_hist.png",
    )

    family_dir = plot_dir / "by_family"
    family_dir.mkdir(exist_ok=True)
    for family, chunks in sorted(family_chunks.items()):
        plot_histogram(
            np.concatenate(chunks),
            f"Distribución de tiempos - familia {family}",
            family_dir / f"{family}_processing_times_hist.png",
        )

    size_dir = plot_dir / "by_size"
    size_dir.mkdir(exist_ok=True)
    for size_family, chunks in sorted(size_chunks.items()):
        plot_histogram(
            np.concatenate(chunks),
            f"Distribución de tiempos - tamaño {size_family}",
            size_dir / f"{size_family}_processing_times_hist.png",
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="ZIP, directorio o fichero individual de instancias JSP.",
    )
    parser.add_argument(
        "--out",
        default=Path("fit_results"),
        type=Path,
        help="Directorio de salida.",
    )
    parser.add_argument("--plots", action="store_true", help="Genera histogramas globales, por familia y por tamaño.")
    args = parser.parse_args()

    _, summary_by_family, summary_by_size, global_summary = analyse_dataset(
        input_path=args.input,
        out_dir=args.out,
        make_plots=args.plots,
    )

    print("\nResumen global:")
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))

    print("\nResumen por familia de benchmark:")
    cols = [
        "benchmark_family",
        "n_files",
        "n",
        "min",
        "mean",
        "std",
        "max",
        "best_discrete",
        "uniform_1_99_pvalue_bins10",
        "best_continuous_exploratory",
    ]
    print(summary_by_family[cols].to_string(index=False))

    print("\nResumen por tamaño:")
    cols_size = [
        "size_family",
        "n_files",
        "benchmark_families",
        "n",
        "min",
        "mean",
        "std",
        "max",
        "best_discrete",
        "uniform_1_99_pvalue_bins10",
    ]
    print(summary_by_size[cols_size].to_string(index=False))

    print(f"\nArchivos generados en: {args.out.resolve()}")


if __name__ == "__main__":
    main()
