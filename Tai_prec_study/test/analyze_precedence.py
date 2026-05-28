#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


def family_from_name(name: str) -> str:
    """Extract family prefix from instance filename: ta01 -> ta, swv01 -> swv."""
    base = Path(name).name
    base = base.split(".")[0]
    m = re.match(r"([A-Za-z]+)", base)
    return m.group(1).lower() if m else "unknown"


@dataclass
class JSPInstance:
    name: str
    family: str
    num_jobs: int
    num_mchs: int
    order: np.ndarray
    proc_by_order: np.ndarray


def _read_text_from_source(input_path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (name, text) from either a directory, a single file, or a zip."""
    if input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = p.read_text(encoding="latin-1")
                yield p.name, text
    elif input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                raw = zf.read(name)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
                yield Path(name).name, text
    elif input_path.is_file():
        try:
            text = input_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = input_path.read_text(encoding="latin-1")
        yield input_path.name, text
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def load_jsp_instance_from_text(name: str, text: str) -> JSPInstance:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)

    if not lines:
        raise ValueError("Empty file or only comments.")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError("First useful line must contain exactly two integers: num_jobs num_machines.")

    num_jobs, num_mchs = map(int, header)

    if len(lines) < num_jobs + 1:
        raise ValueError(
            f"Expected {num_jobs} job lines, got {len(lines) - 1}."
        )

    order = np.empty((num_jobs, num_mchs), dtype=np.int64)
    proc_by_order = np.empty((num_jobs, num_mchs), dtype=np.int64)

    for j in range(num_jobs):
        vals = np.fromstring(lines[j + 1], sep=" ", dtype=np.int64)
        if vals.size != 2 * num_mchs:
            raise ValueError(
                f"Job {j}: expected {2 * num_mchs} integers, got {vals.size}."
            )

        machines = vals[0::2]
        times = vals[1::2]

        # Accept 1-based machine ids and convert them to 0-based.
        if machines.min() >= 1 and machines.max() <= num_mchs:
            machines = machines - 1

        if machines.min() < 0 or machines.max() >= num_mchs:
            raise ValueError(f"Job {j}: machine id out of range.")

        if np.unique(machines).size != num_mchs:
            raise ValueError(f"Job {j}: machines are not a valid permutation.")

        order[j, :] = machines
        proc_by_order[j, :] = times

    return JSPInstance(
        name=Path(name).name,
        family=family_from_name(name),
        num_jobs=num_jobs,
        num_mchs=num_mchs,
        order=order,
        proc_by_order=proc_by_order,
    )


def load_instances(input_path: Path) -> Tuple[List[JSPInstance], List[Dict]]:
    instances: List[JSPInstance] = []
    errors: List[Dict] = []
    for name, text in _read_text_from_source(input_path):
        try:
            inst = load_jsp_instance_from_text(name, text)
            instances.append(inst)
        except Exception as e:
            errors.append({"file": name, "error": str(e)})
    return instances, errors


def safe_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def instance_route_summary(inst: JSPInstance) -> Dict:
    n, m = inst.num_jobs, inst.num_mchs
    routes = [tuple(row.tolist()) for row in inst.order]
    route_counts = Counter(routes)
    unique_routes = len(route_counts)
    top_route_count = max(route_counts.values()) if route_counts else 0

    # Position entropy: for each operation position k, machines should be roughly uniform
    # if routes are random. Normalise by log(m), so 1 = perfectly uniform.
    pos_counts = np.zeros((m, m), dtype=int)  # position x machine
    for route in routes:
        for pos, mach in enumerate(route):
            pos_counts[pos, mach] += 1

    entropies = np.array([safe_entropy_from_counts(pos_counts[pos]) for pos in range(m)])
    max_entropy = math.log(m) if m > 1 else 1.0
    norm_entropies = entropies / max_entropy if max_entropy > 0 else entropies

    # Pairwise precedence: P(a before b)
    before_counts = np.zeros((m, m), dtype=int)
    for route in routes:
        pos_of = {machine: pos for pos, machine in enumerate(route)}
        for a in range(m):
            pa = pos_of[a]
            for b in range(m):
                if a != b and pa < pos_of[b]:
                    before_counts[a, b] += 1

    pair_biases = []
    deterministic_pairs = 0
    for a in range(m):
        for b in range(a + 1, m):
            p_ab = before_counts[a, b] / n
            bias = abs(p_ab - 0.5)
            pair_biases.append(bias)
            if p_ab in (0.0, 1.0):
                deterministic_pairs += 1

    mean_abs_pairwise_bias = float(np.mean(pair_biases)) if pair_biases else 0.0
    max_abs_pairwise_bias = float(np.max(pair_biases)) if pair_biases else 0.0

    # Adjacent transition density.
    adj_edges = set()
    for route in routes:
        for k in range(m - 1):
            adj_edges.add((route[k], route[k + 1]))
    max_directed_edges = m * (m - 1)
    adjacent_edge_density = len(adj_edges) / max_directed_edges if max_directed_edges else 0.0

    # Transitive directed edges observed at least once.
    trans_edges = set()
    for route in routes:
        for i in range(m):
            for j in range(i + 1, m):
                trans_edges.add((route[i], route[j]))
    transitive_edge_density = len(trans_edges) / max_directed_edges if max_directed_edges else 0.0

    # Symmetry in transitive graph: for unordered pair {a,b}, do both directions occur?
    both_dir = 0
    one_dir = 0
    for a in range(m):
        for b in range(a + 1, m):
            ab = (a, b) in trans_edges
            ba = (b, a) in trans_edges
            if ab and ba:
                both_dir += 1
            elif ab or ba:
                one_dir += 1
    unordered_pairs = m * (m - 1) / 2
    bidirectional_pair_ratio = both_dir / unordered_pairs if unordered_pairs else 0.0
    one_direction_pair_ratio = one_dir / unordered_pairs if unordered_pairs else 0.0

    return {
        "file": inst.name,
        "family": inst.family,
        "num_jobs": n,
        "num_mchs": m,
        "size": f"{n}x{m}",
        "unique_routes": unique_routes,
        "route_diversity_ratio": unique_routes / n if n else 0.0,
        "top_route_count": top_route_count,
        "top_route_share": top_route_count / n if n else 0.0,
        "mean_position_entropy_norm": float(np.mean(norm_entropies)) if len(norm_entropies) else 0.0,
        "min_position_entropy_norm": float(np.min(norm_entropies)) if len(norm_entropies) else 0.0,
        "mean_abs_pairwise_bias": mean_abs_pairwise_bias,
        "max_abs_pairwise_bias": max_abs_pairwise_bias,
        "deterministic_pair_ratio": deterministic_pairs / unordered_pairs if unordered_pairs else 0.0,
        "adjacent_edge_density": adjacent_edge_density,
        "transitive_edge_density": transitive_edge_density,
        "bidirectional_pair_ratio": bidirectional_pair_ratio,
        "one_direction_pair_ratio": one_direction_pair_ratio,
    }


def position_rows(inst: JSPInstance) -> List[Dict]:
    rows = []
    n, m = inst.num_jobs, inst.num_mchs
    counts = np.zeros((m, m), dtype=int)  # position x machine
    for route in inst.order:
        for pos, mach in enumerate(route):
            counts[pos, mach] += 1

    for pos in range(m):
        for mach in range(m):
            c = int(counts[pos, mach])
            rows.append({
                "file": inst.name,
                "family": inst.family,
                "size": f"{n}x{m}",
                "num_jobs": n,
                "num_mchs": m,
                "position": pos,
                "machine": mach,
                "count": c,
                "frequency": c / n if n else 0.0,
            })
    return rows


def adjacent_rows(inst: JSPInstance) -> List[Dict]:
    n, m = inst.num_jobs, inst.num_mchs
    counts = np.zeros((m, m), dtype=int)
    for route in inst.order:
        for pos in range(m - 1):
            a, b = route[pos], route[pos + 1]
            counts[a, b] += 1

    rows = []
    denom = n * (m - 1) if m > 1 else 0
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            c = int(counts[a, b])
            rows.append({
                "file": inst.name,
                "family": inst.family,
                "size": f"{n}x{m}",
                "num_jobs": n,
                "num_mchs": m,
                "from_machine": a,
                "to_machine": b,
                "count": c,
                "frequency_over_all_adjacent_arcs": c / denom if denom else 0.0,
                "frequency_per_job": c / n if n else 0.0,
            })
    return rows


def pairwise_rows(inst: JSPInstance) -> List[Dict]:
    n, m = inst.num_jobs, inst.num_mchs
    before_counts = np.zeros((m, m), dtype=int)
    for route in inst.order:
        pos_of = {machine: pos for pos, machine in enumerate(route)}
        for a in range(m):
            pa = pos_of[a]
            for b in range(m):
                if a != b and pa < pos_of[b]:
                    before_counts[a, b] += 1

    rows = []
    for a in range(m):
        for b in range(a + 1, m):
            c_ab = int(before_counts[a, b])
            c_ba = int(before_counts[b, a])
            p_ab = c_ab / n if n else 0.0
            rows.append({
                "file": inst.name,
                "family": inst.family,
                "size": f"{n}x{m}",
                "num_jobs": n,
                "num_mchs": m,
                "machine_a": a,
                "machine_b": b,
                "count_a_before_b": c_ab,
                "count_b_before_a": c_ba,
                "p_a_before_b": p_ab,
                "bias_from_0_5": abs(p_ab - 0.5),
                "dominant_direction": f"{a}->{b}" if p_ab >= 0.5 else f"{b}->{a}",
                "dominant_probability": max(p_ab, 1 - p_ab),
            })
    return rows


def aggregate_position(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(group_cols + ["position", "machine"], as_index=False).agg(
        count=("count", "sum"),
        jobs=("num_jobs", "sum"),
    )
    # jobs is over-counted if grouping across all machines/positions because num_jobs repeats
    # once per row. For position-machine frequency we need denominator = total jobs for that
    # group, not summed per row. Compute denominator separately from original data.
    denom = df[["file"] + group_cols + ["num_jobs"]].drop_duplicates()
    denom = denom.groupby(group_cols, as_index=False).agg(total_jobs=("num_jobs", "sum"))
    g = g.merge(denom, on=group_cols, how="left")
    g["frequency"] = g["count"] / g["total_jobs"]
    return g


def aggregate_adjacent(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(group_cols + ["from_machine", "to_machine"], as_index=False).agg(
        count=("count", "sum"),
    )
    denom = df[["file"] + group_cols + ["num_jobs", "num_mchs"]].drop_duplicates()
    denom["adjacent_arcs_total"] = denom["num_jobs"] * (denom["num_mchs"] - 1)
    denom = denom.groupby(group_cols, as_index=False).agg(
        adjacent_arcs_total=("adjacent_arcs_total", "sum"),
        total_jobs=("num_jobs", "sum"),
    )
    g = g.merge(denom, on=group_cols, how="left")
    g["frequency_over_all_adjacent_arcs"] = g["count"] / g["adjacent_arcs_total"]
    g["frequency_per_job"] = g["count"] / g["total_jobs"]
    return g


def aggregate_pairwise(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    # Pairwise aggregation only makes clean sense when grouped by a fixed number of machines.
    # family_size is therefore recommended. Still works for family if machine ids overlap, but
    # interpretation may be weaker if several sizes are mixed.
    g = df.groupby(group_cols + ["machine_a", "machine_b"], as_index=False).agg(
        count_a_before_b=("count_a_before_b", "sum"),
        count_b_before_a=("count_b_before_a", "sum"),
    )
    g["total_comparisons"] = g["count_a_before_b"] + g["count_b_before_a"]
    g["p_a_before_b"] = g["count_a_before_b"] / g["total_comparisons"].replace(0, np.nan)
    g["bias_from_0_5"] = (g["p_a_before_b"] - 0.5).abs()
    g["dominant_direction"] = np.where(
        g["p_a_before_b"] >= 0.5,
        g["machine_a"].astype(str) + "->" + g["machine_b"].astype(str),
        g["machine_b"].astype(str) + "->" + g["machine_a"].astype(str),
    )
    g["dominant_probability"] = np.maximum(g["p_a_before_b"], 1 - g["p_a_before_b"])
    return g


def summarise_groups(summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    agg = summary.groupby(group_col, as_index=False).agg(
        n_files=("file", "count"),
        total_jobs=("num_jobs", "sum"),
        min_machines=("num_mchs", "min"),
        max_machines=("num_mchs", "max"),
        mean_route_diversity_ratio=("route_diversity_ratio", "mean"),
        mean_top_route_share=("top_route_share", "mean"),
        mean_position_entropy_norm=("mean_position_entropy_norm", "mean"),
        min_position_entropy_norm=("min_position_entropy_norm", "mean"),
        mean_abs_pairwise_bias=("mean_abs_pairwise_bias", "mean"),
        max_abs_pairwise_bias=("max_abs_pairwise_bias", "mean"),
        mean_deterministic_pair_ratio=("deterministic_pair_ratio", "mean"),
        mean_adjacent_edge_density=("adjacent_edge_density", "mean"),
        mean_transitive_edge_density=("transitive_edge_density", "mean"),
        mean_bidirectional_pair_ratio=("bidirectional_pair_ratio", "mean"),
    )
    return agg


def maybe_make_plots(out_dir: Path, summary_by_file: pd.DataFrame,
                     pair_family_size: pd.DataFrame,
                     pos_family_size: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: mean pairwise bias by family.
    if not summary_by_file.empty:
        fam = summary_by_file.groupby("family", as_index=False).agg(
            mean_abs_pairwise_bias=("mean_abs_pairwise_bias", "mean"),
            mean_position_entropy_norm=("mean_position_entropy_norm", "mean"),
            mean_route_diversity_ratio=("route_diversity_ratio", "mean"),
        )
        fam = fam.sort_values("family")

        for metric in [
            "mean_abs_pairwise_bias",
            "mean_position_entropy_norm",
            "mean_route_diversity_ratio",
        ]:
            fig = plt.figure(figsize=(10, 5))
            plt.bar(fam["family"], fam[metric])
            plt.xlabel("Family")
            plt.ylabel(metric)
            plt.title(metric + " by family")
            plt.tight_layout()
            fig.savefig(plots_dir / f"{metric}_by_family.png", dpi=180)
            plt.close(fig)

    # Heatmaps by family_size for position frequencies and pairwise probabilities.
    # Limit to a reasonable number of plots.
    for i, key in enumerate(sorted(pos_family_size["family_size"].unique())[:30] if not pos_family_size.empty else []):
        sub = pos_family_size[pos_family_size["family_size"] == key]
        m = int(sub["machine"].max()) + 1
        matrix = np.full((m, m), np.nan)
        for _, r in sub.iterrows():
            matrix[int(r["position"]), int(r["machine"])] = r["frequency"]
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(matrix, aspect="auto")
        plt.xlabel("Machine")
        plt.ylabel("Position in job route")
        plt.title(f"Position frequency: {key}")
        plt.colorbar(label="Frequency")
        plt.tight_layout()
        safe = key.replace("x", "_").replace("/", "_")
        fig.savefig(plots_dir / f"position_frequency_{safe}.png", dpi=180)
        plt.close(fig)

    for i, key in enumerate(sorted(pair_family_size["family_size"].unique())[:30] if not pair_family_size.empty else []):
        sub = pair_family_size[pair_family_size["family_size"] == key]
        m = int(max(sub["machine_a"].max(), sub["machine_b"].max())) + 1
        matrix = np.full((m, m), np.nan)
        for _, r in sub.iterrows():
            a = int(r["machine_a"])
            b = int(r["machine_b"])
            p = float(r["p_a_before_b"])
            matrix[a, b] = p
            matrix[b, a] = 1 - p
        np.fill_diagonal(matrix, 0.5)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(matrix, aspect="auto", vmin=0, vmax=1)
        plt.xlabel("Machine b")
        plt.ylabel("Machine a")
        plt.title(f"P(machine a before machine b): {key}")
        plt.colorbar(label="P(a before b)")
        plt.tight_layout()
        safe = key.replace("x", "_").replace("/", "_")
        fig.savefig(plots_dir / f"pairwise_precedence_{safe}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input directory, file, or .zip.")
    parser.add_argument("--out", default=Path("precedence_results"), type=Path, help="Output directory.")
    parser.add_argument("--plots", action="store_true", help="Generate summary plots.")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    instances, errors = load_instances(args.input)

    summaries = []
    pos_rows = []
    adj_rows = []
    pair_rows = []

    for inst in instances:
        summaries.append(instance_route_summary(inst))
        pos_rows.extend(position_rows(inst))
        adj_rows.extend(adjacent_rows(inst))
        pair_rows.extend(pairwise_rows(inst))

    summary_by_file = pd.DataFrame(summaries)
    pos_by_file = pd.DataFrame(pos_rows)
    adj_by_file = pd.DataFrame(adj_rows)
    pair_by_file = pd.DataFrame(pair_rows)

    if not summary_by_file.empty:
        summary_by_file["family_size"] = summary_by_file["family"] + "_" + summary_by_file["size"]
    if not pos_by_file.empty:
        pos_by_file["family_size"] = pos_by_file["family"] + "_" + pos_by_file["size"]
    if not adj_by_file.empty:
        adj_by_file["family_size"] = adj_by_file["family"] + "_" + adj_by_file["size"]
    if not pair_by_file.empty:
        pair_by_file["family_size"] = pair_by_file["family"] + "_" + pair_by_file["size"]

    summary_by_family = summarise_groups(summary_by_file, "family")
    summary_by_size = summarise_groups(summary_by_file, "size")
    summary_by_family_size = summarise_groups(summary_by_file, "family_size")

    position_by_family_size = aggregate_position(pos_by_file, ["family_size", "family", "size"])
    adjacent_by_family_size = aggregate_adjacent(adj_by_file, ["family_size", "family", "size"])
    pairwise_by_family_size = aggregate_pairwise(pair_by_file, ["family_size", "family", "size"])

    # Family-level aggregations are exported too, but note: families may mix machine counts.
    position_by_family = aggregate_position(pos_by_file, ["family"])
    adjacent_by_family = aggregate_adjacent(adj_by_file, ["family"])
    pairwise_by_family = aggregate_pairwise(pair_by_file, ["family"])

    # Global summaries where meaningful.
    global_summary = {
        "input": str(args.input),
        "n_files": len(instances),
        "n_parse_errors": len(errors),
        "families": sorted(summary_by_file["family"].unique().tolist()) if not summary_by_file.empty else [],
        "total_jobs": int(summary_by_file["num_jobs"].sum()) if not summary_by_file.empty else 0,
        "min_machines": int(summary_by_file["num_mchs"].min()) if not summary_by_file.empty else None,
        "max_machines": int(summary_by_file["num_mchs"].max()) if not summary_by_file.empty else None,
        "mean_route_diversity_ratio": float(summary_by_file["route_diversity_ratio"].mean()) if not summary_by_file.empty else None,
        "mean_position_entropy_norm": float(summary_by_file["mean_position_entropy_norm"].mean()) if not summary_by_file.empty else None,
        "mean_abs_pairwise_bias": float(summary_by_file["mean_abs_pairwise_bias"].mean()) if not summary_by_file.empty else None,
        "mean_adjacent_edge_density": float(summary_by_file["adjacent_edge_density"].mean()) if not summary_by_file.empty else None,
        "mean_transitive_edge_density": float(summary_by_file["transitive_edge_density"].mean()) if not summary_by_file.empty else None,
        "note": (
            "These metrics analyse technological route precedence between machines inside jobs. "
            "They do not analyse the final scheduled order of jobs on machines; that requires a solved schedule."
        ),
    }

    # Save all CSVs.
    summary_by_file.to_csv(out_dir / "precedence_summary_by_file.csv", index=False)
    summary_by_family.to_csv(out_dir / "precedence_summary_by_family.csv", index=False)
    summary_by_size.to_csv(out_dir / "precedence_summary_by_size.csv", index=False)
    summary_by_family_size.to_csv(out_dir / "precedence_summary_by_family_size.csv", index=False)

    pos_by_file.to_csv(out_dir / "machine_position_by_file.csv", index=False)
    position_by_family.to_csv(out_dir / "machine_position_by_family.csv", index=False)
    position_by_family_size.to_csv(out_dir / "machine_position_by_family_size.csv", index=False)

    adj_by_file.to_csv(out_dir / "adjacent_machine_precedence_by_file.csv", index=False)
    adjacent_by_family.to_csv(out_dir / "adjacent_machine_precedence_by_family.csv", index=False)
    adjacent_by_family_size.to_csv(out_dir / "adjacent_machine_precedence_by_family_size.csv", index=False)

    pair_by_file.to_csv(out_dir / "pairwise_machine_precedence_by_file.csv", index=False)
    pairwise_by_family.to_csv(out_dir / "pairwise_machine_precedence_by_family.csv", index=False)
    pairwise_by_family_size.to_csv(out_dir / "pairwise_machine_precedence_by_family_size.csv", index=False)

    pd.DataFrame(errors).to_csv(out_dir / "parse_errors.csv", index=False)

    with open(out_dir / "global_precedence_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    readme = f"""JSP machine-precedence analysis results
======================================

Input: {args.input}

Parsed instances: {len(instances)}
Parse errors: {len(errors)}
Families: {', '.join(global_summary['families']) if global_summary['families'] else '-'}

What was analysed?
------------------
The route structure of each job:
    job j: machine_1 -> machine_2 -> ... -> machine_m

This is the technological/conjunctive precedence structure of the benchmark.
It is NOT the final sequencing of jobs on each machine in a solved schedule.

Main files
----------
- precedence_summary_by_file.csv
    One row per instance. Includes route diversity, position entropy, pairwise bias,
    adjacent graph density, transitive graph density, etc.

- precedence_summary_by_family.csv
    Aggregated statistics per benchmark family: abz, ft, la, orb, swv, ta, yn...

- precedence_summary_by_family_size.csv
    Aggregated statistics by family and size. This is often the safest comparison
    because some families mix several numbers of machines.

- machine_position_by_family_size.csv
    Frequency of machine m appearing at route position k.

- adjacent_machine_precedence_by_family_size.csv
    Frequency of immediate transitions m_a -> m_b.

- pairwise_machine_precedence_by_family_size.csv
    For each pair of machines {{a,b}}, estimates P(a before b).
    Values close to 0.5 indicate little directional bias.
    Values close to 0 or 1 indicate a strong/near-deterministic precedence relation.

Key metrics
-----------
- route_diversity_ratio:
    distinct machine routes / number of jobs.
    1.0 means every job has a different route.

- mean_position_entropy_norm:
    Entropy of machine positions, normalised to [0,1].
    Near 1.0 means machines are spread almost uniformly across positions.

- mean_abs_pairwise_bias:
    Mean |P(a before b) - 0.5| across machine pairs.
    Near 0 means no systematic machine precedence direction.
    Higher values mean stronger structural bias.

- adjacent_edge_density:
    Fraction of possible directed immediate machine transitions observed.

- transitive_edge_density:
    Fraction of possible directed transitive machine precedences observed.

- bidirectional_pair_ratio:
    Fraction of unordered machine pairs for which both directions appear in different jobs.

Recommended interpretation
--------------------------
Use family_size outputs for rigorous comparisons. Family-only outputs are useful,
but may mix different numbers of machines, especially in la, ta, swv, ft, etc.
"""
    (out_dir / "README_precedence_results.txt").write_text(readme, encoding="utf-8")

    if args.plots:
        maybe_make_plots(out_dir, summary_by_file, pairwise_by_family_size, position_by_family_size)

    print(f"Parsed instances: {len(instances)}")
    print(f"Parse errors: {len(errors)}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
