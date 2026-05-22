import itertools
import json
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd


def row_sorted(M: np.ndarray) -> np.ndarray:
    """Ordena las filas lexicográficamente."""
    return np.array(sorted([tuple(r) for r in M]))


def canonical_key_from_row_sorted(M: np.ndarray, col_perms) -> tuple:
    """
    Dada una matriz ya libre de simetría de filas, elimina también la de columnas:
    prueba todas las permutaciones de columnas, reordena filas y toma la mínima.
    """
    best_key = None
    best_matrix = None
    for p in col_perms:
        Mr = row_sorted(M[:, p])
        key = tuple(Mr.flatten())
        if best_key is None or key < best_key:
            best_key = key
            best_matrix = Mr.copy()
    return best_key, best_matrix


def decode_perm_idx_digits(perm_idx: np.ndarray, base: int, length: int) -> np.ndarray:
    """
    Convierte perm_idx en base=<n_perms> y devuelve sus <length> dígitos.
    Cada dígito identifica una de las permutaciones de tamaño <length>.
    """
    x = np.asarray(perm_idx, dtype=np.int64).copy()
    digits = np.empty((x.size, length), dtype=np.int32)

    for pos in range(length - 1, -1, -1):
        digits[:, pos] = x % base
        x //= base

    return digits


def build_perm_matrices_from_perm_idx(perm_idx: np.ndarray, size: int) -> np.ndarray:
    """
    Reconstruye la matriz size x size de máquinas a partir de perm_idx.
    Asume que perm_idx sigue el orden lexicográfico de:
        product(range(factorial(size)), repeat=size)
    """
    n_perms = factorial(size)
    perm_lookup = np.array(list(itertools.permutations(range(size), size)), dtype=np.int16)
    digits = decode_perm_idx_digits(perm_idx, base=n_perms, length=size)
    return perm_lookup[digits]  # (N, size, size)


def grouped_matrix_stats(class_ids: np.ndarray, perm_mats: np.ndarray):
    """
    Calcula, por exact_class_id:
      1) signed_diff: mean_{i<j} (M_i - M_j).sum(axis=0)
      2) abs_diff: media de diferencias absolutas por columna
      3) mean/std de la matriz size x size por clase
      4) nº de valores distintos por columna

    Devuelve:
      exact_ids, counts, signed_mean, abs_mean, mean_mats, std_mats,
      distinct_col_sums, distinct_col_sums_group
    """
    class_ids = np.asarray(class_ids, dtype=np.int32)
    perm_mats = np.asarray(perm_mats, dtype=np.float64)

    if perm_mats.ndim != 3 or perm_mats.shape[1] != perm_mats.shape[2]:
        raise ValueError("perm_mats debe tener forma (N, size, size).")

    size = perm_mats.shape[1]

    order = np.argsort(class_ids, kind="mergesort")
    cls = class_ids[order]
    mats = perm_mats[order]

    exact_ids, start, counts = np.unique(cls, return_index=True, return_counts=True)
    pair_counts = counts * (counts - 1) // 2

    # media y std de matrices por grupo
    sum_mats = np.add.reduceat(mats, start, axis=0)
    mean_mats = sum_mats / counts[:, None, None]

    sq_sum_mats = np.add.reduceat(mats ** 2, start, axis=0)
    std_mats = np.sqrt(np.maximum(sq_sum_mats / counts[:, None, None] - mean_mats ** 2, 0.0))

    # nº de valores distintos por columna
    distinct_col_sums = []
    distinct_col_sums_group = np.zeros((len(exact_ids), size), dtype=np.int32)

    for g, (s, c) in enumerate(zip(start, counts)):
        block = mats[s:s + c]  # (c, size, size)

        distinct_per_matrix = np.array([
            [np.unique(block[m, :, j]).size for j in range(size)]
            for m in range(c)
        ], dtype=np.int32)

        distinct_col_sums.append(distinct_per_matrix)
        distinct_col_sums_group[g] = distinct_per_matrix.sum(axis=0)

    # signed diff
    col_sums = mats.sum(axis=1)  # (N, size)

    local_pos = np.arange(len(cls)) - np.repeat(start, counts)
    weights = (counts.repeat(counts) - 1 - 2 * local_pos).astype(np.int64)
    signed_sum = np.add.reduceat(col_sums * weights[:, None], start, axis=0)

    signed_mean = np.full((len(exact_ids), size), np.nan, dtype=np.float64)
    valid = pair_counts > 0
    signed_mean[valid] = signed_sum[valid] / pair_counts[valid, None]

    # abs diff
    abs_mean = np.full((len(exact_ids), size), np.nan, dtype=np.float64)
    for g, (s, c) in enumerate(zip(start, counts)):
        if c < 2:
            continue

        block = col_sums[s:s + c]  # (c, size)
        pc = c * (c - 1) // 2
        coeff = 2 * np.arange(c, dtype=np.int64) - c + 1

        for j in range(size):
            x = np.sort(block[:, j])
            abs_mean[g, j] = (coeff @ x) / pc

    return (
        exact_ids,
        counts,
        signed_mean,
        abs_mean,
        mean_mats,
        std_mats,
        distinct_col_sums,
        distinct_col_sums_group,
    )


def build_machine_pattern_df(
    exact_ids,
    counts,
    signed_mean,
    abs_mean,
    mean_mats,
    std_mats,
    distinct_col_sums,
    distinct_col_sums_group,
    size: int,
):
    data = {
        "exact_class_id": exact_ids,
        "n_instances": counts,
    }

    for j in range(size):
        data[f"signed_diff_m{j}"] = signed_mean[:, j]
    for j in range(size):
        data[f"abs_diff_m{j}"] = abs_mean[:, j]

    data["mean_perm_matrix_json"] = [json.dumps(M.tolist()) for M in mean_mats]
    data["std_perm_matrix_json"] = [json.dumps(M.tolist()) for M in std_mats]
    data["distinct_col_sums"] = [json.dumps(M.tolist()) for M in distinct_col_sums]
    data["distinct_col_sums_group"] = [json.dumps(M.tolist()) for M in distinct_col_sums_group]

    return pd.DataFrame(data)


def build_proto_df(exact_class_map: dict, size: int) -> pd.DataFrame:
    proto_rows = []

    for ck, cid in exact_class_map.items():
        M = np.array(ck).reshape(size, size)

        row = {
            "exact_class_id": cid,
            "betweenness_mean": M.mean(),
            "betweenness_std": M.std(),
            "betweenness_range": M.max() - M.min(),
            "row_dispersion": np.std(M.sum(axis=1)),
            "col_dispersion": np.std(M.sum(axis=0)),
            "diag_proxy": np.trace(M),
            "anti_diag_proxy": np.fliplr(M).trace(),
            "matrix_json": json.dumps(M.tolist()),
        }

        for i in range(size):
            for j in range(size):
                row[f"b_{i}{j}"] = M[i, j]

        proto_rows.append(row)

    return pd.DataFrame(proto_rows).sort_values("exact_class_id")


def main(
    size: int = 4,
    base_path: str = ".",
    btw_filename: str | None = None,
    sol_filename: str | None = None,
    all_filename: str | None = None,
) -> None:
    prefix = f"{size}x{size}"

    if btw_filename is None:
        btw_filename = f"{prefix}_perms_btw_map.csv"
    if sol_filename is None:
        sol_filename = f"{prefix}_perms_solutions.csv"
    if all_filename is None:
        all_filename = f"{prefix}_all.csv"

    base = Path(base_path)
    btw_path = base / btw_filename
    sol_path = base / sol_filename
    all_path = base / all_filename

    if not btw_path.exists():
        raise FileNotFoundError(f"No existe {btw_path}")
    if not sol_path.exists():
        raise FileNotFoundError(f"No existe {sol_path}")
    if not all_path.exists():
        raise FileNotFoundError(f"No existe {all_path}")

    btw_all = pd.read_csv(btw_path, usecols=["perm_idx", "job", "machine", "btw"])
    sol = pd.read_csv(sol_path)
    all_df = pd.read_csv(all_path).sort_values("perm_idx")

    # 1) Vector completo -> matriz size x size por perm_idx
    vecs = btw_all.pivot(index="perm_idx", columns=["job", "machine"], values="btw").sort_index(axis=1)
    perm_indices = vecs.index.to_numpy()
    arr = np.round(vecs.values.reshape(-1, size, size), 12)

    # 2) Canonicalización exacta por simetrías fila/columna
    col_perms = list(itertools.permutations(range(size)))

    row_keys = []
    row_to_matrix = {}
    for M in arr:
        rk = tuple(row_sorted(M).flatten())
        row_keys.append(rk)
        row_to_matrix.setdefault(rk, np.array(rk).reshape(size, size))

    canon_cache = {}
    for rk, M in row_to_matrix.items():
        canon_cache[rk] = canonical_key_from_row_sorted(M, col_perms)

    canon_keys = [canon_cache[rk][0] for rk in row_keys]
    unique_canon = list(dict.fromkeys(canon_keys))
    exact_class_map = {ck: i for i, ck in enumerate(unique_canon)}

    perm_to_exact_class = pd.DataFrame({
        "perm_idx": perm_indices,
        "exact_class_id": [exact_class_map[ck] for ck in canon_keys],
    }).sort_values("perm_idx")

    # 2.1) Patrones de cambio por clase exacta
    all_df = all_df.merge(perm_to_exact_class, on="perm_idx", how="left")

    perm_mats = build_perm_matrices_from_perm_idx(all_df["perm_idx"].to_numpy(), size=size)
    (
        exact_ids,
        counts,
        signed_mean,
        abs_mean,
        mean_mats,
        std_mats,
        distinct_col_sums,
        distinct_col_sums_group,
    ) = grouped_matrix_stats(
        all_df["exact_class_id"].to_numpy(),
        perm_mats,
    )

    machine_pattern_df = build_machine_pattern_df(
        exact_ids,
        counts,
        signed_mean,
        abs_mean,
        mean_mats,
        std_mats,
        distinct_col_sums,
        distinct_col_sums_group,
        size=size,
    )

    for row in machine_pattern_df.itertuples(index=False):
        signed_vals = [getattr(row, f"signed_diff_m{j}") for j in range(size)]
        print(
            f"exact_class_id={row.exact_class_id:>3} | "
            f"n={row.n_instances:>4} | signed_mean={signed_vals}"
        )

    # 3) Prototipos de clases exactas
    proto_df = build_proto_df(exact_class_map, size=size)

    # 4) Cruce con resultados del solver
    df = sol.merge(perm_to_exact_class, on="perm_idx", how="left")

    exact_summary = df.groupby("exact_class_id").agg(
        n_permutations=("perm_idx", "size"),
        n_distinct_perm_tuples=("perm", "nunique"),
        makespan_mean=("objectiveBound", "mean"),
        makespan_median=("objectiveBound", "median"),
        makespan_min=("objectiveBound", "min"),
        makespan_max=("objectiveBound", "max"),
        failures_mean=("failures", "mean"),
        failures_median=("failures", "median"),
        failures_max=("failures", "max"),
        nSolutions_mean=("nSolutions", "mean"),
        nSolutions_median=("nSolutions", "median"),
        solveTime_mean=("solveTime", "mean"),
        time_mean=("time", "mean"),
        propagations_mean=("propagations", "mean"),
        opt_rate=("status", lambda s: (s == "OPTIMAL_SOLUTION").mean()),
    ).reset_index()

    exact_summary = exact_summary.merge(proto_df, on="exact_class_id", how="left")
    exact_summary = exact_summary.merge(machine_pattern_df, on="exact_class_id", how="left")

    # 5) Guardado
    out_perm = base / f"betweenness_perm_to_exact_class_{prefix}.csv"
    out_exact = base / f"betweenness_exact_taxonomy_{prefix}.csv"
    out_machine = base / f"betweenness_machine_patterns_{prefix}.csv"
    out_all = base / f"{prefix}_all_with_exact_classes.csv"

    perm_to_exact_class.to_csv(out_perm, index=False)
    machine_pattern_df.to_csv(out_machine, index=False)
    exact_summary.sort_values(["makespan_mean", "exact_class_id"]).to_csv(out_exact, index=False)
    all_df.to_csv(out_all, index=False)

    print(f"[OK] size={size}")
    print(f"[OK] exact classes: {len(unique_canon)}")
    print(f"[OK] saved: {out_perm}")
    print(f"[OK] saved: {out_machine}")
    print(f"[OK] saved: {out_exact}")
    print(f"[OK] saved: {out_all}")


if __name__ == "__main__":
    main(size=3)