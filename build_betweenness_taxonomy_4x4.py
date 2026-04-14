import itertools
import json
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


def decode_perm_idx_digits(perm_idx: np.ndarray, base: int = 24, length: int = 4) -> np.ndarray:
    """
    Convierte perm_idx en base-24 y devuelve sus 4 dígitos.
    Cada dígito identifica una de las 24 permutaciones de tamaño 4.
    """
    x = np.asarray(perm_idx, dtype=np.int64).copy()
    digits = np.empty((x.size, length), dtype=np.int16)
    for pos in range(length - 1, -1, -1):
        digits[:, pos] = x % base
        x //= base
    return digits


def build_perm_matrices_from_perm_idx(perm_idx: np.ndarray) -> np.ndarray:
    """
    Reconstruye directamente la matriz 4x4 de máquinas a partir de perm_idx.
    Evita parsear la columna string `perm` y evita iterrows().
    """
    perm_lookup = np.array(list(itertools.permutations(range(4), 4)), dtype=np.int8)  # (24, 4)
    digits = decode_perm_idx_digits(perm_idx, base=24, length=4)                       # (N, 4)
    return perm_lookup[digits]                                                         # (N, 4, 4)


def grouped_matrix_stats(class_ids: np.ndarray, perm_mats: np.ndarray):
    """
    Calcula, por exact_class_id:
      1) signed_diff: reproduce EXACTAMENTE la lógica actual:
         mean_{i<j} (M_i - M_j).sum(axis=0)
         pero sin construir todas las parejas O(n^2).
      2) abs_diff: media de diferencias absolutas por columna (invariante al orden).
      3) mean/std de la matriz 4x4 por clase.

    Devuelve:
      exact_ids, counts, signed_mean, abs_mean, mean_mats, std_mats,distinct_col_sums_group
    """
    class_ids = np.asarray(class_ids, dtype=np.int32)
    perm_mats = np.asarray(perm_mats)

    order = np.argsort(class_ids, kind="mergesort")
    cls = class_ids[order]
    mats = perm_mats[order].astype(np.float64, copy=False)

    exact_ids, start, counts = np.unique(cls, return_index=True, return_counts=True)
    pair_counts = counts * (counts - 1) // 2

    # --- matrices media y std por clase ---
    sum_mats = np.add.reduceat(mats, start, axis=0)
    mean_mats = sum_mats / counts[:, None, None]

    sq_sum_mats = np.add.reduceat(mats ** 2, start, axis=0)
    std_mats = np.sqrt(np.maximum(sq_sum_mats / counts[:, None, None] - mean_mats ** 2, 0.0))
        
    # --- nº de valores distintos por columna en cada matriz ---
    distinct_col_sums = []  # lista de arrays (uno por grupo)
    distinct_col_sums_group = np.zeros((len(exact_ids), 4), dtype=np.int32)

    for g, (s, c) in enumerate(zip(start, counts)):
        block = mats[s:s+c]  # (c, 4, 4)

        # Para cada matriz (c matrices), contamos valores distintos por columna
        # Resultado: (c, 4)
        distinct_per_matrix = np.array([
            [np.unique(block[m, :, j]).size for j in range(4)]
            for m in range(c)
        ])

        # Guardamos el resultado por matriz (lista de arrays)
        distinct_col_sums.append(distinct_per_matrix)

        # Suma por columnas del grupo entero
        distinct_col_sums_group[g] = distinct_per_matrix.sum(axis=0)
        

    # --- signed diff exacto respecto a tu lógica actual ---
    # Tu código usa: (M_i - M_j).sum(axis=0), así que basta con las sumas por columna.
    col_sums = mats.sum(axis=1)  # (N, 4)

    # Dentro de cada bloque ordenado, sum_{i<j}(c_i - c_j) = sum_k (n - 1 - 2k) * c_k
    local_pos = np.arange(len(cls)) - np.repeat(start, counts)
    weights = (counts.repeat(counts) - 1 - 2 * local_pos).astype(np.int64)
    signed_sum = np.add.reduceat(col_sums * weights[:, None], start, axis=0)

    signed_mean = np.full((len(exact_ids), 4), np.nan, dtype=np.float64)
    valid = pair_counts > 0
    signed_mean[valid] = signed_sum[valid] / pair_counts[valid, None]

    # --- abs diff recomendado: invariante al orden dentro de la clase ---
    abs_mean = np.full((len(exact_ids), 4), np.nan, dtype=np.float64)
    for g, (s, c) in enumerate(zip(start, counts)):
        if c < 2:
            continue
        block = col_sums[s:s + c]  # (c, 4)
        pc = c * (c - 1) // 2
        coeff = 2 * np.arange(c, dtype=np.int64) - c + 1
        for j in range(4):
            x = np.sort(block[:, j])
            abs_mean[g, j] = (coeff @ x) / pc
    print(distinct_col_sums_group)
    quit()
    return exact_ids, counts, signed_mean, abs_mean, mean_mats, std_mats, distinct_col_sums, distinct_col_sums_group


def main(base_path: str = ".",
         btw_filename: str = "4x4_perms_btw_map.csv",
         sol_filename: str = "4x4_perms_solutions.csv",
         all4x4_filename: str = "4x4_all.csv") -> None:

    base = Path(base_path)
    btw_path = base / btw_filename
    sol_path = base / sol_filename
    all4x4_path = base / all4x4_filename

    if not btw_path.exists():
        raise FileNotFoundError(f"No existe {btw_path}")
    if not sol_path.exists():
        raise FileNotFoundError(f"No existe {sol_path}")
    if not all4x4_path.exists():
        raise FileNotFoundError(f"No existe {all4x4_path}")

    btw_all = pd.read_csv(btw_path, usecols=["perm_idx", "job", "machine", "btw"])
    sol = pd.read_csv(sol_path)
    all4x4 = pd.read_csv(all4x4_path).sort_values("perm_idx")

    # 1) Vector completo -> matriz 4x4 por perm_idx
    vecs = btw_all.pivot(index="perm_idx", columns=["job", "machine"], values="btw").sort_index(axis=1)
    perm_indices = vecs.index.to_numpy()
    arr = np.round(vecs.values.reshape(-1, 4, 4), 12)

    # 2) Canonicalización exacta por simetrías job/machine
    col_perms = list(itertools.permutations(range(4)))

    row_keys = []
    row_to_matrix = {}
    for M in arr:
        rk = tuple(row_sorted(M).flatten())
        row_keys.append(rk)
        row_to_matrix.setdefault(rk, np.array(rk).reshape(4, 4))

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

    # 2.1) Análisis rápido de patrones de cambio en máquinas por clase exacta
    all4x4 = all4x4.merge(perm_to_exact_class, on="perm_idx", how="left")

    perm_mats = build_perm_matrices_from_perm_idx(all4x4["perm_idx"].to_numpy())
    exact_ids, counts, signed_mean, abs_mean, mean_mats, std_mats, distinct_col_sums, distinct_col_sums_group = grouped_matrix_stats(
        all4x4["exact_class_id"].to_numpy(),
        perm_mats,
    )

    machine_pattern_df = pd.DataFrame({
        "exact_class_id": exact_ids,
        "n_instances": counts,
        "signed_diff_m0": signed_mean[:, 0],
        "signed_diff_m1": signed_mean[:, 1],
        "signed_diff_m2": signed_mean[:, 2],
        "signed_diff_m3": signed_mean[:, 3],
        "abs_diff_m0": abs_mean[:, 0],
        "abs_diff_m1": abs_mean[:, 1],
        "abs_diff_m2": abs_mean[:, 2],
        "abs_diff_m3": abs_mean[:, 3],
        "mean_perm_matrix_json": [json.dumps(M.tolist()) for M in mean_mats],
        "std_perm_matrix_json": [json.dumps(M.tolist()) for M in std_mats],
        "distinct_col_sums": [json.dumps(M.tolist()) for M in distinct_col_sums],
        "distinct_col_sums_group": distinct_col_sums_group.tolist()
    })

    for row in machine_pattern_df.itertuples(index=False):
        print(
            f"exact_class_id={row.exact_class_id:>3} | n={row.n_instances:>4} | "
            f"signed_mean={[row.signed_diff_m0, row.signed_diff_m1, row.signed_diff_m2, row.signed_diff_m3]}"
        )

    # 3) Prototipos de clases exactas
    proto_rows = []
    for ck, cid in exact_class_map.items():
        M = np.array(ck).reshape(4, 4)
        proto_rows.append({
            "exact_class_id": cid,
            **{f"b_{i}{j}": M[i, j] for i in range(4) for j in range(4)},
            "betweenness_mean": M.mean(),
            "betweenness_std": M.std(),
            "betweenness_range": M.max() - M.min(),
            "row_dispersion": np.std(M.sum(axis=1)),
            "col_dispersion": np.std(M.sum(axis=0)),
            "diag_proxy": np.trace(M),
            "anti_diag_proxy": np.fliplr(M).trace(),
            "matrix_json": json.dumps(M.tolist()),
        })
    proto_df = pd.DataFrame(proto_rows).sort_values("exact_class_id")

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
    out_perm = base / "betweenness_perm_to_exact_class_4x4.csv"
    out_exact = base / "betweenness_exact_taxonomy_4x4.csv"
    out_machine = base / "betweenness_machine_patterns_4x4.csv"
    out_all4x4 = base / "4x4_all_with_exact_classes.csv"

    perm_to_exact_class.to_csv(out_perm, index=False)
    machine_pattern_df.to_csv(out_machine, index=False)
    exact_summary.sort_values(["makespan_mean", "exact_class_id"]).to_csv(out_exact, index=False)
    all4x4.to_csv(out_all4x4, index=False)

    print(f"[OK] exact classes: {len(unique_canon)}")
    print(f"[OK] saved: {out_perm}")
    print(f"[OK] saved: {out_machine}")
    print(f"[OK] saved: {out_exact}")
    print(f"[OK] saved: {out_all4x4}")


if __name__ == "__main__":
    main()
