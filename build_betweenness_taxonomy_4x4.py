import itertools
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from itertools import permutations, product


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


def main(base_path: str = ".",
         btw_filename: str = "4x4_perms_btw_map.csv",
         sol_filename: str = "4x4_perms_solutions.csv",
         all4x4: str = "4x4_all.csv",
         n_macro_classes: int = 4) -> None:

    base = Path(base_path)
    btw_path = base / btw_filename
    sol_path = base / sol_filename
    all4x4_path = base / all4x4

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
    
    
    
    # 2.1) Cruce con matriz original para tener la matriz 4x4 representativa de cada perm_idx
    all4x4 = all4x4.merge(perm_to_exact_class, on="perm_idx", how="left")
    all4x4_perm = all4x4[["perm_idx", "exact_class_id","perm"]].copy().set_index("perm_idx")
    list_of_permutation =  list(permutations(range(4), 4)) 

    all4x4_perm["perm_matrix"] = [np.array(list(map(lambda x:  list(list_of_permutation[int(x)]) ,v["perm"][1:-1].split(",")))) for k, v in all4x4_perm.iterrows()]
    # pbar = tqdm(list(range(len(exact_class_map))), desc="Calculating differences between perm matrices")
    for i in list(range(len(exact_class_map))):
        x1 = all4x4_perm.query(f"exact_class_id == {i}")
        # x1 = all4x4_perm.query(f"exact_class_id == {i}")
        
        diff_x1 = np.array([np.array(x1["perm_matrix"].iloc[i] - x1["perm_matrix"].iloc[j]).sum(axis=0) for i in range(len(x1)) for j in range(i, len(x1)) if i != j])
        # diff_x2 = np.array([np.array(x2["perm_matrix"].iloc[i] - x2["perm_matrix"].iloc[j]).sum(axis=0) for i in range(len(x2)) for j in range(i, len(x2)) if i != j])
        # for l in diff_x1:
        #     print(l)
        # print(np.unique(diff_x1.flatten()), len(np.unique(diff_x1.flatten())))
        print(diff_x1.sum(axis=0) / len(diff_x1))


    
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

    # 5) Macroclases
    X = proto_df[[f"b_{i}{j}" for i in range(4) for j in range(4)]].values
    clustering = AgglomerativeClustering(n_clusters=n_macro_classes, linkage="ward")
    proto_df["macro_class_id"] = clustering.fit_predict(X)

    exact_summary = exact_summary.merge(
        proto_df[["exact_class_id", "macro_class_id"]],
        on="exact_class_id",
        how="left",
    )

    macro_summary = exact_summary.groupby("macro_class_id").agg(
        n_exact_classes=("exact_class_id", "nunique"),
        n_permutations=("n_permutations", "sum"),
        makespan_mean=("makespan_mean", "mean"),
        makespan_median=("makespan_median", "mean"),
        failures_mean=("failures_mean", "mean"),
        failures_median=("failures_median", "mean"),
        nSolutions_mean=("nSolutions_mean", "mean"),
        solveTime_mean=("solveTime_mean", "mean"),
        prop_mean=("propagations_mean", "mean"),
        betweenness_mean=("betweenness_mean", "mean"),
        betweenness_range=("betweenness_range", "mean"),
        row_dispersion=("row_dispersion", "mean"),
        col_dispersion=("col_dispersion", "mean"),
    ).reset_index().sort_values("makespan_mean")

    # representante de cada macroclase
    reps = []
    for mc in sorted(proto_df["macro_class_id"].unique()):
        sub = proto_df[proto_df["macro_class_id"] == mc].copy()
        Xs = sub[[f"b_{i}{j}" for i in range(4) for j in range(4)]].values
        cen = Xs.mean(axis=0)
        idx = np.argmin(((Xs - cen) ** 2).sum(axis=1))
        reps.append((
            mc,
            int(sub.iloc[idx]["exact_class_id"]),
            sub.iloc[idx]["matrix_json"],
        ))
    rep_df = pd.DataFrame(
        reps,
        columns=[
            "macro_class_id",
            "representative_exact_class_id",
            "representative_matrix_json",
        ],
    )
    macro_summary = macro_summary.merge(rep_df, on="macro_class_id", how="left")

    # Etiquetas interpretables simples
    ms = macro_summary.set_index("macro_class_id")
    labels = {}
    for mc, row in ms.iterrows():
        if row["betweenness_mean"] == ms["betweenness_mean"].max():
            labels[mc] = "high-centralized"
        elif row["row_dispersion"] == ms["row_dispersion"].max():
            labels[mc] = "row-asymmetric"
        elif row["col_dispersion"] == ms["col_dispersion"].max():
            labels[mc] = "machine-asymmetric"
        else:
            labels[mc] = "balanced-low"
    macro_summary["macro_label"] = macro_summary["macro_class_id"].map(labels)

    # 6) Guardado
    out_perm = base / "betweenness_perm_to_exact_class_4x4.csv"
    out_exact = base / "betweenness_exact_taxonomy_4x4.csv"
    out_macro = base / "betweenness_macro_taxonomy_4x4.csv"

    perm_to_exact_class.to_csv(out_perm, index=False)
    exact_summary.sort_values(["macro_class_id", "makespan_mean", "exact_class_id"]).to_csv(out_exact, index=False)
    # macro_summary.to_csv(out_macro, index=False)
    all4x4.to_csv(base / "4x4_all_with_exact_classes.csv", index=False)

    print(f"[OK] exact classes: {len(unique_canon)}")
    print(f"[OK] macro classes: {n_macro_classes}")
    print(f"[OK] saved: {out_perm}")
    print(f"[OK] saved: {out_exact}")
    print(f"[OK] saved: {out_macro}")
    print()
    print(macro_summary.to_string(index=False))


if __name__ == "__main__":
    main()
