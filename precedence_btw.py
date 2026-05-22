import numpy as np
import networkx as nx
import pandas as pd
import time
import torch
import os

from tqdm import tqdm
from math import factorial
from functools import lru_cache
from types import SimpleNamespace
from collections import defaultdict
from IGJSP.generador import Generator
from torch_geometric.data import HeteroData
from itertools import permutations, product
from solvers import SOLVER, compute_time_limit
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter


# timeout dinámico
Q_JOBS = range(2, 51)
Q_MACHINES = range(2, 51)

TIME_MIN = 400 / 2
TIME_MAX = 120000 / 2

def build_maxtime(q_jobs, q_machines, time_min, time_max):
    t_job = ((min(q_jobs) * time_min) / 2, (max(q_jobs) * time_max / 100))
    t_mach = ((min(q_machines) * time_min) / 2, (max(q_machines) * time_max / 100))

    maxtime = (
        np.array(
            np.linspace(
                t_job[0] + t_mach[0],
                t_job[1] + t_mach[1],
                len(q_jobs) * len(q_machines),
                dtype=int,
            )
        )
        .reshape(len(q_jobs), len(q_machines))
        .astype(float)
    )

    return maxtime


MAXTIME = build_maxtime(Q_JOBS, Q_MACHINES, TIME_MIN, TIME_MAX)


def dynamic_timeout(instance):
    j_idx = instance.numJobs - min(Q_JOBS)
    m_idx = instance.numMchs - min(Q_MACHINES)

    j_idx = max(0, min(j_idx, MAXTIME.shape[0] - 1))
    m_idx = max(0, min(m_idx, MAXTIME.shape[1] - 1))

    return float(MAXTIME[j_idx, m_idx])


def dict_to_problem(inst_dict):
    inst = dict(inst_dict)

    # valores por defecto por si el generador no los trae
    inst.setdefault("rddd", 0)
    inst.setdefault("speed", 1)

    # asegurar np.array
    inst["Orden"] = np.asarray(inst["Orden"])
    inst["ProcessingTime"] = np.asarray(inst["ProcessingTime"])
    inst["EnergyConsumption"] = np.asarray(inst["EnergyConsumption"])

    return SimpleNamespace(**inst)

def extract_operations_strict(data: Dict[str, Any]) -> List[Tuple[int, int, int, float, float]]:
    """
    Extrae operaciones asumiendo estructura estricta:
      - ProcessingTime: (J, O) o (J, O, 1)
      - Orden:          (J, O)  con id de máquina por operación
      - EnergyConsumption: (J, O) o (J, O, 1)
    """
    PT = np.asarray(data["ProcessingTime"])
    OR = np.asarray(data["Orden"])
    EC = np.asarray(data["EnergyConsumption"])

    num_jobs = int(data["numJobs"])

    # PT
    if PT.ndim == 3:
        num_ops = PT.shape[1]
        get_pt = lambda j, o: float(PT[j, o, 0])
    elif PT.ndim == 2:
        num_ops = PT.shape[1]
        get_pt = lambda j, o: float(PT[j, o])
    else:
        raise ValueError("ProcessingTime tiene una forma inesperada. Esperado (J,O[,1]).")

    # EC
    if EC.ndim == 3:
        get_ec = lambda j, o: float(EC[j, o, 0])
    elif EC.ndim == 2:
        get_ec = lambda j, o: float(EC[j, o])
    else:
        raise ValueError("EnergyConsumption tiene una forma inesperada. Esperado (J,O[,1]).")

    ops = []
    for j in range(num_jobs):
        for o in range(num_ops):
            proc_time = get_pt(j, o)
            energy = get_ec(j, o)
            mach = int(OR[j, o])
            ops.append((j, o, mach, proc_time, energy))
    return ops


def _maybe_get(mat: Dict[str, Any], keys: List[str]) -> Optional[np.ndarray]:
    """Devuelve np.array si existe alguna de las keys y no está vacía."""
    for k in keys:
        if k in mat and mat[k] is not None:
            arr = np.asarray(mat[k])
            if arr.size > 0:
                return arr
    return None


def _flatten_generator_features(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza los nombres de métricas precomputadas del generador y
    las devuelve como escalares Python (o np.* convertidos).
    Mantiene también una copia del dict original 'features' para referencia.
    """
    out: Dict[str, Any] = {}
    # metadatos básicos
    if "numJobs" in inst:  out["n_jobs"] = int(inst["numJobs"])
    if "numMchs" in inst:  out["n_machs"] = int(inst["numMchs"])
    if "speed"   in inst:  out["speed"]  = int(inst["speed"])  # nº de velocidades
    if "rddd"    in inst:  out["rddd"]   = int(inst["rddd"])   # flag de R/D/D presente (si aplica)

    # makespan (nombres del generador)
    if "max_makespan" in inst: out["makespan_max"] = int(inst["max_makespan"])
    if "min_makespan" in inst: out["makespan_min"] = int(inst["min_makespan"])
    # a veces viene como rango o diferencia max-min
    if "max_min_makespan" in inst:
        out["makespan_range"] = int(inst["max_min_makespan"])

    # energía total (sumas)
    # el generador usa dos variantes de nombre: max_energy/min_energy y max_sum_energy/min_sum_energy
    if "max_energy" in inst: out["energy_sum_max"] = int(inst["max_energy"])
    if "min_energy" in inst: out["energy_sum_min"] = int(inst["min_energy"])
    if "max_min_energy" in inst: out["energy_sum_range"] = int(inst["max_min_energy"])

    feats = inst.get("features", {})
    if isinstance(feats, dict):
        # sobreescribimos si están también en 'features' con otros nombres
        if "max_makespan" in feats: out["makespan_max"] = int(feats["max_makespan"])
        if "min_makespan" in feats: out["makespan_min"] = int(feats["min_makespan"])

        if "max_sum_energy" in feats: out["energy_sum_max"] = int(feats["max_sum_energy"])
        if "min_sum_energy" in feats: out["energy_sum_min"] = int(feats["min_sum_energy"])

        # valores puntuales (por operación) agregados
        if "max_processing_time_value" in feats: out["p_value_max"] = int(feats["max_processing_time_value"])
        if "min_processing_time_value" in feats: out["p_value_min"] = int(feats["min_processing_time_value"])
        if "mean_processing_time_value" in feats: out["p_value_mean"] = float(feats["mean_processing_time_value"])

        if "max_energy_value" in feats: out["e_value_max"] = int(feats["max_energy_value"])
        if "min_energy_value" in feats: out["e_value_min"] = int(feats["min_energy_value"])
        if "mean_energy_value" in feats: out["e_value_mean"] = float(feats["mean_energy_value"])

        # ventanas de tiempo + overlap + tardiness (si el generador los rellena)
        if "min_window" in feats: out["window_min"] = float(feats["min_window"])
        if "mean_window" in feats: out["window_mean"] = float(feats["mean_window"])
        if "max_window" in feats: out["window_max"] = float(feats["max_window"]) if feats.get("max_window") is not None else None
        if "overlap" in feats: out["overlap"] = float(feats["overlap"])
        if "max_tardiness" in feats: out["max_tardiness"] = float(feats["max_tardiness"])

        # copia cruda por si quieres depurar/usar directamente
        out["gen_features"] = feats  # se serializa como dict en el HeteroData

    return out


class GraphBuilderStrict:
    def __init__(self, instance_dict: Dict[str, Any]):
        self.data_dict = instance_dict
        self.num_jobs = int(instance_dict["numJobs"])
        self.num_machs = int(instance_dict["numMchs"])
        self.operations = extract_operations_strict(instance_dict)

        self.node_features: List[List[float]] = []
        self.node_mapping: Dict[Tuple[int, int], int] = {}
        self.darcs: List[List[int]] = []
        self.carcs: List[List[int]] = []
        self.data = None

        self.build_graph()

    def _ops_per_job(self) -> int:
        PT = np.asarray(self.data_dict["ProcessingTime"])
        return PT.shape[1] if PT.ndim in (2, 3) else int(round(len(self.operations) / self.num_jobs))

    def build_graph(self):
        # ---------- 1) Nodos ----------
        node_id = 0
        include_virtual = True

        if include_virtual:
            self.node_features.append([0.0, -1.0])              # source
            self.node_mapping[("source", 0)] = node_id

        op_node_index: List[int] = []
        for (job_id, op_id, mac_id, proc_time, energy) in self.operations:
            node_id += 1
            self.node_features.append([float(proc_time), float(mac_id)])
            self.node_mapping[(job_id, op_id)] = node_id
            op_node_index.append(node_id)

        if include_virtual:
            node_id += 1
            self.node_features.append([0.0, -1.0])              # sink
            self.node_mapping[("sink", 0)] = node_id

        # ---------- 2) Arcos conjuntivos ----------
        origen, destino = [], []
        for j in range(self.num_jobs):
            nodes = [self.node_mapping[(j, o)] for o in range(self._ops_per_job())]
            if include_virtual:
                origen.append(self.node_mapping[("source", 0)]); destino.append(nodes[0])
            for i in range(len(nodes) - 1):
                origen.append(nodes[i]); destino.append(nodes[i + 1])
            if include_virtual:
                origen.append(nodes[-1]); destino.append(self.node_mapping[("sink", 0)])
        self.carcs = [origen, destino]

        # ---------- 3) Arcos disyuntivos ----------
        mach_to_nodes = defaultdict(list)
        for (job_id, op_id, mac_id, proc_time, energy) in self.operations:
            nid = self.node_mapping[(job_id, op_id)]
            mach_to_nodes[mac_id].append(nid)

        origen_d, destino_d = [], []
        for nodes in mach_to_nodes.values():
            if len(nodes) < 2:
                continue
            for a, b in permutations(nodes, 2):
                origen_d.append(a); destino_d.append(b)
        self.darcs = [origen_d, destino_d]

        # ---------- 4) Crear HeteroData ----------
        data = HeteroData()
        data["node"].x = torch.tensor(self.node_features, dtype=torch.float)

        data[("node", "conjunctive", "node")].edge_index = (
            torch.tensor(self.carcs, dtype=torch.long) if len(self.carcs[0]) > 0
            else torch.empty((2, 0), dtype=torch.long)
        )
        data[("node", "disjunctive", "node")].edge_index = (
            torch.tensor(self.darcs, dtype=torch.long) if len(self.darcs[0]) > 0
            else torch.empty((2, 0), dtype=torch.long)
        )

        # ---------- 5) Variables de instancia (tesis) ----------
        N_ops = len(self.operations)
        job_ids  = np.fromiter((j for (j, o, m, p, e) in self.operations), dtype=np.int64, count=N_ops)
        op_ids   = np.fromiter((o for (j, o, m, p, e) in self.operations), dtype=np.int64, count=N_ops)
        mach_ids = np.fromiter((m for (j, o, m, p, e) in self.operations), dtype=np.int64, count=N_ops)
        P = np.fromiter((p for (j, o, m, p, e) in self.operations), dtype=np.float64, count=N_ops).reshape(N_ops, 1)
        E = np.fromiter((e for (j, o, m, p, e) in self.operations), dtype=np.float64, count=N_ops).reshape(N_ops, 1)

        # Coste por operación si lo trae el generador (J,O) -> (N_ops,)
        op_cost = _maybe_get(self.data_dict, ["operationCost", "opCost", "operation_cost"])
        if op_cost is not None:
            op_cost = np.asarray(op_cost, dtype=np.float64).reshape(self.num_jobs, self._ops_per_job())
            op_cost = op_cost.reshape(-1)  # aplanamos por filas (job-major)

        # Ventanas de tiempo (si existen)
        R = _maybe_get(self.data_dict, ["ReleaseDate", "Release", "R"])
        D = _maybe_get(self.data_dict, ["DueDate", "Due", "D"])
        if R is not None:
            R = R.reshape(self.num_jobs, -1).astype(np.float64)
            R = R[:, :self._ops_per_job()].reshape(-1)
        if D is not None:
            D = D.reshape(self.num_jobs, -1).astype(np.float64)
            D = D[:, :self._ops_per_job()].reshape(-1)

        # AVJM (si no existe, derivamos del uso de máquinas en cada job)
        avjm = _maybe_get(self.data_dict, ["AllowedMachines", "allowed_machines", "AVJM", "avail_machines", "machine_mask"])
        if avjm is None:
            avjm = np.zeros((self.num_jobs, self.num_machs), dtype=np.int64)
            for j in range(self.num_jobs):
                used = np.unique(mach_ids[j * self._ops_per_job() : (j + 1) * self._ops_per_job()])
                avjm[j, used] = 1
        else:
            avjm = (np.asarray(avjm) > 0).astype(np.int64)
            if avjm.shape == (N_ops, self.num_machs):
                tmp = np.zeros((self.num_jobs, self.num_machs), dtype=np.int64)
                for j in range(self.num_jobs):
                    mask = (job_ids == j)
                    if np.any(mask):
                        tmp[j] = (avjm[mask].max(axis=0) > 0).astype(np.int64)
                avjm = tmp

        # Setup (si lo hubiese)
        setup_mat = _maybe_get(self.data_dict, ["Setup", "SetupTime", "setup", "mtt", "Changeover", "setup_matrix"])
        if setup_mat is not None:
            setup_mat = setup_mat.astype(np.float64)

        # Guardado top-level (para extractor)
        data.n_jobs = int(self.num_jobs)
        data.n_machs = int(self.num_machs)
        data.P = torch.tensor(P, dtype=torch.float)                 # (N_ops, 1)
        data.E = torch.tensor(E, dtype=torch.float)                 # (N_ops, 1)
        data.job = torch.tensor(job_ids, dtype=torch.long)          # (N_ops,)
        data.machine = torch.tensor(mach_ids, dtype=torch.long)     # (N_ops,)
        data.op = torch.tensor(op_ids, dtype=torch.long)            # (N_ops,)
        data.op_node_index = torch.tensor(op_node_index, dtype=torch.long)

        if op_cost is not None:
            data.operation_cost = torch.tensor(op_cost, dtype=torch.float)  # (N_ops,)

        if R is not None: data.R = torch.tensor(R, dtype=torch.float)       # (N_ops,)
        if D is not None: data.D = torch.tensor(D, dtype=torch.float)       # (N_ops,)
        if setup_mat is not None: data.setup = torch.tensor(setup_mat, dtype=torch.float)  # varias formas
        if avjm is not None: data.allowed_machines = torch.tensor(avjm, dtype=torch.long)  # (J,M)

        # Máscaras útiles
        N_nodes = len(self.node_features)
        op_mask = torch.zeros(N_nodes, dtype=torch.bool)
        op_mask[data.op_node_index] = True
        data["node"].op_mask = op_mask

        # Índices de source/sink (si existen)
        if include_virtual:
            data.source_idx = int(self.node_mapping[("source", 0)])
            data.sink_idx = int(self.node_mapping[("sink", 0)])

        # ---------- 6) Inyectar métricas precomputadas del generador ----------
        gf = _flatten_generator_features(self.data_dict)
        # Copiamos cada par k->v como atributo simple (escalares o dict para gen_features)
        for k, v in gf.items():
            setattr(data, k, v if not isinstance(v, np.generic) else v.item())

        self.data = data


@lru_cache(maxsize=4096)
def nth_permutation(pool, k):
    items = list(pool)
    result = []
    n = len(items)

    for i in range(n, 0, -1):
        f = factorial(i - 1)
        pos, k = divmod(k, f)
        result.append(items.pop(pos))

    return tuple(result)


class MyCombinations:
    def __init__(self, iterable, r: int):
        self.pool = tuple(iterable)         # p.ej. (0,1,2,3,4)
        self.m = len(self.pool)             # tamaño de cada permutación
        self.r = r                          # nº de filas de la matriz

        if r <= 0:
            self.finished = True
            self.n = 0
            self.total = 0
            self.indices = iter(())
        else:
            self.finished = False
            self.n = factorial(self.m)      # nº total de permutaciones
            self.total = self.n ** self.r   # nº total de matrices
            self.indices = product(range(self.n), repeat=self.r)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.finished:
            raise StopIteration
        idx = next(self.indices, None)
        if idx is None:
            self.finished = True
            raise StopIteration
        # matriz (r x m), donde cada fila es una permutación
        return np.array([nth_permutation(self.pool, i) for i in idx], dtype=int), idx


def extract_features(graph, k):
    
    features = {}
    
    # --- 2) aristas conj y disj ---
    conj_idx = graph[('node','conjunctive','node')]['edge_index'] .detach().cpu().numpy()
    disj_idx = graph[('node','disjunctive','node')]['edge_index'] .detach().cpu().numpy()
    
    features['num_conjunctive_edges'] = int(conj_idx.shape[1]) if conj_idx is not None else 0
    features['num_disjunctive_edges'] = int(disj_idx.shape[1]) if disj_idx is not None else 0
    
    max_node = -1
    for idx in [conj_idx, disj_idx]:
        if idx is not None and idx.size>0:
            max_node = max(max_node, int(np.max(idx)))
    N = int(max_node + 1) if max_node >= 0 else 0
    features['num_nodes_inferred'] = N
    features['num_nodes_total'] = N
    features['num_edges_total'] = features['num_conjunctive_edges'] + features['num_disjunctive_edges']

    # # --- Grafo no dirigido a partir de conjuntivas ---        
    # G_c = nx.Graph()
    # G_c.add_nodes_from(range(N))
    # if conj_idx is not None:
    #     u = conj_idx[0].astype(int); v = conj_idx[1].astype(int)
    #     edges = list(zip(u.tolist(), v.tolist()))
    #     G_c.add_edges_from(edges)
    
    # --- Grafo no dirigido a partir de conjuntivas y disyuntivas ---        
    G = nx.Graph()
    G.add_nodes_from(range(N))
    if conj_idx is not None:
        u = conj_idx[0].astype(int); v = conj_idx[1].astype(int)
        edges = list(zip(u.tolist(), v.tolist()))
        G.add_edges_from(edges)
        
    if disj_idx is not None:
        u = disj_idx[0].astype(int); v = disj_idx[1].astype(int)
        edges = list(zip(u.tolist(), v.tolist()))
        G.add_edges_from(edges)

    try:
        btw = nx.betweenness_centrality(G, normalized=True)
        btw.pop(0)
        btw.pop(N-1)
        arr = np.array(list(btw.values()))
        features[f'betweenness_mean'] = float(np.mean(arr))
        features[f'betweenness_range'] = float(np.max(arr) - np.min(arr))     
    except Exception as e:
        features['betweenness_error'] = str(e)

    return features, arr

def matrix_symmetry_level(matrix):
    transpose = matrix.T
    
    # Calcular diferencia absoluta media
    if np.allclose(matrix, transpose):
        return 1.0
    
    # Calcular distancia de la transpuesta
    dif = np.abs(matrix - transpose)
    sym = 1.0 - (np.sum(dif) / (matrix.size * np.max(np.abs(matrix)) + 1e-9))
    
    return max(0.0, sym)
        

def main(size=4):
    
    start_time = time.time()
    
    lines = []
    lines.append("=== PERMUTATIONS GENERATOR REPORT ===\n")
    
    fname = f"{size}x{size}_perms"
    report_name = f"{fname}_report.txt"

    g = Generator(json=False)

    instance = g.generate_new_instance(
        jobs=size,
        machines=size,
        tpm=np.ones((size, size))
    )

    precedence = tuple(range(size))
    comb_iter = MyCombinations(precedence, size)
    
    out = [None] * comb_iter.__len__()
    btw_map = [{} for _ in range(len(comb_iter))]
    unique_map = {}
    
    # signatures = {
    #     (4,): "4",
    #     (3,1): "3+1",
    #     (2,2): "2+2",
    #     (2,1,1): "2+1+1",
    #     (1,1,1,1): "1+1+1+1"
    # }
    signatures = {
            (3,): "3",
            (2,1): "2+1",
            (1,1,1): "1+1+1"
    }
    solutions_rows = []
    
    pbar = tqdm(comb_iter, total=len(comb_iter), desc="Matrices / grafos")

    for k, (comb, idx) in enumerate(pbar):
        if isinstance(instance, dict):
            inst_dict = dict(instance)
        else:
            inst_dict = dict(instance.__dict__)

        inst_dict["Orden"] = np.asarray(comb)
        inst_dict["ProcessingTime"] = np.array([[[1]]*size for i in range(size)])
        inst_dict["EnergyConsumption"] = np.array([[[1]]*size for i in range(size)])
        
        ####SOLUTIONS####
        problem = dict_to_problem(inst_dict)

        timeout_ms = dynamic_timeout(problem)
        solver = SOLVER(problem, solver="cp-sat")
        sol = solver.solve(timeout=timeout_ms, verbose=False)

        stats = sol.get("statistics", {}) if isinstance(sol, dict) else {}
        stats.update({
            "perm_idx": k,
            "perm": idx,
            "status": sol.get("status"),
            "objective": sol.get("objective", np.nan),
            # "solveTime": stats.get("solveTime", np.nan),
            # "flatTime": stats.get("flatTime", np.nan),
            # "time": stats.get("time", np.nan),
            "timeout_ms": timeout_ms,
        })
        
        solutions_rows.append(stats)
        
        ####GRAFOS####
        gb = GraphBuilderStrict(inst_dict)
        graph = gb.data
        n_nodes = graph["node"].x.shape[0]
        n_conj = graph[("node", "conjunctive", "node")].edge_index.shape[1]
        n_disj = graph[("node", "disjunctive", "node")].edge_index.shape[1]
    
        btw_map[k]["job"] = np.repeat(range(size), size)
        btw_map[k]["machine"] = inst_dict["Orden"].flatten()
    
        out[k], btw  = extract_features(graph, k) 
        out[k]["perm"] = idx       
        out[k]["perm_idx"] = k    

        multiplicities = sorted(Counter(idx).values(), reverse=True)

        sig = signatures[tuple(multiplicities)]
        
        out[k]["signature"] = sig
        
        btw_matrix = btw.reshape(4, 4)
        
        sym = matrix_symmetry_level(btw_matrix)
        
        out[k]["symmetry_level"] = sym
        
        btw_mean = out[k]["betweenness_mean"]
        
        btw_map[k]["btw"] = btw
        btw_map[k]["perm"] = idx
        btw_map[k]["perm_idx"] = k 
        btw_map[k]["betweenness_mean"] = btw_mean
        btw_map[k]["signature"] = sig
        btw_map[k]["symmetry_level"] = sym
                
        if (btw_mean not in unique_map): unique_map[btw_mean] = btw_map[k]
        
        # pbar.set_postfix(nodes=n_nodes, conj=n_conj, disj=n_disj)
        pbar.set_postfix(instance=k, idx=idx)
    
    ####GRAFOS####
    df = pd.DataFrame(out)
    df.to_csv(f"{fname}.csv", index=False)
    
    df_btw_map = pd.DataFrame(btw_map).explode(["job", "machine", "btw"], ignore_index=True)
    df_btw_map.to_csv(f"{fname}_btw_map.csv", index=False)
    
    df_unique_map = pd.DataFrame.from_dict(unique_map, orient='index').explode(["job", "machine", "btw"], ignore_index=True)
    df_unique_map.to_csv(f"{fname}_unique_map.csv", index=False)
    
    if not os.path.exists(f"{fname}_solutions.csv"):
        # Guardamos las solutions
        df_solutions = pd.DataFrame(solutions_rows)
        df_solutions.to_csv(f"{fname}_solutions.csv", index=False)
    else:
        # Leemos las solutions
        df_solutions = pd.read_csv(f"{fname}_solutions.csv", index_col=None)
    
    
    df_solutions = df_solutions.drop(columns=["perm", "perm_idx"])
    
    # Guardamos todos los datos juntos
    df_all = df.join(df_solutions)
    df_all.to_csv(f"{size}x{size}_all.csv", index=None)
    
    
    ####REPORT####
    lines.append(f"Number of combinations performed: {comb_iter.__len__()}")    
    lines.append(f"Generated file: {fname}")
    lines.append(f"\nLISTA DE POSIBLE PERMUTACIONES PARA TAMAÑO {size}\n")
    
    for i, p in enumerate(permutations(range(size), size)):
        lines.append(f"{i:>3}.- {p}") 
    lines.append(f"\n")
    
    execution_time = time.time() - start_time
    
    hours, rest = divmod(execution_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    lines.append(f"Tiempo de ejecución: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    with open(report_name, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main(size=5)