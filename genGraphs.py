import os
from collections import defaultdict
from itertools import permutations
from typing import Any, Dict, List, Tuple, Optional

import itertools
import numpy as np
import torch
from IGJSP.generador import Generator  # mantiene el import que ya usas
from torch_geometric.data import HeteroData
from tqdm import tqdm


# operación: (jobId, opId, macId, procTime, energy)
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


def main(jobs, machines, seed=1, ReleaseDateDueDate=0, pbar=None):
    try:
        save_path = "graphs"
        os.makedirs(save_path, exist_ok=True)

        gen = Generator(dzn=True, savepath="./instances/",single_folder_output=True)

        for s in range(seed):
            inst = gen.generate_new_instance(
                jobs=jobs,
                machines=machines,
                ReleaseDateDueDate=ReleaseDateDueDate,
                seed=s
            )

            inst_dict = inst.__dict__ if not isinstance(inst, dict) and hasattr(inst, "__dict__") else inst
            gb = GraphBuilderStrict(inst_dict)

            inst_name = f"{jobs}-{machines}-{s}.pt"
            save_file = os.path.join(save_path, inst_name)
            torch.save(gb.data, save_file)

            if pbar is not None:
                pbar.set_description(
                    f"Guardado {save_file} (jobs={gb.num_jobs}, machs={gb.num_machs}, seed={s})"
                )
            else:
                print(f"Guardado {save_file} (jobs={jobs}, machs={machines}, seed={s})")

    except Exception as e:
        print("Excepción:", repr(e))


if __name__ == "__main__":
    configs = list(itertools.product(range(2, 51), range(2, 51), repeat=1))
    pbar = tqdm(configs)
    for j, m in pbar:
        pbar.set_description(f"Generating instance for j: {j}, m: {m}")
        # Activa R/D si quieres que el generador incluya ventanas de tiempo
        main(j, m, ReleaseDateDueDate=0, pbar=pbar)
