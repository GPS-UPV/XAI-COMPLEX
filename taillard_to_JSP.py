from pathlib import Path

import numpy as np
import torch
from IGJSP.generador import JSP
from torch_geometric.data import HeteroData
from tqdm import tqdm



def load_taillard_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)

    if not lines:
        raise ValueError("El fichero está vacío o solo contiene comentarios.")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError("La primera línea útil debe contener exactamente dos enteros.")

    num_jobs, num_mchs = map(int, header)
    if len(lines) < num_jobs + 1:
        raise ValueError(f"Se esperaban {num_jobs} líneas de jobs, pero solo hay {len(lines) - 1}.")

    orden = np.empty((num_jobs, num_mchs), dtype=np.int64)
    proc_by_order = np.empty((num_jobs, num_mchs), dtype=np.int64)

    for j in range(num_jobs):
        vals = np.fromstring(lines[j + 1], sep=" ", dtype=np.int64)
        if vals.size != 2 * num_mchs:
            raise ValueError(f"La línea del job {j} tiene {vals.size} elementos, pero se esperaban {2 * num_mchs}.")
        machines = vals[0::2]
        times = vals[1::2]
        if machines.min() >= 1 and machines.max() <= num_mchs:
            machines = machines - 1
        if machines.min() < 0 or machines.max() >= num_mchs:
            raise ValueError(f"El job {j} contiene máquinas fuera de rango.")
        if np.unique(machines).size != num_mchs:
            raise ValueError(f"El job {j} no contiene una permutación válida de máquinas.")
        orden[j] = machines
        proc_by_order[j] = times

    energy_by_order = proc_by_order.copy()
    processing_time = np.zeros((num_jobs, num_mchs, 1), dtype=np.int64)
    energy_consumption = np.zeros((num_jobs, num_mchs, 1), dtype=np.int64)

    rows = np.arange(num_jobs)[:, None]
    processing_time[rows, orden, 0] = proc_by_order
    energy_consumption[rows, orden, 0] = energy_by_order

    jsp = JSP(
        jobs=num_jobs,
        machines=num_mchs,
        ProcessingTime=processing_time,
        EnergyConsumption=energy_consumption,
        ReleaseDateDueDate=np.array([]),
        Orden=orden,
    )

    return jsp, {
        "numJobs": num_jobs,
        "numMchs": num_mchs,
        "Orden": orden,
        "proc_by_order": proc_by_order,
        "energy_by_order": energy_by_order,
    }


def build_graph(parsed, inst_meta=None):
    num_jobs = int(parsed["numJobs"])
    num_mchs = int(parsed["numMchs"])
    orden = np.asarray(parsed["Orden"], dtype=np.int64)
    proc_by_order = np.asarray(parsed["proc_by_order"], dtype=np.float32)
    energy_by_order = np.asarray(parsed["energy_by_order"], dtype=np.float32)
    num_ops = orden.shape[1]
    n_ops_total = num_jobs * num_ops

    op_nodes = np.arange(1, n_ops_total + 1, dtype=np.int64).reshape(num_jobs, num_ops)
    source_idx = 0
    sink_idx = n_ops_total + 1

    node_x = np.empty((n_ops_total + 2, 2), dtype=np.float32)
    node_x[0] = (0.0, -1.0)
    node_x[-1] = (0.0, -1.0)
    node_x[1:-1, 0] = proc_by_order.reshape(-1)
    node_x[1:-1, 1] = orden.reshape(-1)

    conj_edges = np.empty((2, num_jobs * (num_ops + 1)), dtype=np.int64)
    idx = 0
    for j in range(num_jobs):
        nodes = op_nodes[j]
        conj_edges[0, idx] = source_idx
        conj_edges[1, idx] = nodes[0]
        idx += 1
        if num_ops > 1:
            span = num_ops - 1
            conj_edges[0, idx:idx + span] = nodes[:-1]
            conj_edges[1, idx:idx + span] = nodes[1:]
            idx += span
        conj_edges[0, idx] = nodes[-1]
        conj_edges[1, idx] = sink_idx
        idx += 1

    machine_to_nodes = [[] for _ in range(num_mchs)]
    flat_nodes = op_nodes.reshape(-1)
    flat_machines = orden.reshape(-1)
    for nid, mid in zip(flat_nodes, flat_machines):
        machine_to_nodes[int(mid)].append(int(nid))

    dis_src = []
    dis_dst = []
    for nodes in machine_to_nodes:
        k = len(nodes)
        if k < 2:
            continue
        for i in range(k):
            src = nodes[i]
            dis_src.extend([src] * (k - 1))
            dis_dst.extend(nodes[:i] + nodes[i + 1:])

    data = HeteroData()
    data["node"].x = torch.from_numpy(node_x)
    data[("node", "conjunctive", "node")].edge_index = torch.from_numpy(conj_edges)
    if dis_src:
        data[("node", "disjunctive", "node")].edge_index = torch.tensor([dis_src, dis_dst], dtype=torch.long)
    else:
        data[("node", "disjunctive", "node")].edge_index = torch.empty((2, 0), dtype=torch.long)

    job_ids = np.repeat(np.arange(num_jobs, dtype=np.int64), num_ops)
    op_ids = np.tile(np.arange(num_ops, dtype=np.int64), num_jobs)
    mach_ids = orden.reshape(-1)
    proc_flat = proc_by_order.reshape(-1, 1)
    energy_flat = energy_by_order.reshape(-1, 1)

    data.n_jobs = num_jobs
    data.n_machs = num_mchs
    data.P = torch.from_numpy(proc_flat)
    data.E = torch.from_numpy(energy_flat)
    data.job = torch.from_numpy(job_ids)
    data.machine = torch.from_numpy(mach_ids)
    data.op = torch.from_numpy(op_ids)
    data.op_node_index = torch.from_numpy(flat_nodes)
    data.source_idx = source_idx
    data.sink_idx = sink_idx

    allowed = np.zeros((num_jobs, num_mchs), dtype=np.int64)
    allowed[np.arange(num_jobs)[:, None], orden] = 1
    data.allowed_machines = torch.from_numpy(allowed)

    op_mask = torch.zeros(n_ops_total + 2, dtype=torch.bool)
    op_mask[data.op_node_index] = True
    data["node"].op_mask = op_mask

    if inst_meta is not None:
        raw_features = None
        for k in ["speed","rddd","max_makespan","min_makespan","max_min_makespan","max_energy","min_energy","max_min_energy"]:
            if k in inst_meta and inst_meta[k] is not None:
                v = inst_meta[k]
                if isinstance(v, np.generic):
                    v = v.item()
                setattr(data, k, v)
        if isinstance(inst_meta.get("features"), dict):
            raw_features = inst_meta["features"]
            data.gen_features = raw_features
            mapping = {
                "max_makespan": "makespan_max",
                "min_makespan": "makespan_min",
                "max_sum_energy": "energy_sum_max",
                "min_sum_energy": "energy_sum_min",
                "max_processing_time_value": "p_value_max",
                "min_processing_time_value": "p_value_min",
                "mean_processing_time_value": "p_value_mean",
                "max_energy_value": "e_value_max",
                "min_energy_value": "e_value_min",
                "mean_energy_value": "e_value_mean",
                "min_window": "window_min",
                "mean_window": "window_mean",
                "max_window": "window_max",
                "overlap": "overlap",
                "max_tardiness": "max_tardiness",
            }
            for src, dst in mapping.items():
                if src in raw_features and raw_features[src] is not None:
                    setattr(data, dst, raw_features[src])

    return data



def main(input_dir=Path("./TaillardInstances"), json_dir=Path("./TaillardInstancesJSON"), graph_dir=Path("./TaillardInstancesGRAPHS")):
    json_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.is_file())

    for path in tqdm(files, desc="Taillard", unit="file"):
        jsp, parsed = load_taillard_file(path)
        jsp.generate_maxmin_objective_values()
        jsp.saveJsonFile(json_dir / f"{path.stem}.json")
        graph = build_graph(parsed, jsp.__dict__)
        torch.save(graph, graph_dir / f"{path.stem}.pt")


if __name__ == "__main__":
    main()
