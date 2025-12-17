import glob
import json
import os
import traceback
from collections import defaultdict
import time
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

# ---------------- utilidades ----------------

def safe_torch_load(path, map_location='cpu'):
    load = torch.load
    last_exc = None
    try:
        return load(path, map_location=map_location)
    except Exception as e:
        last_exc = e
        try:
            return load(path, map_location=map_location, weights_only=False)
        except TypeError:
            raise last_exc
        except Exception as e2:
            raise e2

def _agg_stats(values):
    """Devuelve un dict con min,max,mean,median,std,range,q1,q3,gini"""
    if values is None or len(values) == 0:
        return dict(min=None, max=None, mean=None, median=None, std=None, range=None, q1=None, q3=None, gini=None)
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    gini = _gini(arr)
    return dict(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(float(np.std(arr, ddof=0))) if arr.size>0 else None,
        range=float(np.max(arr) - np.min(arr)),
        q1=q1,
        q3=q3,
        gini=gini
    )

def _gini(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if total == 0.0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=float)
    g = (2.0 * np.dot(idx, x)) / (n * total) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))

def _get_edge_index_from_hdata(hdata, key):
    try:
        part = hdata[key]
        e = getattr(part, 'edge_index', None)
        if e is None:
            return None
        if isinstance(e, torch.Tensor):
            e_np = e.detach().cpu().numpy()
        else:
            e_np = np.array(e)
        if e_np.ndim == 2 and e_np.shape[0] == 2:
            return e_np
        if e_np.ndim == 2 and e_np.shape[1] == 2:
            return e_np.T
    except Exception:
        return None
    return None

def _as_numpy(x, dtype=float):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.array(x)
    return x.astype(dtype, copy=False)

def _get_node_attr(hdata, names):
    """
    Busca un atributo por varios alias en:
      - hdata['node'] (NodeStorage o dict): getattr, dict.get y __getitem__
      - hdata a nivel superior (getattr y __getitem__)
    Devuelve np.array (con tamaño > 0) o None. Los “storages vacíos” => None.
    """
    def _pull(container, nm):
        try:
            v = getattr(container, nm)
            if v is not None:
                return v
        except Exception:
            pass
        try:
            if isinstance(container, dict) and nm in container:
                return container[nm]
        except Exception:
            pass
        try:
            return container[nm]
        except Exception:
            return None

    node = None
    try:
        node = hdata['node']
    except Exception:
        pass
    if node is not None:
        for nm in names:
            v = _pull(node, nm)
            if v is not None:
                arr = _as_numpy(v)
                if arr is not None and arr.size > 0:
                    return arr

    for nm in names:
        v = _pull(hdata, nm)
        if v is not None:
            arr = _as_numpy(v)
            if arr is not None and arr.size > 0:
                return arr
    return None

def _ensure_2d(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _infer_jobs_machines(N, n_jobs, n_machs, job_ids=None, mach_ids=None):
    if job_ids is None or mach_ids is None:
        if n_jobs and n_machs and N == n_jobs * n_machs:
            if job_ids is None:
                job_ids = np.repeat(np.arange(n_jobs), n_machs)
            if mach_ids is None:
                mach_ids = np.tile(np.arange(n_machs), n_jobs)
        else:
            if job_ids is None:
                job_ids = np.zeros(N, dtype=int)
            if mach_ids is None:
                mach_ids = np.full(N, -1, dtype=int)
    return job_ids, mach_ids

def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    return float(np.mean(x))

def _sum_per_group(vals, group_ids, K=None):
    vals = np.asarray(vals, dtype=float).ravel()
    group_ids = np.asarray(group_ids, dtype=int).ravel()
    if K is None:
        K = int(group_ids.max()) + 1 if group_ids.size else 0
    out = np.zeros(K, dtype=float)
    np.add.at(out, group_ids, vals)
    return out

def _probe_thesis_inputs(hdata):
    keys = {
        'P': ['P','p','proc_time','processing_time','duration','dur','proc_times','times'],
        'E': ['E','energy','energy_per_speed','energy_cost','energies'],
        'R': ['R','release','r','release_date','release_time'],
        'D': ['D','due','d','due_date','due_time'],
        'start': ['start','st','s','start_time'],
        'end': ['end','finish','f','end_time'],
        'job': ['job','job_id','j','op_job','job_idx','job_index'],
        'mach': ['machine','mach','m','machine_id','op_machine','machine_idx','machine_index'],
        'setup': ['setup','mtt','setup_time','changeover','setup_matrix'],
        'allowed': ['allowed_machines','avail_machines','avjm','machine_mask','allowed_machs'],
    }
    rep = {}
    for k, aliases in keys.items():
        arr = _get_node_attr(hdata, aliases)
        rep[k] = None if arr is None else list(arr.shape)

    nx_shape = None
    try:
        node = hdata['node']
        x = None
        try:
            x = getattr(node, 'x', None)
        except Exception:
            x = None
        if x is None:
            try:
                x = node['x']
            except Exception:
                x = None
        if x is not None:
            xn = _as_numpy(x)
            if xn is not None:
                nx_shape = list(xn.shape)
    except Exception:
        pass
    rep['node.x'] = nx_shape

    try:
        conj = _get_edge_index_from_hdata(hdata, ('node','conjunctive','node'))
        disj = _get_edge_index_from_hdata(hdata, ('node','disjunctive','node'))
        rep['edges_conj'] = None if conj is None else list(conj.shape)
        rep['edges_disj'] = None if disj is None else list(disj.shape)
    except Exception:
        pass
    return rep

def _getattr_safe(hdata, name):
    try:
        v = getattr(hdata, name)
        return v
    except Exception:
        return None

def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else None
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (int, float)):
        return v
    return None

def _maybe_jsonify(d):
    try:
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        try:
            # conversión aproximada si hay np types
            def conv(x):
                if isinstance(x, (np.generic,)):
                    return x.item()
                return x
            return json.dumps({k: conv(v) for k, v in d.items()}, ensure_ascii=False)
        except Exception:
            return str(d)

def _extract_generator_features(hdata):
    """
    Copia al dict plano las métricas precomputadas por el generador
    (si existen como atributos top-level del HeteroData).
    """
    out = {}

    # metadatos
    n_jobs = _getattr_safe(hdata, 'n_jobs')
    n_machs = _getattr_safe(hdata, 'n_machs')
    speed = _getattr_safe(hdata, 'speed')
    rddd = _getattr_safe(hdata, 'rddd')

    if _to_scalar(n_jobs) is not None: out['n_jobs'] = int(_to_scalar(n_jobs))
    if _to_scalar(n_machs) is not None: out['n_machines'] = int(_to_scalar(n_machs))
    if _to_scalar(speed) is not None: out['speed'] = int(_to_scalar(speed))
    if _to_scalar(rddd) is not None: out['rddd'] = int(_to_scalar(rddd))

    # makespan / energía agregada
    for src, dst in [
        ('makespan_max', 'makespan_max'),
        ('makespan_min', 'makespan_min'),
        ('makespan_range', 'makespan_range'),
        ('energy_sum_max', 'energy_sum_max'),
        ('energy_sum_min', 'energy_sum_min'),
        ('energy_sum_range', 'energy_sum_range'),
    ]:
        val = _getattr_safe(hdata, src)
        s = _to_scalar(val)
        if s is not None:
            out[dst] = s

    # valores por operación (agregados por el generador)
    for src, dst in [
        ('p_value_max', 'p_value_max'),
        ('p_value_min', 'p_value_min'),
        ('p_value_mean', 'p_value_mean'),
        ('e_value_max', 'e_value_max'),
        ('e_value_min', 'e_value_min'),
        ('e_value_mean', 'e_value_mean'),
        ('window_min', 'window_min'),
        ('window_mean', 'window_mean'),
        ('window_max', 'window_max'),
        ('overlap', 'overlap'),
        ('max_tardiness', 'max_tardiness'),
    ]:
        val = _getattr_safe(hdata, src)
        s = _to_scalar(val)
        if s is not None:
            out[dst] = float(s) if isinstance(s, float) or isinstance(s, np.floating) else s

    # copia cruda del dict de features del generador (si existe)
    gen_feats = _getattr_safe(hdata, 'gen_features')
    if isinstance(gen_feats, dict):
        out['gen_features_raw'] = _maybe_jsonify(gen_feats)

    # campos útiles adicionales
    op_cost = _getattr_safe(hdata, 'operation_cost')
    if isinstance(op_cost, torch.Tensor) and op_cost.numel() > 0:
        out['operation_cost_mean'] = float(op_cost.detach().cpu().numpy().mean())
        out['operation_cost_std'] = float(op_cost.detach().cpu().numpy().std())

    return out

def _compute_lb_makespan(P, job_ids, mach_ids, n_machs):
    res = {}
    if P is None:
        return res
    p_mean = np.mean(P, axis=1) if P.ndim == 2 else np.asarray(P, dtype=float).ravel()
    # jobs
    if job_ids is not None and job_ids.size:
        nJ = int(job_ids.max()) + 1
        load_jobs = _sum_per_group(p_mean, job_ids, K=nJ)
        if load_jobs.size:
            res['makespan_lb_job_sum'] = float(np.max(load_jobs))
    # machines
    nM = int(n_machs) if n_machs else None
    if nM is None or nM == 0:
        if mach_ids is not None and mach_ids.size and mach_ids.max() >= 0:
            nM = int(mach_ids.max()) + 1
    if nM and mach_ids is not None and mach_ids.size:
        load_machs = _sum_per_group(p_mean, np.clip(mach_ids, 0, None), K=nM)
        if load_machs.size:
            res['makespan_lb_machine_sum'] = float(np.max(load_machs))
            res['makespan_lb_meanload'] = float(np.sum(p_mean) / nM)
    return res

def _compute_energy_stats(E):
    res = {}
    if E is None:
        return res
    Emin = np.min(E, axis=1)
    Eavg = np.mean(E, axis=1)
    Emax = np.max(E, axis=1)
    res['energy_min_mean'] = float(np.mean(Emin))
    res['energy_avg_mean'] = float(np.mean(Eavg))
    res['energy_max_mean'] = float(np.mean(Emax))
    return res

def _compute_windows_ratio(P, R, D):
    res = {}
    if P is None or R is None or D is None:
        return res
    N = P.shape[0] if P.ndim == 2 else P.size
    if R.size != N: R = np.resize(R, N)
    if D.size != N: D = np.resize(D, N)
    P2 = P if P.ndim == 2 else P.reshape(-1, 1)
    P2 = np.maximum(P2, 1e-12)
    window = np.maximum(D - R, 0.0).reshape(-1, 1)
    ratio = window / P2
    res['time_windows_ratio_mean'] = float(np.mean(ratio))
    res['time_windows_ratio_min']  = float(np.min(ratio))
    res['time_windows_ratio_max']  = float(np.max(ratio))
    return res

# ---------------- extracción principal de features ----------------

def extract_features_from_heterodata(n_jobs_from_name, n_machs_from_name, seed, hdata,
                                     obj=None, use_approx_betweenness=False, betweenness_samples=100):
    """
    Extrae el conjunto de características del HeteroData. Mezcla:
      - features de grafo
      - métricas precomputadas por el generador (top-level)
      - métricas 'ligeras' de tesis solo si faltan y hay datos
    """
    features = {}

    # --- 0) metadatos preferentemente del HeteroData ---
    n_jobs = _getattr_safe(hdata, 'n_jobs')
    n_machs = _getattr_safe(hdata, 'n_machs')
    if _to_scalar(n_jobs) is None: n_jobs = n_jobs_from_name
    else: n_jobs = int(_to_scalar(n_jobs))
    if _to_scalar(n_machs) is None: n_machs = n_machs_from_name
    else: n_machs = int(_to_scalar(n_machs))

    features['n_jobs'] = int(n_jobs) if n_jobs is not None else None
    features['n_machines'] = int(n_machs) if n_machs is not None else None
    features['seed'] = int(seed) if seed is not None else None
    
    #? GRAPH FEATURES
    # --- 1) node.x stats (como antes) ---
    N = None
    try:
        x = hdata['node'].x
    except Exception:
        x = None
    if x is not None:
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
        if x_np.ndim == 1: x_np = x_np.reshape(-1, 1)
        N = int(x_np.shape[0]); F = int(x_np.shape[1]) if x_np.ndim == 2 else 1
        features['num_nodes'] = N
        features['node_x_cols'] = F
        for col in range(F):
            colvals = x_np[:, col].astype(float)
            stats = _agg_stats(colvals)
            for k, v in stats.items():
                features[f'node_x_col{col}_{k}'] = v
        features['node_x_overall_mean'] = float(np.mean(x_np))
        features['node_x_overall_std'] = float(np.std(x_np))
    else:
        features['node_x_cols'] = 0

    # --- 2) aristas conj y disj ---
    conj_idx = _get_edge_index_from_hdata(hdata, ('node','conjunctive','node'))
    disj_idx = _get_edge_index_from_hdata(hdata, ('node','disjunctive','node'))

    features['num_conjunctive_edges'] = int(conj_idx.shape[1]) if conj_idx is not None else 0
    features['num_disjunctive_edges'] = int(disj_idx.shape[1]) if disj_idx is not None else 0

    if N is None:
        max_node = -1
        for idx in [conj_idx, disj_idx]:
            if idx is not None and idx.size>0:
                max_node = max(max_node, int(np.max(idx)))
        N = int(max_node + 1) if max_node >= 0 else 0
        features['num_nodes_inferred'] = N
    features['num_nodes_total'] = N
    features['num_edges_total'] = features['num_conjunctive_edges'] + features['num_disjunctive_edges']

    # --- 3) Grafo no dirigido a partir de disyuntivas ---
    G_d = nx.Graph()
    G_d.add_nodes_from(range(N))
    if disj_idx is not None:
        u = disj_idx[0].astype(int); v = disj_idx[1].astype(int)
        edges = list(zip(u.tolist(), v.tolist()))
        G_d.add_edges_from(edges)

    n_d = G_d.number_of_nodes()
    m_d = G_d.number_of_edges()
    features['disj_graph_density'] = None if n_d <= 1 else 2.0 * m_d / (n_d * (n_d - 1))

    degrees = [d for _, d in G_d.degree()]
    for k, v in _agg_stats(degrees).items():
        features[f'deg_d_{k}'] = v

    if G_d.number_of_nodes() > 0:
        clustering = nx.clustering(G_d)
        for k, v in _agg_stats(list(clustering.values())).items():
            features[f'clustering_{k}'] = v
    else:
        for name in ['min','max','mean','median','std','range','q1','q3','gini']:
            features[f'clustering_{name}'] = None

    # --- 4) Grafo no dirigido a partir de conjuntivas ---
    G_c = nx.Graph()
    G_c.add_nodes_from(range(N))
    if conj_idx is not None:
        u = conj_idx[0].astype(int); v = conj_idx[1].astype(int)
        edges = list(zip(u.tolist(), v.tolist()))
        G_c.add_edges_from(edges)
        
    n_c = G_c.number_of_nodes()
    m_c = G_c.number_of_edges()
    features['conj_graph_density'] = None if n_c <= 1 else 2.0 * m_c / (n_c * (n_c - 1))

    degrees = [d for _, d in G_c.degree()]
    for k, v in _agg_stats(degrees).items():
        features[f'deg_c_{k}'] = v

    try:
        if use_approx_betweenness and G_c.number_of_nodes() > betweenness_samples:
            btw = nx.betweenness_centrality(G_c, normalized=True, k=betweenness_samples)
        else:
            btw = nx.betweenness_centrality(G_c, normalized=True)
        for k, v in _agg_stats(list(btw.values())).items():
            features[f'betweenness_{k}'] = v
    except Exception as e:
        features['betweenness_error'] = str(e)

    # --- 5) tesis christian ---
    genf = _extract_generator_features(hdata)
    features.update(genf)

    # --- 6) solución errores datos  ---
    P = _ensure_2d(_get_node_attr(hdata, ['P','p','proc_time','processing_time','duration','dur','proc_times','times']))
    E = _ensure_2d(_get_node_attr(hdata, ['E','energy','energy_per_speed','energy_cost','energies']))
    R = _get_node_attr(hdata, ['R','release','r','release_date','release_time'])
    D = _get_node_attr(hdata, ['D','due','d','due_date','due_time'])
    job_ids = _get_node_attr(hdata, ['job','job_id','j','op_job','job_idx','job_index'])
    mach_ids = _get_node_attr(hdata, ['machine','mach','m','machine_id','op_machine','machine_idx','machine_index'])

    #TODO: revisar si hace falta
    # LBs makespan si no hay makespan_min/max
    if 'makespan_min' not in features or 'makespan_max' not in features:
        features.update(_compute_lb_makespan(P, job_ids, mach_ids, n_machs))

    # Energía por operación si faltan agregados de energía
    if not any(k in features for k in ['energy_min_mean','energy_avg_mean','energy_max_mean']):
        features.update(_compute_energy_stats(E))

    # Ventanas ratio si el generador no las calculó (o puso -1)
    need_windows = ('window_mean' not in features) or (features.get('window_mean', None) in (-1, None))
    if need_windows:
        features.update(_compute_windows_ratio(P, R, D))

    # Nodos de operación (opcional)
    try:
        op_mask = getattr(hdata['node'], 'op_mask', None)
        if isinstance(op_mask, torch.Tensor):
            features['num_op_nodes'] = int(op_mask.sum().item())
    except Exception:
        # fallback: si P existe, asumimos N_ops = P.shape[0]
        if P is not None:
            features['num_op_nodes'] = int(P.shape[0])

    return features

# ---------------- main: leer carpeta ./graphs ----------------

def main(graphs_folder='./graphs', out_csv=None, verbose=False):
    
    start_time = time.time()
    
    if out_csv is None:
        out_csv = os.path.join(graphs_folder, 'features.csv')
    os.makedirs(graphs_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(graphs_folder, '*.pt')))
    rows = []
    debug = {}

    if len(files) == 0:
        print(f"No se han encontrado archivos .pt en {graphs_folder}")
        return
    pbar = tqdm(files)
    for fpath in pbar:
        fname = os.path.basename(fpath)
        if verbose:
            print(f"Procesando {fname} ...")
        pbar.set_description(f"Procesando {fname}")
        file_record = {'file': fname}
        try:
            obj = safe_torch_load(fpath)
            hdata = None
            if HeteroData is not None and isinstance(obj, HeteroData):
                hdata = obj
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if HeteroData is not None and isinstance(v, HeteroData):
                        hdata = v
                        break
                if hdata is None:
                    if 'data' in obj and (HeteroData is not None and isinstance(obj['data'], HeteroData)):
                        hdata = obj['data']
                    elif 'hetero' in obj and (HeteroData is not None and isinstance(obj['hetero'], HeteroData)):
                        hdata = obj['hetero']
            else:
                if hasattr(obj, 'data') and (HeteroData is not None and isinstance(getattr(obj,'data'), HeteroData)):
                    hdata = getattr(obj, 'data')

            if hdata is None:
                try:
                    if isinstance(obj, dict) and ('node' in obj or ('node','disjunctive','node' in obj)):
                        hdata = obj
                except Exception:
                    hdata = None

            if hdata is None:
                raise ValueError("No se identificó un HeteroData ni una estructura compatible dentro del .pt")

            # n_jobs, n_machs preferentemente del HeteroData; si no, del nombre
            try:
                n_jobs_h = getattr(hdata, 'n_jobs', None)
                n_machs_h = getattr(hdata, 'n_machs', None)
                n_jobs = int(n_jobs_h) if n_jobs_h is not None else None
                n_machs = int(n_machs_h) if n_machs_h is not None else None
            except Exception:
                n_jobs = n_machs = None

            if n_jobs is None or n_machs is None:
                try:
                    n_jobs_f, n_machs_f, seed = [int(x) for x in fname.split(".")[0].split("-")]
                except Exception:
                    # fallback razonable
                    n_jobs_f, n_machs_f, seed = (None, None, 0)
            else:
                try:
                    _parts = fname.split(".")[0].split("-")
                    seed = int(_parts[-1]) if len(_parts) >= 3 else 0
                except Exception:
                    seed = 0
                n_jobs_f, n_machs_f = n_jobs, n_machs

            if verbose:
                probe = _probe_thesis_inputs(hdata)
                print('>> PROBE:', json.dumps(probe, indent=2, default=str))

            feats = extract_features_from_heterodata(n_jobs_f, n_machs_f, seed, hdata, obj=obj)
            file_record.update(feats)
            debug[fname] = feats
            rows.append(file_record)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR procesando {fname}: {e}")
            if verbose:
                print(f"  Traceback {fname}: {tb}")
            debug[fname] =  str(e)
            file_record.update({'error': str(e)})
            rows.append(file_record)

    # DataFrame y guardado
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    debug_path = os.path.join(graphs_folder, 'features.json')
    with open(debug_path, 'w', encoding='utf-8') as fh:
        json.dump(debug, fh, indent=2, ensure_ascii=False)

    print(f"\nTerminado. Guardado {out_csv} (filas: {len(df)}) y debug -> {debug_path}")
    
    execution_time = time.time() - start_time
    
    hours, rest = divmod(execution_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    report_path = os.path.join(graphs_folder, 'graph_features_report.txt')
    with open(report_path, "w", encoding='utf-8') as fr:
        fr.write(f"Tiempo de ejecución: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

if __name__ == '__main__':
    # pon verbose=True si quieres el PROBE
    main(verbose=False)
