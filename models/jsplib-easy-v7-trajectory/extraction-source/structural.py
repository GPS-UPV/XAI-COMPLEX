"""Fixed-route JSP descriptors independent of machine and job identifiers."""
import numpy as np

from .io import load_instance


def structural_features(instance, *, speed_index=None):
    inst = load_instance(instance)
    if inst["instance_type"] != "JSP":
        raise ValueError("Structural JSP features do not model flexible machine alternatives")
    if speed_index is None:
        if inst["speed"] != 1:
            raise ValueError("Select speed_index explicitly for multiple speeds")
        speed_index = 0
    if type(speed_index) is not int or not 0 <= speed_index < inst["speed"]:
        raise ValueError("Invalid speed_index")
    n, m = len(inst["jobs"]), inst["n_machines"]
    if any(sorted(o['machine'] for o in j['operations']) != list(range(m)) for j in inst['jobs']):
        raise ValueError("Structural v1 requires permutation JSP routes")
    if any(k in record for job in inst['jobs'] for record in [job, *job['operations']] for k in ('release-date', 'due-date')):
        raise ValueError("Structural v1 reference does not model release or due dates")
    route = np.array([[o['machine'] for o in j['operations']] for j in inst['jobs']])
    p = np.array([[o['speeds'][speed_index]['processing_time'] for o in j['operations']] for j in inst['jobs']], dtype=float)
    machine_p = np.zeros((n, m)); ranks = np.zeros((n, m))
    machine_p[np.arange(n)[:, None], route] = p
    ranks[np.arange(n)[:, None], route] = np.arange(m)
    loads, jobloads = machine_p.sum(axis=0), p.sum(axis=1)
    lb = max(float(loads.max()), float(jobloads.max()), 1e-12)
    features = {'n_jobs': float(n), 'n_machines': float(m), 'n_operations': float(n*m),
                'jobs_per_machine': n/m, 'log_operations': float(np.log1p(n*m)),
                'load_lower_bound': lb, 'machine_job_bound_ratio': float(loads.max()/max(jobloads.max(),1e-12))}
    def describe(name, values):
        values = np.asarray(values, dtype=float).ravel()
        mean, std = float(values.mean()), float(values.std())
        for key, value in zip(('min','q10','q25','median','q75','q90','max'), np.quantile(values,[0,.1,.25,.5,.75,.9,1])):
            features[f'{name}_{key}'] = float(value)
        features[f'{name}_cv'] = std/max(abs(mean),1e-12)
        features[f'{name}_mean'] = mean
        features[f'{name}_skew'] = float(np.mean(((values-mean)/max(std,1e-12))**3))
    describe('duration',p)
    describe('job_load_relative',jobloads/lb)
    describe('machine_load_relative',loads/lb)
    describe('job_duration_cv',p.std(axis=1)/np.maximum(p.mean(axis=1),1e-12))
    describe('machine_duration_cv',machine_p.std(axis=0)/np.maximum(machine_p.mean(axis=0),1e-12))
    describe('position_relative',ranks.mean(axis=0)/max(m-1,1))
    describe('position_spread',ranks.std(axis=0)/max(m-1,1))
    features['zero_fraction'] = float(np.mean(p==0))
    features['duration_unique_fraction'] = len(np.unique(p))/(n*m)
    features['duration_integer_fraction'] = float(np.mean(p==np.floor(p)))
    for stage, chunk in enumerate(np.array_split(np.arange(m), min(m,4))):
        part = np.zeros((n,m))
        part[np.arange(n)[:,None],route[:,chunk]] = p[:,chunk]
        describe(f'stage_{stage}_load',part.sum(axis=0)/lb)
    # La orientación entre pares detecta rutas comunes sin usar el número de máquina.
    if m > 1:
        agreement = np.mean(ranks[:,:,None] < ranks[:,None,:],axis=0)
        describe('route_pair_agreement',np.abs(agreement[np.triu_indices(m,1)]-.5)*2)
    else:
        describe('route_pair_agreement',[1.])
    total = max(float(p.sum()),1e-12)
    features['load_utilization_bound'] = total/(m*lb)
    singular = np.linalg.svd(machine_p, compute_uv=False)
    features['duration_rank_concentration'] = float(singular[0]**2/max(np.sum(singular**2),1e-12))
    return features
