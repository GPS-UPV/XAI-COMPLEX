"""Bound strength from fixed-budget nonpreemptive bottleneck relaxations."""
import numpy as np
from .io import load_instance
from .search_features import search_features


def machine_relaxation(releases, durations, tails, budget=.02):
    """A proven lower bound on 1|r_i,q_i|Cmax, never an incumbent as JSP bound."""
    from ortools.sat.python import cp_model
    r,p,q = (np.asarray(v,dtype=float) for v in (releases,durations,tails))
    if not np.isfinite(budget) or budget<=0:
        raise ValueError('Relaxation budget must be finite and positive')
    if not (r.shape==p.shape==q.shape) or not len(p):
        raise ValueError('Expected aligned nonempty machine operations')
    if any(not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all() for v in (r,p,q)):
        raise ValueError('Relaxations require nonnegative integer values')
    horizon=int(r.max()+p.sum()+q.max())
    model=cp_model.CpModel();intervals=[];completion=[]
    for i,(release,duration,tail) in enumerate(zip(r,p,q)):
        start=model.new_int_var(int(release),horizon-int(duration)-int(tail),f's{i}')
        end=model.new_int_var(int(release+duration),horizon-int(tail),f'e{i}')
        intervals.append(model.new_interval_var(start,int(duration),end,f'i{i}'))
        completion.append(end+int(tail))
    model.add_no_overlap(intervals)
    makespan=model.new_int_var(0,horizon,'makespan')
    model.add_max_equality(makespan,completion);model.minimize(makespan)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.random_seed=42;solver.parameters.max_deterministic_time=float(budget)
    status=solver.solve(model)
    if status in (cp_model.INFEASIBLE,cp_model.MODEL_INVALID):
        raise ValueError('Invalid machine relaxation')
    response=solver.response_proto
    return {'lower_bound':max(0.,solver.best_objective_bound),
            'proved':float(status==cp_model.OPTIMAL),
            'effort':float(response.deterministic_time),
            'conflicts':float(solver.num_conflicts),
            'branches':float(solver.num_branches)}


def relaxation_features(instance, *, speed_index=None, base_features=None, budget=.02):
    inst=load_instance(instance)
    features=dict(base_features) if base_features is not None else search_features(inst,speed_index=speed_index)
    level=0 if speed_index is None else speed_index
    route=np.array([[o['machine'] for o in j['operations']] for j in inst['jobs']])
    p=np.array([[o['speeds'][level]['processing_time'] for o in j['operations']] for j in inst['jobs']],dtype=float)
    head=np.cumsum(p,axis=1)-p;tail=p.sum(axis=1)[:,None]-head-p
    results=[machine_relaxation(head[route==m],p[route==m],tail[route==m],budget) for m in range(inst['n_machines'])]
    lower=features['load_lower_bound']*features['schrage_lb_relative']
    upper=lower*(1+features['dispatch_best_gap']);scale=max(lower,1.)
    bounds=np.array([r['lower_bound'] for r in results]);tight=max(lower,float(bounds.max()))
    features['bottleneck_bound_gain']=(tight-lower)/scale
    features['bottleneck_proved_fraction']=float(np.mean([r['proved'] for r in results]))
    features['bottleneck_dispatch_gap']=(upper-tight)/max(upper,1.)
    features['bottleneck_near_bound_fraction']=float(np.mean(bounds>=.95*tight))
    for key in ('effort','conflicts','branches'):
        values=np.log1p([r[key] for r in results])
        features[f'bottleneck_{key}_mean']=float(values.mean())
        features[f'bottleneck_{key}_max']=float(values.max())
    for quantile in (.25,.5,.75):
        features[f'bottleneck_bound_q{quantile}']=float(np.quantile(bounds,quantile))/scale
    return features
