"""Scheduling bounds, constructive probes and optional bounded CP-SAT search."""
import heapq
import numpy as np
from .io import load_instance
from .structural import structural_features


def preemptive_bound(releases, durations, tails):
    """Schrage preemptive lower bound for one machine with delivery tails."""
    events=sorted(zip(releases,durations,tails))
    queue=[]; time=0.; bound=0.; i=0; remaining=0.; tail=0.
    while i<len(events) or queue or remaining>0:
        if not queue and remaining<=0 and i<len(events):
            time=max(time,float(events[i][0]))
        while i<len(events) and events[i][0]<=time:
            r,p,q=events[i]; heapq.heappush(queue,(-float(q),float(p))); i+=1
        if remaining>0:
            heapq.heappush(queue,(-tail,remaining))
        if not queue:
            continue
        minus_tail,remaining=heapq.heappop(queue); tail=-minus_tail
        next_event=float(events[i][0]) if i<len(events) else float('inf')
        run=min(remaining,max(0.,next_event-time))
        time+=run; remaining-=run
        if remaining<=1e-12:
            remaining=0.; bound=max(bound,time+tail)
    return bound


def dispatch_makespan(route,p,rule):
    n,m=p.shape
    positions=np.zeros(n,dtype=int); ready=np.zeros(n); machines=np.zeros(m)
    remaining=np.flip(np.cumsum(np.flip(p,axis=1),axis=1),axis=1)
    canonical=sorted(range(n),key=lambda j:tuple(p[j]))
    rank=np.empty(n,dtype=int); rank[canonical]=np.arange(n)
    for _ in range(n*m):
        active=np.flatnonzero(positions<m)
        slots=positions[active]; durations=p[active,slots]
        starts=np.maximum(ready[active],machines[route[active,slots]])
        priority=durations if rule=='spt' else -durations if rule=='lpt' else -remaining[active,slots] if rule=='mwkr' else remaining[active,slots]
        selected=int(np.lexsort((rank[active],priority,starts))[0]); j=int(active[selected]); step=positions[j]
        end=starts[selected]+p[j,step]
        ready[j]=end; machines[route[j,step]]=end; positions[j]+=1
    return float(ready.max())


def search_features(instance, *, speed_index=None, probe_budget=None):
    inst=load_instance(instance)
    features=structural_features(inst,speed_index=speed_index)
    level=0 if speed_index is None else speed_index
    route=np.array([[o['machine'] for o in j['operations']] for j in inst['jobs']])
    p=np.array([[o['speeds'][level]['processing_time'] for o in j['operations']] for j in inst['jobs']],dtype=float)
    n,m=p.shape; lb=features['load_lower_bound']; safe=lambda v:max(float(v),1e-12)
    head=np.cumsum(p,axis=1)-p; tail=p.sum(axis=1)[:,None]-head-p
    bounds=[]; head_dispersion=[]; tail_dispersion=[]
    for machine in range(m):
        mask=route==machine
        bounds.append(preemptive_bound(head[mask],p[mask],tail[mask]))
        head_dispersion.append(float(head[mask].std())/lb); tail_dispersion.append(float(tail[mask].std())/lb)
    tight=max(lb,max(bounds))
    if not p.any():
        tight=0.
    features.update({'schrage_lb_relative':tight/lb,'machine_bound_mean':float(np.mean(bounds))/lb,
                     'machine_bound_std':float(np.std(bounds))/lb,'head_dispersion_mean':float(np.mean(head_dispersion)),
                     'tail_dispersion_mean':float(np.mean(tail_dispersion)),
                     'critical_job_fraction':float(np.mean(p.sum(axis=1)>=.9*lb))})
    makespans=[]
    for rule in ('spt','lpt','mwkr','lwkr'):
        value=dispatch_makespan(route,p,rule); makespans.append(value)
        features[f'dispatch_{rule}_gap']=(value-tight)/safe(tight)
    best=min(makespans)
    features['dispatch_spread_relative']=(max(makespans)-best)/safe(best)
    features['dispatch_best_gap']=(best-tight)/safe(tight)
    features['dispatch_mean_gap']=(float(np.mean(makespans))-tight)/safe(tight)
    if probe_budget is not None:
        features.update(cp_sat_probe(route,p,tight,best,probe_budget))
    return features


def cp_sat_probe(route,p,lower_bound,upper_bound,budget):
    from ortools.sat.python import cp_model
    if not np.equal(p,np.floor(p)).all():
        raise ValueError('CP-SAT feature probe requires integer durations; no implicit rounding')
    if not np.isfinite(budget) or budget<=0:
        raise ValueError('probe_budget must be finite and positive')
    n,m=p.shape; model=cp_model.CpModel(); intervals=[[] for _ in range(m)]; ends=[]
    horizon=int(p.sum()); previous=[]
    for j in range(n):
        last=None
        for step in range(m):
            duration=int(p[j,step]); start=model.new_int_var(0,horizon,f's{j}_{step}'); end=model.new_int_var(0,horizon,f'e{j}_{step}')
            interval=model.new_interval_var(start,duration,end,f'i{j}_{step}')
            intervals[int(route[j,step])].append(interval)
            if last is not None:
                model.add(start>=last)
            last=end
        ends.append(last)
    for values in intervals:
        model.add_no_overlap(values)
    makespan=model.new_int_var(0,horizon,'makespan');model.add_max_equality(makespan,ends);model.minimize(makespan)
    solver=cp_model.CpSolver();solver.parameters.max_deterministic_time=float(budget)
    solver.parameters.num_search_workers=1;solver.parameters.random_seed=42
    status=solver.solve(model)
    if status in (cp_model.MODEL_INVALID,cp_model.INFEASIBLE):
        raise ValueError(f'Invalid fixed-route probe result: {solver.status_name(status)}')
    has=status in (cp_model.OPTIMAL,cp_model.FEASIBLE)
    objective=solver.objective_value if has else upper_bound
    bound=max(lower_bound,solver.best_objective_bound)
    response=solver.response_proto
    # No se usan tiempos de pared: dependen de la carga y del paralelismo externo.
    return {'probe_has_solution':float(has),'probe_proved_optimal':float(status==cp_model.OPTIMAL),
            'probe_gap':max(0.,(objective-bound)/max(objective,1e-12)),
            'probe_bound_relative':bound/max(lower_bound,1e-12),
            'probe_improvement_relative':(upper_bound-objective)/max(upper_bound,1e-12),
            'probe_conflicts_log':float(np.log1p(solver.num_conflicts)),
            'probe_branches_log':float(np.log1p(solver.num_branches)),
            'probe_propagations_log':float(np.log1p(response.num_binary_propagations+response.num_integer_propagations)),
            'probe_deterministic_time':float(response.deterministic_time)}
