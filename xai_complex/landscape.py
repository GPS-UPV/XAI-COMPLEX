"""Route geometry and critical blocks of a deterministic constructive schedule."""
import numpy as np
from .io import load_instance
from .search_features import search_features,cp_sat_probe


def landscape_features(instance,*,speed_index=None,probe_budget=None):
    inst=load_instance(instance);features=search_features(inst,speed_index=speed_index,probe_budget=.2 if probe_budget else None)
    level=0 if speed_index is None else speed_index
    route=np.array([[o['machine'] for o in j['operations']] for j in inst['jobs']])
    p=np.array([[o['speeds'][level]['processing_time'] for o in j['operations']] for j in inst['jobs']],dtype=float)
    n,m=p.shape;rank=np.empty_like(route);rank[np.arange(n)[:,None],route]=np.arange(m)
    precedence=np.mean(rank[:,:,None]<rank[:,None,:],axis=0)
    pairs=precedence[np.triu_indices(m,1)]
    features['geometry_fixed_pair_fraction']=float(np.mean((pairs==0)|(pairs==1))) if len(pairs) else 1.
    matrix=precedence-precedence.T
    singular=np.linalg.svd(matrix,compute_uv=False);energy=singular**2;total=max(float(energy.sum()),1e-12)
    features['geometry_order_concentration']=float(energy[:2].sum()/total)
    normalized=energy/total
    features['geometry_order_entropy']=float(-np.sum(normalized*np.log(np.maximum(normalized,1e-15))))
    counts=np.zeros((m,m))
    for row in route:
        for a,b in zip(row,row[1:]):counts[a,b]+=1
    values=counts.ravel()/max(float(counts.sum()),1.)
    features['geometry_transition_entropy']=float(-np.sum(values*np.log(np.maximum(values,1e-15))))/max(float(np.log(max(m*(m-1),2))),1.)
    features['geometry_transition_coverage']=float(np.count_nonzero(counts)/max(m*(m-1),1))
    for fraction in (.25,.5,.75):
        length=max(1,int(round(m*fraction)));indicator=(rank<length).astype(float)
        overlap=(indicator@indicator.T)[np.triu_indices(n,1)]/length
        for name,stat in (('mean',np.mean),('std',np.std),('max',np.max)):
            features[f'geometry_prefix_{fraction}_{name}']=float(stat(overlap)) if len(overlap) else 1.
    canonical=sorted(range(n),key=lambda j:tuple(p[j]));tie=np.empty(n,dtype=int);tie[canonical]=np.arange(n)
    step=np.zeros(n,dtype=int);ready=np.zeros(n);machine_ready=np.zeros(m);machine_nodes=[[] for _ in range(m)]
    order=[];starts=np.zeros(n*m);ends=np.zeros(n*m)
    remaining=np.flip(np.cumsum(np.flip(p,axis=1),axis=1),axis=1)
    for _ in range(n*m):
        active=np.flatnonzero(step<m);slots=step[active];begin=np.maximum(ready[active],machine_ready[route[active,slots]])
        idx=int(np.lexsort((tie[active],-remaining[active,slots],begin))[0]);j=int(active[idx]);s=int(step[j]);node=j*m+s;machine=int(route[j,s])
        starts[node]=begin[idx];ends[node]=begin[idx]+p[j,s];ready[j]=ends[node];machine_ready[machine]=ends[node]
        machine_nodes[machine].append(node);order.append(node);step[j]+=1
    successors=[[] for _ in order]
    for j in range(n):
        for s in range(m-1):successors[j*m+s].append(j*m+s+1)
    for nodes in machine_nodes:
        for a,b in zip(nodes,nodes[1:]):successors[a].append(b)
    after=np.zeros(n*m)
    durations=p.ravel()
    for node in reversed(order):
        after[node]=max((durations[v]+after[v] for v in successors[node]),default=0.)
    makespan=float(ready.max());critical=np.abs(makespan-ends-after)<=1e-8
    blocks=[];machines_critical=0
    for nodes in machine_nodes:
        if any(critical[v] for v in nodes):machines_critical+=1
        run=[]
        for node in nodes:
            if critical[node] and (not run or abs(starts[node]-ends[run[-1]])<=1e-8):run.append(node)
            else:
                if len(run)>1:blocks.append(run)
                run=[node] if critical[node] else []
        if len(run)>1:blocks.append(run)
    lengths=[len(b) for b in blocks]
    features.update({'critical_operation_fraction':float(critical.mean()),'critical_machine_fraction':machines_critical/m,
                     'critical_duration_fraction':float(durations[critical].sum()/max(durations.sum(),1e-12)),
                     'critical_blocks_per_operation':len(blocks)/(n*m),'critical_block_max_relative':max(lengths,default=0)/max(n,1),
                     'critical_block_mean':float(np.mean(lengths)) if lengths else 0.,'critical_block_std':float(np.std(lengths)) if lengths else 0.,
                     'critical_slack_mean':float(np.mean(np.maximum(0,makespan-ends-after))/max(makespan,1e-12))})
    if probe_budget:
        short={k:v for k,v in features.items() if k.startswith('probe_')}
        lb=features['load_lower_bound']*features['schrage_lb_relative']
        ub=lb*(1+features['dispatch_best_gap'])
        late=short if short['probe_proved_optimal'] else cp_sat_probe(route,p,lb,ub,probe_budget)
        features.update({f'late_{k}':v for k,v in late.items()})
        features['trajectory_gap_reduction']=short['probe_gap']-late['probe_gap']
        features['trajectory_bound_gain']=late['probe_bound_relative']-short['probe_bound_relative']
        features['trajectory_improvement_gain']=late['probe_improvement_relative']-short['probe_improvement_relative']
        features['trajectory_proved_later']=late['probe_proved_optimal']-short['probe_proved_optimal']
    return features
