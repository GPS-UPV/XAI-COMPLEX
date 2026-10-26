"""Fixed-budget optimality-proof probes; historical solver times are never inputs."""
import numpy as np
from .io import load_instance
from .landscape import landscape_features


def proof_probe(route, durations, lower_bound, upper_bound, budget, *, strong=False):
    from ortools.sat.python import cp_model
    p = np.asarray(durations, dtype=float)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Proof budget must be finite and positive')
    if not np.isfinite(p).all() or (p < 0).any() or not np.equal(p, np.floor(p)).all():
        raise ValueError('Proof probes require nonnegative integer durations')
    n, m = p.shape
    model = cp_model.CpModel()
    horizon = int(round(upper_bound))
    intervals = [[] for _ in range(m)]
    ends = []
    for j in range(n):
        last = None
        for s in range(m):
            head = int(p[j, :s].sum())
            tail = int(p[j, s + 1:].sum())
            duration = int(p[j, s])
            start = model.new_int_var(head, horizon - tail - duration, f's{j}_{s}')
            end = model.new_int_var(head + duration, horizon - tail, f'e{j}_{s}')
            intervals[int(route[j, s])].append(model.new_interval_var(start, duration, end, f'i{j}_{s}'))
            if last is not None:
                model.add(start >= last)
            last = end
        ends.append(last)
    for machine in intervals:
        model.add_no_overlap(machine)
    makespan = model.new_int_var(int(np.ceil(lower_bound - 1e-9)), horizon, 'makespan')
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = float(budget)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    solver.parameters.use_strong_propagation_in_disjunctive = strong
    events = []

    class Trace(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            events.append((self.deterministic_time, self.objective_value, self.best_objective_bound))

    status = solver.solve(model, Trace())
    if status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise ValueError(f'Invalid proof probe: {solver.status_name(status)}')
    response = solver.response_proto
    elapsed = float(response.deterministic_time)
    found = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    incumbent = float(solver.objective_value) if found else float(upper_bound)
    bound = max(float(lower_bound), float(solver.best_objective_bound))
    scale = max(float(upper_bound), 1.)
    integral = 0.
    previous_time, previous_objective = 0., float(upper_bound)
    for t, obj, _ in events:
        integral += max(0., t - previous_time) * (previous_objective - lower_bound) / scale
        previous_time, previous_objective = t, obj
    integral += max(0., elapsed - previous_time) * (previous_objective - lower_bound) / scale
    # El esfuerzo está censurado al presupuesto; no equivale al tiempo óptimo histórico.
    result = {
        'proved': float(status == cp_model.OPTIMAL), 'found': float(found),
        'effort_log': float(np.log1p(elapsed)), 'budget_fraction': min(elapsed / budget, 1.),
        'gap': max(0., (incumbent - bound) / max(incumbent, 1.)),
        'bound_gain': (bound - lower_bound) / max(lower_bound, 1.),
        'incumbent_gain': (upper_bound - incumbent) / scale,
        'solutions_log': float(np.log1p(len(events))),
        'first_solution_effort_log': float(np.log1p(events[0][0] if events else budget)),
        'last_improvement_effort_log': float(np.log1p(events[-1][0] if events else budget)),
        'primal_integral': float(integral / budget),
        'conflicts_log': float(np.log1p(solver.num_conflicts)),
        'branches_log': float(np.log1p(solver.num_branches)),
        'propagations_log': float(np.log1p(response.num_binary_propagations + response.num_integer_propagations)),
        'lp_iterations_log': float(np.log1p(response.num_lp_iterations)),
        'boolean_variables_log': float(np.log1p(response.num_booleans)),
    }
    for checkpoint in (.01, .1, 1., 10.):
        eligible = [(obj, lb) for t, obj, lb in events if t <= checkpoint]
        result[f'incumbent_gain_at_{checkpoint}'] = (upper_bound - min((obj for obj, _ in eligible), default=upper_bound)) / scale
    return result


def proof_features(instance, *, speed_index=None, probe_budget=10., base_features=None):
    inst = load_instance(instance)
    # Solo se reutilizan cálculos previos de la misma instancia en los experimentos.
    features = dict(base_features) if base_features is not None else landscape_features(inst, speed_index=speed_index, probe_budget=3.)
    route = np.array([[o['machine'] for o in job['operations']] for job in inst['jobs']])
    level = 0 if speed_index is None else speed_index
    p = np.array([[o['speeds'][level]['processing_time'] for o in job['operations']] for job in inst['jobs']], dtype=float)
    lower = features['load_lower_bound'] * features['schrage_lb_relative']
    upper = lower * (1 + features['dispatch_best_gap'])
    for strong, prefix, budget in ((False, 'proof_', probe_budget), (True, 'disjunctive_', min(probe_budget, 1.))):
        result = proof_probe(route, p, lower, upper, budget, strong=strong)
        features.update({prefix + k: v for k, v in result.items()})
    return features
