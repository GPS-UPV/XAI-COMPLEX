"""Nested evaluation of proof-effort features on the frozen JSPLIB cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from train_jsplib import score
from cejsp import CATEGORIES
from xai_complex.proof_features import proof_features

ROOT = Path(__file__).resolve().parents[1]


def extract(path, base, budget):
    started = time.perf_counter()
    cpu = time.process_time()
    with threadpool_limits(limits=1):
        row = proof_features(path, probe_budget=budget, base_features=base)
    return row, {'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}


def grid():
    configs = []
    for view in ('all', 'proof', 'effort'):
        for fraction in (.5, 1.):
            for balanced in (False, True):
                configs.append({'kind': 'et', 'view': view, 'fraction': fraction, 'balanced': balanced})
        for depth in (3, 5):
            configs.append({'kind': 'cat', 'view': view, 'depth': depth})
        for c in (1., 10., 100.):
            configs.append({'kind': 'svm', 'view': view, 'C': c})
    configs.append({'kind': 'rf', 'view': 'all'})
    configs.append({'kind': 'hist', 'view': 'all'})
    return configs


def estimator(cfg):
    kind = cfg['kind']
    if kind == 'et':
        model = ExtraTreesClassifier(n_estimators=300, max_features=cfg['fraction'], class_weight='balanced' if cfg['balanced'] else None, random_state=42, n_jobs=1)
    elif kind == 'rf':
        model = RandomForestClassifier(n_estimators=300, max_features=.8, class_weight=None, random_state=42, n_jobs=1)
    elif kind == 'cat':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=450, depth=cfg['depth'], learning_rate=.05, l2_leaf_reg=6, thread_count=1, verbose=False, allow_writing_files=False, random_seed=42)
    elif kind == 'hist':
        model = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=5, random_state=42)
    else:
        model = make_pipeline(StandardScaler(), SVC(C=cfg['C'], probability=True, random_state=42))
    return make_pipeline(SimpleImputer(strategy='median'), model)


def fit_prob(cfg, matrices, y, tr, te):
    model = estimator(cfg)
    model.fit(matrices[cfg['view']][tr], y[tr])
    prob = model.predict_proba(matrices[cfg['view']][te])
    return prob[:, [list(model.classes_).index(c) for c in CATEGORIES]]


def evaluate_config(cfg, matrices, y, ids, folds):
    pred = np.zeros((len(ids), 5))
    with threadpool_limits(limits=1):
        for tr, te in folds:
            pred[te] = fit_prob(cfg, matrices, y, ids[tr], ids[te])
    return {'config': cfg, 'probabilities': pred}


def choose(results, y):
    options = []
    for i, result in enumerate(results):
        p = np.array(CATEGORIES)[np.argmax(result['probabilities'], axis=1)]
        options.append((accuracy_score(y, p), f1_score(y, p, average='macro', zero_division=0), i))
    ranked = sorted(options, reverse=True)
    choices = [{'indices': [i], 'weights': [1.], 'accuracy': float(acc), 'macro_f1': float(f1)} for acc, f1, i in ranked]
    for count in (2, 3, 5):
        indices = [i for _, _, i in ranked[:count]]
        prob = np.mean([results[i]['probabilities'] for i in indices], axis=0)
        p = np.array(CATEGORIES)[np.argmax(prob, axis=1)]
        choices.append({'indices': indices, 'weights': [1. / count] * count, 'accuracy': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, average='macro', zero_division=0))})
    return max(choices, key=lambda r: (r['accuracy'], r['macro_f1']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--budget', type=float, default=10.)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.workers <= 32 or not np.isfinite(args.budget) or args.budget <= 0:
        raise ValueError('Invalid workers or deterministic budget')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'evaluation.json').exists():
        raise ValueError('Choose a new output directory; completed studies are preserved')
    started = time.perf_counter()
    manifest = json.loads((ROOT / 'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    examples = manifest['examples']
    y = np.array([e['label'] for e in examples])
    if len({e['base_instance_id'] for e in examples}) != len(examples):
        raise ValueError('Use grouped folds when repeated base instances exist')
    base_dir = ROOT / 'models/jsplib-easy-v7-trajectory'
    base = json.loads((base_dir / 'features.json').read_text())
    provenance = json.loads((base_dir / 'features-provenance.json').read_text())
    if provenance['instance_hashes'] != [e['sha256'] for e in examples]:
        raise ValueError('Base feature rows do not align with this manifest')
    for name, digest in provenance['source_hashes'].items():
        if sha256((base_dir / 'extraction-source' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Archived base extraction source differs')
    for e in examples:
        if sha256(Path(e['path']).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('Dataset content changed')
    key = {'instance_hashes': provenance['instance_hashes'], 'base_feature_sha256': sha256((base_dir / 'features.json').read_bytes()).hexdigest(), 'proof_source_sha256': sha256((ROOT / 'pypi/src/xai_complex/proof_features.py').read_bytes()).hexdigest(), 'ortools': importlib.metadata.version('ortools'), 'budget': args.budget}
    meta = out / 'features-provenance.json'
    partial = out / 'features-partial.json'
    if meta.exists() and json.loads(meta.read_text()) != key:
        raise ValueError('Partial features differ from the current extraction protocol')
    meta.write_text(json.dumps(key, indent=2))
    full = json.loads(partial.read_text()) if partial.exists() else [None] * len(y)
    timings = json.loads((out / 'extraction-cost.json').read_text()) if (out / 'extraction-cost.json').exists() else {}
    if (out / 'features.json').exists():
        full = json.loads((out / 'features.json').read_text())
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract, e['path'], base[i], args.budget): i for i, e in enumerate(examples) if full[i] is None}
            for future in as_completed(futures):
                i = futures[future]
                full[i], timings[str(i)] = future.result()
                count = sum(r is not None for r in full)
                (out / 'features-partial.pending').write_text(json.dumps(full))
                (out / 'features-partial.pending').replace(partial)
                (out / 'extraction-cost.json').write_text(json.dumps(timings, indent=2))
                print(json.dumps({'phase': 'features', 'complete': count, 'total': len(y), 'wall_seconds': round(time.perf_counter() - started, 1), 'completed_worker_cpu_seconds': round(sum(t['cpu_seconds'] for t in timings.values()), 1)}), flush=True)
        (out / 'features.json').write_text(json.dumps(full))
    columns = sorted(full[0])
    views = {'all': columns,
             'proof': [c for c in columns if c in base[0] and not c.startswith(('geometry_', 'critical_')) or c.startswith(('proof_', 'disjunctive_'))],
             'effort': [c for c in columns if c.startswith(('proof_', 'disjunctive_', 'probe_', 'late_probe_', 'trajectory_')) or c in ('n_jobs', 'n_machines', 'log_operations', 'aspect_ratio', 'n_operations')]}
    matrices = {view: np.array([[r[c] for c in cols] for r in full]) for view, cols in views.items()}
    configs = grid()
    plan = {'goal_accuracy': .9, 'budget': args.budget, 'workers': args.workers, 'solver_workers': 1, 'outer_seed': 20261024, 'configs': configs, 'views': views, 'objective': 'accuracy; macro-F1 breaks ties', 'selection': 'nested 5x3 CV including inner choice of top-2/3/5 mean-vote ensemble', 'warning': 'Previously explored cohort; internal estimate, not external validation'}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    predicted = np.empty(len(y), dtype=object)
    control = np.empty(len(y), dtype=object)
    history = []
    with Parallel(n_jobs=args.workers, backend='loky', inner_max_num_threads=1) as parallel:
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20261024).split(full, y)):
            folds = list(StratifiedKFold(3, shuffle=True, random_state=20269000 + fold).split(tr, y[tr]))
            results = parallel(delayed(evaluate_config)(c, matrices, y, tr, folds) for c in configs)
            chosen = choose(results, y[tr])
            probability = np.mean([fit_prob(configs[i], matrices, y, tr, te) for i in chosen['indices']], axis=0)
            predicted[te] = np.array(CATEGORIES)[np.argmax(probability, axis=1)]
            base_matrix = np.array([[base[i][c] for c in sorted(base[0])] for i in range(len(y))])
            baseline = ExtraTreesClassifier(n_estimators=200, max_features=.8, class_weight='balanced', random_state=42, n_jobs=1).fit(base_matrix[tr], y[tr])
            control[te] = baseline.predict(base_matrix[te])
            detail = {'fold': fold, 'train': tr.tolist(), 'test': te.tolist(), 'chosen': chosen, 'configs': [configs[i] for i in chosen['indices']], 'scores': [{'config': r['config'], 'accuracy': float(accuracy_score(y[tr], np.array(CATEGORIES)[r['probabilities'].argmax(axis=1)]))} for r in results]}
            history.append(detail)
            (out / 'progress.json').write_text(json.dumps(history, indent=2))
            print(json.dumps({'phase': 'nested', 'fold': fold, 'chosen': detail['configs'], 'accuracy': float(np.mean(y[te] == predicted[te])), 'wall_seconds': round(time.perf_counter() - started, 1)}), flush=True)
        ids = np.arange(len(y))
        folds = list(StratifiedKFold(3, shuffle=True, random_state=20269999).split(ids, y))
        results = parallel(delayed(evaluate_config)(c, matrices, y, ids, folds) for c in configs)
    chosen = choose(results, y)
    import joblib
    fitted = []
    for i in chosen['indices']:
        cfg = configs[i]
        model = estimator(cfg).fit(matrices[cfg['view']], y)
        fitted.append({'config': cfg, 'columns': views[cfg['view']], 'model': model})
    # Es un artefacto de investigación: la API pública se integra tras la evaluación.
    joblib.dump({'members': fitted, 'categories': CATEGORIES, 'budget': args.budget, 'protocol': manifest['protocol'], 'version': out.name}, out / 'research-reference.joblib')
    report = {'nested': score(y, predicted), 'control': score(y, control), 'chosen': chosen, 'chosen_configs': [configs[i] for i in chosen['indices']], 'seconds': time.perf_counter() - started, 'goal_90_achieved': bool(np.mean(y == predicted) >= .9), 'instances': len(y), 'history': history, 'warning': plan['warning']}
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    import pandas as pd
    pd.DataFrame({'instance': [e['source_instance'] for e in examples], 'actual': y, 'control': control, 'predicted': predicted}).to_csv(out / 'predictions.csv', index=False)
    print(json.dumps({'complete': True, 'accuracy': report['nested']['accuracy'], 'easy_recall': report['nested']['per_class']['easy']['recall'], 'goal_90_achieved': report['goal_90_achieved'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
