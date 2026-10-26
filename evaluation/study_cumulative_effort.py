"""Nested comparison of cumulative time-threshold and direct classifiers."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import time
import numpy as np
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xai_complex.cumulative import CumulativeDifficultyClassifier
from cejsp import CATEGORIES
from train_jsplib import score

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    configs = []
    for view in ('all', 'effort'):
        for kind in ('direct', 'cumulative'):
            for balance in (False, True):
                configs.append(dict(view=view, kind=kind, balanced=balance))
        configs.append(dict(view=view, kind='hist', balanced=False))
    return configs


def estimator(config):
    if config['kind'] == 'cumulative':
        core = CumulativeDifficultyClassifier(n_estimators=180, balanced=config['balanced'])
    elif config['kind'] == 'direct':
        core = ExtraTreesClassifier(n_estimators=300, max_features=.8,
            class_weight='balanced' if config['balanced'] else None, random_state=42, n_jobs=1)
    else:
        core = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=7,
            min_samples_leaf=5, l2_regularization=5, learning_rate=.05, random_state=42)
    return make_pipeline(SimpleImputer(strategy='median'),core)


def predict(config, matrices, y, tr, te):
    with threadpool_limits(limits=1):
        model = estimator(config).fit(matrices[config['view']][tr], y[tr])
        p = model.predict_proba(matrices[config['view']][te])
    return p[:,[list(model.classes_).index(c) for c in CATEGORIES]]


def inner(config, matrices, y, ids, folds):
    p = np.zeros((len(ids),5))
    for tr,te in folds:
        p[te] = predict(config,matrices,y,ids[tr],ids[te])
    labels = np.array(CATEGORIES)[p.argmax(axis=1)]
    result = score(y[ids],labels)
    return {'config':config, 'accuracy':result['accuracy'], 'macro_f1':result['macro_f1']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists():
        raise ValueError('Preserve completed studies')
    manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    examples=manifest['examples']; y=np.array([e['label'] for e in examples])
    provenance=json.loads(args.features.with_name('features-provenance.json').read_text())
    if provenance['instance_hashes'] != [e['sha256'] for e in examples]:
        raise ValueError('Feature provenance does not align with source labels')
    for e in examples:
        if sha256(Path(e['path']).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('Dataset content changed')
    full=json.loads(args.features.read_text())
    if isinstance(full,dict): full=full['rows']
    if len(full)!=len(y) or any(row is None for row in full):
        raise ValueError('Feature rows are incomplete or not aligned')
    if len(set(e['base_instance_id'] for e in examples))!=len(y):
        raise ValueError('Repeated base instances require grouped CV')
    # Only predeclared instance-derived features enter the matrices.
    columns=sorted(full[0]); views={'all':columns,'effort':[c for c in columns if c.startswith(('proof_','disjunctive_','probe_','late_probe_','trajectory_')) or c in ('n_jobs','n_machines','n_operations','aspect_ratio')]}
    matrices={v:np.array([[r[c] for c in cols] for r in full]) for v,cols in views.items()}
    configs=configurations(); seed=20261024; started=time.perf_counter()
    (out/'plan.json').write_text(json.dumps({'configs':configs,'views':views,'outer_seed':seed,'inner_seed_base':20271000,'goal_accuracy':.9,'feature_file':str(args.features.resolve()),'feature_sha256':sha256(args.features.read_bytes()).hexdigest(),'warning':'Previously explored cohort; internal evaluation only; open is not an observed runtime'},indent=2))
    oof=np.empty(len(y),dtype=object); control=np.empty(len(y),dtype=object); history=[]
    with Parallel(n_jobs=args.workers, backend='loky',inner_max_num_threads=1) as parallel:
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(full,y)):
            folds=list(StratifiedKFold(3,shuffle=True,random_state=20271000+fold).split(tr,y[tr]))
            scores=parallel(delayed(inner)(c,matrices,y,tr,folds) for c in configs)
            chosen=max(scores,key=lambda r:(r['accuracy'],r['macro_f1']))
            oof[te]=np.array(CATEGORIES)[predict(chosen['config'],matrices,y,tr,te).argmax(axis=1)]
            control[te]=np.array(CATEGORIES)[predict(configs[1],matrices,y,tr,te).argmax(axis=1)]
            history.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'chosen':chosen,'inner':scores})
            (out/'progress.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({'fold':fold,'accuracy':float(np.mean(oof[te]==y[te])),'chosen':chosen['config']}),flush=True)
        ids=np.arange(len(y));folds=list(StratifiedKFold(3,shuffle=True,random_state=20271999).split(ids,y))
        scores=parallel(delayed(inner)(c,matrices,y,ids,folds) for c in configs)
    chosen=max(scores,key=lambda r:(r['accuracy'],r['macro_f1']));cfg=chosen['config']
    with threadpool_limits(limits=1): final=estimator(cfg).fit(matrices[cfg['view']],y)
    dump({'model':final,'columns':views[cfg['view']],'config':cfg,'protocol':manifest['protocol'],'feature_source':str(args.features.resolve()),'categories':CATEGORIES},out/'research-reference.joblib')
    report={'nested':score(y,oof),'control':score(y,control),'chosen':chosen,'final_scores':scores,'goal_90_achieved':bool(np.mean(oof==y)>=.9),'history':history,'seconds':time.perf_counter()-started,'warning':'Internal estimate on repeatedly explored cohort; no external IGJSP validation'}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'predicted':oof,'control':control}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'accuracy':report['nested']['accuracy'],'goal_90_achieved':report['goal_90_achieved']}),flush=True)


if __name__=='__main__': main()
