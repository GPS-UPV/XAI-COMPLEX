"""Export a completed research fit through the ordinary CEJSP classify API."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from cejsp import JobShopComplexityClassifier,CATEGORIES

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',type=Path);ap.add_argument('--features',type=Path,required=True)
    args=ap.parse_args();out=args.study.resolve()
    if (out/'reference.joblib').exists():raise ValueError('Preserve existing references')
    report=json.loads((out/'evaluation.json').read_text())
    bundle=joblib.load(out/'research-reference.joblib')
    rows=json.loads(args.features.read_text())
    manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    y=np.array([e['label'] for e in manifest['examples']]);columns=sorted(rows[0])
    backend='proof_bottleneck' if any(c.startswith('proof_') for c in columns) and any(c.startswith('bottleneck_') for c in columns) else 'proof' if any(c.startswith('proof_') for c in columns) else 'bottleneck' if any(c.startswith('bottleneck_') for c in columns) else 'trajectory'
    budget=bundle.get('budget',10. if backend.startswith('proof') else 3.)
    if 'members' in bundle:
        members=[{'config':m['config'],'indices':[columns.index(c) for c in m['columns']]} for m in bundle['members']]
        kind='subset_voting';trees=300;params={'members':members};selected=rows
        expected=np.mean([m['model'].predict_proba(np.array([[r[c] for c in m['columns']] for r in rows]))[:,[list(m['model'].classes_).index(c) for c in CATEGORIES]] for m in bundle['members']],axis=0)
    else:
        config=bundle['config'];columns=bundle['columns']
        kind={'direct':'extra_trees','hist':'hist_gradient','cumulative':'cumulative'}[config['kind']]
        trees={'direct':300,'hist':200,'cumulative':180}[config['kind']]
        params={'max_features':.8,'class_weight':'balanced' if config['balanced'] else None} if kind=='extra_trees' else {'balanced':config['balanced']} if kind=='cumulative' else {'max_iter':200,'max_leaf_nodes':7,'min_samples_leaf':5,'l2_regularization':5,'learning_rate':.05}
        selected=[{c:r[c] for c in columns} for r in rows]
        p=bundle['model'].predict_proba(np.array([[r[c] for c in columns] for r in rows]))
        expected=p[:,[list(bundle['model'].classes_).index(c) for c in CATEGORIES]]
    with threadpool_limits(limits=1):
        model=JobShopComplexityClassifier(feature_backend=backend,probe_budget=budget,estimator_kind=kind,n_estimators=trees,estimator_params=params,n_jobs=1)
        model.fit(selected,y,protocol=manifest['protocol'],model_version=out.name)
        model.training_groups_=[e['base_instance_id'] for e in manifest['examples']]
        model.previous_version_='jsplib-easy-v7-trajectory'
        actual=np.array([[r['class_probabilities'][c] for c in CATEGORIES] for r in model.predict(selected)])
        maximum=float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        model.save(out/'reference.joblib')
        loaded=model.load(out/'reference.joblib',trusted=True)
        assert loaded.predict(selected[:3])==model.predict(selected[:3])
    manifest.update({'model_version':out.name,'feature_backend':backend,'estimator_kind':kind,'n_estimators':trees,'estimator_params':params,'probe_budget':budget,'n_jobs':1})
    (out/'training-manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in model.dependencies_.items()))
    result={'public_api_export':True,'max_research_public_probability_difference':maximum,'persistence':True,'model':str((out/'reference.joblib').resolve()),'nested_accuracy':report['nested']['accuracy']}
    (out/'export-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
