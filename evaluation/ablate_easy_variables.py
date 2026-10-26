"""Hold the classifier fixed while comparing groups of input variables."""
from pathlib import Path
import json
import numpy as np
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedKFold
from cejsp import JobShopComplexityClassifier
from train_jsplib import score


def run_group(name,rows,y,folds,protocol):
    from threadpoolctl import threadpool_limits
    predicted=np.empty(len(y),dtype=object)
    with threadpool_limits(limits=1):
        for tr,te in folds:
            model=JobShopComplexityClassifier(feature_backend='structural',estimator_kind='extra_trees',n_estimators=200).fit([rows[i] for i in tr],y[tr],protocol=protocol,model_version='feature-ablation-only')
            predicted[te]=[r['category'] for r in model.predict([rows[i] for i in te])]
    return {'group':name,'features':len(rows[0]),'metrics':score(y,predicted)}


def main():
    root=Path(__file__).resolve().parents[1];out=root/'models/jsplib-easy-v7-trajectory'
    manifest=json.loads((root/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());y=np.array([e['label'] for e in manifest['examples']])
    full=json.loads((out/'features.json').read_text());short=json.loads((root/'models/jsplib-easy-v5/features.json').read_text())['rows']
    from xai_complex.structural import structural_features
    keys=set(structural_features(manifest['examples'][0]['path']))
    groups={'structural':[{k:v for k,v in row.items() if k in keys} for row in full],
            'bounds_dispatch':[{k:v for k,v in row.items() if not k.startswith('probe_')} for row in short],
            'short_probe':short,
            'geometry_critical_blocks':[{k:v for k,v in row.items() if not k.startswith(('probe_','late_probe_','trajectory_'))} for row in full],
            'long_probe_trajectory':full}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=20261024).split(full,y))
    results=Parallel(n_jobs=4)(delayed(run_group)(name,rows,y,folds,manifest['protocol']) for name,rows in groups.items())
    (out/'feature-group-ablation.json').write_text(json.dumps({'classifier_fixed':'ExtraTrees 200, max_features .8, balanced, no decision reweighting','split_seed':20261024,'interpretation':'Exploratory OOF ablation; not a new independent test','results':results},indent=2))
    for r in results:print(r['group'],r['metrics']['accuracy'],r['metrics']['macro_f1'],r['metrics']['per_class']['easy']['recall'])


if __name__=='__main__':main()
