"""Nested model selection on the existing JSPLIB cohort; old test is diagnostic."""
import json
from pathlib import Path
import time
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits
from cejsp import JobShopComplexityClassifier, CATEGORIES
from xai_complex.structural import structural_features
from train_jsplib import score


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'models/jsplib-8a6e36ef3126-v1'
OUT = ROOT / 'models/jsplib-8a6e36ef3126-v2'


def objective(y, predicted):
    return .5*accuracy_score(y,predicted) + .5*f1_score(y,predicted,labels=list(CATEGORIES),average='macro',zero_division=0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'reference.joblib').exists():
        raise ValueError('Preserve the existing v2 model; choose another output directory')
    manifest=json.loads((OLD/'training-manifest.json').read_text())
    examples=manifest['examples']
    y=np.array([e['label'] for e in examples])
    families=np.array([e['family'] for e in examples])
    caches=list((ROOT/'data/jsplib'/manifest['protocol']['source_commit']).glob('features-*.json'))
    if len(caches)!=1:
        raise ValueError('Expected the single verified v1 feature cache')
    graph=json.loads(caches[0].read_text())
    if len(graph)!=len(examples):
        raise ValueError('Graph cache cohort mismatch')
    started=time.perf_counter()
    structural=[structural_features(e['path']) for e in examples]
    rows={'graph_approx':graph,'structural':structural,
          'hybrid':[{**g,**{f'struct_{k}':v for k,v in s.items()}} for g,s in zip(graph,structural)]}
    candidates=[('graph_approx','random_forest')]+[(backend,kind) for backend in ('structural','hybrid') for kind in ('random_forest','extra_trees','hist_gradient','rbf_svm','ensemble')]
    (OUT/'experiment-plan.json').write_text(json.dumps({'candidates':candidates,'outer_folds':5,'inner_folds':3,'seed':20261023,'selection_objective':'0.5 accuracy + 0.5 macro-F1','old_test':'diagnostic only; already observed','features_seconds':time.perf_counter()-started},indent=2))
    def fit(candidate, train, version='comparison'):
        backend,kind=candidate
        return JobShopComplexityClassifier(feature_backend=backend,estimator_kind=kind,n_estimators=200).fit(
            [rows[backend][i] for i in train],y[train],protocol=manifest['protocol'],model_version=version)
    def predict(model, candidate, test):
        return [r['category'] for r in model.predict([rows[candidate[0]][i] for i in test])]
    def select(ids, seed):
        folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(ids,y[ids]))
        scores=[]
        for candidate in candidates:
            values=[]
            for tr,te in folds:
                model=fit(candidate,ids[tr])
                values.append(objective(y[ids[te]],predict(model,candidate,ids[te])))
            scores.append({'backend':candidate[0],'estimator':candidate[1],'objective':float(np.mean(values))})
        best=int(np.argmax([s['objective'] for s in scores]))
        return candidates[best],scores
    nested=np.empty(len(y),dtype=object); baseline=np.empty(len(y),dtype=object)
    splits=[]; choices=[]
    outer=StratifiedKFold(5,shuffle=True,random_state=20261023)
    with threadpool_limits(limits=1):
        for fold,(tr,te) in enumerate(outer.split(examples,y)):
            chosen,ranking=select(tr,20261100+fold)
            model=fit(chosen,tr)
            nested[te]=predict(model,chosen,te)
            base=fit(candidates[0],tr)
            baseline[te]=predict(base,candidates[0],te)
            splits.append({'fold':fold,'train':[examples[i]['source_instance'] for i in tr],'test':[examples[i]['source_instance'] for i in te]})
            choices.append({'fold':fold,'chosen':chosen,'inner_scores':ranking})
            (OUT/'progress.json').write_text(json.dumps({'folds':choices},indent=2))
            print(json.dumps({'fold':fold,'chosen':chosen,'accuracy':accuracy_score(y[te],nested[te]),'macro_f1':f1_score(y[te],nested[te],average='macro')}),flush=True)
        selected,ranking=select(np.arange(len(y)),20261200)
        print(json.dumps({'final_candidate':selected}),flush=True)
        family_pred=np.empty(len(y),dtype=object); family_base=np.empty(len(y),dtype=object)
        family_choices=[]
        # La selección también se repite sin consultar la familia externa.
        for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(examples,y,families)):
            chosen,scores=select(tr,20261300+fold)
            family_pred[te]=predict(fit(chosen,tr),chosen,te)
            family_base[te]=predict(fit(candidates[0],tr),candidates[0],te)
            family_choices.append({'family':str(families[te[0]]),'selected':chosen,'inner_scores':scores})
            print(json.dumps({'family':str(families[te[0]]),'selected':chosen,'accuracy':accuracy_score(y[te],family_pred[te])}),flush=True)
        model=fit(selected,np.arange(len(y)),'jsplib-8a6e36ef3126-v2')
        model.previous_version_=manifest['model_version']
        model.training_groups_=[e['base_instance_id'] for e in examples]
        model.save(OUT/'reference.joblib')
        restored=model.load(OUT/'reference.joblib',trusted=True)
        backend=selected[0]
        assert model.predict(rows[backend][:3])==restored.predict(rows[backend][:3])
        assert model.predict(rows[backend][:1])[0]==model.predict(rows[backend][:3])[0]
    manifest.update({'model_version':model.model_version_,'feature_backend':selected[0],'estimator_kind':selected[1],'n_estimators':200})
    (OUT/'training-manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in model.dependencies_.items()))
    import pandas as pd
    pd.DataFrame([{'instance':e['source_instance'],'family':e['family'],'actual':a,'v1_oof':b,'v2_nested_oof':p,'v2_family_oof':fp} for e,a,b,p,fp in zip(examples,y,baseline,nested,family_pred)]).to_csv(OUT/'predictions.csv',index=False)
    report={'selected':selected,'dataset_n':len(y),'class_counts':dict(Counter(y.tolist())),
            'nested_v1':score(y,baseline),'nested_v2':score(y,nested),'family_nested_v2':score(y,family_pred),
            'family_v1':score(y,family_base),'choices':choices,'final_ranking':ranking,'family_choices':family_choices,
            'seconds':time.perf_counter()-started,'checks':{'persistence':True,'batch_invariance':True},
            'limitations':['Historical labels and all cohort labels already used in prior exploratory study; no new external test', 'Tiny minority classes; uncalibrated votes','Nested folds estimate selection procedure; final refit model is not externally validated','Family grouping is a stricter domain-shift diagnostic; inner tuning uses stratified instances','No claim of a global optimum over all possible classifiers']}
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2))
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    print(json.dumps({'complete':True,'selected':selected,'v1_macro_f1':report['nested_v1']['macro_f1'],'v2_macro_f1':report['nested_v2']['macro_f1']}),flush=True)


if __name__=='__main__':
    main()
