"""Ablate extra, genuinely labeled JSPLIB cases using fold-local selection."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from cejsp import JobShopComplexityClassifier,CATEGORIES
from xai_complex.structural import structural_features
from train_jsplib import score


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'models/jsplib-8a6e36ef3126-v4-expanded'
    out.mkdir(parents=True,exist_ok=True)
    original=json.loads((root/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    expanded=json.loads((root/'evaluation/expanded-import/training-manifest.json').read_text())
    known={e['base_instance_id'] for e in original['examples']}
    extra=[e for e in expanded['examples'] if e['base_instance_id'] not in known]
    examples=original['examples']; y=np.array([e['label'] for e in examples]); families=np.array([e['family'] for e in examples])
    rows=[structural_features(e['path']) for e in examples]
    extra_rows=[structural_features(e['path']) for e in extra]
    def fit(ids,addition,excluded_family=None):
        use=[i for i,e in enumerate(extra) if addition and e['family']!=excluded_family]
        model=JobShopComplexityClassifier(feature_backend='structural',estimator_kind='extra_trees',n_estimators=200)
        model.fit([rows[i] for i in ids]+[extra_rows[i] for i in use],y[ids].tolist()+[extra[i]['label'] for i in use],protocol=original['protocol'],model_version='expanded-evaluation')
        return model
    def pred(model,ids):
        return [p['category'] for p in model.predict([rows[i] for i in ids])]
    def select(ids,seed,excluded_family=None):
        ranking=[]
        folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(ids,y[ids]))
        for addition in (False,True):
            values=[]
            for a,b in folds:
                p=pred(fit(ids[a],addition,excluded_family),ids[b])
                values.append(.5*accuracy_score(y[ids[b]],p)+.5*f1_score(y[ids[b]],p,labels=list(CATEGORIES),average='macro',zero_division=0))
            ranking.append(float(np.mean(values)))
        return bool(np.argmax(ranking)),ranking
    n=len(y); plain=np.empty(n,dtype=object); extended=np.empty(n,dtype=object); nested=np.empty(n,dtype=object)
    folds=[]; family_result=np.empty(n,dtype=object)
    with threadpool_limits(limits=1):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261023).split(rows,y)):
            chosen,ranking=select(tr,20261600+fold)
            plain[te]=pred(fit(tr,False),te); extended[te]=pred(fit(tr,True),te)
            nested[te]=extended[te] if chosen else plain[te]
            folds.append({'fold':fold,'extra_selected':chosen,'inner_objectives':ranking})
            print(json.dumps(folds[-1]),flush=True)
        for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(rows,y,families)):
            family=str(families[te[0]])
            chosen,ranking=select(tr,20261700+fold,family)
            family_result[te]=pred(fit(tr,chosen,family),te)
        chosen,ranking=select(np.arange(n),20261800)
        final=fit(np.arange(n),chosen)
        final.model_version_='jsplib-8a6e36ef3126-v4-expanded'
        final.training_groups_=[e['base_instance_id'] for e in examples+(extra if chosen else [])]
        final.save(out/'reference.joblib')
    report={'extra_n':len(extra),'final_uses_extra':chosen,'final_inner_objectives':ranking,'same_original_cohort_n':n,'fixed_original':score(y,plain),'fixed_extended':score(y,extended),'nested':score(y,nested),'family_nested':score(y,family_result),'folds':folds,'label_policy':'Original JSPLIB truth labels only; no pseudo-labels','limitations':['Previously explored cohort; no new external test','Fixed extended results are an ablation, not an independent post-selection test']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    expanded.update({'examples':examples+(extra if chosen else []),'model_version':final.model_version_,'feature_backend':'structural','estimator_kind':'extra_trees','n_estimators':200})
    (out/'training-manifest.json').write_text(json.dumps(expanded,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'plain':plain,'extra':extended,'nested':nested}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'extra_n':len(extra),'final_extra':chosen,'nested_accuracy':report['nested']['accuracy']}),flush=True)


if __name__=='__main__':
    main()
