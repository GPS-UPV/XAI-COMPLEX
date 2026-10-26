"""IGJSP weak-label augmentation evaluated only against held-out real benchmarks."""
from pathlib import Path
import json
from hashlib import sha256
import importlib.metadata
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from IGJSP.generador import Generator
from cejsp import JobShopComplexityClassifier, CATEGORIES
from cejsp.io import load_instance
from xai_complex.structural import structural_features
from train_jsplib import score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'models/jsplib-8a6e36ef3126-v1'
OUT=ROOT/'models/jsplib-8a6e36ef3126-v3-igjsp'
DATA=ROOT/'data/igjsp-augmentation-v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    if (OUT/'experiment.json').exists():
        raise ValueError('Preserve completed augmentation experiment')
    started=time.perf_counter()
    manifest=json.loads((BASE/'training-manifest.json').read_text())
    examples=manifest['examples']; y=np.array([e['label'] for e in examples])
    families=np.array([e['family'] for e in examples]); rows=[structural_features(e['path']) for e in examples]
    generator=Generator(json=False,dzn=False,taillard=False)
    generated=[]; synthetic=[]; parents=[]
    for i,e in enumerate(examples):
        real=load_instance(e['path'])
        target=np.median([o['speeds'][0]['processing_time'] for j in real['jobs'] for o in j['operations']])
        for repeat in range(3):
            seed=710000+3*i+repeat
            path=DATA/f'igjsp-{seed}.json'
            if path.exists():
                instance=load_instance(path)
            else:
                obj=generator.generate_new_instance(jobs=len(real['jobs']),machines=real['n_machines'],speed=1,ReleaseDateDueDate=0,distribution='uniform',seed=seed,instance_type='JSP')
                instance=load_instance(obj)
                p=[o['speeds'][0]['processing_time'] for j in instance['jobs'] for o in j['operations']]
                scale=max(float(target),1)/max(float(np.median(p)),1)
                # Se registra el reescalado externo; no se modifica IGJSP ni se hereda la clase.
                for j in instance['jobs']:
                    for o in j['operations']:
                        o['speeds'][0]['processing_time']=float(max(1,round(o['speeds'][0]['processing_time']*scale)))
                instance['metadata']={'generator':'IGJSP','version':importlib.metadata.version('IGJSP'),'seed':seed,'external_time_scale':scale}
                instance['instance_id']=sha256(json.dumps(instance['jobs'],sort_keys=True).encode()).hexdigest()
                path.write_text(json.dumps(instance),encoding='utf-8')
            synthetic.append(structural_features(instance)); parents.append(i)
            generated.append({'path':str(path.relative_to(ROOT)),'parent_base_id':e['base_instance_id'],'source_instance':e['source_instance'],'seed':seed,'ground_truth_label':None,'sha256':sha256(path.read_bytes()).hexdigest()})
        if (i+1)%40==0:
            print(f'IGJSP generated {(i+1)*3}',flush=True)
    parents=np.array(parents)
    (DATA/'manifest.json').write_text(json.dumps({'generator':'IGJSP','version':importlib.metadata.version('IGJSP'),'matched':'dimensions and median processing time; uniform distribution, independent routes','truth_labels':False,'instances':generated},indent=2))
    configurations=[('extra_trees',0.),('random_forest',0.),('extra_trees',.15),('random_forest',.15)]
    def fit(config,tr):
        kind,weight=config
        model=JobShopComplexityClassifier(feature_backend='structural',estimator_kind=kind,n_estimators=200)
        x=[rows[i] for i in tr]; labels=y[tr].tolist()
        model.fit(x,labels,protocol=manifest['protocol'],model_version='teacher-training-fold-only')
        used=[]
        if weight:
            candidates=np.flatnonzero(np.isin(parents,tr))
            predictions=model.predict([synthetic[i] for i in candidates])
            # El cupo por clase evita que las pseudoetiquetas toy dominen las clases escasas.
            for label in CATEGORIES:
                selected=sorted([(max(p['class_probabilities'].values()),int(i)) for i,p in zip(candidates,predictions) if p['category']==label and max(p['class_probabilities'].values())>=.7],reverse=True)[:30]
                used.extend((idx,label) for confidence,idx in selected)
            if used:
                x += [synthetic[i] for i,label in used]
                labels += [label for i,label in used]
                model.fit(x,labels,protocol=manifest['protocol'],model_version='augmented-training-fold-only',sample_weight=[1.]*len(tr)+[weight]*len(used))
        return model,len(used)
    def pred(model,te):
        return [r['category'] for r in model.predict([rows[i] for i in te])]
    def objective(actual,predicted):
        return .5*accuracy_score(actual,predicted)+.5*f1_score(actual,predicted,labels=list(CATEGORIES),average='macro',zero_division=0)
    def select(ids,seed):
        splits=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(ids,y[ids]))
        ranking=[]
        for config in configurations:
            values=[]
            for a,b in splits:
                model,used=fit(config,ids[a]); values.append(objective(y[ids[b]],pred(model,ids[b])))
            ranking.append({'configuration':config,'objective':float(np.mean(values))})
        return configurations[int(np.argmax([r['objective'] for r in ranking]))],ranking
    n=len(y); baseline=np.empty(n,dtype=object); augmented=np.empty(n,dtype=object); nested=np.empty(n,dtype=object)
    folds=[]
    with threadpool_limits(limits=1):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261023).split(rows,y)):
            chosen,ranking=select(tr,20261400+fold)
            model,used=fit(chosen,tr); nested[te]=pred(model,te)
            plain,_=fit(('extra_trees',0.),tr); baseline[te]=pred(plain,te)
            aug,count=fit(('extra_trees',.15),tr); augmented[te]=pred(aug,te)
            assert not set(parents[np.isin(parents,tr)]) & set(te)
            folds.append({'fold':fold,'selected':chosen,'inner_scores':ranking,'pseudo_labels_used':used,'train':[examples[i]['source_instance'] for i in tr],'test':[examples[i]['source_instance'] for i in te]})
            print(json.dumps({'fold':fold,'chosen':chosen,'pseudo_labels':used}),flush=True)
        selected,ranking=select(np.arange(n),20261500)
        model,used=fit(selected,np.arange(n))
        model.model_version_='jsplib-8a6e36ef3126-v3-igjsp'
        model.previous_version_=manifest['model_version']
        model.training_groups_=[e['base_instance_id'] for e in examples]
        model.augmentation_={'generator':'IGJSP 2.0.1','weak_labels':True,'synthetic_candidates':len(synthetic),'used':used,'sample_weight':selected[1],'confidence_threshold':.7,'per_class_cap':30}
        model.save(OUT/'experimental-reference.joblib')
    report={'generated_n':len(synthetic),'selected':selected,'pseudo_labels_used_final':used,'baseline_structural_et':score(y,baseline),'fixed_augmented_et':score(y,augmented),'nested_selection':score(y,nested),'folds':folds,'final_ranking':ranking,'seconds':time.perf_counter()-started,'label_policy':'Fold-trained teacher pseudo-labels, NOT solver-verified truth; no label copied from parent','limitations':['No independent IGJSP ground truth','Source cohort already explored; nested estimates are provisional','Do not label a generated instance open because a short solver run timed out']}
    (OUT/'experiment.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'baseline':baseline,'augmented':augmented,'nested':nested}).to_csv(OUT/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'baseline_accuracy':report['baseline_structural_et']['accuracy'],'augmented_accuracy':report['fixed_augmented_et']['accuracy'],'nested_accuracy':report['nested_selection']['accuracy']}),flush=True)


if __name__=='__main__':
    main()
