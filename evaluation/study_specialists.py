"""Third alternative: ordinal forests and a separate easy-versus-rest expert."""
import argparse,json,time
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from cejsp import CATEGORIES
from study_easy import ROOT,fit,probabilities,objective,evaluate_choice
from train_jsplib import score


def inner(config,rows,y,ids,folds,protocol):
    values=np.zeros((len(ids),5))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for a,b in folds:
            model=fit(config,rows,y,ids[a],protocol,.2);values[b]=probabilities(model,rows,ids[b])
    candidates=[]
    for easy in (.3,.6,1.,1.5,2.):
        for middle in (1.,1.3):
            weights=np.array([1.,easy,middle,middle,1.]);pred=np.array(CATEGORIES)[np.argmax(values*weights,axis=1)]
            candidates.append({'weights':weights.tolist(),'objective':float(objective(y[ids],pred))})
    return {'config':config,'choice':max(candidates,key=lambda c:c['objective'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--out',type=Path,default=ROOT/'models/jsplib-easy-v6-specialists');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'reference.joblib').exists():raise ValueError('Preserve completed reference')
    start=time.perf_counter();manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());examples=manifest['examples'];y=np.array([e['label'] for e in examples])
    full=json.loads((ROOT/'models/jsplib-easy-v5/features.json').read_text())['rows']
    keys={k for k in full[0] if not k.startswith(('probe_','dispatch_','schrage_','machine_bound_','head_dispersion_','tail_dispersion_','critical_job_fraction'))}
    rows={'structural':[{k:v for k,v in row.items() if k in keys} for row in full],'probe':full}
    configs=[{'id':'baseline','backend':'structural','kind':'extra_trees','trees':200,'params':{}}]
    for backend in ('structural','probe'):
        for leaf in (1,2):configs.append({'id':f'{backend}-ordinal-{leaf}','backend':backend,'kind':'ordinal_forest','trees':300,'params':{'min_samples_leaf':leaf}})
        for kind in ('forest','svm'):configs.append({'id':f'{backend}-specialist-{kind}','backend':backend,'kind':'easy_specialist','trees':300,'params':{'specialist_kind':kind}})
    (out/'plan.json').write_text(json.dumps({'configs':configs,'workers':args.workers,'round':3,'objective':'.5 accuracy + .4 macro-F1 + .1 easy recall','split_seed':20261024,'inner_folds':3,'outer_folds':5,'ordinal_assumption':'An experimental ordered proxy for the historical classes, not a mathematical hardness order'},indent=2))
    predicted=np.empty(len(y),dtype=object);baseline=np.empty(len(y),dtype=object);history=[]
    with Parallel(n_jobs=args.workers,backend='loky',inner_max_num_threads=1) as parallel:
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261024).split(examples,y)):
            folds=list(StratifiedKFold(3,shuffle=True,random_state=20265000+fold).split(tr,y[tr]))
            results=parallel(delayed(inner)(c,rows,y,tr,folds,manifest['protocol']) for c in configs)
            chosen=max(results,key=lambda r:r['choice']['objective'])
            predicted[te]=evaluate_choice(chosen,rows,y,tr,te,manifest['protocol'],.2)
            base=fit(configs[0],rows,y,tr,manifest['protocol'],.2);baseline[te]=[r['category'] for r in base.predict([rows['structural'][i] for i in te])]
            history.append({'fold':fold,'chosen':chosen,'inner_scores':results})
            print(json.dumps({'fold':fold,'chosen':chosen['config']['id'],'accuracy':accuracy_score(y[te],predicted[te]),'easy_recall':float(np.mean(predicted[te][y[te]=='easy']=='easy'))}),flush=True)
            (out/'progress.json').write_text(json.dumps(history,indent=2))
        ids=np.arange(len(y));folds=list(StratifiedKFold(3,shuffle=True,random_state=20266000).split(ids,y))
        results=parallel(delayed(inner)(c,rows,y,ids,folds,manifest['protocol']) for c in configs)
    chosen=max(results,key=lambda r:r['choice']['objective']);weights=dict(zip(CATEGORIES,chosen['choice']['weights']))
    model=fit(chosen['config'],rows,y,np.arange(len(y)),manifest['protocol'],.2,'jsplib-easy-v6-specialists',weights)
    model.training_groups_=[e['base_instance_id'] for e in examples];model.save(out/'reference.joblib')
    loaded=model.load(out/'reference.joblib',trusted=True);assert loaded.predict(rows[model.feature_backend][:2])==model.predict(rows[model.feature_backend][:2])
    cfg=chosen['config'];manifest.update({'model_version':model.model_version_,'feature_backend':cfg['backend'],'estimator_kind':cfg['kind'],'n_estimators':cfg['trees'],'estimator_params':cfg['params'],'class_decision_weights':weights,'probe_budget':.2,'n_jobs':1})
    (out/'training-manifest.json').write_text(json.dumps(manifest,indent=2));(out/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in model.dependencies_.items()))
    report={'baseline':score(y,baseline),'round_c':score(y,predicted),'chosen':chosen,'final_scores':results,'history':history,'seconds':time.perf_counter()-start,'target_achieved':bool(accuracy_score(y,predicted)>=.95),'checks':{'persistence':True},'limitations':['Previously explored data; internal nested estimates only','Third family of models examined after earlier experiments; no external validation','Ordinal class order is an experimental modeling assumption']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'baseline':baseline,'round_c':predicted}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'accuracy':report['round_c']['accuracy'],'easy_recall':report['round_c']['per_class']['easy']['recall'],'target_achieved':report['target_achieved']}),flush=True)


if __name__=='__main__':main()
