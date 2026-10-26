"""Two-round nested search with instance-only features and an explicit easy objective."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
from sklearn.metrics import accuracy_score,f1_score,recall_score
from threadpoolctl import threadpool_limits
from cejsp import JobShopComplexityClassifier,CATEGORIES
from xai_complex.search_features import search_features
from train_jsplib import score

ROOT=Path(__file__).resolve().parents[1]


def extract(path,budget):
    with threadpool_limits(limits=1):
        return search_features(path,probe_budget=budget)


def candidates():
    result=[{'id':'baseline','backend':'structural','kind':'extra_trees','trees':200,'params':{}}]
    for backend in ('structural','search','probe'):
        for fraction in (.5,.8,1.):
            for leaf in (1,2):
                result.append({'id':f'{backend}-et-{fraction}-{leaf}','backend':backend,'kind':'extra_trees','trees':400,'params':{'max_features':fraction,'min_samples_leaf':leaf}})
        for fraction in ('sqrt',.7):
            result.append({'id':f'{backend}-rf-{fraction}','backend':backend,'kind':'random_forest','trees':400,'params':{'max_features':fraction,'min_samples_leaf':1}})
        for depth in (3,5):
            for balance in ('None','Balanced'):
                result.append({'id':f'{backend}-cb-{depth}-{balance}','backend':backend,'kind':'catboost','trees':160,'params':{'depth':depth,'auto_class_weights':balance,'learning_rate':.08,'l2_leaf_reg':6}})
        for c in (1,10):
            result.append({'id':f'{backend}-svm-{c}','backend':backend,'kind':'rbf_svm','trees':200,'params':{'C':c}})
        result.append({'id':f'{backend}-hist','backend':backend,'kind':'hist_gradient','trees':200,'params':{'max_iter':250,'max_leaf_nodes':7,'min_samples_leaf':5,'l2_regularization':5}})
    return result


def fit(config,rows,y,ids,protocol,budget,version='inner-selection',weights=None):
    model=JobShopComplexityClassifier(feature_backend=config['backend'],estimator_kind=config['kind'],n_estimators=config['trees'],estimator_params=config['params'],probe_budget=budget,n_jobs=1,class_decision_weights=weights)
    return model.fit([rows[config['backend']][i] for i in ids],y[ids],protocol=protocol,model_version=version)


def probabilities(model,rows,ids):
    raw=model.model_.predict_proba(model._matrix([rows[model.feature_backend][i] for i in ids]))
    return raw[:,[list(model.model_.classes_).index(c) for c in CATEGORIES]]


def objective(y,p):
    easy=float(np.mean(np.asarray(p)[y=='easy']=='easy'))
    return .5*accuracy_score(y,p)+.4*f1_score(y,p,labels=list(CATEGORIES),average='macro',zero_division=0)+.1*easy


def inner_candidate(config,rows,y,ids,folds,protocol,budget):
    values=np.zeros((len(ids),5))
    with threadpool_limits(limits=1):
        for tr,te in folds:
            model=fit(config,rows,y,ids[tr],protocol,budget)
            values[te]=probabilities(model,rows,ids[te])
    choices=[]
    for easy in (1.,1.5,2.,3.):
        for middle in (1.,1.3):
            weights=np.array([1.,easy,middle,middle,1.])
            pred=np.array(CATEGORIES)[np.argmax(values*weights,axis=1)]
            choices.append({'weights':weights.tolist(),'objective':float(objective(y[ids],pred)),
                            'accuracy':float(accuracy_score(y[ids],pred)),
                            'easy_recall':float(np.mean(pred[y[ids]=='easy']=='easy'))})
    best=max(choices,key=lambda c:c['objective'])
    return {'config':config,'choice':best,'unweighted':choices[0]}


def select(results,allow_probe):
    eligible=[r for r in results if allow_probe or r['config']['backend']!='probe']
    return max(eligible,key=lambda r:r['choice']['objective'])


def evaluate_choice(choice,rows,y,tr,te,protocol,budget):
    weights=dict(zip(CATEGORIES,choice['choice']['weights']))
    model=fit(choice['config'],rows,y,tr,protocol,budget,weights=weights)
    return np.array([r['category'] for r in model.predict([rows[model.feature_backend][i] for i in te])])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--budget',type=float,default=.2)
    parser.add_argument('--out',type=Path,default=ROOT/'models/jsplib-easy-v5');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'reference.joblib').exists():
        raise ValueError('Choose a new output directory; references are preserved')
    if args.workers<1 or args.workers>32:
        raise ValueError('workers must be between 1 and 32')
    started=time.perf_counter()
    manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());examples=manifest['examples'];y=np.array([e['label'] for e in examples])
    for e in examples:
        if sha256(Path(e['path']).read_bytes()).hexdigest()!=e['sha256']:
            raise ValueError('Source instance hash mismatch')
    configs=candidates()
    plan={'target_accuracy':.95,'workers':args.workers,'probe_budget_deterministic':args.budget,'solver_workers_per_probe':1,'seed':20261024,
          'objective':'.5 accuracy + .4 macro-F1 + .1 easy recall','outer_folds':5,'inner_folds':3,'candidates':configs,
          'decision_weight_grid':{'easy':[1,1.5,2,3],'medium_hard':[1,1.3]},'rounds':['static bounds and constructive probes','add deterministic CP-SAT probes'],
          'data_n':len(y),'class_counts':dict(Counter(y.tolist())),'evaluation_warning':'Previously explored cohort; nested estimates remain internal; never use training accuracy as target evidence'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    cache=out/'features.json'; source_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pypi/src/xai_complex/structural.py',ROOT/'pypi/src/xai_complex/search_features.py')}
    cache_key={'source_hashes':source_hashes,'budget':args.budget,'ortools':importlib.metadata.version('ortools'),'instance_hashes':[e['sha256'] for e in examples]}
    if cache.exists():
        stored=json.loads(cache.read_text())
        if stored['key']!=cache_key:
            raise ValueError('Feature cache does not match current code or data')
        full=stored['rows']
    else:
        full=[None]*len(y)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(extract,e['path'],args.budget):i for i,e in enumerate(examples)}
            for done,future in enumerate(as_completed(futures),1):
                full[futures[future]]=future.result()
                if done%20==0:
                    print(json.dumps({'features':done,'total':len(y)}),flush=True)
        cache.write_text(json.dumps({'key':cache_key,'rows':full}))
    from xai_complex.structural import structural_features
    keys=set(structural_features(examples[0]['path']))
    rows={'structural':[{k:v for k,v in r.items() if k in keys} for r in full],
          'search':[{k:v for k,v in r.items() if not k.startswith('probe_')} for r in full],'probe':full}
    results_a=np.empty(len(y),dtype=object);results_b=np.empty(len(y),dtype=object);baseline=np.empty(len(y),dtype=object)
    history=[];split_records=[]
    with Parallel(n_jobs=args.workers,backend='loky',inner_max_num_threads=1) as parallel:
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261024).split(examples,y)):
            folds=list(StratifiedKFold(3,shuffle=True,random_state=20262000+fold).split(tr,y[tr]))
            scores=parallel(delayed(inner_candidate)(config,rows,y,tr,folds,manifest['protocol'],args.budget) for config in configs)
            a=select(scores,False);b=select(scores,True)
            results_a[te]=evaluate_choice(a,rows,y,tr,te,manifest['protocol'],args.budget)
            results_b[te]=evaluate_choice(b,rows,y,tr,te,manifest['protocol'],args.budget)
            base=fit(configs[0],rows,y,tr,manifest['protocol'],args.budget)
            baseline[te]=[r['category'] for r in base.predict([rows['structural'][i] for i in te])]
            history.append({'fold':fold,'round_a':a,'round_b':b,'inner_scores':scores})
            split_records.append({'fold':fold,'train':[examples[i]['source_instance'] for i in tr],'test':[examples[i]['source_instance'] for i in te]})
            (out/'progress.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({'fold':fold,'round_a':a['config']['id'],'round_b':b['config']['id'],'round_b_accuracy':accuracy_score(y[te],results_b[te]),'round_b_easy_recall':float(np.mean(results_b[te][y[te]=='easy']=='easy'))}),flush=True)
        ids=np.arange(len(y));folds=list(StratifiedKFold(3,shuffle=True,random_state=20263000).split(ids,y))
        final_scores=parallel(delayed(inner_candidate)(c,rows,y,ids,folds,manifest['protocol'],args.budget) for c in configs)
    chosen=select(final_scores,True)
    weights=dict(zip(CATEGORIES,chosen['choice']['weights']))
    with threadpool_limits(limits=1):
        final=fit(chosen['config'],rows,y,np.arange(len(y)),manifest['protocol'],args.budget,'jsplib-easy-v5',weights)
        final.training_groups_=[e['base_instance_id'] for e in examples];final.previous_version_='jsplib-8a6e36ef3126-v2'
        final.save(out/'reference.joblib')
        loaded=final.load(out/'reference.joblib',trusted=True)
        assert loaded.predict(rows[final.feature_backend][:3])==final.predict(rows[final.feature_backend][:3])
        assert loaded.predict(rows[final.feature_backend][:1])[0]==loaded.predict(rows[final.feature_backend][:3])[0]
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'family':[e['family'] for e in examples],'actual':y,'baseline':baseline,'round_a':results_a,'round_b':results_b}).to_csv(out/'predictions.csv',index=False)
    report={'baseline':score(y,baseline),'round_a':score(y,results_a),'round_b':score(y,results_b),'chosen':chosen,'final_scores':final_scores,'seconds':time.perf_counter()-started,
            'target_achieved':float(accuracy_score(y,results_b))>=.95,'checks':{'persistence':True,'batch_invariance':True},'feature_source_hashes':source_hashes,
            'limitations':['All source instances have been explored before; no new independent external test','15 easy cases cannot establish stable 95% class accuracy','CP probe signals require bounded solving during inference and are solver-version-dependent','Source labels describe historical multi-engine performance; local proof or timeout does not replace source labels','Class score adjustment is not probability calibration']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2));(out/'splits.json').write_text(json.dumps(split_records,indent=2))
    cfg=chosen['config'];manifest.update({'model_version':final.model_version_,'feature_backend':cfg['backend'],'estimator_kind':cfg['kind'],'n_estimators':cfg['trees'],'estimator_params':cfg['params'],'class_decision_weights':weights,'probe_budget':args.budget,'n_jobs':1})
    (out/'training-manifest.json').write_text(json.dumps(manifest,indent=2));(out/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in final.dependencies_.items()))
    print(json.dumps({'complete':True,'accuracy':report['round_b']['accuracy'],'easy_recall':report['round_b']['per_class']['easy']['recall'],'chosen':cfg['id'],'target_achieved':report['target_achieved']}),flush=True)


if __name__=='__main__':
    main()
