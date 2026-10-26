"""Fourth round: route geometry, critical blocks and a longer bounded search."""
import argparse,json,time
from hashlib import sha256
import importlib.metadata
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedKFold
from cejsp import CATEGORIES
from xai_complex.landscape import landscape_features
from study_easy import ROOT,fit,probabilities,objective,evaluate_choice
from train_jsplib import score


def extract(path):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):return landscape_features(path,probe_budget=3.)


def inner(config,rows,y,ids,folds,protocol):
    values=np.zeros((len(ids),5))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for a,b in folds:
            model=fit(config,rows,y,ids[a],protocol,3.);values[b]=probabilities(model,rows,ids[b])
    choices=[]
    for easy in (.6,1.,1.5,2.,3.):
        for middle in (1.,1.3):
            weights=np.array([1.,easy,middle,middle,1.]);p=np.array(CATEGORIES)[np.argmax(values*weights,axis=1)]
            choices.append({'weights':weights.tolist(),'objective':float(objective(y[ids],p))})
    return {'config':config,'choice':max(choices,key=lambda c:c['objective'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--out',type=Path,default=ROOT/'models/jsplib-easy-v7-trajectory');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'reference.joblib').exists():raise ValueError('Preserve existing reference')
    started=time.perf_counter();manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());examples=manifest['examples'];y=np.array([e['label'] for e in examples])
    cache=out/'features.json';partial=out/'features-partial.json';meta=out/'features-provenance.json'
    key={'source_hashes':{p.name:sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pypi/src/xai_complex/structural.py',ROOT/'pypi/src/xai_complex/search_features.py',ROOT/'pypi/src/xai_complex/landscape.py')},'instance_hashes':[e['sha256'] for e in examples],'budgets':[.2,3.],'ortools':importlib.metadata.version('ortools')}
    if meta.exists() and json.loads(meta.read_text())!=key:raise ValueError('Cached features differ from current code or inputs')
    if (cache.exists() or partial.exists()) and not meta.exists():raise ValueError('Feature cache lacks provenance')
    meta.write_text(json.dumps(key,indent=2))
    if cache.exists():full=json.loads(cache.read_text())
    else:
        full=json.loads(partial.read_text()) if partial.exists() else [None]*len(y)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(extract,e['path']):i for i,e in enumerate(examples) if full[i] is None}
            for done,future in enumerate(as_completed(futures),1):
                full[futures[future]]=future.result()
                if done%5==0:
                    pending=out/'features-partial.pending';pending.write_text(json.dumps(full));pending.replace(partial)
                    print(json.dumps({'features':sum(r is not None for r in full),'total':len(y)}),flush=True)
        cache.write_text(json.dumps(full))
    from xai_complex.structural import structural_features
    keys=set(structural_features(examples[0]['path']))
    rows={'structural':[{k:v for k,v in row.items() if k in keys} for row in full],
          'landscape':[{k:v for k,v in row.items() if not k.startswith(('probe_','late_probe_','trajectory_'))} for row in full],'trajectory':full}
    configs=[{'id':'baseline','backend':'structural','kind':'extra_trees','trees':200,'params':{}}]
    for backend in ('landscape','trajectory'):
        for fraction in (.5,.8):
            for leaf in (1,2):configs.append({'id':f'{backend}-et-{fraction}-{leaf}','backend':backend,'kind':'extra_trees','trees':400,'params':{'max_features':fraction,'min_samples_leaf':leaf}})
        configs.append({'id':f'{backend}-catboost','backend':backend,'kind':'catboost','trees':160,'params':{'depth':4,'learning_rate':.08,'auto_class_weights':'Balanced','l2_leaf_reg':6}})
        configs.append({'id':f'{backend}-hist','backend':backend,'kind':'hist_gradient','trees':200,'params':{'max_iter':250,'max_leaf_nodes':7,'min_samples_leaf':5,'l2_regularization':5}})
    (out/'plan.json').write_text(json.dumps({'round':4,'probe_budgets':[.2,3.],'workers':args.workers,'configs':configs,'objective':'.5 accuracy + .4 macro-F1 + .1 easy recall','split_seed':20261024,'feature_count':len(full[0])},indent=2))
    predicted=np.empty(len(y),dtype=object);baseline=np.empty(len(y),dtype=object);history=[]
    with Parallel(n_jobs=args.workers,backend='loky',inner_max_num_threads=1) as parallel:
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261024).split(examples,y)):
            folds=list(StratifiedKFold(3,shuffle=True,random_state=20267000+fold).split(tr,y[tr]))
            results=parallel(delayed(inner)(c,rows,y,tr,folds,manifest['protocol']) for c in configs)
            chosen=max(results,key=lambda r:r['choice']['objective']);predicted[te]=evaluate_choice(chosen,rows,y,tr,te,manifest['protocol'],3.)
            base=fit(configs[0],rows,y,tr,manifest['protocol'],3.);baseline[te]=[r['category'] for r in base.predict([rows['structural'][i] for i in te])]
            history.append({'fold':fold,'chosen':chosen,'inner_scores':results})
            (out/'progress.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({'fold':fold,'chosen':chosen['config']['id'],'accuracy':float(np.mean(y[te]==predicted[te])),'easy_recall':float(np.mean(predicted[te][y[te]=='easy']=='easy'))}),flush=True)
        ids=np.arange(len(y));folds=list(StratifiedKFold(3,shuffle=True,random_state=20268000).split(ids,y))
        results=parallel(delayed(inner)(c,rows,y,ids,folds,manifest['protocol']) for c in configs)
    chosen=max(results,key=lambda r:r['choice']['objective']);weights=dict(zip(CATEGORIES,chosen['choice']['weights']))
    model=fit(chosen['config'],rows,y,np.arange(len(y)),manifest['protocol'],3.,'jsplib-easy-v7-trajectory',weights)
    model.training_groups_=[e['base_instance_id'] for e in examples];model.save(out/'reference.joblib')
    cfg=chosen['config'];manifest.update({'model_version':model.model_version_,'feature_backend':cfg['backend'],'estimator_kind':cfg['kind'],'n_estimators':cfg['trees'],'estimator_params':cfg['params'],'class_decision_weights':weights,'probe_budget':3.,'n_jobs':1})
    (out/'training-manifest.json').write_text(json.dumps(manifest,indent=2));(out/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in model.dependencies_.items()))
    loaded=model.load(out/'reference.joblib',trusted=True);assert loaded.predict(rows[model.feature_backend][:3])==model.predict(rows[model.feature_backend][:3])
    report={'baseline':score(y,baseline),'round_d':score(y,predicted),'chosen':chosen,'final_scores':results,'history':history,'seconds':time.perf_counter()-started,'target_achieved':bool(np.mean(y==predicted)>=.95),'checks':{'persistence':True},'limitations':['Internal exploratory nested CV, not an independent new test','Four rounds of model-family exploration create selection uncertainty','A longer bounded solve is required for trajectory inference','Historical class open is not certified by a timeout']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'baseline':baseline,'round_d':predicted}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'accuracy':report['round_d']['accuracy'],'easy_recall':report['round_d']['per_class']['easy']['recall'],'target_achieved':report['target_achieved']}),flush=True)


if __name__=='__main__':main()
