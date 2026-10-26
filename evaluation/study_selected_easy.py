"""Fifth round: select variables for easy inside every training fold."""
import argparse,json,time
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedKFold
from cejsp import CATEGORIES
from study_easy import ROOT,fit,evaluate_choice
from study_trajectory import inner
from train_jsplib import score


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--out',type=Path,default=ROOT/'models/jsplib-easy-v8-selection');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'reference.joblib').exists():raise ValueError('Preserve completed reference')
    start=time.perf_counter();manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());examples=manifest['examples'];y=np.array([e['label'] for e in examples])
    full=json.loads((ROOT/'models/jsplib-easy-v7-trajectory/features.json').read_text())
    from xai_complex.structural import structural_features
    keys=set(structural_features(examples[0]['path']));rows={'structural':[{k:v for k,v in row.items() if k in keys} for row in full],'trajectory':full}
    configs=[{'id':'baseline','backend':'structural','kind':'extra_trees','trees':200,'params':{}},{'id':'all-trajectory','backend':'trajectory','kind':'extra_trees','trees':200,'params':{}}]
    for backend in ('structural','trajectory'):
        for k in (16,32,64):
            for weight in (.5,1.):configs.append({'id':f'{backend}-select-{k}-{weight}','backend':backend,'kind':'selected_easy','trees':200,'params':{'k':k,'easy_weight':weight}})
    (out/'plan.json').write_text(json.dumps({'round':5,'workers':args.workers,'configs':configs,'selection':'Mutual information, joint class/easy or easy only, fitted inside each fold','split_seed':20261024},indent=2))
    predicted=np.empty(len(y),dtype=object);history=[]
    with Parallel(n_jobs=args.workers,backend='loky',inner_max_num_threads=1) as parallel:
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261024).split(examples,y)):
            folds=list(StratifiedKFold(3,shuffle=True,random_state=20269000+fold).split(tr,y[tr]))
            results=parallel(delayed(inner)(c,rows,y,tr,folds,manifest['protocol']) for c in configs)
            chosen=max(results,key=lambda r:r['choice']['objective']);predicted[te]=evaluate_choice(chosen,rows,y,tr,te,manifest['protocol'],3.)
            history.append({'fold':fold,'chosen':chosen,'inner_scores':results})
            print(json.dumps({'fold':fold,'chosen':chosen['config']['id'],'accuracy':float(np.mean(y[te]==predicted[te])),'easy_recall':float(np.mean(predicted[te][y[te]=='easy']=='easy'))}),flush=True)
            (out/'progress.json').write_text(json.dumps(history,indent=2))
        ids=np.arange(len(y));folds=list(StratifiedKFold(3,shuffle=True,random_state=20270000).split(ids,y))
        results=parallel(delayed(inner)(c,rows,y,ids,folds,manifest['protocol']) for c in configs)
    chosen=max(results,key=lambda r:r['choice']['objective']);weights=dict(zip(CATEGORIES,chosen['choice']['weights']))
    model=fit(chosen['config'],rows,y,np.arange(len(y)),manifest['protocol'],3.,'jsplib-easy-v8-selection',weights)
    model.training_groups_=[e['base_instance_id'] for e in examples];model.save(out/'reference.joblib')
    loaded=model.load(out/'reference.joblib',trusted=True);assert loaded.predict(rows[model.feature_backend][:3])==model.predict(rows[model.feature_backend][:3])
    cfg=chosen['config'];manifest.update({'model_version':model.model_version_,'feature_backend':cfg['backend'],'estimator_kind':cfg['kind'],'n_estimators':cfg['trees'],'estimator_params':cfg['params'],'class_decision_weights':weights,'probe_budget':3.,'n_jobs':1})
    (out/'training-manifest.json').write_text(json.dumps(manifest,indent=2));(out/'requirements-model.txt').write_text(''.join(f'{k}=={v}\n' for k,v in model.dependencies_.items()))
    report={'round_e':score(y,predicted),'chosen':chosen,'history':history,'final_scores':results,'seconds':time.perf_counter()-start,'target_achieved':bool(np.mean(y==predicted)>=.95),'checks':{'persistence':True},'limitations':['Previously explored data and several model searches; no new independent test','Variable ranking is fitted only within training folds','15 easy cases yield unstable estimates']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    pd.DataFrame({'instance':[e['source_instance'] for e in examples],'actual':y,'round_e':predicted}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'complete':True,'accuracy':report['round_e']['accuracy'],'easy_recall':report['round_e']['per_class']['easy']['recall'],'target_achieved':report['target_achieved']}),flush=True)


if __name__=='__main__':main()
