"""Cross-fitted single-feature associations with easy, never causal attribution."""
from pathlib import Path
import argparse
from html.parser import HTMLParser
import json
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


def association(name,x,y,folds,labels):
    probabilities=np.zeros(len(y));nonlinear=np.zeros(len(y))
    with threadpool_limits(limits=1):
        for tr,te in folds:
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(class_weight='balanced',solver='liblinear',C=1))
            model.fit(x[tr,None],y[tr]);probabilities[te]=model.predict_proba(x[te,None])[:,1]
            tree=make_pipeline(SimpleImputer(strategy='median'),DecisionTreeClassifier(max_depth=2,min_samples_leaf=5,class_weight='balanced',random_state=42))
            tree.fit(x[tr,None],y[tr]);nonlinear[te]=tree.predict_proba(x[te,None])[:,1]
    return {'feature':name,'crossfit_easy_auc':float(roc_auc_score(y,probabilities)),
            'crossfit_easy_nonlinear_auc':float(roc_auc_score(y,nonlinear)),
            **{f'median_{label}':float(np.median(x[labels==label])) for label in sorted(set(labels))}}


class Bounds(HTMLParser):
    def __init__(self):
        super().__init__();self.cells=[];self.cell=None;self.bounds={}
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.cells=[]
        if tag=='td':self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag=='td' and self.cell is not None:self.cells.append(''.join(self.cell).strip());self.cell=None
        if tag=='tr' and len(self.cells)==6:
            try:self.bounds[self.cells[0]]=float(self.cells[-1].replace(',',''))
            except ValueError:pass


def main():
    root=Path(__file__).resolve().parents[1]
    parser_args=argparse.ArgumentParser();parser_args.add_argument('--study',type=Path,default=root/'models/jsplib-easy-v5');args=parser_args.parse_args();out=args.study.resolve()
    stored=json.loads((out/'features.json').read_text());full=stored['rows'] if isinstance(stored,dict) else stored
    manifest=json.loads((root/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text());examples=manifest['examples']
    labels=np.array([e['label'] for e in examples]);y=labels=='easy'
    folds=list(StratifiedKFold(5,shuffle=True,random_state=20264000).split(full,y))
    results=Parallel(n_jobs=4)(delayed(association)(name,np.array([r[name] for r in full]),y,folds,labels) for name in sorted(full[0]))
    frame=pd.DataFrame(results).sort_values('crossfit_easy_nonlinear_auc',ascending=False)
    frame.to_csv(out/'easy-feature-associations.csv',index=False)
    parser=Bounds();parser.feed((root/'data/jsplib'/manifest['protocol']['source_commit']/'README.upstream.md').read_text(encoding='utf-8'))
    checked=[]
    for e,features in zip(examples,full):
        if e['source_instance'] in parser.bounds:
            lower=features['load_lower_bound']*features['schrage_lb_relative'];upper=parser.bounds[e['source_instance']]
            if lower>upper+1e-7:raise AssertionError(f'Invalid lower bound: {e["source_instance"]}')
            checked.append({'instance':e['source_instance'],'computed_lower_bound':lower,'source_bks_upper_bound':upper})
    (out/'bounds-validation.json').write_text(json.dumps({'checked':len(checked),'passed':True,'source_bks_used_for_validation_only':True,'cases':checked},indent=2))
    (out/'feature-study.json').write_text(json.dumps({'features':len(full[0]),'easy_cases':int(y.sum()),'non_easy_cases':int((~y).sum()),'crossfit_folds':5,'top_associations':results if len(results)<10 else frame.head(20).to_dict(orient='records'),
                                                    'interpretation':'Post-hoc exploratory binary associations; correlated features and family effects; not independent causal determinants or new test evidence'},indent=2))
    print(frame[['feature','crossfit_easy_auc','crossfit_easy_nonlinear_auc']].head(12).to_string(index=False))
    print('Lower bounds checked:',len(checked))


if __name__=='__main__':main()
