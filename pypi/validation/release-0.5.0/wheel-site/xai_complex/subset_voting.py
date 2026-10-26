"""A soft-vote ensemble with explicit, persisted feature subsets."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def member_estimator(config):
    kind=config['kind']
    if kind=='et':
        core=ExtraTreesClassifier(n_estimators=300,max_features=config['fraction'],class_weight='balanced' if config['balanced'] else None,random_state=42,n_jobs=1)
    elif kind=='rf':
        core=RandomForestClassifier(n_estimators=300,max_features=.8,random_state=42,n_jobs=1)
    elif kind=='cat':
        from catboost import CatBoostClassifier
        core=CatBoostClassifier(iterations=450,depth=config['depth'],learning_rate=.05,l2_leaf_reg=6,thread_count=1,verbose=False,allow_writing_files=False,random_seed=42)
    elif kind=='hist':
        core=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=7,min_samples_leaf=5,l2_regularization=5,random_state=42)
    elif kind=='svm':
        core=make_pipeline(StandardScaler(),SVC(C=config['C'],probability=True,random_state=42))
    else:
        raise ValueError('Unsupported ensemble member')
    return make_pipeline(SimpleImputer(strategy='median'),core)


class SubsetVotingClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,members):
        self.members=members

    def fit(self,X,y,sample_weight=None):
        if not self.members:
            raise ValueError('Specify at least one member')
        self.classes_=np.unique(y);self.n_features_in_=X.shape[1]
        self.models_=[]
        for member in self.members:
            indices=np.asarray(member['indices'],dtype=int)
            if not len(indices) or (indices<0).any() or (indices>=X.shape[1]).any():
                raise ValueError('Invalid ensemble feature subset')
            model=member_estimator(member['config'])
            options={}
            if sample_weight is not None:
                step=model.steps[-1][0]
                if member['config']['kind']=='svm':step+='__svc'
                options[step+'__sample_weight']=sample_weight
            model.fit(X[:,indices],y,**options)
            self.models_.append((indices,model))
        return self

    def predict_proba(self,X):
        probabilities=[]
        for indices,model in self.models_:
            p=model.predict_proba(X[:,indices])
            probabilities.append(p[:,[list(model.classes_).index(c) for c in self.classes_]])
        return np.mean(probabilities,axis=0)

    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
