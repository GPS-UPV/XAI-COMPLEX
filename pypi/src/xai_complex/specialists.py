"""Explicit ordinal and easy-specialist alternatives to direct classification."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .categories import CATEGORIES


class OrdinalForestClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_estimators=300,random_state=42,n_jobs=1,max_features=.8,min_samples_leaf=1,sigma=.6):
        self.n_estimators=n_estimators;self.random_state=random_state;self.n_jobs=n_jobs
        self.max_features=max_features;self.min_samples_leaf=min_samples_leaf;self.sigma=sigma

    def fit(self,X,y,sample_weight=None):
        if not np.isfinite(self.sigma) or self.sigma<=0:
            raise ValueError('sigma must be finite and positive')
        y=np.asarray(y);rank=np.array([CATEGORIES.index(v) for v in y],dtype=float)
        counts={c:np.sum(y==c) for c in CATEGORIES}
        weights=np.array([len(y)/(5*counts[c]) for c in y])
        if sample_weight is not None:weights*=sample_weight
        self.model_=ExtraTreesRegressor(n_estimators=self.n_estimators,random_state=self.random_state,n_jobs=self.n_jobs,max_features=self.max_features,min_samples_leaf=self.min_samples_leaf).fit(X,rank,sample_weight=weights)
        self.classes_=np.array(sorted(CATEGORIES));self.n_features_in_=X.shape[1]
        return self

    def predict_proba(self,X):
        scores=self.model_.predict(X)
        centers=np.array([CATEGORIES.index(c) for c in self.classes_])
        logits=-((scores[:,None]-centers[None,:])/self.sigma)**2/2
        logits-=logits.max(axis=1,keepdims=True);values=np.exp(logits)
        return values/values.sum(axis=1,keepdims=True)

    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


class EasySpecialistClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_estimators=300,random_state=42,n_jobs=1,specialist_kind='forest',max_features=.8):
        self.n_estimators=n_estimators;self.random_state=random_state;self.n_jobs=n_jobs
        self.specialist_kind=specialist_kind;self.max_features=max_features

    def fit(self,X,y,sample_weight=None):
        y=np.asarray(y)
        self.base_=ExtraTreesClassifier(n_estimators=self.n_estimators,random_state=self.random_state,n_jobs=self.n_jobs,max_features=self.max_features,class_weight='balanced').fit(X,y,sample_weight=sample_weight)
        binary=(y=='easy').astype(int)
        if self.specialist_kind=='svm':
            self.specialist_=make_pipeline(StandardScaler(),SVC(C=3,class_weight='balanced',probability=True,random_state=self.random_state))
            self.specialist_.fit(X,binary,**({'svc__sample_weight':sample_weight} if sample_weight is not None else {}))
        elif self.specialist_kind=='forest':
            self.specialist_=ExtraTreesClassifier(n_estimators=self.n_estimators,random_state=self.random_state,n_jobs=self.n_jobs,max_features=self.max_features,class_weight='balanced').fit(X,binary,sample_weight=sample_weight)
        else:raise ValueError('Unsupported easy specialist')
        self.classes_=self.base_.classes_;self.n_features_in_=X.shape[1]
        return self

    def predict_proba(self,X):
        values=self.base_.predict_proba(X);idx=list(self.classes_).index('easy')
        easy=self.specialist_.predict_proba(X)[:,list(self.specialist_.classes_).index(1)]
        values[:,idx]=0
        denominator=values.sum(axis=1,keepdims=True)
        values=np.divide(values,denominator,out=np.zeros_like(values),where=denominator>0)
        empty=denominator[:,0]==0
        if empty.any():values[empty]=np.array([0. if c=='easy' else .25 for c in self.classes_])
        values*=1-easy[:,None];values[:,idx]=easy
        return values

    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
