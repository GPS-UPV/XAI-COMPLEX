"""Feature selection fitted only on each training fold, with an easy component."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif


class SelectedEasyClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_estimators=200,random_state=42,n_jobs=1,k=32,easy_weight=.5,max_features=.8):
        self.n_estimators=n_estimators;self.random_state=random_state;self.n_jobs=n_jobs
        self.k=k;self.easy_weight=easy_weight;self.max_features=max_features

    def fit(self,X,y,sample_weight=None):
        if type(self.k) is not int or self.k<1 or not 0<=self.easy_weight<=1:
            raise ValueError('Invalid feature selection parameters')
        y=np.asarray(y)
        multi=mutual_info_classif(X,y,random_state=self.random_state)
        easy=mutual_info_classif(X,y=='easy',random_state=self.random_state)
        # The binary component prioritizes signals of the minority class.
        self.selection_scores_=(1-self.easy_weight)*multi+self.easy_weight*easy
        self.selected_=np.argsort(-self.selection_scores_,kind='stable')[:min(self.k,X.shape[1])]
        self.model_=ExtraTreesClassifier(n_estimators=self.n_estimators,random_state=self.random_state,n_jobs=self.n_jobs,max_features=self.max_features,class_weight='balanced').fit(X[:,self.selected_],y,sample_weight=sample_weight)
        self.classes_=self.model_.classes_;self.n_features_in_=X.shape[1]
        return self

    def predict_proba(self,X):return self.model_.predict_proba(X[:,self.selected_])
    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
