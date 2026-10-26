"""Ordered difficulty thresholds learned from labels, not benchmark times."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import isotonic_regression
from .categories import CATEGORIES


class CumulativeDifficultyClassifier(ClassifierMixin, BaseEstimator):
    """Learn P(class > k); enforce ordered tails before taking differences.

    Thresholds refer to the source protocol, not local seconds. The last
    threshold separates historical open labels; it is not a runtime cap.
    """
    def __init__(self, kind='extra_trees', n_estimators=300, max_features=.8,
                 min_samples_leaf=1, balanced=False, random_state=42, n_jobs=1):
        self.kind = kind
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.balanced = balanced
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        rank = np.array([CATEGORIES.index(v) for v in y])
        self.classes_ = np.array(sorted(CATEGORIES))
        self.n_features_in_ = X.shape[1]
        self.models_ = []
        for threshold in range(4):
            target = (rank > threshold).astype(int)
            if np.unique(target).size != 2:
                raise ValueError('Each cumulative threshold requires both outcomes')
            if self.kind == 'extra_trees':
                model = ExtraTreesClassifier(
                    n_estimators=self.n_estimators, max_features=self.max_features,
                    min_samples_leaf=self.min_samples_leaf, n_jobs=self.n_jobs,
                    class_weight='balanced' if self.balanced else None,
                    random_state=self.random_state)
            elif self.kind == 'hist_gradient':
                model = HistGradientBoostingClassifier(
                    max_iter=self.n_estimators, max_leaf_nodes=7,
                    min_samples_leaf=max(self.min_samples_leaf, 5),
                    l2_regularization=5., learning_rate=.05,
                    class_weight='balanced' if self.balanced else None,
                    random_state=self.random_state)
            else:
                raise ValueError('Unsupported cumulative estimator')
            model.fit(X, target, sample_weight=sample_weight)
            self.models_.append(model)
        return self

    @staticmethod
    def ordered_probabilities(tails):
        tails = np.asarray(tails, dtype=float)
        if tails.ndim != 2 or tails.shape[1] != 4 or not np.isfinite(tails).all():
            raise ValueError('Expected four finite cumulative probabilities per row')
        ordered = np.array([isotonic_regression(np.clip(row, 0., 1.), increasing=False)
                            for row in tails])
        boundaries = np.column_stack([np.ones(len(tails)), ordered, np.zeros(len(tails))])
        return boundaries[:, :-1] - boundaries[:, 1:]

    def predict_proba(self, X):
        tails = np.column_stack([m.predict_proba(X)[:, list(m.classes_).index(1)]
                                 for m in self.models_])
        probabilities = self.ordered_probabilities(tails)
        return probabilities[:, [CATEGORIES.index(c) for c in self.classes_]]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
