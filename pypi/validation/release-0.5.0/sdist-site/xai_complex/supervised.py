"""Reproducible supervised candidates; preprocessing is fitted in each fold."""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_estimator(kind, trees=200, seed=42, *, n_jobs=1, parameters=None):
    parameters=dict(parameters or {})
    if kind == 'subset_voting':
        from .subset_voting import SubsetVotingClassifier
        return SubsetVotingClassifier(**parameters)
    if kind == 'cumulative':
        from .cumulative import CumulativeDifficultyClassifier
        return CumulativeDifficultyClassifier(n_estimators=trees, random_state=seed, n_jobs=n_jobs, **parameters)
    if kind=='selected_easy':
        from .feature_selector import SelectedEasyClassifier
        return SelectedEasyClassifier(n_estimators=trees,random_state=seed,n_jobs=n_jobs,**parameters)
    if kind in ('ordinal_forest','easy_specialist'):
        from .specialists import OrdinalForestClassifier,EasySpecialistClassifier
        cls=OrdinalForestClassifier if kind=='ordinal_forest' else EasySpecialistClassifier
        return cls(n_estimators=trees,random_state=seed,n_jobs=n_jobs,**parameters)
    if kind == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=trees,random_seed=seed,thread_count=n_jobs,verbose=False,allow_writing_files=False,**parameters)
    rf = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=n_jobs, class_weight="balanced")
    et = ExtraTreesClassifier(n_estimators=trees, random_state=seed, n_jobs=n_jobs, class_weight="balanced", max_features=.8)
    gb = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, min_samples_leaf=5, learning_rate=.06, l2_regularization=2, random_state=seed)
    if kind == "random_forest":
        return rf.set_params(**parameters)
    if kind == "extra_trees":
        return et.set_params(**parameters)
    if kind == "hist_gradient":
        return gb.set_params(**parameters)
    if kind == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=seed).set_params(**parameters))
    if kind == "ensemble":
        return VotingClassifier([("rf",rf),("et",et),("gb",gb)], voting="soft").set_params(**parameters)
    raise ValueError("Unsupported estimator_kind")
