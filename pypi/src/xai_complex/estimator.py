"""Inductive version of the research ensemble, with frozen calibration."""

from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .labels import SolverTargetCalibrator


def _records(rows):
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict(orient="records")
    rows = list(rows)
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty sequence of feature dictionaries")
    return rows


class ComplexityEstimator:
    """fit learns reference scales; predict never refits on incoming instances.

    Unsupervised scores measure structural atypicality, not proven solver hardness.
    Use y from an independent solver protocol to add supervised calibration.
    """

    def __init__(self, random_state=42, n_estimators=200):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def _matrix(self, rows, fitting=False):
        rows = _records(rows)
        if fitting:
            self.columns_ = sorted(set().union(*(row.keys() for row in rows)))
            if not self.columns_ or any(not isinstance(c, str) for c in self.columns_):
                raise ValueError("Feature names must be nonempty strings")
        expected = set(self.columns_)
        if any(set(row) != expected for row in rows):
            raise ValueError("Feature schema differs from training; missing and extra columns are rejected")
        try:
            matrix = np.asarray([[row[c] for c in self.columns_] for row in rows], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Features must be numeric or missing (None/NaN)") from exc
        if np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
            raise ValueError("Infinite or entirely missing feature row")
        return matrix

    def fit(self, rows, y=None):
        matrix = self._matrix(rows, fitting=True)
        if len(matrix) < 3:
            raise ValueError("At least three reference instances are required")
        if np.isnan(matrix).all(axis=0).any():
            raise ValueError("Entirely missing feature column")
        self.preprocessor_ = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
        x = self.preprocessor_.fit_transform(matrix)
        if np.all(np.ptp(x, axis=0) == 0):
            raise ValueError("Reference features are all constant")
        k = min(20, len(x) - 1)
        self.iforest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=1).fit(x)
        self.lof_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(x)
        self.covariance_ = MinCovDet(random_state=self.random_state).fit(x)
        self.knn_ = NearestNeighbors(n_neighbors=k).fit(x)
        raw = self._raw_scores(x)
        self.score_low_ = raw.min(axis=0)
        self.score_range_ = raw.max(axis=0) - self.score_low_
        self.supervised_ = None
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.shape != (len(x),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError("y must contain one finite [0,1] target per reference row")
            if np.ptp(y) == 0:
                raise ValueError("Supervised target is constant")
            self.supervised_ = RandomForestRegressor(n_estimators=self.n_estimators, max_features="sqrt", random_state=self.random_state, n_jobs=1).fit(x, y)
        scores = self._combine(x)
        self.thresholds_ = np.quantile(scores, [.33, .66])
        self.dependencies_ = {k: version(k) for k in ("numpy", "scikit-learn", "joblib", "scipy")}
        self.schema_version_ = 1
        return self

    def _raw_scores(self, x):
        return np.column_stack((-self.iforest_.score_samples(x), -self.lof_.score_samples(x),
                                np.sqrt(np.maximum(self.covariance_.mahalanobis(x), 0)),
                                self.knn_.kneighbors(x)[0].mean(axis=1)))

    def _combine(self, x):
        raw = self._raw_scores(x)
        normalized = np.divide(raw - self.score_low_, self.score_range_, out=np.zeros_like(raw), where=self.score_range_ > 0)
        score = np.clip(normalized, 0, 1).mean(axis=1)
        if self.supervised_ is not None:
            score = .6 * score + .4 * self.supervised_.predict(x)
        return np.clip(score, 0, 1)

    def predict(self, rows):
        if not hasattr(self, "thresholds_"):
            raise ValueError("Fit or load a reference model before predicting")
        x = self.preprocessor_.transform(self._matrix(rows))
        scores = self._combine(x)
        categories = np.where(scores <= self.thresholds_[0], "easy", np.where(scores <= self.thresholds_[1], "medium", "hard"))
        return [{"complexity_0_1": float(s), "category": str(c),
                 "signal": "solver_calibrated_blend" if self.supervised_ is not None else "structural_atypicality"} for s, c in zip(scores, categories)]

    def fit_instances(self, instances, y=None, *, speed_index=None):
        from .graphs import extract_instance_features
        return self.fit([extract_instance_features(i, speed_index=speed_index) for i in instances], y)

    def predict_instances(self, instances, *, speed_index=None):
        from .graphs import extract_instance_features
        instances = list(instances)
        results = self.predict([extract_instance_features(i, speed_index=speed_index) for i in instances])
        for inst, result in zip(instances, results):
            if isinstance(inst, dict) and "instance_id" in inst:
                result["instance_id"] = inst["instance_id"]
        return results

    def save(self, path):
        if not hasattr(self, "thresholds_"):
            raise ValueError("Cannot save an unfitted estimator")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path, *, trusted=False):
        """joblib can execute code: explicit trust is required before deserialization."""
        if not trusted:
            raise ValueError("Only load trusted model files; pass trusted=True")
        obj = joblib.load(path)
        if not isinstance(obj, cls) or getattr(obj, "schema_version_", None) != 1:
            raise ValueError("Unsupported model artifact")
        for package, saved in obj.dependencies_.items():
            if saved != version(package):
                raise ValueError(f"Model needs {package}=={saved}, found {version(package)}")
        return obj


def evaluate_supervised(features, solver_rows, *, groups=None, n_splits=5, random_state=42, n_estimators=200):
    """Evaluate solver target prediction with target and preprocessing fit per fold.

    Pass base-instance groups so variants stay in the same fold. Returned errors
    concern the solver target regressor, not the unsupervised ensemble.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.feature_extraction import DictVectorizer

    rows = _records(features)
    solver_rows = list(solver_rows)
    if groups is None and any("base_instance_id" in row for row in solver_rows):
        if not all(row.get("base_instance_id") is not None for row in solver_rows):
            raise ValueError("Every solver record must supply base_instance_id")
        groups = [row["base_instance_id"] for row in solver_rows]
    if len(rows) != len(solver_rows) or len(rows) < n_splits * 2 or n_splits < 2:
        raise ValueError("Need at least two samples per fold and aligned solver records")
    if groups is None:
        warnings.warn("No base-instance groups supplied; variants may cross folds", UserWarning, stacklevel=2)
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(rows)
    else:
        if len(groups) != len(rows):
            raise ValueError("Group count differs from row count")
        if any(g is None or (isinstance(g, float) and not np.isfinite(g)) for g in groups):
            raise ValueError("Base-instance groups cannot be missing")
        splits = GroupKFold(n_splits=n_splits).split(rows, groups=groups)
    oof, targets = np.empty(len(rows)), np.empty(len(rows))
    folds = []
    fold_metrics = {"r2": [], "mae": [], "rmse": []}
    for train, test in splits:
        if len(test) < 2:
            raise ValueError("Every validation fold needs at least two records for R2")
        folds.append({"train": train.tolist(), "test": test.tolist()})
        calibrator = SolverTargetCalibrator().fit([solver_rows[i] for i in train])
        y_train = calibrator.transform([solver_rows[i] for i in train])
        targets[test] = calibrator.transform([solver_rows[i] for i in test])
        if np.ptp(y_train) == 0:
            raise ValueError("Constant target in a training fold")
        model = make_pipeline(DictVectorizer(sparse=False), SimpleImputer(strategy="median"), RobustScaler(),
                              RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt", random_state=random_state, n_jobs=1))
        model.fit([rows[i] for i in train], y_train)
        oof[test] = model.predict([rows[i] for i in test])
        fold_metrics["r2"].append(r2_score(targets[test], oof[test]))
        fold_metrics["mae"].append(mean_absolute_error(targets[test], oof[test]))
        fold_metrics["rmse"].append(np.sqrt(mean_squared_error(targets[test], oof[test])))
    result = {"r2": float(r2_score(targets, oof)), "mae": float(mean_absolute_error(targets, oof)),
            "rmse": float(np.sqrt(mean_squared_error(targets, oof))), "oof": oof.tolist(),
            "targets": targets.tolist(), "grouped": groups is not None, "folds": folds,
            "target_protocol": "training-fold solver calibration v1"}
    for name, values in fold_metrics.items():
        result[f"fold_{name}_mean"] = float(np.mean(values))
        result[f"fold_{name}_std"] = float(np.std(values, ddof=1))
    return result
