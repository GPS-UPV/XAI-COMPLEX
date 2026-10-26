"""Supervised five-class classifier with a persisted labeling protocol."""
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .categories import CATEGORIES, validate_protocol
from .io import load_instance


class JobShopComplexityClassifier:
    """Predict empirical labels; never infer historical open-problem status."""

    def __init__(self, *, n_estimators=200, random_state=42, feature_backend="graph", speed_index=None, betweenness_samples=100, estimator_kind="random_forest", n_jobs=1, estimator_params=None, class_decision_weights=None, probe_budget=.2):
        if feature_backend not in ("graph", "graph_approx", "summary", "structural", "hybrid", "search", "probe", "landscape", "trajectory", "proof", "bottleneck", "proof_bottleneck"):
            raise ValueError("Unsupported feature_backend")
        if estimator_kind not in ("random_forest", "extra_trees", "hist_gradient", "rbf_svm", "ensemble", "catboost", "ordinal_forest", "easy_specialist", "selected_easy", "cumulative", "subset_voting"):
            raise ValueError("Unsupported estimator_kind")
        self.estimator_kind = estimator_kind
        if type(n_jobs) is not int or n_jobs==0 or n_jobs < -1:
            raise ValueError('n_jobs must be -1 or a positive integer')
        self.n_jobs=n_jobs
        self.estimator_params=deepcopy(estimator_params or {})
        self.class_decision_weights=deepcopy(class_decision_weights)
        if class_decision_weights is not None and (set(class_decision_weights)!=set(CATEGORIES) or any(not np.isfinite(v) or v<=0 for v in class_decision_weights.values())):
            raise ValueError('Decision weights must be finite and positive for all classes')
        if not np.isfinite(probe_budget) or probe_budget<=0:
            raise ValueError('probe_budget must be finite and positive')
        self.probe_budget=probe_budget
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_backend = feature_backend
        self.speed_index = speed_index
        if type(betweenness_samples) is not int or betweenness_samples < 1:
            raise ValueError("betweenness_samples must be a positive integer")
        self.betweenness_samples = betweenness_samples

    def _features(self, instance):
        instance = load_instance(instance)
        if instance["instance_type"] != "JSP":
            raise ValueError("Five-class JSP classification does not model flexible alternatives")
        if self.feature_backend == 'proof_bottleneck':
            from .proof_features import proof_features
            from .bottleneck import relaxation_features
            base = proof_features(instance, speed_index=self.speed_index, probe_budget=self.probe_budget)
            features = relaxation_features(instance, speed_index=self.speed_index, base_features=base, budget=.02)
        elif self.feature_backend == 'bottleneck':
            from .landscape import landscape_features
            from .bottleneck import relaxation_features
            base = landscape_features(instance, speed_index=self.speed_index, probe_budget=self.probe_budget)
            features = relaxation_features(instance, speed_index=self.speed_index, base_features=base, budget=.02)
        elif self.feature_backend == 'proof':
            from .proof_features import proof_features
            features = proof_features(instance, speed_index=self.speed_index, probe_budget=self.probe_budget)
        elif self.feature_backend in ('landscape','trajectory'):
            from .landscape import landscape_features
            features=landscape_features(instance,speed_index=self.speed_index,probe_budget=self.probe_budget if self.feature_backend=='trajectory' else None)
        elif self.feature_backend in ('search','probe'):
            from .search_features import search_features
            features=search_features(instance,speed_index=self.speed_index,probe_budget=self.probe_budget if self.feature_backend=='probe' else None)
        elif self.feature_backend == "structural":
            from .structural import structural_features
            features = structural_features(instance, speed_index=self.speed_index)
        elif self.feature_backend in ("graph", "graph_approx", "hybrid"):
            from .graphs import extract_instance_features
            features = extract_instance_features(instance, speed_index=self.speed_index,
                                                 approximate_betweenness=self.feature_backend in ("graph_approx", "hybrid"),
                                                 betweenness_samples=getattr(self, "betweenness_samples", 100))
            if self.feature_backend == "hybrid":
                from .structural import structural_features
                features.update({f"struct_{k}": v for k,v in structural_features(instance, speed_index=self.speed_index).items()})
        else:
            from .instances import summarize_instance
            features = summarize_instance(instance)
        # Exclude energy because Taillard inputs do not declare it.
        return {k: v for k, v in features.items() if "energy" not in k.lower() and not k.startswith("e_value") and k not in ("operation_cost_mean", "operation_cost_std", "seed")}

    def _matrix(self, rows, *, fitting=False):
        rows = list(rows)
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Expected nonempty feature dictionaries")
        if fitting:
            columns = set().union(*(row.keys() for row in rows))
            if not columns or any(not isinstance(k, str) or not k for k in columns):
                raise ValueError("Features require nonempty string names")
            self.columns_ = sorted(columns)
        matrix = np.array([[row.get(k, np.nan) for k in self.columns_] for row in rows], dtype=float)
        if np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
            raise ValueError("Features are infinite or entirely missing")
        return matrix

    def fit(self, features, labels, *, protocol, model_version, sample_weight=None):
        protocol = deepcopy(validate_protocol(protocol))
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version is required")
        labels = list(labels)
        if set(labels) != set(CATEGORIES):
            raise ValueError("Training must contain all five defined classes")
        matrix = self._matrix(features, fitting=True)
        if len(labels) != len(matrix) or any(labels.count(label) < 2 for label in CATEGORIES):
            raise ValueError("Need aligned labels and at least two examples per class; validation requires more")
        if np.isnan(matrix).all(axis=0).any():
            raise ValueError("An entire training feature is missing")
        if np.all(np.nanmax(matrix, axis=0) == np.nanmin(matrix, axis=0)):
            raise ValueError("Training features are all constant")
        from .supervised import make_estimator
        model = make_pipeline(SimpleImputer(strategy="median"), make_estimator(
            getattr(self, "estimator_kind", "random_forest"), self.n_estimators, self.random_state,
            n_jobs=getattr(self,'n_jobs',1),parameters=getattr(self,'estimator_params',None)))
        fit_options = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(labels),) or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("sample_weight must be aligned, finite and positive")
            step = model.steps[-1][0]
            if getattr(self, "estimator_kind", "random_forest") == "rbf_svm":
                step += "__svc"
            fit_options[f"{step}__sample_weight"] = weights
        model.fit(matrix, labels, **fit_options)
        self.model_ = model
        self.protocol_ = protocol
        self.model_version_ = model_version
        self.feature_min_ = np.nanmin(matrix, axis=0)
        self.feature_max_ = np.nanmax(matrix, axis=0)
        self.class_counts_ = {label: labels.count(label) for label in CATEGORIES}
        payload = json.dumps({"columns": self.columns_, "features": matrix.tolist(), "labels": labels, "protocol": protocol}, sort_keys=True)
        self.training_digest_ = sha256(payload.encode()).hexdigest()
        if sample_weight is not None:
            self.sample_weight_digest_ = sha256(np.asarray(sample_weight,dtype=float).tobytes()).hexdigest()
        self.dependencies_ = {p: version(p) for p in ("numpy", "scikit-learn", "joblib", "scipy")}
        if self.feature_backend in ("graph", "graph_approx", "hybrid"):
            self.dependencies_.update({p: version(p) for p in ("torch", "torch-geometric", "networkx", "pandas")})
        if self.feature_backend in ('probe','trajectory','proof','bottleneck','proof_bottleneck'):
            self.dependencies_['ortools']=version('ortools')
        if getattr(self,'estimator_kind','random_forest')=='catboost':
            self.dependencies_['catboost']=version('catboost')
        if getattr(self, 'estimator_kind', None) == 'subset_voting' and any(m['config']['kind']=='cat' for m in self.estimator_params['members']):
            self.dependencies_['catboost'] = version('catboost')
        self.schema_ = "cejsp.five-class.v3" if self.feature_backend in ("structural", "hybrid") or getattr(self, "estimator_kind", "random_forest") != "random_forest" else "cejsp.five-class.v2" if self.feature_backend == "graph_approx" else "cejsp.five-class.v1"
        if self.feature_backend in ('search','probe','landscape','trajectory','proof','bottleneck','proof_bottleneck') or getattr(self,'estimator_kind',None) in ('catboost','ordinal_forest','easy_specialist','selected_easy') or getattr(self,'estimator_params',None) or getattr(self,'class_decision_weights',None):
            self.schema_='cejsp.five-class.v4'
        return self

    def fit_instances(self, instances, labels, *, protocol, model_version):
        return self.fit([self._features(instance) for instance in instances], labels, protocol=protocol, model_version=model_version)

    def predict(self, features):
        if not hasattr(self, "model_"):
            raise ValueError("Train or load a five-class reference first")
        matrix = self._matrix(features)
        probabilities = self.model_.predict_proba(matrix)
        classes = self.model_.classes_
        decision_weights=getattr(self,'class_decision_weights',None)
        if decision_weights:
            probabilities=probabilities*np.array([decision_weights[c] for c in classes])
            probabilities=probabilities/probabilities.sum(axis=1,keepdims=True)
        result = []
        for row, votes in zip(matrix, probabilities):
            outside = np.any((row < self.feature_min_) | (row > self.feature_max_))
            result.append({"category": str(classes[np.argmax(votes)]),
                           "class_probabilities": {label: float(votes[list(classes).index(label)]) for label in CATEGORIES},
                           "probability_calibration": "uncalibrated_weighted_model_votes" if decision_weights else "uncalibrated_forest_votes" if getattr(self,'estimator_kind','random_forest') in ('random_forest','extra_trees') else "uncalibrated_model_votes",
                           "applicability": "incomplete_features" if np.isnan(row).any() else "outside_training_bounds" if outside else "within_training_bounds",
                           "missing_feature_count": int(np.isnan(row).sum()),
                           "model_version": self.model_version_, "label_protocol": self.protocol_["id"],
                           "open_semantics": "predicted_protocol_class_not_verified_open_problem"})
            if decision_weights:
                result[-1]['decision_class_weights']=dict(decision_weights)
        return result

    def classify(self, instance):
        normalized = load_instance(instance)
        result = self.predict([self._features(normalized)])[0]
        result["instance_id"] = normalized["instance_id"]
        return result

    def save(self, path):
        if not hasattr(self, "model_"):
            raise ValueError("Cannot save an unfitted classifier")
        path = Path(path)
        if path.exists():
            raise ValueError("Choose a new path; saved models are not overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path, *, trusted=False):
        if not trusted:
            raise ValueError("Only load trusted joblib models; pass trusted=True")
        result = joblib.load(path)
        if not isinstance(result, cls) or getattr(result, "schema_", None) not in ("cejsp.five-class.v1", "cejsp.five-class.v2", "cejsp.five-class.v3", "cejsp.five-class.v4"):
            raise ValueError("Not a supported five-class model")
        for name, saved in result.dependencies_.items():
            if saved != version(name):
                raise ValueError(f"Model requires {name}=={saved}")
        return result


def train_manifest(path, output, *, previous=None, trusted=False):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema") != "cejsp.training.v1":
        raise ValueError("Expected cejsp.training.v1 manifest")
    validate_protocol(data.get("protocol"))
    examples = data["examples"]
    if not examples or any(not e.get("base_instance_id") for e in examples):
        raise ValueError("Each labeled example requires base_instance_id")
    if set(e.get("label") for e in examples) != set(CATEGORIES):
        raise ValueError("Manifest must contain real labels for all five classes")
    group_labels = {}
    for e in examples:
        group = e["base_instance_id"]
        if group in group_labels and group_labels[group] != e["label"]:
            raise ValueError("Conflicting labels for variants of the same base instance")
        group_labels[group] = e["label"]
    if any(sum(label == category for label in group_labels.values()) < 2 for category in CATEGORIES):
        raise ValueError("Each class needs at least two distinct base instances")
    if Path(output).exists():
        raise ValueError("Choose a new output path; existing model artifacts are not overwritten")
    if previous is not None and Path(previous).resolve() == Path(output).resolve():
        raise ValueError("Retraining must write a new model artifact")
    model = JobShopComplexityClassifier(feature_backend=data.get("feature_backend", "graph"), speed_index=data.get("speed_index"), betweenness_samples=data.get("betweenness_samples", 100), estimator_kind=data.get("estimator_kind", "random_forest"), n_estimators=data.get("n_estimators", 200), n_jobs=data.get('n_jobs',1),estimator_params=data.get('estimator_params'),class_decision_weights=data.get('class_decision_weights'),probe_budget=data.get('probe_budget',.2))
    if previous is not None:
        old = model.load(previous, trusted=trusted)
        if old.model_version_ == data["model_version"]:
            raise ValueError("Use a new model_version for retraining")
        if old.protocol_["id"] == data["protocol"]["id"] and old.protocol_ != data["protocol"]:
            raise ValueError("Changed labeling rules require a new protocol id")
    instances = [path.parent / e["path"] for e in examples]
    model.fit_instances(instances, [e["label"] for e in examples], protocol=data["protocol"], model_version=data["model_version"])
    model.training_manifest_sha256_ = sha256(path.read_bytes()).hexdigest()
    model.training_groups_ = [e["base_instance_id"] for e in examples]
    if previous is not None:
        model.previous_version_ = old.model_version_
    model.save(output)
    return model
