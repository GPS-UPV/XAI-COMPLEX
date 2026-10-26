"""Train CEJSP on source JSPLIB labels with frozen, reproducible splits."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import re
import time


class LabelTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.labels = {}
        self.cells = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        if tag == "td":
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.cells.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and len(self.cells) >= 7:
            name, label = self.cells[0], self.cells[5].lower()
            if label in ("toy", "easy", "medium", "hard", "open", "closed"):
                if name in self.labels and self.labels[name] != label:
                    raise ValueError(f"Conflicting source labels for {name}")
                self.labels[name] = label


def import_dataset(snapshot, out, *, max_operations=2500):
    from cejsp.io import load_instance
    parser = LabelTable()
    parser.feed((snapshot / "README.upstream.md").read_text(encoding="utf-8"))
    index = json.loads((snapshot / "upstream-json-index.json").read_text(encoding="utf-8"))
    expected_hashes = {entry["name"]: entry["sha"] for entry in index}
    if not parser.labels:
        raise ValueError("No source classification rows found")
    (out / "source-labels.json").write_text(json.dumps(parser.labels, indent=2), encoding="utf-8")
    examples, excluded = [], []
    groups = {}
    for name, label in sorted(parser.labels.items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe instance identifier in source classification")
        path = snapshot / "raw" / f"{name}.json"
        reason = None
        if label == "closed":
            reason = "closed_is_not_one_of_the_five_target_classes"
        elif not path.exists():
            reason = "no_uncompressed_JSON_available_in_downloaded_snapshot"
        else:
            try:
                raw = path.read_bytes()
                git_sha = sha256(raw).hexdigest()
                from hashlib import sha1
                if sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != expected_hashes.get(path.name):
                    raise ValueError("raw_file_differs_from_pinned_source_hash")
                source = json.loads(raw.decode("utf-8-sig"))
                if source.get("instance") != name:
                    raise ValueError("Instance name differs from source file")
                if len(source["operations"]) > max_operations:
                    raise ValueError(f"resource_limit_above_{max_operations}_operations")
                normalized = load_instance(source)
                m = normalized["n_machines"]
                if any(sorted(o["machine"] for o in job["operations"]) != list(range(m)) for job in normalized["jobs"]):
                    raise ValueError("non_permutation_or_reentrant_routing_outside_original_graph_cohort")
                group = normalized["instance_id"]
                if group in groups:
                    if groups[group] != label:
                        raise ValueError("conflicting_labels_for_identical_routing_and_durations")
                    raise ValueError("duplicate_instance_content")
                groups[group] = label
                examples.append({"path": str(path.resolve()), "label": label, "base_instance_id": group,
                                 "source_instance": name, "family": str(source["family"]),
                                 "sha256": git_sha})
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                reason = str(exc)
        if reason:
            excluded.append({"instance": name, "label": label, "reason": reason})
    protocol = {
        "id": f"jsplib-{snapshot.name[:12]}",
        "source_url": "https://scheduleopt.github.io/benchmarks/jsplib",
        "source_commit": snapshot.name,
        "source_license": "CC-BY-SA-4.0",
        "class_definitions": {
            "toy": "Optimality proved within one minute by at least one JSPLIB reference engine.",
            "easy": "Optimality proved within ten minutes by at least one reference engine, source easy class.",
            "medium": "Optimality proved within one hour by at least one reference engine, source medium class.",
            "hard": "Optimality proved after more than one hour by at least one reference engine.",
            "open": "No proof of optimality reported in the pinned JSPLIB classification."},
        "label_policy": "Use source table labels verbatim; never relabel closed; no lookup of instance name during prediction."}
    manifest = {"schema": "cejsp.training.v1", "model_version": f"jsplib-{snapshot.name[:12]}-v1",
                "feature_backend": "graph_approx", "protocol": protocol, "examples": examples}
    (out / "training-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"source_labels": len(parser.labels), "source_class_counts": dict(Counter(parser.labels.values())),
              "training_cohort": len(examples), "cohort_class_counts": dict(Counter(e["label"] for e in examples)),
              "excluded": excluded, "max_operations": max_operations, "source_commit": snapshot.name}
    (out / "dataset-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest, report


def score(y, pred):
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, accuracy_score
    from cejsp import CATEGORIES
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(CATEGORIES), average="macro", zero_division=0)),
            "class_order": list(CATEGORIES), "confusion_matrix": confusion_matrix(y, pred, labels=list(CATEGORIES)).tolist(),
            "per_class": classification_report(y, pred, labels=list(CATEGORIES), zero_division=0, output_dict=True)}


def train(snapshot, out, *, max_operations, trees):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
    from threadpoolctl import threadpool_limits
    from cejsp import JobShopComplexityClassifier, CATEGORIES
    from cejsp.io import load_instance
    manifest, dataset_report = import_dataset(snapshot, out, max_operations=max_operations)
    print(json.dumps({k: dataset_report[k] for k in ("source_labels", "training_cohort", "cohort_class_counts")}), flush=True)
    examples = manifest["examples"]
    labels = np.array([e["label"] for e in examples])
    if set(labels) != set(CATEGORIES) or min(Counter(labels).values()) < 5:
        raise ValueError("Need at least five independent base instances per class for this protocol")
    split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=20261022)
    train_ids, test_ids = next(split.split(examples, labels))
    if set(examples[i]["base_instance_id"] for i in train_ids) & set(examples[i]["base_instance_id"] for i in test_ids):
        raise AssertionError("Base-instance leakage")
    split_manifest = {"seed": 20261022, "train": [examples[i]["source_instance"] for i in train_ids],
                      "test": [examples[i]["source_instance"] for i in test_ids],
                      "test_use": "single fixed evaluation, no hyperparameter selection"}
    (out / "split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    feature_model = JobShopComplexityClassifier(n_estimators=trees, feature_backend="graph_approx")
    import importlib.metadata
    source_root = Path(__file__).resolve().parents[1] / "pypi/src/xai_complex"
    source_hashes = {name: sha256((source_root / name).read_bytes()).hexdigest() for name in
                     ("_graph_features.py", "_graph_builder.py", "graphs.py", "io.py", "instances.py", "classifier.py")}
    cache_key = sha256(json.dumps({"examples": examples, "backend": "graph_approx", "betweenness_samples": 100,
                                 "source_hashes": source_hashes,
                                 "versions": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "torch-geometric", "networkx")}}, sort_keys=True).encode()).hexdigest()
    feature_cache = snapshot / f"features-{cache_key}.json"
    start = time.perf_counter()
    if feature_cache.exists():
        features = json.loads(feature_cache.read_text(encoding="utf-8"))
    else:
        features = []
        for i, example in enumerate(examples):
            features.append(feature_model._features(example["path"]))
            if (i + 1) % 10 == 0:
                print(f"Graph features {i + 1}/{len(examples)}", flush=True)
        feature_cache.write_text(json.dumps(features), encoding="utf-8")
    feature_seconds = time.perf_counter() - start
    sizes = np.array([[load_instance(e["path"])["n_machines"], len(load_instance(e["path"])["jobs"])] for e in examples])
    sizes = np.column_stack([sizes, sizes[:, 0] * sizes[:, 1]])
    def new_model():
        return JobShopComplexityClassifier(n_estimators=trees, random_state=42, feature_backend="graph_approx")
    with threadpool_limits(limits=1):
        evaluation = new_model().fit([features[i] for i in train_ids], labels[train_ids], protocol=manifest["protocol"], model_version=manifest["model_version"] + "-evaluation")
        predicted = [r["category"] for r in evaluation.predict([features[i] for i in test_ids])]
        baselines = {}
        for name, model in [("majority", DummyClassifier(strategy="most_frequent")),
                            ("size_forest", RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=42, n_jobs=1))]:
            model.fit(sizes[train_ids], labels[train_ids])
            baselines[name] = score(labels[test_ids], model.predict(sizes[test_ids]))
            if name == "size_forest":
                size_predictions = model.predict(sizes[test_ids]).tolist()
        evaluation.save(out / "evaluation-reference.joblib")
        family_pred, family_true, family_names = [], [], []
        families = np.array([e["family"] for e in examples])
        family_details = []
        for tr, te in LeaveOneGroupOut().split(features, labels, groups=families):
            if set(labels[tr]) != set(CATEGORIES) or min(Counter(labels[tr]).values()) < 2:
                family_details.append({"family": str(families[te[0]]), "state": "not_evaluable_missing_training_class"})
                continue
            model = new_model().fit([features[i] for i in tr], labels[tr], protocol=manifest["protocol"], model_version="family-evaluation")
            p = [r["category"] for r in model.predict([features[i] for i in te])]
            family_pred.extend(p)
            family_true.extend(labels[te].tolist())
            family_names.extend(families[te].tolist())
            family_details.append({"family": str(families[te[0]]), "test_n": len(te), "accuracy": sum(a == b for a, b in zip(labels[te], p)) / len(te), "state": "evaluated"})
        final = new_model().fit(features, labels, protocol=manifest["protocol"], model_version=manifest["model_version"])
        final.training_manifest_sha256_ = sha256((out / "training-manifest.json").read_bytes()).hexdigest()
        final.training_groups_ = [e["base_instance_id"] for e in examples]
        final_path = final.save(out / "reference.joblib")
        loaded = final.load(final_path, trusted=True)
        query = features[int(test_ids[0])]
        if loaded.predict([query]) != final.predict([query]):
            raise AssertionError("Persistence changed class predictions")
        if final.predict([query])[0] != final.predict([query, features[0]])[0]:
            raise AssertionError("Batch-dependent class predictions")
        example_result = loaded.classify(examples[int(test_ids[0])]["path"])
    import importlib.metadata
    class_ids = {name: i for i, name in enumerate(CATEGORIES)}
    actual_ids = np.array([class_ids[v] for v in labels[test_ids]])
    pred_ids = np.array([class_ids[v] for v in predicted])
    size_ids = np.array([class_ids[v] for v in size_predictions])
    def macro_f1(y, p):
        confusion = np.bincount(y * 5 + p, minlength=25).reshape(5, 5)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        return float(np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(5), where=denominator > 0).mean())
    rng = np.random.default_rng(42)
    bootstrap, difference = [], []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(np.flatnonzero(actual_ids == k), size=np.sum(actual_ids == k), replace=True) for k in range(5)])
        value = macro_f1(actual_ids[indices], pred_ids[indices])
        bootstrap.append(value)
        difference.append(value - macro_f1(actual_ids[indices], size_ids[indices]))
    report = {"source_commit": snapshot.name, "model_path": str(final_path.resolve()),
              "model_version": final.model_version_, "dataset": {k: dataset_report[k] for k in ("training_cohort", "cohort_class_counts", "source_labels")},
              "feature_backend": "graph_approx", "betweenness_samples": 100, "betweenness_seed": 42, "trees": trees,
              "feature_seconds": feature_seconds, "heldout_n": len(test_ids), "heldout_cejsp": score(labels[test_ids], predicted),
              "heldout_baselines": baselines, "leave_family_out": score(family_true, family_pred) if family_true else None,
              "heldout_uncertainty": {"method": "2000 paired stratified bootstrap resamples of independent test instances; conditional on fitted models",
                                      "macro_f1_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                      "macro_f1_minus_size_forest_ci95": np.quantile(difference, [.025, .975]).tolist()},
              "family_folds": family_details, "model_file_sha256": sha256(final_path.read_bytes()).hexdigest(),
              "checks": {"persistence": True, "batch_invariance": True, "single_file_classification": example_result},
              "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "scipy", "torch", "torch-geometric")},
              "feature_source_hashes": source_hashes,
              "limitations": ["Source difficulty labels are snapshot- and reference-engine-dependent", "open is a prediction, not a proof of unsolved status", "No IGJSP difficulty ground truth evaluated", "Reentrant, compressed-only and resource-exceeding cases excluded with reasons", "Final deployment reference refitted on all eligible cases after evaluation; its training accuracy is not a test metric", "Forest votes are not calibrated probabilities"]}
    pd.DataFrame([{"instance": examples[i]["source_instance"], "actual": labels[i], "predicted": pred} for i, pred in zip(test_ids, predicted)]).to_csv(out / "heldout-predictions.csv", index=False)
    (out / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"state": "training_complete", "model": str(final_path), "test_macro_f1": report["heldout_cejsp"]["macro_f1"], "test_balanced_accuracy": report["heldout_cejsp"]["balanced_accuracy"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-operations", type=int, default=2500)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.import_only:
        _, report = import_dataset(args.snapshot.resolve(), args.out.resolve(), max_operations=args.max_operations)
        print(json.dumps({k: report[k] for k in ("source_labels", "training_cohort", "cohort_class_counts")}), flush=True)
    else:
        train(args.snapshot.resolve(), args.out.resolve(), max_operations=args.max_operations, trees=args.trees)


if __name__ == "__main__":
    main()
