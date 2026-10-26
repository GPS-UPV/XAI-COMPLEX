import argparse
import json
from pathlib import Path



def main(argv=None):
    parser = argparse.ArgumentParser(prog="cejsp", description="CEJSP: JSP Complexity Estimator")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("input", help="JSON list of numeric feature dictionaries")
    fit.add_argument("model")
    fit.add_argument("--targets", help="Optional JSON list of calibrated [0,1] solver targets")
    predict = sub.add_parser("predict")
    predict.add_argument("input")
    predict.add_argument("model")
    predict.add_argument("--trust-model", action="store_true", help="Allow loading a trusted joblib file")
    classify = sub.add_parser("classify", help="Classify one JSON, DZN or Taillard instance into five empirical classes")
    classify.add_argument("input")
    classify.add_argument("model")
    classify.add_argument("--trust-model", action="store_true")
    for name in ("train", "retrain"):
        train = sub.add_parser(name, help="Fit a versioned five-class reference from a labeled manifest")
        train.add_argument("manifest")
        train.add_argument("model")
        if name == "retrain":
            train.add_argument("--previous", required=True)
            train.add_argument("--trust-model", action="store_true")
    inspect = sub.add_parser("inspect", help="Read and normalize an instance without training or scientific dependencies")
    inspect.add_argument("input")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            from .io import load_instance
            print(json.dumps(load_instance(args.input), indent=2, allow_nan=False))
            return
        if args.command == "classify":
            from .classifier import JobShopComplexityClassifier
            model = JobShopComplexityClassifier.load(args.model, trusted=args.trust_model)
            print(json.dumps(model.classify(args.input), indent=2, allow_nan=False))
            return
        if args.command in ("train", "retrain"):
            from .classifier import train_manifest
            model = train_manifest(args.manifest, args.model, previous=getattr(args, "previous", None), trusted=getattr(args, "trust_model", False))
            print(json.dumps({"model_version": model.model_version_, "classes": model.class_counts_, "training_digest": model.training_digest_}, indent=2))
            return
        from .estimator import ComplexityEstimator
        rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if args.command == "fit":
            y = json.loads(Path(args.targets).read_text(encoding="utf-8")) if args.targets else None
            ComplexityEstimator().fit(rows, y).save(args.model)
        else:
            model = ComplexityEstimator.load(args.model, trusted=args.trust_model)
            print(json.dumps(model.predict(rows), indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"cejsp: {exc}\n")


if __name__ == "__main__":
    main()
