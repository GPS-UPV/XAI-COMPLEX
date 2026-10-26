# CEJSP · JSP Complexity Estimator

**One instance. One difficulty class. A fixed, bundled reference.**

CEJSP estimates the empirical difficulty of a Job Shop Scheduling instance and returns **`toy`**, **`easy`**, **`medium`**, **`hard`**, or **`open`**. Read existing **IGJSP JSON**, **DZN**, and **Taillard** files without converting them yourself. The trained reference ships with the package: no model download, model path, training step, or API key is required.

## Install

```bash
python -m pip install cejsp==0.5.0
```

Python **3.11 or newer** is required. Core dependencies are pinned to the versions used by the bundled reference. To generate optional test instances with IGJSP:

```bash
python -m pip install "cejsp[integration]==0.5.0"
```

## Start with one line

```python
from cejsp import CEJSP

label = CEJSP("instance.json")
print(label)  # toy, easy, medium, hard, or open
```

Use the same call for other supported formats:

```python
label = CEJSP("instance.dzn")
label = CEJSP("instance.txt")  # Taillard machine-duration format
```

`CEJSP(instance)` returns a **string**. The instance is its only argument. A supported instance dictionary or IGJSP object can also be passed directly. Invalid inputs raise an explanatory exception rather than returning a guessed class.

## Inspect the full result

The professional inference API uses the **same fixed reference** and exposes its output rather than its training configuration:

```python
from cejsp import classify, classify_many

report = classify("instance.json")
print(report["category"])
print(report["class_probabilities"])
print(report["applicability"])
print(report["model_version"])

reports = classify_many(
    ["instance.json", "another.dzn", "third.txt"],
    workers=2,
)
```

Batch results preserve input order. `workers` controls concurrency only. None of these inference functions accepts a model path, changes model weights, or retrains the reference. Returned votes are **uncalibrated**; they are not certified probabilities. An `outside_training_bounds` result flags features outside the ranges seen during fitting and should be treated cautiously.

Earlier `JobShopComplexityClassifier` and `xai_complex` interfaces remain available for compatibility with research scripts. They are separate low-level research tools, not configuration hooks for `CEJSP`, `classify`, or `classify_many`.

## Use the command line

```bash
cejsp classify instance.json
cejsp classify instance.dzn --details
cejsp batch instance.json another.txt --workers 2
```

The first command prints the class. `--details` and `batch` print JSON. The public CLI uses the bundled model and has no model-selection or training options.

## What the labels mean

The reference was trained on the original labels of a pinned [JSPLIB snapshot](https://github.com/ScheduleOpt/benchmarks/tree/8a6e36ef31263d36975e56fb481e0ba04e188d6f/jobshop). Those labels depend on the time needed to **prove optimality** using the benchmark's reference engines:

| Class | Source protocol |
|---|---|
| `toy` | Proof within 1 minute |
| `easy` | Proof within 10 minutes, beyond the toy category |
| `medium` | Proof within 1 hour, beyond the easy category |
| `hard` | Proof requiring more than 1 hour |
| `open` | No proof recorded in that snapshot |

For a new instance, these are **predicted protocol classes**. In particular, `open` is not a certificate that the instance is an unresolved research problem. The class is not the makespan and is not a computational-complexity theorem.

## Evidence and limits

| Item | Bundled reference |
|---|---|
| Model | Extra Trees with structural, bound, dispatch, and bounded CP-SAT search features |
| Training cohort | 272 compatible JSPLIB instances |
| Exploratory out-of-fold accuracy | **84.19%** |
| Exploratory macro-F1 | **0.606** |
| `easy` recall in that evaluation | **1/15** |
| Independent validation on newly generated IGJSP labels | **Not established** |

The 84.19% result comes from an exploratory ablation selected after examining the same cohort. It is not an independent test or a guaranteed accuracy on new inputs. The retained model prioritizes global accuracy and performs poorly on the small `easy` class. Later experiments, including checkpoint training, did not justify replacing it.

Inference extracts features and runs bounded CP-SAT probes. It can take longer on large instances; the search budget is deterministic solver effort, not a wall-clock deadline. No external solver service or network call is used for inference.

## Supported instances

Use fixed-route **permutation JSP** instances with nonnegative integer processing times and a single speed. Every job must visit every machine once. Flexible machine alternatives, reentrant routes, release/due-date constraints, and ambiguous multiple-speed inputs are outside the bundled reference's contract and are rejected. DZN uses the supported IGJSP declarations, not arbitrary MiniZinc programs. Taillard inputs contain machine-duration pairs.

## Documentation and examples

- [User guide](docs/USER_GUIDE.md): inputs, output fields, batching, and error handling.
- [Model card](docs/MODEL_CARD.md): provenance, integrity, performance, and limitations.
- [Release guide](docs/RELEASING.md): local build and explicit Twine upload commands.
- [Quick start](examples/quickstart.py): minimal Python usage.
- [IGJSP integration test](examples/test_igjsp_classifier.py): generate a real instance, save its native JSON, and classify it.

These relative documentation links refer to files in the source distribution and repository. The wheel contains the runtime code, one bundled reference, its metadata, and license notices. Datasets, abandoned models, checkpoints, build logs, and experimental reports are not included.

## License and attribution

The code is distributed under the **MIT License**. The bundled reference and its metadata are distributed under **CC BY-SA 4.0** with attribution to the [ScheduleOpt JSPLIB benchmark](https://scheduleopt.github.io/benchmarks/jsplib), whose upstream license is included as `MODEL_LICENSE`. See `NOTICE` for provenance and attribution.
