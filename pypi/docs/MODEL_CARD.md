# Model card

The bundled reference is `jsplib-easy-v7-accuracy`, retained after subsequent experiments failed to improve the accuracy target. Its original serialized bytes are preserved in `cejsp/data/reference.joblib`. `reference.json` records the SHA-256 digest, dependency versions, source snapshot, backend, and evaluation scope. Inference checks the digest and dependency versions before loading.

## Provenance

Labels and training instances originate from ScheduleOpt JSPLIB, snapshot `8a6e36ef31263d36975e56fb481e0ba04e188d6f`. The cohort contains 272 compatible permutation JSP instances. JSPLIB is credited at https://scheduleopt.github.io/benchmarks/jsplib and https://github.com/ScheduleOpt/benchmarks. Its upstream CC BY-SA 4.0 license is reproduced in MODEL_LICENSE. The learned reference and its metadata are distributed under that license; software remains MIT-licensed.

## Evidence

Exploratory five-fold out-of-fold accuracy is 84.19%, macro-F1 is approximately 0.606, and easy recall is 1/15. This fixed-model ablation was examined during repeated experimentation and selected afterward. It does not establish independent external performance. The cohort was also used for model development. There is no independently labeled IGJSP transfer evaluation supporting this accuracy claim. Do not interpret the figure as a guarantee.

## Inference

The model uses Extra Trees and a fixed trajectory feature backend. Features include route/duration statistics, bounds, dispatch rules, and CP-SAT probes with deterministic budgets 0.2 and 3. One solver worker and fixed seeds are used within each probe. Time is spent extracting these features; deterministic effort does not impose a fixed wall-clock latency. Model votes are uncalibrated. A training-range check is a limited diagnostic, not a comprehensive out-of-distribution detector.

## Limitations

The small easy class is poorly recognized. Historical source labels reflect reference engines, their settings, and their runtime protocol rather than intrinsic complexity. Open labels are snapshot-dependent and predictions do not establish an instance's scientific status. Inputs outside the supported representation are rejected. No attempt is made to estimate flexible/reentrant scheduling difficulty using this fixed-route reference.

## Updating the reference

Public inference functions have no custom-model or retraining option. A future reference requires an explicitly evaluated, versioned package release. Low-level historical research classes are retained for compatibility; they do not change the bundled reference used by the public API.
