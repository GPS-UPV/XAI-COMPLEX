# User guide

CEJSP offers two levels of inference with one bundled reference. Use `CEJSP(instance)` for a class string; use `classify(instance)` or `classify_many(instances, workers=...)` for detailed reports. These APIs do not accept alternative models or training parameters.

## Inputs

Pass a file path (`str` or `pathlib.Path`), a supported instance dictionary, or an IGJSP instance object. Files can be IGJSP JSON, supported IGJSP DZN declarations, or Taillard text. DZN is parsed as data; it is not executed. The reference requires permutation JSP routes, one speed, integer durations, and no release/due-date constraints. Keep native operation order when constructing dictionaries.

## Reports

| Field | Meaning |
|---|---|
| `category` | Predicted toy/easy/medium/hard/open label |
| `class_probabilities` | Uncalibrated model votes for all five classes |
| `probability_calibration` | Explicit description of the vote interpretation |
| `applicability` | Whether features lie within the training ranges |
| `missing_feature_count` | Number of unavailable model inputs |
| `model_version` | Identity of the bundled reference |
| `label_protocol` | Identity of the source labeling protocol |
| `open_semantics` | Reminder that open is a predicted class, not a certificate |
| `instance_id` | Content-derived identity from normalization |

Each call returns a new report. Modifying the report does not modify the model. Batch results preserve order. Invalid inputs raise their normal exception and fail the batch; partial results are not silently returned. Positive integer worker counts are required. More workers increase simultaneous feature extraction and resource use.

## CLI and errors

Run `cejsp classify instance.json` for a label, or add `--details` for a JSON report. Use `cejsp batch first.json second.dzn --workers 2` for a batch. Errors exit with status 2 and an English explanation.

The bundled reference is integrity-checked before loading and cached within each Python process. The reference is part of the installed distribution; there is no model-selection argument or environment-variable override. Updating CEJSP is the supported way to receive a revised reference. This is an API constraint, not a security boundary against someone editing their own Python installation.

## Generating a test instance

Install the integration extra and run `examples/test_igjsp_classifier.py --output run-01` from the source distribution. Optional `--jobs`, `--machines`, and `--seed` parameters control IGJSP, not the classifier. Use an empty output directory. The script writes `instance.json` and `result.json`. Passing this test proves the software flow works; the generated instance has no independently verified difficulty label.
