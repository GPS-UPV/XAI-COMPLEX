# CEJSP 0.5.0 release verification

The 0.5.0 build is complete. Public inference now uses one fixed, bundled reference. No publication has been performed.

## Public API

```python
from cejsp import CEJSP, classify, classify_many
label = CEJSP('instance.json')
report = classify('instance.dzn')
reports = classify_many(['first.json', 'second.txt'], workers=2)
```

The simple function takes only the instance and returns a string. The detailed and batch functions use the same reference and expose no model-selection or retraining argument. The retained model's bytes are unchanged. Earlier research interfaces are kept for compatibility and do not configure the public inference reference.

## Verification

The repository test suite passes 72 tests and 22 subtests without skips. Each final artifact was installed into a separate target and tested with isolated Python imports: six public API checks and the real IGJSP generation/classification example passed for both wheel and sdist. Those checks share the verified dependency environment; they do not claim a fresh online dependency resolution. Twine strict metadata checks pass for both artifacts.

Archive inspection checked 101 members across both distributions. The wheel includes one retained model and its metadata; its model digest matches the original reference. There are no raw datasets, abandoned models, experiment checkpoints, validation logs, bytecode caches, tests, or PyPI tokens in the release archives. Source documentation and examples are in English. The source archive contains only the three current English guides.

## Artifacts

Use only the two files in pypi/dist/0.5.0. The exact hashes and verification scope are recorded in [release-0.5.0-final/report.json](release-0.5.0-final/report.json); the archive index is in the same directory. The dependency versions required by the reference are declared by the package. Code is MIT-licensed; the bundled reference and metadata carry CC BY-SA 4.0 with upstream attribution and the included model license.

## Publishing

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m twine upload pypi/dist/0.5.0/cejsp-0.5.0-py3-none-any.whl pypi/dist/0.5.0/cejsp-0.5.0.tar.gz
```

Enter the token only at the Twine prompt. The build does not establish ownership or availability of the PyPI name/version. The reference's exploratory 84.19% JSPLIB accuracy and weak easy recall remain documented limitations, not guarantees for new instances.
