# Validation and release status

The 0.5.0 release introduces fixed-reference inference through CEJSP, classify and classify_many. The bundled model preserves the retained reference's original bytes. Functional checks cover direct calls, JSON/DZN/Taillard equivalence, ordered parallel batches, independent output dictionaries, model integrity, and rejection of custom model arguments.

Validation logs and release reports remain outside the distribution under pypi/validation/. The build contains no experiment checkpoints, raw benchmark datasets, build logs or authentication configuration. English user documentation and one bundled model are included.

See the [model card](MODEL_CARD.md) for scientific limitations. Passing integration tests does not establish difficulty-prediction accuracy on newly generated instances. See the [release guide](RELEASING.md) for building and publishing; publishing is an explicit maintainer action and is not performed by the build.
