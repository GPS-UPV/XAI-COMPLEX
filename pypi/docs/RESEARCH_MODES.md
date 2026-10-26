# Historical research interfaces

CEJSP keeps the xai_complex implementation namespace because saved models reference its class identities and older experiments import it. This compatibility layer includes training, graph extraction, batch-relative scoring and reference calibration. These are research interfaces, separate from the fixed-reference inference functions exported by the current public API.

Use `CEJSP`, `classify` or `classify_many` for production inference. They always use the bundled retained reference. A custom estimator fitted through a historical class does not replace that reference.

Graph-based research workflows require the graphs extra. Research-only candidate comparisons can require the research extra. Neither extra is necessary to classify supported inputs with the bundled reference. IGJSP is optional and belongs to the integration extra.

Historical experiment metrics and source data remain in the repository under evaluation/ and models/. Failed checkpoint experiments and alternative references are not shipped. The public release makes no claim that any alternative reached 90% independent accuracy.
