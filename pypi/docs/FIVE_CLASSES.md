# Five-class classification

The supported inference API uses one bundled reference. Call `CEJSP(instance)` for a class string, `classify(instance)` for a detailed report, or `classify_many(instances, workers=...)` for a batch. No model path is required or accepted by these functions.

See the [user guide](USER_GUIDE.md) for accepted formats and output fields, and the [model card](MODEL_CARD.md) for provenance and limitations. The classes are toy, easy, medium, hard and open. They predict the pinned JSPLIB protocol rather than certifying mathematical complexity or current open-problem status.

Earlier manifests and train/retrain commands belong to the historical research interfaces. They do not select or change the reference used by the supported public inference API.
