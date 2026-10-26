"""Use the project's original graph features on already generated instances."""

from .instances import normalize_instance


def extract_instance_features(instance, *, speed_index=None, approximate_betweenness=False, betweenness_samples=100):
    """Retain the research graph; speed projection must be explicitly selected.

    The research disjunctive graph models fixed routing. Flexible instances are
    rejected instead of silently treating their nominal route as fixed.
    """
    import numpy as np
    from ._graph_builder import GraphBuilderStrict
    from ._graph_features import extract_features_from_heterodata

    inst = instance if isinstance(instance, dict) and instance.get("schema") == "xai-complex.instance.v1" else normalize_instance(instance)
    if inst["instance_type"] != "JSP":
        raise ValueError("The original XAI-COMPLEX graph supports fixed-route JSP; FJSP needs a validated flexible graph model")
    if speed_index is None:
        if inst["speed"] != 1:
            raise ValueError("Multiple speeds: select speed_index explicitly for the research graph")
        speed_index = 0
    if isinstance(speed_index, bool) or not isinstance(speed_index, int) or not 0 <= speed_index < inst["speed"]:
        raise ValueError("Invalid speed_index")
    jobs = inst["jobs"]
    counts = {len(j["operations"]) for j in jobs}
    if len(counts) != 1:
        raise ValueError("The research graph requires equal operation counts per job")
    data = {"numJobs": len(jobs), "numMchs": inst["n_machines"], "speed": 1,
            "Orden": [[o["machine"] for o in j["operations"]] for j in jobs],
            "ProcessingTime": [[[o["speeds"][speed_index]["processing_time"]] for o in j["operations"]] for j in jobs],
            "EnergyConsumption": [[[o["speeds"][speed_index]["energy"]] for o in j["operations"]] for j in jobs]}
    for key, array_key in (("release-date", "ReleaseDate"), ("due-date", "DueDate")):
        values = [[o.get(key, j.get(key)) for o in j["operations"]] for j in jobs]
        if any(v is not None for row in values for v in row):
            if any(v is None for row in values for v in row):
                raise ValueError("Partially missing release/due date arrays")
            data[array_key] = values
    data = {k: np.asarray(v) if k in ("Orden", "ProcessingTime", "EnergyConsumption", "ReleaseDate", "DueDate") else v for k, v in data.items()}
    graph = GraphBuilderStrict(data).data
    if type(betweenness_samples) is not int or betweenness_samples < 1:
        raise ValueError("betweenness_samples must be positive")
    features = extract_features_from_heterodata(len(jobs), inst["n_machines"], 0, graph,
                                              use_approx_betweenness=approximate_betweenness,
                                              betweenness_samples=betweenness_samples)
    failures = {k: v for k, v in features.items() if k.endswith("_error")}
    if failures:
        raise ValueError(f"Graph feature extraction failed: {failures}")
    return {k: float(v) for k, v in features.items() if isinstance(v, (int, float, np.number)) and k != "seed"}
