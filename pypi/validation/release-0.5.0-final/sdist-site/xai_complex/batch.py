"""Public batch entry point using the original weighted research methodology."""


def estimate_features(features, labels=None):
    """Return scores with batch-calibrated scales, as in the research scripts.

    For inference with frozen reference scales, use ComplexityEstimator instead.
    """
    import numpy as np
    import pandas as pd
    from . import research

    df = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    if not df.index.is_unique or len(df) < 3:
        raise ValueError("Need three or more uniquely indexed feature records")
    df = research._clean_and_engineer(df)
    df = df.drop(columns=[c for c in df if "energy" in c.lower() or c == "num_nodes"])
    cols = [c for c in df if "betweenness_c_" not in c and not (("deg" in c or "clustering" in c) and "_c_" not in c and "_d_" not in c) and c != "graph_density"]
    df = df[cols].select_dtypes(include=[np.number])
    if df.empty or np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError("No numeric features or infinite feature values")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("All features are missing")
    _, scaled, _, _ = research._build_matrix(df)
    if np.all(np.ptp(scaled, axis=0) == 0):
        raise ValueError("Batch features are all constant")
    unsup = research.unsupervised_complexity(scaled)
    prior = research.prior_complexity_from_generator(df)
    scores = unsup["complexity_unsup"]
    if prior is not None:
        scores = research._safe_minmax(.7 * scores + .3 * prior)
    sup = research.supervised_calibration(df, scaled, labels) if labels is not None else {"used": False}
    if sup.get("used"):
        scores = research._safe_minmax((.5 if prior is not None else .6) * scores + (.5 if prior is not None else .4) * sup["complexity_sup_pred"])
    q1, q2 = np.quantile(scores, [.33, .66])
    return pd.DataFrame({"complexity_0_1": scores, "category": np.where(scores <= q1, "easy", np.where(scores <= q2, "medium", "hard"))}, index=df.index)
