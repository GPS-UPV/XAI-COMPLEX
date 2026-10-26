"""Supervised calibration shared by the corrected research scripts."""


def supervised_calibration(df_feats, X_scl, labels, dump_weighted_components_path=None,
                           n_splits=5, random_state=42):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from .estimator import evaluate_supervised
    from .labels import SolverTargetCalibrator

    if labels is None or len(labels) == 0:
        return {"used": False, "reason": "No labels"}
    labels = labels.copy()
    if "instance_id" in labels:
        labels = labels.set_index("instance_id")
    target_index = set(df_feats.index)
    ids = []
    for raw in labels.index:
        value = str(raw)
        if raw in target_index:
            ids.append(raw)
            continue
        ids.append(None)
    if any(i is None for i in ids):
        raise ValueError("Unmatched label identifiers; provide explicit full instance_id values")
    if len(set(ids)) != len(ids):
        raise ValueError("Several solver records map to one instance; aggregate explicitly by solver protocol")
    labels.index = ids
    common = df_feats.index.intersection(labels.index, sort=False)
    if len(common) < max(20, n_splits * 3):
        return {"used": False, "reason": "Too few aligned solver records"}
    numeric = df_feats.select_dtypes(include=[np.number])
    aligned = numeric.loc[common]
    records = labels.loc[common].to_dict(orient="records")
    groups = labels.loc[common, "base_instance_id"].tolist() if "base_instance_id" in labels else None
    metrics = evaluate_supervised(aligned.to_dict(orient="records"), records,
                                  groups=groups, n_splits=n_splits, random_state=random_state, n_estimators=600)
    calibrator = SolverTargetCalibrator().fit(records)
    y = calibrator.transform(records)
    if np.ptp(y) == 0:
        raise ValueError("Constant solver target")
    model = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                          RandomForestRegressor(n_estimators=600, max_features="sqrt", random_state=random_state, n_jobs=1))
    model.fit(aligned.to_numpy(dtype=float), y)
    predictions = model.predict(numeric.to_numpy(dtype=float))
    actual, oof = np.asarray(metrics["targets"]), np.asarray(metrics["oof"])
    rho = spearmanr(actual, oof)
    if dump_weighted_components_path:
        from pathlib import Path
        path = Path(dump_weighted_components_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"target_final_reference": y, "target_fold_reference": actual, "oof_pred": oof}, index=common).to_csv(path)
    out = {"used": True, "model": "RandomForestRegressor", "target_from": "weighted_solver_fold_calibration_v1",
           "complexity_sup_pred": np.clip(predictions, 0, 1),
           "oof_pred_aligned": pd.Series(oof, index=common, name="oof_pred"),
           "y_true_aligned": pd.Series(actual, index=common, name="y_true"),
           "feature_importances": pd.Series(model.steps[-1][1].feature_importances_, index=numeric.columns),
           "oof_r2": metrics["r2"], "oof_mae": metrics["mae"], "oof_rmse": metrics["rmse"],
           "oof_spearman_rho": float(rho.statistic), "oof_spearman_p": float(rho.pvalue),
           "n_labels_aligned": len(common), "weighted_weights": calibrator.weights}
    for name in ("r2", "mae", "rmse"):
        out[f"cv_{name}_mean"] = metrics[f"fold_{name}_mean"]
        out[f"cv_{name}_std"] = metrics[f"fold_{name}_std"]
    return out
