"""Feature importance + SHAP explanations for fitted tree models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

log = get_logger(__name__)


def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def feature_importance(model, features: list[str]) -> list[dict]:
    est = _final_estimator(model)
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "get_feature_importance"):
        imp = est.get_feature_importance()
    if imp is None:
        return []
    imp = np.asarray(imp, float)
    imp = imp / imp.sum() if imp.sum() else imp
    return sorted([{"feature": f, "importance": float(v)} for f, v in zip(features, imp)],
                  key=lambda r: -r["importance"])


def shap_summary(model, X: pd.DataFrame, sample: int = 2000, class_index: int | None = None) -> dict:
    """Mean |SHAP| per feature plus a compact sample for beeswarm-style plots."""
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}
    est = _final_estimator(model)
    Xs = X.sample(min(sample, len(X)), random_state=42)
    if hasattr(model, "steps"):  # imputer pipeline (RandomForest): transform first
        Xt = pd.DataFrame(model[:-1].transform(Xs), columns=Xs.columns, index=Xs.index)
    else:
        Xt = Xs
    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xt, check_additivity=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP failed: %s", exc)
        return {"error": str(exc)}
    sv = np.asarray(sv)
    if sv.ndim == 3:  # multiclass: (n, features, classes) or (classes, n, features)
        sv = sv[..., class_index or 0] if sv.shape[1] == Xt.shape[1] else sv[class_index or 0]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    top = order[:15]
    pts = min(300, len(Xt))
    return {
        "mean_abs_shap": [{"feature": Xt.columns[i], "value": float(mean_abs[i])} for i in order],
        "beeswarm": [{"feature": Xt.columns[i],
                      "shap": np.round(sv[:pts, i], 4).tolist(),
                      "feature_value": np.round(pd.to_numeric(Xt.iloc[:pts, i], errors="coerce")
                                                .fillna(np.nan).to_numpy(float), 4).tolist()}
                     for i in top],
        "n_explained": int(len(Xt)),
        "class_index": class_index,
    }
