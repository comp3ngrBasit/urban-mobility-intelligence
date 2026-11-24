"""Model Performance: metrics, comparisons, importance, SHAP, error analysis."""
from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.core.errors import NotReady
from backend.app.services import store

router = APIRouter(prefix="/api/models", tags=["models"])

HEADLINE = {"reg": ("mae", "rmse", "r2", "mape_pct"), "clf": ("f1_macro", "accuracy", "roc_auc_ovr_macro")}


@router.get("")
def list_models():
    out = []
    for task in store.list_tasks():
        m = store.model_json(task, "metrics")
        fc = store.model_json(task, "feature_config") or {}
        meta = store.model_json(task, "metadata") or {}
        if not m:
            continue
        if "best_model" not in m:   # classical forecasting summary
            out.append({"task": task, "kind": "forecast_classical", "version": store.model_dir(task).name,
                        "aggregate": m.get("aggregate")})
            continue
        kind = fc.get("kind", "reg")
        best = m["models"][m["best_model"]]["test"]
        base = {k: {h: v.get(h) for h in HEADLINE[kind]} for k, v in m.get("baselines", {}).items()}
        out.append({"task": task, "kind": kind, "best_model": m["best_model"], "version": store.model_dir(task).name,
                    "created_at": meta.get("created_at"), "test_period": m["split"]["test_period"],
                    "best_test": {h: best.get(h) for h in HEADLINE[kind]}, "baselines": base,
                    "candidates": {k: {h: v["test"].get(h) for h in HEADLINE[kind]} | {"cv_mean": v["cv_mean"], "cv_std": v["cv_std"]}
                                   for k, v in m["models"].items()}})
    if not out:
        raise NotReady("models", "python -m src.pipeline train")
    return out


@router.get("/{task}")
def model_detail(task: str):
    if task not in store.list_tasks():
        raise NotReady(task, "unknown or untrained task")
    return {"task": task, "version": store.model_dir(task).name,
            **{k: store.model_json(task, k) for k in ("metrics", "feature_config", "metadata", "importance",
                                                       "shap", "errors", "intervals")}}


@router.get("/{task}/predictions")
def model_predictions(task: str, sample: int = Query(3000, le=20000)):
    tp = store.model_frame(task, "test_predictions")
    if tp is None:
        raise NotReady(f"{task} predictions", "retrain")
    d = tp.sample(min(sample, len(tp)), random_state=0) if len(tp) > sample else tp
    return {"task": task, "n_total": int(len(tp)), "rows": store.records(d.round(4))}
