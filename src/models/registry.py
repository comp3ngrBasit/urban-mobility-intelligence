"""Versioned model store on disk + MLflow tracking.

artifacts/models/<task>/<version>/
    model.joblib            fitted estimator (best by validation score)
    metrics.json            test metrics of every candidate + baselines
    feature_config.json     feature list, target, horizon, preprocessing notes
    metadata.json           data period, row counts, timestamps, library versions
    *.parquet / *.json      test predictions, SHAP, importance, error analysis
artifacts/models/<task>/LATEST   -> name of the current version
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)


def new_version_dir(task: str) -> Path:
    v = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S")
    d = get_settings().models_dir / task / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_version(d: Path, *, model=None, metrics: dict, feature_config: dict, metadata: dict,
                 extras: dict[str, Any] | None = None, frames: dict[str, pd.DataFrame] | None = None,
                 models: dict[str, Any] | None = None) -> None:
    if model is not None:
        joblib.dump(model, d / "model.joblib")
    for name, m in (models or {}).items():
        joblib.dump(m, d / f"{name}.joblib")
    import sklearn
    metadata = {**metadata, "created_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "sklearn": sklearn.__version__}
    _json(d / "metrics.json", metrics)
    _json(d / "feature_config.json", feature_config)
    _json(d / "metadata.json", metadata)
    for name, obj in (extras or {}).items():
        _json(d / f"{name}.json", obj)
    for name, df in (frames or {}).items():
        df.to_parquet(d / f"{name}.parquet", index=False)
    (d.parent / "LATEST").write_text(d.name)
    log.info("saved model version %s", d)


def latest_dir(task: str) -> Path | None:
    root = get_settings().models_dir / task
    pointer = root / "LATEST"
    if pointer.exists():
        d = root / pointer.read_text().strip()
        if d.exists():
            return d
    versions = sorted(p for p in root.glob("v*") if p.is_dir()) if root.exists() else []
    return versions[-1] if versions else None


def load_json(task: str, name: str) -> Any | None:
    d = latest_dir(task)
    p = d / f"{name}.json" if d else None
    return json.loads(p.read_text()) if p and p.exists() else None


def load_model(task: str, name: str = "model"):
    d = latest_dir(task)
    p = d / f"{name}.joblib" if d else None
    return joblib.load(p) if p and p.exists() else None


def load_frame(task: str, name: str) -> pd.DataFrame | None:
    d = latest_dir(task)
    p = d / f"{name}.parquet" if d else None
    return pd.read_parquet(p) if p and p.exists() else None


class MlflowRun:
    """Thin optional wrapper so training still works if MLflow isn't reachable."""

    def __init__(self, experiment: str, run_name: str):
        self.experiment, self.run_name, self._active = experiment, run_name, False

    def __enter__(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment)
            mlflow.start_run(run_name=self.run_name)
            self._active = True
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow unavailable (%s); continuing without tracking", exc)
        return self

    def log_params(self, params: dict) -> None:
        if self._active:
            import mlflow
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        if self._active:
            import mlflow
            flat = {f"{prefix}{k}": float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float)) and v is not None}
            mlflow.log_metrics(flat)

    def log_artifacts(self, path: Path) -> None:
        if self._active:
            import mlflow
            mlflow.log_artifacts(str(path))

    def __exit__(self, *exc):
        if self._active:
            import mlflow
            mlflow.end_run(status="FAILED" if exc[0] else "FINISHED")
        return False
