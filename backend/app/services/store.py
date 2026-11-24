"""Read-side data access for the API: processed Parquet, analytics JSON and model
artifacts, cached per file modification time so refreshed pipeline outputs are picked up
without a restart and unchanged files are never re-read."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.errors import NotReady
from src.common.config import get_settings
from src.models.registry import latest_dir

_lock = threading.Lock()
_cache: dict[tuple[str, float], Any] = {}


def _cached(path: Path, loader):
    key = (str(path), path.stat().st_mtime)
    with _lock:
        if key in _cache:
            return _cache[key]
    val = loader(path)
    with _lock:
        for k in [k for k in _cache if k[0] == str(path)]:
            del _cache[k]
        _cache[key] = val
    return val


def processed(name: str, hint: str = "python -m src.pipeline ingest && python -m src.pipeline clean",
              columns: list[str] | None = None) -> pd.DataFrame:
    p = get_settings().processed_dir / name
    if not p.exists():
        raise NotReady(name, hint)
    df = _cached(p, pd.read_parquet)
    return df[columns] if columns else df


def processed_exists(name: str) -> bool:
    return (get_settings().processed_dir / name).exists()


def analytics(name: str, hint: str = "python -m src.pipeline analyze") -> Any:
    p = get_settings().analytics_dir / f"{name}.json"
    if not p.exists():
        raise NotReady(name, hint)
    return _cached(p, lambda q: json.loads(q.read_text()))


def analytics_frame(name: str, hint: str = "python -m src.pipeline analyze") -> pd.DataFrame:
    p = get_settings().analytics_dir / name
    if not p.exists():
        raise NotReady(name, hint)
    return _cached(p, pd.read_parquet)


def external(name: str) -> Path | None:
    p = get_settings().external_dir / name
    return p if p.exists() else None


def model_dir(task: str) -> Path:
    d = latest_dir(task)
    if d is None:
        raise NotReady(task, f"python -m src.pipeline train  (task {task})")
    return d


def model_json(task: str, name: str) -> Any | None:
    p = model_dir(task) / f"{name}.json"
    return _cached(p, lambda q: json.loads(q.read_text())) if p.exists() else None


def model_frame(task: str, name: str) -> pd.DataFrame | None:
    p = model_dir(task) / f"{name}.parquet"
    return _cached(p, pd.read_parquet) if p.exists() else None


def model_object(task: str, name: str = "model"):
    import joblib
    p = model_dir(task) / f"{name}.joblib"
    return _cached(p, joblib.load) if p.exists() else None


def list_tasks() -> list[str]:
    root = get_settings().models_dir
    return sorted(p.name for p in root.iterdir() if p.is_dir() and latest_dir(p.name)) if root.exists() else []


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))
