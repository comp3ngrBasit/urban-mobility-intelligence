"""Pipeline run ledger and incremental-ingestion watermarks.

Each ingestion run appends a JSON line to data/raw/_runs.jsonl. The monitoring page and
the API's fallback logic ("serve last successful ingestion") are driven from this ledger.
"""
from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    source: str
    started_at: str = field(default_factory=utcnow_iso)
    finished_at: str | None = None
    status: str = "running"          # running | success | failed | partial
    records: int = 0
    data_min_ts: str | None = None   # earliest observation timestamp in the pulled data
    data_max_ts: str | None = None   # latest observation timestamp (data freshness)
    duration_s: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _ledger() -> Path:
    p = get_settings().raw_dir / "_runs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def track_run(source: str) -> Iterator[RunRecord]:
    rec = RunRecord(source=source)
    t0 = time.perf_counter()
    try:
        yield rec
        if rec.status == "running":
            rec.status = "success"
    except Exception as exc:  # recorded, then re-raised for the caller/scheduler
        rec.status = "failed"
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.details["traceback"] = traceback.format_exc(limit=5)
        raise
    finally:
        rec.finished_at = utcnow_iso()
        rec.duration_s = round(time.perf_counter() - t0, 2)
        with open(_ledger(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), default=str) + "\n")
        log.info("run %s finished: %s (%d records)", source, rec.status, rec.records)


def read_runs(limit: int | None = None) -> list[dict[str, Any]]:
    p = _ledger()
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:] if limit else rows


def last_success(source: str) -> dict[str, Any] | None:
    for r in reversed(read_runs()):
        if r["source"] == source and r["status"] in ("success", "partial"):
            return r
    return None


# ---------------- watermarks (incremental loads) ----------------

def _wm_path() -> Path:
    return get_settings().raw_dir / "_watermarks.json"


def get_watermark(source: str) -> str | None:
    p = _wm_path()
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(source)


def set_watermark(source: str, value: str) -> None:
    p = _wm_path()
    data = json.loads(p.read_text()) if p.exists() else {}
    data[source] = value
    p.write_text(json.dumps(data, indent=2))
