"""Freshness / liveness labelling. Live = NYC DOT data younger than the configured
threshold; otherwise the newest stored data is served and labelled 'latest available'.
Chicago traffic is always 'historical' (archived source)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from backend.app.schemas.models import Freshness
from src.common.config import sources
from src.common.runs import last_success, read_runs


def _last_run(source: str) -> dict | None:
    runs = [r for r in read_runs() if r["source"] == source]
    return runs[-1] if runs else None


def nyc_dot(data_as_of: pd.Timestamp | None) -> Freshness:
    cfg = sources()["nyc_dot_speeds"]
    run = _last_run("nyc_dot_speeds")
    if data_as_of is None or pd.isna(data_as_of):
        return Freshness(source="nyc_dot_speeds", mode="unavailable", note="No NYC DOT data ingested yet.")
    now_ny = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    age_min = (now_ny - data_as_of.to_pydatetime()).total_seconds() / 60
    live = age_min <= cfg["stale_after_minutes"]
    note = None if live else (f"Showing the latest stored observations ({age_min / 60:.1f} h old). "
                              + ("The last refresh failed: " + (run.get("error") or "")
                                 if run and run["status"] == "failed" else "Live feed not refreshed recently."))
    return Freshness(source="nyc_dot_speeds", data_as_of=str(data_as_of), mode="live" if live else "latest_available",
                     note=note, last_ingestion=run["finished_at"] if run else None,
                     last_ingestion_status=run["status"] if run else None)


def chicago_traffic(data_end: pd.Timestamp | None) -> Freshness:
    ok = last_success("chicago_traffic")
    return Freshness(source="chicago_traffic", data_as_of=str(data_end) if data_end is not None else None,
                     mode="historical",
                     note="Chicago Traffic Tracker is archived (no updates since mid-2026). "
                          "Values are historical observations, not current conditions.",
                     last_ingestion=ok["finished_at"] if ok else None,
                     last_ingestion_status=ok["status"] if ok else None)
