"""Executive overview, pipeline monitoring and health."""
from __future__ import annotations

import json
from collections import defaultdict

from fastapi import APIRouter, Query

from backend.app.core.errors import NotReady
from backend.app.schemas.models import Health
from backend.app.services import store
from src.common.config import get_settings
from src.common.runs import read_runs

router = APIRouter(prefix="/api", tags=["system"])


def _try(fn):
    try:
        return fn()
    except NotReady:
        return None


@router.get("/overview")
def overview():
    """Each block is independent: missing inputs yield null blocks, never fabricated values."""
    mob = _try(lambda: store.analytics("mobility"))
    tr = _try(lambda: store.analytics("traffic"))
    acc = _try(lambda: store.analytics("accidents"))

    def live():
        from backend.app.api.live import nyc_live
        r = nyc_live()
        return {"freshness": r["freshness"], **r["summary"]}

    def predicted():
        from backend.app.api.predict import _forecast_at_horizon
        cg, t = _forecast_at_horizon("congestion_h60")
        return {"target_time": t, "distribution": cg["predicted"].value_counts(normalize=True).round(4).to_dict()}

    return {
        "mobility": mob and {"total_trips": mob["total_trips"], "period": mob["period"], "daily": mob["daily"],
                             "borough_peak_hours": mob["borough_peak_hours"], "scope_note": mob["scope_note"]},
        "nyc_live": _try(live),
        "chicago_traffic": tr and {"mean_speed": tr["mean_speed"], "period": tr["period"],
                                   "congestion_distribution": tr["congestion_distribution"],
                                   "segments": tr["segments"], "scope_note": tr["scope_note"]},
        "accidents": acc and {k: acc[k] for k in ("total_crashes", "crashes_per_day", "severity_counts",
                                                  "fatalities", "injuries_total", "period", "monthly")},
        "predicted_congestion": _try(predicted),
    }


@router.get("/pipeline/status")
def pipeline_status():
    runs = read_runs()
    by: dict[str, list] = defaultdict(list)
    for r in runs:
        by[r["source"]].append(r)
    sources = []
    for src, rs in sorted(by.items()):
        last = rs[-1]
        ok = next((r for r in reversed(rs) if r["status"] in ("success", "partial")), None)
        sources.append({"source": src, "last_status": last["status"], "last_run": last["finished_at"],
                        "last_success": ok["finished_at"] if ok else None,
                        "data_max_ts": ok["data_max_ts"] if ok else None,
                        "records_last_success": ok["records"] if ok else 0,
                        "total_records": sum(r["records"] for r in rs if r["status"] == "success"),
                        "last_error": last.get("error"), "runs": len(rs),
                        "failures": sum(r["status"] == "failed" for r in rs)})
    ver = get_settings().analytics_dir / "source_verification.json"
    return {"sources": sources, "verification": json.loads(ver.read_text()) if ver.exists() else None}


@router.get("/pipeline/runs")
def pipeline_runs(limit: int = Query(100, le=1000), source: str | None = None):
    rs = [r for r in read_runs() if source in (None, r["source"])]
    for r in rs:
        r.get("details", {}).pop("traceback", None)
    return list(reversed(rs[-limit:]))


@router.get("/pipeline/quality")
def data_quality():
    d = get_settings().analytics_dir / "quality"
    if not d.exists():
        raise NotReady("quality reports", "python -m src.pipeline clean")
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


@router.get("/health", response_model=Health)
def health():
    s = get_settings()
    db = "not configured"
    if s.database_url:
        try:
            from sqlalchemy import create_engine, text
            with create_engine(s.database_url, pool_pre_ping=True).connect() as c:
                c.execute(text("SELECT 1"))
            db = "ok"
        except Exception as exc:  # noqa: BLE001
            db = f"unreachable: {type(exc).__name__}"
    tables = sorted(p.name for p in s.processed_dir.glob("*.parquet")) if s.processed_dir.exists() else []
    return Health(status="ok", processed_tables=tables, model_tasks=store.list_tasks(), database=db)
