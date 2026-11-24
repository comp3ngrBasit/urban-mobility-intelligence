"""End-to-end pipeline orchestration.

    python -m src.pipeline verify             # check endpoints, schemas, freshness
    python -m src.pipeline ingest [--source X]
    python -m src.pipeline clean
    python -m src.pipeline analyze            # EDA, statistics, hotspots, anomalies
    python -m src.pipeline train [--model X]  # traffic | travel_time | demand | classical | lstm
    python -m src.pipeline load-db            # push processed tables into PostGIS
    python -m src.pipeline all

Each stage isolates failures per source: one unavailable API never blocks the others,
and the previously ingested partitions remain in place for the API to serve.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from src.common.config import get_settings
from src.common.http import SourceUnavailable
from src.common.logging import get_logger

log = get_logger("pipeline")


def _safe(name: str, fn: Callable, *a, **kw):
    try:
        return fn(*a, **kw)
    except (SourceUnavailable, FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("stage %s failed: %s — continuing with remaining stages", name, exc)
        return None


def ingest(source: str | None = None) -> None:
    from src.ingestion import chicago, nyc_dot, nyc_tlc, osm, weather

    jobs: dict[str, Callable] = {
        "nyc_dot": nyc_dot.ingest_dot_speeds,
        "chicago_crashes": chicago.ingest_crashes,
        "chicago_traffic": chicago.ingest_traffic_segments,
        "tlc_zones": nyc_tlc.ingest_zones,
        "tlc_trips": nyc_tlc.ingest_trips,
        "osm_chicago": lambda: osm.ingest_road_network("chicago"),
        "osm_nyc": lambda: osm.ingest_road_network("nyc"),
        "osm_infra_chicago": lambda: osm.ingest_infrastructure("chicago", "Chicago"),
        "weather": ingest_weather_for_data,
        "weather_forecast": lambda: [weather.ingest_weather_forecast(c) for c in ("nyc", "chicago")],
    }
    for name, fn in jobs.items():
        if source in (None, name):
            log.info("== ingest %s", name)
            _safe(name, fn)


def ingest_weather_for_data() -> None:
    """Pull archive weather covering exactly the periods present in the traffic/taxi data."""
    from src.ingestion.weather import ingest_weather_archive

    raw = get_settings().raw_dir
    spans = {"chicago": [], "nyc": []}
    for part in (raw / "chicago_traffic").glob("*.parquet"):
        t = pd.read_parquet(part, columns=["time"])["time"]
        spans["chicago"] += [pd.to_datetime(t).min(), pd.to_datetime(t).max()]
    for part in (raw / "chicago_crashes").glob("*.parquet"):
        t = pd.to_datetime(pd.read_parquet(part, columns=["crash_date"])["crash_date"])
        spans["chicago"] += [t.min(), t.max()]
    for part in (raw / "nyc_tlc").glob("*.parquet"):
        y, m = map(int, part.stem.split("_")[1].split("-"))
        spans["nyc"] += [pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(1)]
    for city, s in spans.items():
        if s:
            ingest_weather_archive(city, min(s).date() - timedelta(days=8), max(s).date())
        else:
            ingest_weather_archive(city, date.today() - timedelta(days=60), date.today())


def clean() -> None:
    from src.preprocessing import cleaners as c

    for name, fn in [("crashes", c.clean_crashes), ("traffic", c.clean_traffic),
                     ("dot", c.clean_dot_speeds), ("tlc", c.clean_tlc), ("weather", c.clean_weather)]:
        log.info("== clean %s", name)
        _safe(name, fn)


def analyze() -> None:
    from src.analytics import eda
    from src.geospatial import hotspots
    from src.models import anomaly

    _safe("hotspots", hotspots.run)
    _safe("anomaly", anomaly.run)
    eda.run_all()


def train(model: str | None = None) -> None:
    from src.forecasting import demand
    from src.models import traffic, travel_time

    jobs = {"traffic": traffic.train_all, "travel_time": travel_time.train,
            "demand": demand.train_ml, "classical": demand.train_classical, "lstm": demand.train_lstm}
    for name, fn in jobs.items():
        if model in (None, name):
            log.info("== train %s", name)
            _safe(name, fn)


def load_db() -> None:
    from backend.app.database.loader import load_all
    load_all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["verify", "ingest", "clean", "analyze", "train", "load-db", "all"])
    ap.add_argument("--source")
    ap.add_argument("--model")
    a = ap.parse_args()
    get_settings().ensure_dirs()
    if a.stage == "verify":
        from src.ingestion.verify import verify_all
        for r in verify_all():
            log.info("%-22s %-12s %s", r["source"], r["status"], r.get("rows_updated_at") or "")
    if a.stage in ("ingest", "all"):
        ingest(a.source)
    if a.stage in ("clean", "all"):
        clean()
    # training runs before analysis so traffic EDA reuses the train-period free-flow speeds
    if a.stage in ("train", "all"):
        train(a.model)
    if a.stage in ("analyze", "all"):
        analyze()
    if a.stage == "load-db" or (a.stage == "all" and get_settings().database_url):
        _safe("load-db", load_db)


if __name__ == "__main__":
    main()
