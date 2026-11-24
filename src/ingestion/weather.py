"""Open-Meteo historical archive + forecast, hourly, per city."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.common.config import sources
from src.common.http import build_session, get_json
from src.common.runs import track_run
from src.ingestion.storage import raw_path, write_parquet_atomic


def _to_frame(payload: dict, city: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])   # local wall-clock time of the requested timezone
    df["city"] = city
    return df


def ingest_weather_archive(city: str, start: date, end: date) -> int:
    cfg = sources()["open_meteo"]
    loc = cfg["locations"][city]
    session = build_session()
    with track_run(f"weather_archive_{city}") as run:
        # archive lags real time by ~5 days; clamp end accordingly
        end = min(end, date.today() - timedelta(days=5))
        params = {"latitude": loc["latitude"], "longitude": loc["longitude"],
                  "start_date": start.isoformat(), "end_date": end.isoformat(),
                  "hourly": ",".join(cfg["hourly"]), "timezone": loc["timezone"]}
        df = _to_frame(get_json(session, cfg["archive_url"], params), city)
        write_parquet_atomic(df, raw_path("weather", f"archive_{city}_{start}_{end}.parquet"))
        run.records = len(df)
        run.data_min_ts, run.data_max_ts = str(df["time"].min()), str(df["time"].max())
        return len(df)


def ingest_weather_forecast(city: str, days: int = 3) -> int:
    cfg = sources()["open_meteo"]
    loc = cfg["locations"][city]
    session = build_session()
    with track_run(f"weather_forecast_{city}") as run:
        params = {"latitude": loc["latitude"], "longitude": loc["longitude"],
                  "hourly": ",".join(cfg["hourly"] + cfg["forecast_extra_hourly"]),
                  "forecast_days": days, "timezone": loc["timezone"]}
        df = _to_frame(get_json(session, cfg["forecast_url"], params), city)
        write_parquet_atomic(df, raw_path("weather", f"forecast_{city}.parquet"))
        run.records = len(df)
        run.data_min_ts, run.data_max_ts = str(df["time"].min()), str(df["time"].max())
        return len(df)
