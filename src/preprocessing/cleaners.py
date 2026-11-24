"""Raw -> processed cleaners for every source. Outputs land in data/processed/.

Design rules
* Timestamps are standardised to naive *local* time of the city (the sources publish local
  wall-clock time; weather is requested in the same timezone so joins line up).
* Rows are dropped only for hard failures (unparseable time, impossible values); softer
  problems are flagged and counted in the quality report.
* NYC taxi demand, Chicago arterial speeds and Chicago crashes stay in separate tables —
  they measure different things and are never merged into a single "traffic" measure.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.common.config import get_settings, model_config, sources
from src.common.logging import get_logger
from src.ingestion.storage import list_partitions
from src.preprocessing.validation import ColumnRule as R
from src.preprocessing.validation import Schema, coerce, save_report, validate

log = get_logger(__name__)


def _processed(name: str):
    p = get_settings().processed_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_parts(source: str) -> pd.DataFrame:
    parts = list_partitions(source)
    if not parts:
        raise FileNotFoundError(f"no raw partitions for {source}; run ingestion first")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# --------------------------------------------------------------------------- Chicago crashes
CRASH_SCHEMA = Schema(
    name="chicago_crashes",
    columns={"crash_record_id": R("str"), "crash_date": R("datetime"),
             "latitude": R("float", min=41.6, max=42.1), "longitude": R("float", min=-88.0, max=-87.5),
             "posted_speed_limit": R("float", required=False, min=0, max=70),
             "injuries_total": R("float", required=False, min=0),
             "injuries_fatal": R("float", required=False, min=0),
             "injuries_incapacitating": R("float", required=False, min=0),
             "num_units": R("float", required=False, min=0, max=20),
             "weather_condition": R("str", required=False),
             "roadway_surface_cond": R("str", required=False),
             "lighting_condition": R("str", required=False),
             "most_severe_injury": R("str", required=False),
             "prim_contributory_cause": R("str", required=False),
             "first_crash_type": R("str", required=False),
             "trafficway_type": R("str", required=False)},
    key=["crash_record_id"], lat_col="latitude", lon_col="longitude",
    outlier_cols=["injuries_total", "num_units"],
)

SEVERITY_ORDER = ["No injury", "Minor", "Serious", "Fatal"]


def crash_severity(df: pd.DataFrame) -> pd.Series:
    sev = np.select(
        [df["injuries_fatal"].fillna(0) > 0,
         df["injuries_incapacitating"].fillna(0) > 0,
         df["injuries_total"].fillna(0) > 0],
        ["Fatal", "Serious", "Minor"], default="No injury")
    return pd.Series(pd.Categorical(sev, categories=SEVERITY_ORDER, ordered=True), index=df.index)


def clean_crashes(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = _read_parts("chicago_crashes") if raw is None else raw
    df = coerce(raw, CRASH_SCHEMA)
    bbox = sources()["geo"]["bbox"]["chicago"]
    CRASH_SCHEMA.bbox = bbox
    report = validate(df, CRASH_SCHEMA)
    df = df.dropna(subset=["crash_date"]).drop_duplicates("crash_record_id", keep="last")
    in_box = df["latitude"].between(bbox["min_lat"], bbox["max_lat"]) & \
        df["longitude"].between(bbox["min_lon"], bbox["max_lon"])
    df["has_valid_location"] = in_box
    df["severity"] = crash_severity(df)
    for c in ["weather_condition", "roadway_surface_cond", "lighting_condition"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").str.upper()
    df["hour"] = df["crash_date"].dt.hour
    df["dow"] = df["crash_date"].dt.dayofweek
    df["date"] = df["crash_date"].dt.normalize()
    report["rows_after_cleaning"] = len(df)
    report["rows_with_valid_location"] = int(in_box.sum())
    save_report(report)
    df.to_parquet(_processed("chicago_crashes.parquet"), index=False)
    return df


# --------------------------------------------------------------------------- Chicago traffic
TRAFFIC_SCHEMA = Schema(
    name="chicago_traffic",
    columns={"time": R("datetime"), "segment_id": R("int"), "speed": R("float", min=0, max=80),
             "length": R("float", required=False, min=0),
             "bus_count": R("float", required=False, min=0),
             "message_count": R("float", required=False, min=0),
             "start_latitude": R("float"), "start_longitude": R("float"),
             "end_latitude": R("float"), "end_longitude": R("float"),
             "street": R("str", required=False), "direction": R("str", required=False),
             "from_street": R("str", required=False), "to_street": R("str", required=False)},
    key=["segment_id", "time"], lat_col="start_latitude", lon_col="start_longitude",
    outlier_cols=["speed"],
)


def clean_traffic(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Speed == -1 means 'no estimate' in the tracker (too few bus GPS pings) -> NaN."""
    raw = _read_parts("chicago_traffic") if raw is None else raw
    df = coerce(raw, TRAFFIC_SCHEMA)
    TRAFFIC_SCHEMA.bbox = sources()["geo"]["bbox"]["chicago"]
    df.loc[df["speed"] < 0, "speed"] = np.nan
    report = validate(df, TRAFFIC_SCHEMA)
    df = df.dropna(subset=["time", "segment_id"]).drop_duplicates(["segment_id", "time"], keep="last")
    df["segment_id"] = df["segment_id"].astype("int64")
    report["no_estimate_pct"] = round(float(df["speed"].isna().mean() * 100), 3)
    report["segments"] = int(df["segment_id"].nunique())
    save_report(report)
    df.to_parquet(_processed("chicago_traffic.parquet"), index=False)

    seg_cols = ["segment_id", "street", "direction", "from_street", "to_street", "length",
                "start_latitude", "start_longitude", "end_latitude", "end_longitude"]
    segs = (df.sort_values("time").groupby("segment_id")[[c for c in seg_cols if c != "segment_id"]]
            .last().reset_index())
    segs.to_parquet(_processed("chicago_segments.parquet"), index=False)
    return df


def traffic_grid(df: pd.DataFrame, bin_minutes: int = 15) -> pd.DataFrame:
    """Resample irregular ~10-minute tracker observations onto a regular per-segment grid
    (mean speed per bin; empty bins stay NaN and are *not* forward-filled)."""
    df = df.dropna(subset=["speed"]).copy()
    df["ts"] = df["time"].dt.floor(f"{bin_minutes}min")
    g = df.groupby(["segment_id", "ts"], as_index=False).agg(speed=("speed", "mean"),
                                                             bus_count=("bus_count", "sum"))
    full_idx = pd.MultiIndex.from_product(
        [g["segment_id"].unique(), pd.date_range(g["ts"].min(), g["ts"].max(), freq=f"{bin_minutes}min")],
        names=["segment_id", "ts"])
    return g.set_index(["segment_id", "ts"]).reindex(full_idx).reset_index()


# --------------------------------------------------------------------------- NYC DOT speeds
DOT_SCHEMA = Schema(
    name="nyc_dot_speeds",
    columns={"id": R("str"), "speed": R("float", min=0, max=100), "travel_time": R("float", min=0),
             "data_as_of": R("datetime"), "link_id": R("str", required=False),
             "link_points": R("str", required=False), "borough": R("str", required=False),
             "link_name": R("str", required=False), "status": R("str", required=False)},
    key=["id", "data_as_of"], outlier_cols=["speed", "travel_time"],
)


def _parse_link_points(s: str | None) -> list[list[float]] | None:
    """link_points is a space-separated list of 'lat,lon' pairs (sometimes truncated)."""
    if not isinstance(s, str):
        return None
    pts = []
    for tok in s.split():
        parts = tok.split(",")
        if len(parts) != 2:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if 40.4 < lat < 41.0 and -74.3 < lon < -73.6:
            pts.append([lon, lat])
    return pts if len(pts) >= 2 else None


def clean_dot_speeds(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = _read_parts("nyc_dot_speeds") if raw is None else raw
    df = coerce(raw, DOT_SCHEMA)
    report = validate(df, DOT_SCHEMA)
    df = df.dropna(subset=["data_as_of", "speed"]).drop_duplicates(["id", "data_as_of"], keep="last")
    df = df[df["speed"].between(0, 100)]
    df["path"] = df["link_points"].map(_parse_link_points)
    report["links_with_geometry"] = int(df.dropna(subset=["path"])["id"].nunique())
    save_report(report)
    df.drop(columns=["path"]).to_parquet(_processed("nyc_dot_speeds.parquet"), index=False)
    links = df.dropna(subset=["path"]).sort_values("data_as_of").groupby("id").last().reset_index()
    links[["id", "link_name", "borough", "path"]].to_parquet(_processed("nyc_dot_links.parquet"), index=False)
    return df


# --------------------------------------------------------------------------- NYC TLC
def clean_tlc() -> dict[str, int]:
    """Unify yellow/green (and fhvhv if present) trips with DuckDB directly over Parquet,
    then build: a cleaned trip sample, zone-hour demand panel and hourly OD matrix."""
    raw_dir = get_settings().raw_dir / "nyc_tlc"
    cfg = model_config()["travel_time"]
    con = duckdb.connect()
    selects = []
    if list(raw_dir.glob("yellow_*.parquet")):
        selects.append(f"""SELECT 'yellow' AS service, tpep_pickup_datetime AS pickup, tpep_dropoff_datetime AS dropoff,
            trip_distance AS distance_mi, PULocationID AS pu, DOLocationID AS "do", passenger_count
            FROM read_parquet('{raw_dir}/yellow_*.parquet')""")
    if list(raw_dir.glob("green_*.parquet")):
        selects.append(f"""SELECT 'green', lpep_pickup_datetime, lpep_dropoff_datetime, trip_distance,
            PULocationID, DOLocationID, passenger_count FROM read_parquet('{raw_dir}/green_*.parquet')""")
    if list(raw_dir.glob("fhvhv_*.parquet")):
        selects.append(f"""SELECT 'fhvhv', pickup_datetime, dropoff_datetime, trip_miles,
            PULocationID, DOLocationID, NULL FROM read_parquet('{raw_dir}/fhvhv_*.parquet')""")
    if not selects:
        raise FileNotFoundError("no TLC partitions; run ingestion first")
    con.execute("CREATE TEMP VIEW trips_raw AS " + " UNION ALL ".join(selects))
    stats = con.execute("""SELECT count(*), min(pickup), max(pickup) FROM trips_raw""").fetchone()
    con.execute(f"""
        CREATE TEMP TABLE trips AS
        SELECT *, date_diff('second', pickup, dropoff)/60.0 AS duration_min
        FROM trips_raw
        WHERE pickup IS NOT NULL AND dropoff > pickup
          AND distance_mi > 0 AND distance_mi < 100
          AND pu BETWEEN 1 AND 263 AND "do" BETWEEN 1 AND 263
          AND date_diff('second', pickup, dropoff)/60.0 BETWEEN {cfg['min_duration_min']} AND {cfg['max_duration_min']}
          AND distance_mi / (date_diff('second', pickup, dropoff)/3600.0) <= {cfg['max_speed_mph']}
    """)
    # keep only the months the files are named for (TLC files contain a few stray timestamps)
    con.execute("""DELETE FROM trips WHERE pickup < (SELECT date_trunc('month', quantile_cont(pickup, 0.001)) FROM trips)
                   OR pickup >= (SELECT date_trunc('month', quantile_cont(pickup, 0.999)) + INTERVAL 1 MONTH FROM trips)""")
    kept = con.execute("SELECT count(*) FROM trips").fetchone()[0]
    out = get_settings().processed_dir
    con.execute(f"""COPY (SELECT date_trunc('hour', pickup) AS ts, pu AS zone, count(*) AS pickups
                         FROM trips GROUP BY 1, 2 ORDER BY 1, 2)
                    TO '{out}/tlc_pickups_zone_hour.parquet' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT date_trunc('hour', dropoff) AS ts, "do" AS zone, count(*) AS dropoffs
                         FROM trips GROUP BY 1, 2 ORDER BY 1, 2)
                    TO '{out}/tlc_dropoffs_zone_hour.parquet' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT hour(pickup) AS hour, isodow(pickup) AS dow, pu, "do", count(*) AS trips,
                                median(duration_min) AS median_duration_min, median(distance_mi) AS median_distance_mi
                         FROM trips GROUP BY 1,2,3,4)
                    TO '{out}/tlc_od_hour.parquet' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT * FROM trips USING SAMPLE reservoir({int(cfg['sample_trips'])} ROWS) REPEATABLE (42))
                    TO '{out}/tlc_trips_sample.parquet' (FORMAT PARQUET)""")
    report = {"dataset": "nyc_tlc", "checked_at": pd.Timestamp.now('UTC').isoformat(),
              "rows": int(stats[0]), "rows_after_cleaning": int(kept),
              "removed_pct": round(100 * (1 - kept / max(stats[0], 1)), 3),
              "pickup_min": str(stats[1]), "pickup_max": str(stats[2])}
    save_report(report)
    return {"raw": int(stats[0]), "clean": int(kept)}


# --------------------------------------------------------------------------- Weather
WEATHER_SCHEMA = Schema(
    name="weather",
    columns={"time": R("datetime"), "city": R("str"), "temperature_2m": R("float", min=-50, max=55),
             "precipitation": R("float", min=0, max=200), "rain": R("float", required=False, min=0),
             "snowfall": R("float", required=False, min=0),
             "wind_speed_10m": R("float", min=0, max=250),
             "relative_humidity_2m": R("float", min=0, max=100), "weather_code": R("float")},
    key=["city", "time"],
)

# WMO weather interpretation codes -> coarse categories
def weather_category(code: pd.Series) -> pd.Series:
    c = code.fillna(-1)
    return pd.Series(np.select(
        [c <= 1, c <= 3, c.isin([45, 48]), c.between(51, 67) | c.between(80, 82),
         c.between(71, 77) | c.isin([85, 86]), c >= 95],
        ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Thunderstorm"], default="Unknown"), index=code.index)


def clean_weather(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None:
        parts = list_partitions("weather", "archive_*.parquet")
        if not parts:
            raise FileNotFoundError("no weather archive partitions")
        raw = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = coerce(raw, WEATHER_SCHEMA)
    report = validate(df, WEATHER_SCHEMA)
    df = df.dropna(subset=["time"]).drop_duplicates(["city", "time"], keep="last").sort_values(["city", "time"])
    # short gaps only: interpolate up to 3 hours, never extrapolate
    num = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]
    df[num] = df.groupby("city")[num].transform(lambda s: s.interpolate(limit=3, limit_area="inside"))
    df["precipitation"] = df["precipitation"].fillna(0)
    df["weather_category"] = weather_category(df["weather_code"])
    df["rain_intensity"] = pd.cut(df["precipitation"], [-0.01, 0, 2.5, 7.6, 1e9],
                                  labels=["None", "Light", "Moderate", "Heavy"]).astype(str)
    save_report(report)
    df.to_parquet(_processed("weather_hourly.parquet"), index=False)
    return df
