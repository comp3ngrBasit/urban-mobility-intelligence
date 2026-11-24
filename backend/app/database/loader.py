"""Load processed Parquet into PostGIS with upserts (ON CONFLICT DO UPDATE), so repeated
pipeline runs accumulate history instead of duplicating it."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPolygon
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from backend.app.database.models import Base
from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)


def engine():
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(url, pool_pre_ping=True)


def init_db(eng) -> None:
    with eng.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(eng)


def _upsert(eng, table: str, df: pd.DataFrame, keys: list[str], chunk: int = 20_000) -> int:
    tbl = Base.metadata.tables[table]
    cols = [c.name for c in tbl.columns if c.name in df.columns]
    df = df[cols].astype(object).where(pd.notna(df[cols]), None)
    n = 0
    with eng.begin() as conn:
        for i in range(0, len(df), chunk):
            rows = df.iloc[i:i + chunk].to_dict("records")
            stmt = insert(tbl).values(rows)
            upd = {c: stmt.excluded[c] for c in cols if c not in keys}
            conn.execute(stmt.on_conflict_do_update(index_elements=keys, set_=upd) if upd
                         else stmt.on_conflict_do_nothing(index_elements=keys))
            n += len(rows)
    log.info("upserted %d rows into %s", n, table)
    return n


def load_all() -> None:
    eng = engine()
    init_db(eng)
    p = get_settings().processed_dir
    if (p / "chicago_segments.parquet").exists():
        s = pd.read_parquet(p / "chicago_segments.parquet")
        s["geom"] = [f"SRID=4326;{LineString([(r.start_longitude, r.start_latitude), (r.end_longitude, r.end_latitude)]).wkt}"
                     for r in s.itertuples()]
        _upsert(eng, "chicago_segments", s, ["segment_id"])
    if (p / "chicago_traffic.parquet").exists():
        _upsert(eng, "chicago_speeds", pd.read_parquet(p / "chicago_traffic.parquet"), ["segment_id", "time"])
    if (p / "chicago_crashes.parquet").exists():
        c = pd.read_parquet(p / "chicago_crashes.parquet")
        c["severity"] = c["severity"].astype(str)
        c["geom"] = [f"SRID=4326;POINT({lo} {la})" if ok else None
                     for lo, la, ok in zip(c.longitude, c.latitude, c.has_valid_location)]
        _upsert(eng, "chicago_crashes", c, ["crash_record_id"])
    if (p / "nyc_dot_speeds.parquet").exists():
        _upsert(eng, "nyc_dot_speeds", pd.read_parquet(p / "nyc_dot_speeds.parquet"), ["id", "data_as_of"])
    if (p / "tlc_pickups_zone_hour.parquet").exists():
        _upsert(eng, "tlc_pickups_zone_hour", pd.read_parquet(p / "tlc_pickups_zone_hour.parquet"), ["zone", "ts"])
    if (p / "weather_hourly.parquet").exists():
        _upsert(eng, "weather_hourly", pd.read_parquet(p / "weather_hourly.parquet"), ["city", "time"])
    zp = get_settings().external_dir / "taxi_zones.parquet"
    if zp.exists():
        z = gpd.read_parquet(zp)
        z["geom"] = [f"SRID=4326;{(g if g.geom_type == 'MultiPolygon' else MultiPolygon([g])).wkt}" for g in z.geometry]
        z = pd.DataFrame(z.drop(columns="geometry")).rename(columns={"LocationID": "location_id"})
        _upsert(eng, "taxi_zones", z, ["location_id"])
