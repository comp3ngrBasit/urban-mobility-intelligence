"""NYC TLC trip records (Parquet) + taxi zone lookup and boundaries.

Downloads are month-partitioned and skipped when already present (incremental). After
download, only the configured columns are kept to bound disk and memory use."""
from __future__ import annotations

import zipfile
from datetime import date

import geopandas as gpd
import pyarrow.parquet as pq

from src.common.config import get_settings, sources
from src.common.http import build_session, download_file
from src.common.logging import get_logger
from src.common.runs import track_run
from src.ingestion.storage import raw_path

log = get_logger(__name__)


def _months(latest: str, n: int) -> list[tuple[int, int]]:
    y, m = map(int, latest.split("-"))
    out = []
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def _probe_latest(session, cfg) -> str:
    """Probe up to 3 months beyond the configured latest month (TLC publishes ~monthly)."""
    latest = cfg["latest_published"]
    y, m = map(int, latest.split("-"))
    today = date.today()
    for _ in range(3):
        m2, y2 = (1, y + 1) if m == 12 else (m + 1, y)
        if (y2, m2) >= (today.year, today.month):
            break
        url = cfg["base_url"] + cfg["trip_path"].format(service="yellow", year=y2, month=m2)
        if session.head(url, timeout=30).status_code != 200:
            break
        y, m = y2, m2
    return f"{y}-{m:02d}"


def ingest_zones() -> None:
    cfg = sources()["nyc_tlc"]
    session = build_session()
    with track_run("nyc_tlc_zones") as run:
        ext = get_settings().external_dir
        lookup = download_file(session, cfg["base_url"] + cfg["zone_lookup_path"], ext / "taxi_zone_lookup.csv")
        zpath = download_file(session, cfg["base_url"] + cfg["zone_shapes_path"], ext / "taxi_zones.zip")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ext / "taxi_zones")
        shp = next((ext / "taxi_zones").rglob("*.shp"))
        zones = gpd.read_file(shp).to_crs(4326)
        zones.to_parquet(ext / "taxi_zones.parquet")
        run.records = len(zones)
        run.details["lookup"] = lookup.name


def ingest_trips(services: list[str] | None = None, months_back: int | None = None) -> int:
    cfg = sources()["nyc_tlc"]
    session = build_session()
    total = 0
    with track_run("nyc_tlc_trips") as run:
        latest = _probe_latest(session, cfg)
        run.details["latest_month"] = latest
        months = _months(latest, months_back or cfg["months_back"])
        for service in services or cfg["services"]:
            for y, m in months:
                final = raw_path("nyc_tlc", f"{service}_{y}-{m:02d}.parquet")
                if final.exists():
                    log.info("skip %s (already ingested)", final.name)
                    continue
                url = cfg["base_url"] + cfg["trip_path"].format(service=service, year=y, month=m)
                full = final.with_suffix(".full.parquet")
                try:
                    download_file(session, url, full)
                except FileNotFoundError:
                    log.warning("not published yet: %s", url)
                    continue
                cols = [c for c in cfg["columns"][service] if c in pq.read_schema(full).names]
                table = pq.read_table(full, columns=cols)
                pq.write_table(table, final, compression="zstd")
                full.unlink()
                total += table.num_rows
        run.records = total
        run.data_min_ts = f"{months[-1][0]}-{months[-1][1]:02d}-01"
        run.data_max_ts = latest
    return total
