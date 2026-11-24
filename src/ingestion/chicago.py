"""Chicago Traffic Crashes (85ca-t3if) and Chicago Traffic Tracker historical segment
speeds (4g9f-3jbs). Both pulls are incremental against a stored watermark."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.common.config import sources
from src.common.logging import get_logger
from src.common.runs import get_watermark, set_watermark, track_run
from src.ingestion.socrata import SocrataClient, SocrataDataset
from src.ingestion.storage import raw_path, write_parquet_atomic

log = get_logger(__name__)


def _parse_ts(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=None)


def ingest_crashes(client: SocrataClient | None = None, days_back: int | None = None) -> int:
    cfg = sources()["chicago_crashes"]
    client = client or SocrataClient()
    ds = SocrataDataset(cfg["domain"], cfg["dataset_id"])
    with track_run("chicago_crashes") as run:
        select = client.verify_columns(ds, cfg["select"])
        wm = get_watermark("chicago_crashes")
        start = (_parse_ts(wm) if wm
                 else datetime.now() - timedelta(days=days_back or cfg["days_back"]))
        df = client.fetch_window(ds, select=select, date_column=cfg["date_column"],
                                 start=start, page_size=cfg["page_size"])
        run.records = len(df)
        if df.empty:
            run.details["note"] = "no new rows since watermark"
            return 0
        ts = pd.to_datetime(df[cfg["date_column"]])
        run.data_min_ts, run.data_max_ts = str(ts.min()), str(ts.max())
        part = raw_path("chicago_crashes", f"crashes_{ts.min():%Y%m%d}_{ts.max():%Y%m%d%H%M}.parquet")
        write_parquet_atomic(df, part)
        set_watermark("chicago_crashes", run.data_max_ts)
        run.details["partition"] = part.name
        return len(df)


def ingest_traffic_segments(client: SocrataClient | None = None, days_back: int | None = None) -> int:
    """The tracker is archived, so the window is anchored on the dataset's own last
    timestamp (not on today) and every row is labelled as historical downstream."""
    cfg = sources()["chicago_traffic"]
    client = client or SocrataClient()
    ds = SocrataDataset(cfg["domain"], cfg["dataset_id"])
    with track_run("chicago_traffic") as run:
        meta = client.metadata(ds)
        run.details.update({"dataset_name": meta["name"], "rows_updated_at": meta["rows_updated_at"],
                            "deprecated": meta["deprecated"], "is_live": False})
        select = client.verify_columns(ds, cfg["select"])
        last = client.max_value(ds, cfg["date_column"])
        if not last:
            raise RuntimeError("could not determine last timestamp of traffic dataset")
        end = _parse_ts(last)
        wm = get_watermark("chicago_traffic")
        start = _parse_ts(wm) if wm else end - timedelta(days=days_back or cfg["days_back"])
        if start >= end:
            run.details["note"] = "archive already fully ingested for configured window"
            return 0
        # pull week by week so one failure doesn't lose the whole window
        total, cursor = 0, start
        while cursor < end:
            stop = min(cursor + timedelta(days=7), end)
            df = client.fetch_window(ds, select=select, date_column=cfg["date_column"],
                                     start=cursor, end=stop, page_size=cfg["page_size"])
            if not df.empty:
                part = raw_path("chicago_traffic", f"segments_{cursor:%Y%m%d}_{stop:%Y%m%d}.parquet")
                write_parquet_atomic(df, part)
                total += len(df)
                set_watermark("chicago_traffic", stop.isoformat())
            cursor = stop
        run.records = total
        run.data_min_ts, run.data_max_ts = start.isoformat(), end.isoformat()
        return total
