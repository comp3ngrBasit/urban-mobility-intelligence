"""NYC DOT Traffic Speeds (i4gi-tjb9) — the platform's live traffic component.

Each pull is saved as its own snapshot partition. If the API is down, the API layer serves
the newest snapshot on disk and labels it with its real `data_as_of` timestamp."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.common.config import sources
from src.common.runs import get_watermark, set_watermark, track_run
from src.ingestion.socrata import SocrataClient, SocrataDataset
from src.ingestion.storage import raw_path, write_parquet_atomic


def ingest_dot_speeds(client: SocrataClient | None = None, hours_back: int | None = None) -> int:
    cfg = sources()["nyc_dot_speeds"]
    client = client or SocrataClient()
    ds = SocrataDataset(cfg["domain"], cfg["dataset_id"])
    with track_run("nyc_dot_speeds") as run:
        select = client.verify_columns(ds, cfg["select"])
        last = client.max_value(ds, cfg["date_column"])
        anchor = pd.Timestamp(last).to_pydatetime() if last else datetime.now()
        wm = get_watermark("nyc_dot_speeds")
        start = (pd.Timestamp(wm).to_pydatetime() if wm
                 else anchor - timedelta(hours=hours_back or cfg["hours_back"]))
        df = client.fetch_window(ds, select=select, date_column=cfg["date_column"], start=start,
                                 page_size=cfg["page_size"])
        run.records = len(df)
        run.details["source_last_timestamp"] = last
        if df.empty:
            run.details["note"] = "no new observations"
            return 0
        ts = pd.to_datetime(df[cfg["date_column"]])
        run.data_min_ts, run.data_max_ts = str(ts.min()), str(ts.max())
        part = raw_path("nyc_dot_speeds", f"speeds_{ts.max():%Y%m%dT%H%M%S}.parquet")
        write_parquet_atomic(df, part)
        set_watermark("nyc_dot_speeds", run.data_max_ts)
        return len(df)
