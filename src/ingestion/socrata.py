"""Generic Socrata (SODA 2.x) client: metadata check, date-window filtering, column
selection and stable keyset-style pagination with $order + $offset."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd

from src.common.config import get_settings
from src.common.http import build_session, get_json
from src.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class SocrataDataset:
    domain: str
    dataset_id: str

    @property
    def resource_url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.json"

    @property
    def metadata_url(self) -> str:
        return f"https://{self.domain}/api/views/{self.dataset_id}.json"


class SocrataClient:
    def __init__(self) -> None:
        token = get_settings().socrata_app_token
        self.session = build_session({"X-App-Token": token} if token else None)

    # ---- metadata / verification ----
    def metadata(self, ds: SocrataDataset) -> dict[str, Any]:
        meta = get_json(self.session, ds.metadata_url)
        rows_updated = meta.get("rowsUpdatedAt")
        return {
            "name": meta.get("name"),
            "rows_updated_at": (datetime.fromtimestamp(rows_updated, tz=timezone.utc).isoformat()
                                if rows_updated else None),
            "columns": [c["fieldName"] for c in meta.get("columns", [])],
            "deprecated": "no longer updated" in (meta.get("description") or "").lower(),
        }

    def verify_columns(self, ds: SocrataDataset, wanted: list[str]) -> list[str]:
        """Return the subset of requested columns that the live schema actually has."""
        available = set(self.metadata(ds)["columns"])
        missing = [c for c in wanted if c not in available]
        if missing:
            log.warning("%s: columns missing from live schema, dropped: %s", ds.dataset_id, missing)
        return [c for c in wanted if c in available]

    def max_value(self, ds: SocrataDataset, column: str) -> str | None:
        rows = get_json(self.session, ds.resource_url, {"$select": f"max({column}) AS mx"})
        return rows[0].get("mx") if rows else None

    # ---- paginated pull ----
    def iter_pages(self, ds: SocrataDataset, *, select: list[str], where: str | None,
                   order: str, page_size: int = 50_000,
                   max_rows: int | None = None) -> Iterator[pd.DataFrame]:
        offset = 0
        while True:
            params: dict[str, Any] = {"$select": ",".join(select), "$order": order,
                                      "$limit": page_size, "$offset": offset}
            if where:
                params["$where"] = where
            rows = get_json(self.session, ds.resource_url, params)
            if not rows:
                break
            yield pd.DataFrame.from_records(rows)
            offset += len(rows)
            log.info("%s: fetched %d rows", ds.dataset_id, offset)
            if len(rows) < page_size or (max_rows and offset >= max_rows):
                break

    def fetch_window(self, ds: SocrataDataset, *, select: list[str], date_column: str,
                     start: datetime, end: datetime | None = None, page_size: int = 50_000,
                     extra_where: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
        fmt = "%Y-%m-%dT%H:%M:%S"
        where = f"{date_column} > '{start.strftime(fmt)}'"
        if end is not None:
            where += f" AND {date_column} <= '{end.strftime(fmt)}'"
        if extra_where:
            where += f" AND ({extra_where})"
        # order by date then a tie-breaker to keep offset pagination stable
        order = f"{date_column} ASC, :id"
        frames = list(self.iter_pages(ds, select=select, where=where, order=order,
                                      page_size=page_size, max_rows=max_rows))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=select)
