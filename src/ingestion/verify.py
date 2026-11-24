"""Source verification: checks each endpoint, schema and freshness before ingestion.
Writes artifacts/analytics/source_verification.json for the monitoring page."""
from __future__ import annotations

import json

from src.common.config import get_settings, sources
from src.common.http import SourceUnavailable, build_session
from src.common.runs import utcnow_iso
from src.ingestion.socrata import SocrataClient, SocrataDataset


def verify_all() -> list[dict]:
    cfg = sources()
    client = SocrataClient()
    out: list[dict] = []
    for key in ("chicago_crashes", "chicago_traffic", "nyc_dot_speeds"):
        c = cfg[key]
        ds = SocrataDataset(c["domain"], c["dataset_id"])
        entry = {"source": key, "dataset_id": c["dataset_id"], "endpoint": ds.resource_url,
                 "checked_at": utcnow_iso(), "is_live": c.get("is_live", False)}
        try:
            meta = client.metadata(ds)
            missing = sorted(set(c["select"]) - set(meta["columns"]))
            entry.update(status="ok", name=meta["name"], rows_updated_at=meta["rows_updated_at"],
                         deprecated=meta["deprecated"], missing_columns=missing,
                         last_observation=client.max_value(ds, c["date_column"]))
        except SourceUnavailable as exc:
            entry.update(status="unreachable", error=str(exc))
        out.append(entry)

    session = build_session()
    simple = {
        "nyc_tlc": cfg["nyc_tlc"]["base_url"] + cfg["nyc_tlc"]["zone_lookup_path"],
        "open_meteo_archive": cfg["open_meteo"]["archive_url"]
        + "?latitude=41.88&longitude=-87.63&start_date=2024-01-01&end_date=2024-01-01&hourly=temperature_2m",
        "open_meteo_forecast": cfg["open_meteo"]["forecast_url"]
        + "?latitude=41.88&longitude=-87.63&hourly=temperature_2m&forecast_days=1",
        "overpass": cfg["osm"]["overpass_url"] + "?data=[out:json];node(1);out;",
    }
    for key, url in simple.items():
        entry = {"source": key, "endpoint": url.split("?")[0], "checked_at": utcnow_iso()}
        try:
            r = session.get(url, timeout=30)
            entry.update(status="ok" if r.status_code < 400 else "error", http_status=r.status_code)
        except Exception as exc:  # noqa: BLE001 — report any connectivity problem
            entry.update(status="unreachable", error=str(exc))
        out.append(entry)

    dest = get_settings().analytics_dir / "source_verification.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    for row in verify_all():
        print(f"{row['source']:<22} {row['status']:<12} {row.get('rows_updated_at') or ''}"
              f"{' DEPRECATED' if row.get('deprecated') else ''}")
