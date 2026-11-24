"""Live Traffic Intelligence: NYC DOT live speeds (with freshness label) and the latest
archived Chicago snapshot (always labelled historical)."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.core.errors import NotReady
from backend.app.services import freshness, store

router = APIRouter(prefix="/api/live", tags=["live"])


def nyc_latest() -> tuple[pd.DataFrame, pd.Timestamp | None]:
    df = store.processed("nyc_dot_speeds.parquet")
    latest = df.sort_values("data_as_of").groupby("id").tail(1)
    links = store.processed("nyc_dot_links.parquet") if store.processed_exists("nyc_dot_links.parquet") else None
    if links is not None:
        latest = latest.merge(links[["id", "path"]], on="id", how="left")
    return latest, df["data_as_of"].max()


@router.get("/nyc")
def nyc_live():
    latest, as_of = nyc_latest()
    fr = freshness.nyc_dot(as_of)
    hist = store.processed("nyc_dot_speeds.parquet")
    # per-link typical speed for the same hour (from stored history) -> relative condition
    hist = hist.assign(hour=hist["data_as_of"].dt.hour)
    typical = hist.groupby(["id", "hour"])["speed"].median().rename("typical_speed").reset_index()
    latest = latest.assign(hour=latest["data_as_of"].dt.hour).merge(typical, on=["id", "hour"], how="left")
    latest["relative_speed"] = latest["speed"] / latest["typical_speed"]
    try:
        an = store.analytics_frame("anomalies_nyc.parquet")
        recent = an[an["data_as_of"] >= as_of - pd.Timedelta(hours=1)]
        latest["anomaly"] = latest["id"].isin(recent["id"])
    except NotReady:
        latest["anomaly"] = False
    cols = ["id", "link_name", "borough", "speed", "travel_time", "data_as_of", "typical_speed",
            "relative_speed", "anomaly"] + (["path"] if "path" in latest.columns else [])
    latest["path"] = latest["path"].map(lambda p: p.tolist() if hasattr(p, "tolist") else p) \
        if "path" in latest.columns else None
    return {"freshness": fr.model_dump(), "links": store.records(latest[cols].round(3)),
            "summary": {"links": int(len(latest)), "median_speed_mph": float(latest["speed"].median()),
                        "slow_links": int((latest["relative_speed"] < 0.6).sum())}}


@router.get("/nyc/{link_id}/history")
def nyc_link_history(link_id: str, hours: int = Query(72, ge=1, le=720)):
    df = store.processed("nyc_dot_speeds.parquet")
    d = df[df["id"] == link_id].sort_values("data_as_of")
    if d.empty:
        raise NotReady(f"link {link_id}", "unknown link id")
    d = d[d["data_as_of"] >= d["data_as_of"].max() - pd.Timedelta(hours=hours)]
    return {"link_id": link_id, "link_name": d["link_name"].iloc[-1],
            "series": store.records(d[["data_as_of", "speed", "travel_time"]])}


@router.get("/chicago")
def chicago_latest():
    t = store.processed("chicago_traffic.parquet")
    segs = store.processed("chicago_segments.parquet")
    end = t["time"].max()
    snap = t[t["time"] >= end - pd.Timedelta(minutes=30)].dropna(subset=["speed"])
    snap = snap.sort_values("time").groupby("segment_id").tail(1)
    if store.processed_exists("chicago_free_flow.parquet"):
        ff = store.processed("chicago_free_flow.parquet")
        snap = snap.merge(ff, on="segment_id", how="left")
        from src.features.traffic import congestion_label
        snap["congestion_level"] = congestion_label(snap["speed"] / snap["free_flow_speed"]).astype(str)
    snap = snap.drop(columns=[c for c in segs.columns if c in snap.columns and c != "segment_id"]) \
        .merge(segs, on="segment_id", how="left")
    return {"freshness": freshness.chicago_traffic(end).model_dump(), "segments": store.records(snap.round(5))}


@router.get("/anomalies")
def anomalies(city: str = Query("nyc", pattern="^(nyc|chicago)$"), limit: int = Query(200, le=2000)):
    df = store.analytics_frame(f"anomalies_{city}.parquet")
    ts = "data_as_of" if city == "nyc" else "time"
    idc = "id" if city == "nyc" else "segment_id"
    d = df.sort_values(ts, ascending=False).head(limit)
    cols = [idc, ts, "speed", "profile_median", "robust_z", "iso_score", "anomaly_type", "z_flag", "iso_flag", "lof_flag"]
    summary = store.analytics("anomaly_summary").get(city)
    return {"city": city, "summary": summary, "mode": "live" if city == "nyc" else "historical",
            "anomalies": store.records(d[[c for c in cols if c in d.columns]].round(3))}
