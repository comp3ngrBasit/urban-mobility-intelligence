"""Mobility, accident and weather analytics endpoints (all backed by computed artifacts)."""
from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.services import store
from src.evaluation.metrics import regression_metrics

router = APIRouter(prefix="/api", tags=["analytics"])


# ----------------------------------------------------------------- mobility (NYC TLC)
@router.get("/mobility")
def mobility(borough: str | None = None):
    m = dict(store.analytics("mobility"))
    if borough:
        m["zones"] = [z for z in m["zones"] if z.get("borough") == borough]
        m["od_top"] = [o for o in m["od_top"] if borough in (o.get("pu_borough"), o.get("do_borough"))]
    m.pop("decomposition_weekly", None)
    return m


@router.get("/mobility/decomposition")
def mobility_decomposition():
    return store.analytics("mobility").get("decomposition_weekly")


@router.get("/mobility/od")
def od_flows(min_trips: int = Query(0, ge=0), borough: str | None = None, hour: int | None = Query(None, ge=0, le=23),
             limit: int = Query(150, le=1000)):
    od = store.processed("tlc_od_hour.parquet", hint="python -m src.pipeline clean")
    if hour is not None:
        od = od[od["hour"] == hour]
    agg = od.groupby(["pu", "do"]).agg(trips=("trips", "sum"),
                                       median_duration_min=("median_duration_min", "median")).reset_index()
    agg = agg[(agg["pu"] != agg["do"]) & (agg["trips"] >= min_trips)]
    zl = store.external("taxi_zone_lookup.csv")
    if zl:
        lk = pd.read_csv(zl).rename(columns={"LocationID": "zone", "Zone": "zone_name", "Borough": "borough"})
        agg = agg.merge(lk.add_prefix("pu_"), left_on="pu", right_on="pu_zone", how="left") \
                 .merge(lk.add_prefix("do_"), left_on="do", right_on="do_zone", how="left")
        if borough:
            agg = agg[(agg["pu_borough"] == borough) | (agg["do_borough"] == borough)]
    zp = store.external("taxi_zones.parquet")
    if zp:
        import geopandas as gpd
        z = gpd.read_parquet(zp)
        c = z.set_index("LocationID").geometry.to_crs(2263).centroid.to_crs(4326)
        agg["pu_lon"], agg["pu_lat"] = agg["pu"].map(c.x), agg["pu"].map(c.y)
        agg["do_lon"], agg["do_lat"] = agg["do"].map(c.x), agg["do"].map(c.y)
    return {"hour": hour, "flows": store.records(agg.nlargest(limit, "trips").round(5))}


# ----------------------------------------------------------------- accidents (Chicago)
@router.get("/accidents/summary")
def accidents_summary():
    return store.analytics("accidents")


@router.get("/accidents/hotspots")
def hotspots(method: str = Query("gi", pattern="^(gi|dbscan|hdbscan|kmeans|kde)$")):
    h = store.analytics("accident_hotspots")
    base = {"n_crashes": h["n_crashes"], "period": h["period"], "morans_i": h["morans_i"],
            "significant_hotspots": h["significant_hotspots"], "method": method}
    if method == "gi":
        base["cells"] = h["h3_cells"]
    elif method == "kde":
        base["kde"] = h["kde"]
    else:
        base["clusters"] = h[method]
    return base


@router.get("/accidents/points")
def crash_points(start: date | None = None, end: date | None = None, severity: list[str] | None = Query(None),
                 weather: str | None = None, hour_from: int = Query(0, ge=0, le=23),
                 hour_to: int = Query(23, ge=0, le=23), limit: int = Query(20000, le=100000)):
    c = store.processed("chicago_crashes.parquet")
    c = c[c["has_valid_location"]]
    if start:
        c = c[c["crash_date"] >= pd.Timestamp(start)]
    if end:
        c = c[c["crash_date"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if severity:
        c = c[c["severity"].astype(str).isin(severity)]
    if weather:
        c = c[c["weather_condition"] == weather.upper()]
    c = c[c["hour"].between(hour_from, hour_to)]
    total = len(c)
    if total > limit:
        c = c.sample(limit, random_state=0)
    cols = ["crash_record_id", "crash_date", "latitude", "longitude", "severity", "weather_condition",
            "prim_contributory_cause", "injuries_total"]
    out = c[cols].copy()
    out["severity"] = out["severity"].astype(str)
    return {"total_matching": int(total), "returned": int(len(out)), "points": store.records(out)}


# ----------------------------------------------------------------- weather impact
@router.get("/weather/impact")
def weather_impact():
    return store.analytics("weather_impact")


@router.get("/weather/predictions-by-condition")
def predictions_by_weather(horizon: int = Query(60)):
    """How well the speed model performs, and what it predicts, under each observed
    weather condition in the held-out test period."""
    tp = store.model_frame(f"speed_h{horizon}", "test_predictions")
    out = {}
    for col in ("rain_intensity", "weather_category"):
        if col in tp.columns:
            rows = []
            for k, g in tp.dropna(subset=[col]).groupby(col):
                if len(g) >= 20:
                    rows.append({"condition": k, "n": int(len(g)), "actual_mean_speed": float(g["actual"].mean()),
                                 "predicted_mean_speed": float(g["predicted"].mean()),
                                 **{f"model_{k2}": v for k2, v in regression_metrics(g["actual"], g["predicted"]).items()
                                    if k2 in ("mae", "rmse")}})
            out[col] = rows
    return {"horizon_min": horizon, "by_condition": out}


# ----------------------------------------------------------------- road traffic (Chicago, archived)
@router.get("/traffic")
def traffic_summary(limit: int = Query(30, le=100)):
    t = store.analytics("traffic")
    out = {k: v for k, v in t.items() if k not in ("segment_stats", "spatial", "decomposition_daily")}
    out["bottlenecks"] = t["bottlenecks"][:limit]
    out["spatial"] = {"morans_i": t.get("spatial", {}).get("morans_i")}
    return out
