"""Geospatial Explorer: GeoJSON layers built from processed data."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.core.errors import NotReady
from backend.app.services import store
from src.geospatial.h3utils import cell_polygon

router = APIRouter(prefix="/api/geo", tags=["geo"])


def _fc(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


@router.get("/taxi-zones")
def taxi_zones(metric: str = Query("pickups_per_day")):
    import geopandas as gpd
    zp = store.external("taxi_zones.parquet")
    if not zp:
        raise NotReady("taxi_zones", "python -m src.pipeline ingest --source tlc_zones")
    z = gpd.read_parquet(zp)
    z["geometry"] = z.geometry.simplify(0.0003)
    try:
        stats = pd.DataFrame(store.analytics("mobility")["zones"])
        z = z.merge(stats.drop(columns=[c for c in ("zone_name", "borough") if c in stats]),
                    left_on="LocationID", right_on="zone", how="left")
    except NotReady:
        pass
    keep = [c for c in z.columns if c in ("LocationID", "zone", "borough", "pickups", "dropoffs",
                                          "pickups_per_day", "net_flow_per_day", "cluster", "geometry")]
    return json.loads(z[keep].to_json())


@router.get("/chicago-segments")
def chicago_segments():
    tr = store.analytics("traffic")
    feats = []
    for s in tr["segment_stats"]:
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": [[s["start_longitude"], s["start_latitude"]],
                                                                         [s["end_longitude"], s["end_latitude"]]]},
                      "properties": {k: s.get(k) for k in ("segment_id", "street", "direction", "from_street",
                                                           "to_street", "mean_speed", "mean_ratio", "p_congested")}})
    return _fc(feats)


@router.get("/road-network")
def road_network(city: str = Query("chicago", pattern="^(chicago|nyc)$"), min_rank: int = Query(3, ge=0, le=6)):
    import geopandas as gpd
    from src.geospatial.road_features import ROAD_RANK
    p = store.external(f"osm_edges_{city}.parquet")
    if not p:
        raise NotReady(f"osm_edges_{city}", f"python -m src.pipeline ingest --source osm_{city}")
    e = gpd.read_parquet(p)
    e["road_class"] = e["highway"].astype(str).str.replace("_link", "", regex=False)
    e["road_rank"] = e["road_class"].map(ROAD_RANK).fillna(0)
    e = e[e["road_rank"] >= min_rank].copy()
    e["geometry"] = e.geometry.simplify(0.00005)
    return json.loads(e[["road_class", "road_rank", "name", "bridge", "tunnel", "geometry"]].to_json())


@router.get("/h3")
def h3_layer(layer: str = Query("crashes", pattern="^(crashes|congestion|priority)$")):
    if layer == "crashes":
        rows = store.analytics("accident_hotspots")["h3_cells"]
        props = ("crashes", "serious", "gi_z", "gi_class", "lisa_cluster")
    elif layer == "congestion":
        rows = store.analytics("traffic").get("spatial", {}).get("local", [])
        props = ("value", "gi_z", "gi_class", "lisa_cluster")
    else:
        rows = store.analytics("infrastructure_priority")["cells"]
        props = ("priority_index", "serious", "congestion_share", "z_serious", "z_congestion", "gi_class")
    feats = [{"type": "Feature", "geometry": cell_polygon(r["h3"]).__geo_interface__,
              "properties": {"h3": r["h3"], **{k: r.get(k) for k in props}}} for r in rows]
    return _fc(feats)


@router.get("/infrastructure")
def infrastructure(city: str = "chicago"):
    import geopandas as gpd
    p = store.external(f"osm_infra_{city}.parquet")
    if not p:
        raise NotReady(f"osm_infra_{city}", "python -m src.pipeline ingest --source osm_infra_chicago")
    g = gpd.read_parquet(p)
    return {"counts": g["kind"].value_counts().to_dict(),
            "points": store.records(pd.DataFrame(g.drop(columns="geometry")))}


@router.get("/infrastructure-priority")
def infrastructure_priority(limit: int = Query(50, le=500)):
    d = store.analytics("infrastructure_priority")
    return {"method": d["method"], "cells": d["cells"][:limit]}
