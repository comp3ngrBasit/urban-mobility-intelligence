"""OpenStreetMap road networks (OSMnx) and infrastructure features (Overpass)."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.common.config import get_settings, sources
from src.common.http import build_session
from src.common.runs import track_run


def ingest_road_network(city: str) -> int:
    import osmnx as ox  # heavy import, only when needed

    cfg = sources()["osm"]
    with track_run(f"osm_network_{city}") as run:
        G = ox.graph_from_place(cfg["places"][city], network_type=cfg["network_type"], simplify=True)
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()
        keep = [c for c in ["u", "v", "key", "osmid", "highway", "name", "length", "maxspeed",
                            "lanes", "oneway", "bridge", "tunnel", "geometry"] if c in edges.columns]
        edges = edges[keep].copy()
        # list-valued OSM tags -> first value, string-typed for Parquet
        for c in ["osmid", "highway", "name", "maxspeed", "lanes", "bridge", "tunnel"]:
            if c in edges.columns:
                edges[c] = edges[c].apply(lambda v: v[0] if isinstance(v, list) else v).astype("string")
        nodes = nodes.reset_index()[["osmid", "y", "x", "street_count", "geometry"]]
        ext = get_settings().external_dir
        edges.to_parquet(ext / f"osm_edges_{city}.parquet")
        nodes.to_parquet(ext / f"osm_nodes_{city}.parquet")
        run.records = len(edges)
        run.details["nodes"] = len(nodes)
        return len(edges)


OVERPASS_QUERY = """
[out:json][timeout:180];
area["name"="{name}"]["boundary"="administrative"]->.a;
(
  way["bridge"="yes"]["highway"](area.a);
  way["tunnel"="yes"]["highway"](area.a);
  node["highway"="traffic_signals"](area.a);
);
out center tags;
"""


def ingest_infrastructure(city: str, area_name: str) -> int:
    cfg = sources()["osm"]
    session = build_session()
    with track_run(f"osm_infra_{city}") as run:
        resp = session.post(cfg["overpass_url"], data={"data": OVERPASS_QUERY.format(name=area_name)},
                            timeout=240)
        resp.raise_for_status()
        rows = []
        for el in resp.json().get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            tags = el.get("tags", {})
            kind = ("traffic_signal" if tags.get("highway") == "traffic_signals"
                    else "bridge" if tags.get("bridge") == "yes" else "tunnel")
            rows.append({"osm_id": el["id"], "kind": kind, "lat": lat, "lon": lon,
                         "name": tags.get("name"), "highway": tags.get("highway")})
        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
        gdf.to_parquet(get_settings().external_dir / f"osm_infra_{city}.parquet")
        run.records = len(gdf)
        return len(gdf)
