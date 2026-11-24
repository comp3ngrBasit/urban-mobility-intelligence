"""Road-network features for Chicago tracker segments from the OSM drive network:
road class of the nearest OSM edge, intersection density, connectivity (mean node degree)
and distance to the nearest major road. All distances in metres (EPSG:26916, UTM 16N)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.common.config import get_settings, sources
from src.common.logging import get_logger

log = get_logger(__name__)
UTM_CHICAGO = 26916

ROAD_RANK = {"motorway": 6, "trunk": 5, "primary": 4, "secondary": 3, "tertiary": 2,
             "unclassified": 1, "residential": 1}


def segments_gdf(segs: pd.DataFrame) -> gpd.GeoDataFrame:
    geom = [LineString([(r.start_longitude, r.start_latitude), (r.end_longitude, r.end_latitude)])
            for r in segs.itertuples()]
    return gpd.GeoDataFrame(segs.copy(), geometry=geom, crs=4326)


def build_segment_road_features(segs: pd.DataFrame, city: str = "chicago",
                                radius_m: float = 500) -> pd.DataFrame:
    ext = get_settings().external_dir
    edges_p, nodes_p = ext / f"osm_edges_{city}.parquet", ext / f"osm_nodes_{city}.parquet"
    g = segments_gdf(segs).to_crs(UTM_CHICAGO)
    out = pd.DataFrame({"segment_id": segs["segment_id"].values})
    out["segment_length_m"] = g.geometry.length.values
    if not edges_p.exists():
        log.warning("OSM network not ingested for %s; road features left empty", city)
        return out
    edges = gpd.read_parquet(edges_p).to_crs(UTM_CHICAGO)
    nodes = gpd.read_parquet(nodes_p).to_crs(UTM_CHICAGO)
    edges["road_class"] = edges["highway"].astype(str).str.replace("_link", "", regex=False)
    edges["road_rank"] = edges["road_class"].map(ROAD_RANK).fillna(0)

    mids = g.copy()
    mids["geometry"] = g.geometry.interpolate(0.5, normalized=True)
    near = gpd.sjoin_nearest(mids[["segment_id", "geometry"]],
                             edges[["road_class", "road_rank", "lanes", "maxspeed", "geometry"]],
                             how="left", distance_col="dist_nearest_edge_m")
    near = near.drop_duplicates("segment_id")
    out = out.merge(near[["segment_id", "road_class", "road_rank", "lanes", "maxspeed"]],
                    on="segment_id", how="left")
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce")
    out["osm_maxspeed_mph"] = pd.to_numeric(out["maxspeed"].astype(str).str.extract(r"(\d+)")[0],
                                            errors="coerce")
    out = out.drop(columns=["maxspeed"])

    intersections = nodes[nodes["street_count"] >= 3]
    buf = mids[["segment_id", "geometry"]].copy()
    buf["geometry"] = buf.buffer(radius_m)
    area_km2 = np.pi * (radius_m / 1000) ** 2
    cnt = gpd.sjoin(intersections[["geometry"]], buf, predicate="within").groupby("segment_id").size()
    out["intersection_density_km2"] = out["segment_id"].map(cnt).fillna(0) / area_km2
    deg = gpd.sjoin(nodes[["street_count", "geometry"]], buf, predicate="within") \
        .groupby("segment_id")["street_count"].mean()
    out["connectivity_mean_degree"] = out["segment_id"].map(deg)

    major = edges[edges["road_class"].isin(sources()["osm"]["major_highway_classes"])]
    if len(major):
        d = gpd.sjoin_nearest(mids[["segment_id", "geometry"]], major[["geometry"]],
                              distance_col="dist_major_road_m").drop_duplicates("segment_id")
        out["dist_major_road_m"] = out["segment_id"].map(d.set_index("segment_id")["dist_major_road_m"])
    return out
