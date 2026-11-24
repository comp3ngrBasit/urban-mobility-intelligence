"""Safety features per Chicago segment: historical crash frequency and severity mix
within a buffer, counted ONLY from crashes before `cutoff` (the end of the training
period) to prevent leakage into validation/test."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.geospatial.road_features import UTM_CHICAGO, segments_gdf


def crash_features(segments: pd.DataFrame, crashes: pd.DataFrame, cutoff: pd.Timestamp,
                   radius_m: float = 150) -> pd.DataFrame:
    c = crashes[(crashes["crash_date"] < cutoff) & crashes["has_valid_location"]]
    out = pd.DataFrame({"segment_id": segments["segment_id"].values})
    if c.empty:
        out["crash_count_150m"] = 0.0
        out["serious_crash_share_150m"] = 0.0
        return out
    pts = gpd.GeoDataFrame(pd.DataFrame({"serious": c["severity"].astype(str).isin(["Serious", "Fatal"]).astype(int).values}),
                           geometry=gpd.points_from_xy(c["longitude"].values, c["latitude"].values), crs=4326).to_crs(UTM_CHICAGO)
    buf = segments_gdf(segments).to_crs(UTM_CHICAGO)[["segment_id", "geometry"]]
    buf["geometry"] = buf.buffer(radius_m)
    j = gpd.sjoin(pts, buf, predicate="within")
    years = max((cutoff - c["crash_date"].min()).days / 365.25, 1 / 12)
    agg = j.groupby("segment_id").agg(n=("serious", "size"), serious=("serious", "sum"))
    out["crash_count_150m"] = out["segment_id"].map(agg["n"] / years).fillna(0)   # crashes per year
    out["serious_crash_share_150m"] = out["segment_id"].map(agg["serious"] / agg["n"]).fillna(0)
    return out
