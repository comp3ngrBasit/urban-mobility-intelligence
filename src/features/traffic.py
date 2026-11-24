"""Horizon-aware traffic features for the Chicago segment speed / congestion models.

For a target bin at time t and horizon H, the forecast is *issued* at T = t - H.
Every historical feature is built from the series shifted by at least H, so nothing
observed after T can enter a feature. `tests/test_features.py` asserts this property.

Free-flow speed and the segment×hour×weekday profile are fit on the training period only
and passed in, so validation/test periods never inform them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import model_config
from src.features.temporal import TEMPORAL_FEATURES, add_temporal
from src.features.weather import join_weather
from src.geospatial.h3utils import neighbors, to_cells


def free_flow_speed(grid: pd.DataFrame) -> pd.Series:
    """85th percentile night-time (22:00-05:59) speed per segment."""
    night = grid[grid["ts"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])].dropna(subset=["speed"])
    ff = night.groupby("segment_id")["speed"].quantile(0.85)
    overall = grid.dropna(subset=["speed"]).groupby("segment_id")["speed"].quantile(0.85)
    return ff.reindex(overall.index).fillna(overall).clip(lower=5).rename("free_flow_speed")


def speed_profile(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.dropna(subset=["speed"]).assign(hour=lambda d: d["ts"].dt.hour,
                                             dow=lambda d: d["ts"].dt.dayofweek)
    return (g.groupby(["segment_id", "dow", "hour"])["speed"].mean()
            .rename("hist_profile_speed").reset_index())


def congestion_label(ratio: pd.Series) -> pd.Series:
    cfg = model_config()["congestion"]
    return pd.cut(ratio, bins=cfg["ratio_bins"], labels=cfg["labels"], right=False)


def build_traffic_features(grid: pd.DataFrame, horizon_min: int, *, free_flow: pd.Series,
                           profile: pd.DataFrame, segments: pd.DataFrame,
                           weather: pd.DataFrame | None = None,
                           static_features: pd.DataFrame | None = None) -> pd.DataFrame:
    bin_min = model_config()["congestion"]["bin_minutes"]
    if horizon_min % bin_min:
        raise ValueError("horizon must be a multiple of the bin size")
    h = horizon_min // bin_min
    per_day = 1440 // bin_min
    df = grid.sort_values(["segment_id", "ts"]).copy()
    g = df.groupby("segment_id", sort=False)["speed"]

    df["speed_at_issue"] = g.shift(h)
    df["speed_prev_15m"] = g.shift(h + 15 // bin_min)
    df["speed_prev_30m"] = g.shift(h + 30 // bin_min)
    df["speed_prev_1h"] = g.shift(h + 60 // bin_min)
    df["speed_prev_24h"] = g.shift(max(per_day, h))             # same clock time yesterday (or earlier)
    df["speed_prev_7d"] = g.shift(7 * per_day)
    shifted = g.shift(h)
    sg = shifted.groupby(df["segment_id"], sort=False)
    df["roll_mean_1h"] = sg.transform(lambda s: s.rolling(60 // bin_min, min_periods=1).mean())
    df["roll_mean_3h"] = sg.transform(lambda s: s.rolling(180 // bin_min, min_periods=1).mean())
    df["roll_std_1h"] = sg.transform(lambda s: s.rolling(60 // bin_min, min_periods=2).std())
    df["obs_count_3h"] = sg.transform(lambda s: s.notna().rolling(180 // bin_min, min_periods=1).sum())

    df = df.merge(free_flow, left_on="segment_id", right_index=True, how="left")
    df["congestion_ratio_at_issue"] = df["speed_at_issue"] / df["free_flow_speed"]
    df["roll_congestion_index_1h"] = 1 - (df["roll_mean_1h"] / df["free_flow_speed"]).clip(upper=1.5)

    # neighbouring segments (same H3 k=1 disk) observed at issue time
    res = 8
    seg = segments.set_index("segment_id")
    mid_lat = (seg["start_latitude"] + seg["end_latitude"]) / 2
    mid_lon = (seg["start_longitude"] + seg["end_longitude"]) / 2
    cell = to_cells(mid_lat, mid_lon, res)
    df["h3"] = df["segment_id"].map(cell)
    cell_mean = df.groupby(["h3", "ts"])["speed_at_issue"].mean()
    nb_frames = []
    for c in df["h3"].dropna().unique():
        nbs = [n for n in neighbors(c, 1)]
        vals = cell_mean[cell_mean.index.get_level_values(0).isin(nbs)]
        if len(vals):
            m = vals.groupby(level=1).mean()
            nb_frames.append(pd.DataFrame({"h3": c, "ts": m.index, "neighbor_speed_at_issue": m.values}))
    if nb_frames:
        df = df.merge(pd.concat(nb_frames), on=["h3", "ts"], how="left")
    else:
        df["neighbor_speed_at_issue"] = np.nan

    df = add_temporal(df, "ts", subdiv="IL")
    df = df.merge(profile, on=["segment_id", "dow", "hour"], how="left")
    if weather is not None and len(weather):
        df["issue_ts"] = df["ts"] - pd.Timedelta(minutes=horizon_min)
        df = join_weather(df, weather, "chicago", "issue_ts").drop(columns="issue_ts")
    if static_features is not None:
        df = df.merge(static_features, on="segment_id", how="left")

    df["congestion_ratio"] = df["speed"] / df["free_flow_speed"]
    df["congestion_level"] = congestion_label(df["congestion_ratio"])
    df["horizon_min"] = horizon_min
    return df


HIST_FEATURES = ["speed_at_issue", "speed_prev_15m", "speed_prev_30m", "speed_prev_1h", "speed_prev_24h",
                 "speed_prev_7d", "roll_mean_1h", "roll_mean_3h", "roll_std_1h", "obs_count_3h",
                 "congestion_ratio_at_issue", "roll_congestion_index_1h", "neighbor_speed_at_issue",
                 "hist_profile_speed", "free_flow_speed"]
WEATHER_FEATURES = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m",
                    "snowfall"]
STATIC_FEATURES = ["segment_length_m", "road_rank", "lanes", "osm_maxspeed_mph",
                   "intersection_density_km2", "connectivity_mean_degree", "dist_major_road_m",
                   "crash_count_150m", "serious_crash_share_150m"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = HIST_FEATURES + TEMPORAL_FEATURES + WEATHER_FEATURES + STATIC_FEATURES
    return [c for c in cols if c in df.columns]
