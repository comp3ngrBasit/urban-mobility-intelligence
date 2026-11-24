"""Zone-hour taxi demand features (NYC TLC pickups) for direct multi-horizon forecasting.
Lags start at the horizon H, so the model issued at T = t - H only uses data <= T."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.temporal import TEMPORAL_FEATURES, add_temporal
from src.features.weather import join_weather


def complete_panel(pickups: pd.DataFrame) -> pd.DataFrame:
    """Fill the zone × hour grid with zeros (an hour with no pickups is a real zero)."""
    hours = pd.date_range(pickups["ts"].min(), pickups["ts"].max(), freq="h")
    idx = pd.MultiIndex.from_product([sorted(pickups["zone"].unique()), hours], names=["zone", "ts"])
    return (pickups.set_index(["zone", "ts"])["pickups"].reindex(idx, fill_value=0)
            .reset_index())


def build_demand_features(panel: pd.DataFrame, horizon_h: int, weather: pd.DataFrame | None = None,
                          zone_neighbors: dict[int, list[int]] | None = None) -> pd.DataFrame:
    df = panel.sort_values(["zone", "ts"]).copy()
    g = df.groupby("zone", sort=False)["pickups"]
    H = horizon_h
    df["lag_at_issue"] = g.shift(H)
    df["lag_issue_minus_1h"] = g.shift(H + 1)
    df["lag_issue_minus_2h"] = g.shift(H + 2)
    df["lag_24h"] = g.shift(max(24, H))
    df["lag_48h"] = g.shift(max(48, H + 24))
    df["lag_168h"] = g.shift(168)
    sh = g.shift(H)
    sg = sh.groupby(df["zone"], sort=False)
    df["roll_mean_6h"] = sg.transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["roll_mean_24h"] = sg.transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["roll_std_24h"] = sg.transform(lambda s: s.rolling(24, min_periods=2).std())
    df["roll_mean_168h"] = sg.transform(lambda s: s.rolling(168, min_periods=24).mean())

    if zone_neighbors:
        wide = df.pivot(index="ts", columns="zone", values="lag_at_issue")
        nb = {z: wide[[n for n in nbs if n in wide.columns]].mean(axis=1)
              for z, nbs in zone_neighbors.items() if z in wide.columns and nbs}
        nb_long = pd.DataFrame(nb).stack().rename("neighbor_lag_at_issue").reset_index()
        nb_long.columns = ["ts", "zone", "neighbor_lag_at_issue"]
        df = df.merge(nb_long, on=["ts", "zone"], how="left")

    df = add_temporal(df, "ts", subdiv="NY")
    if weather is not None and len(weather):
        df["issue_ts"] = df["ts"] - pd.Timedelta(hours=H)
        df = join_weather(df, weather, "nyc", "issue_ts").drop(columns="issue_ts")
    df["log_target"] = np.log1p(df["pickups"])
    df["horizon_h"] = H
    return df


DEMAND_FEATURES = ["zone", "lag_at_issue", "lag_issue_minus_1h", "lag_issue_minus_2h", "lag_24h",
                   "lag_48h", "lag_168h", "roll_mean_6h", "roll_mean_24h", "roll_std_24h",
                   "roll_mean_168h", "neighbor_lag_at_issue", "temperature_2m", "precipitation",
                   "wind_speed_10m", "relative_humidity_2m"] + TEMPORAL_FEATURES


def demand_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in DEMAND_FEATURES if c in df.columns]
