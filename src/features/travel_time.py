"""Trip-level features for travel-time prediction (NYC TLC). Everything is known at
departure: distance, origin/destination zones, departure time, weather at departure
hour, plus historical zone/OD speed statistics fit on the training period only."""
from __future__ import annotations

import pandas as pd

from src.features.temporal import TEMPORAL_FEATURES, add_temporal
from src.features.weather import join_weather


def fit_history(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = train.assign(hour=train["pickup"].dt.hour,
                     mph=train["distance_mi"] / (train["duration_min"] / 60))
    zone_hour = (t.groupby(["pu", "hour"])["mph"].median().rename("hist_pu_zone_mph").reset_index())
    od = (t.groupby(["pu", "do"]).agg(hist_od_median_min=("duration_min", "median"),
                                      hist_od_trips=("duration_min", "size")).reset_index())
    od = od[od["hist_od_trips"] >= 5]
    city_hour = t.groupby("hour")["mph"].median().rename("hist_city_hour_mph").reset_index()
    return {"zone_hour": zone_hour, "od": od, "city_hour": city_hour}


def build_trip_features(trips: pd.DataFrame, history: dict[str, pd.DataFrame],
                        weather: pd.DataFrame | None = None) -> pd.DataFrame:
    df = add_temporal(trips.rename(columns={"pickup": "ts"}), "ts", subdiv="NY")
    df = df.merge(history["zone_hour"], on=["pu", "hour"], how="left")
    df = df.merge(history["od"], on=["pu", "do"], how="left")
    df = df.merge(history["city_hour"], on="hour", how="left")
    df["same_zone"] = (df["pu"] == df["do"]).astype(int)
    df["naive_eta_min"] = df["distance_mi"] / df["hist_city_hour_mph"] * 60
    if weather is not None and len(weather):
        df = join_weather(df, weather, "nyc", "ts")
    return df.rename(columns={"ts": "pickup"})


TRIP_FEATURES = ["distance_mi", "pu", "do", "same_zone", "hist_pu_zone_mph", "hist_od_median_min",
                 "hist_od_trips", "hist_city_hour_mph", "naive_eta_min", "temperature_2m",
                 "precipitation", "wind_speed_10m"] + TEMPORAL_FEATURES


def trip_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in TRIP_FEATURES if c in df.columns]
