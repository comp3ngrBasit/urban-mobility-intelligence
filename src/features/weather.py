"""Weather join. For forecasting, weather is taken at the *issue* time (the last hour
fully observed before the prediction is made), never at the target time — target-time
observations would not be available in production. Swap in Open-Meteo forecasts to use
target-time weather legitimately."""
from __future__ import annotations

import pandas as pd

WEATHER_FEATURES = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m",
                    "snowfall", "weather_code"]


def join_weather(df: pd.DataFrame, weather: pd.DataFrame, city: str, ts_col: str) -> pd.DataFrame:
    w = weather[weather["city"] == city].copy()
    keep = [c for c in WEATHER_FEATURES + ["weather_category", "rain_intensity"] if c in w.columns]
    w = w[["time"] + keep].rename(columns={"time": "_wx_hour"})
    out = df.copy()
    out["_wx_hour"] = pd.to_datetime(out[ts_col]).dt.floor("h")
    out = out.merge(w, on="_wx_hour", how="left").drop(columns="_wx_hour")
    return out
