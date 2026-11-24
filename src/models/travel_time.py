"""Model 3 — travel-time prediction on NYC TLC trips.

Note on scope: taxi trips are a sample of *taxi* movements, not of all road traffic. The
model estimates taxi journey duration between taxi zones, which is what the data supports."""
from __future__ import annotations

import pandas as pd

from src.common.config import get_settings, model_config
from src.features.travel_time import build_trip_features, fit_history, trip_feature_columns
from src.models.registry import latest_dir
from src.models.tabular import TaskSpec, chronological_split, train_task


def train() -> dict:
    cfg = model_config()
    p = get_settings().processed_dir
    trips = pd.read_parquet(p / "tlc_trips_sample.parquet",
                            columns=["pickup", "pu", "do", "distance_mi", "duration_min", "service"])
    weather = pd.read_parquet(p / "weather_hourly.parquet") if (p / "weather_hourly.parquet").exists() else None
    trips = trips.sort_values("pickup")
    train_part, _, _, (t1, _) = chronological_split(trips, "pickup", cfg["split"]["train_frac"],
                                                    cfg["split"]["valid_frac"])
    history = fit_history(train_part)
    df = build_trip_features(trips, history, weather)
    spec = TaskSpec(
        task="travel_time", kind="reg", target="duration_min", features=trip_feature_columns(df),
        ts_col="pickup", algorithms=["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest"],
        baselines={"distance_over_hourly_city_speed": lambda d: d["naive_eta_min"].to_numpy(),
                   "od_historical_median": lambda d: d["hist_od_median_min"].to_numpy()},
        error_groups=["hour", "pu", "is_peak"], max_train_rows=1_500_000)
    res = train_task(df, spec, {"source": "NYC TLC yellow/green sample", "history_fit_end": str(t1)})
    d = latest_dir("travel_time")
    for name, frame in history.items():
        frame.to_parquet(d / f"history_{name}.parquet", index=False)
    return res
