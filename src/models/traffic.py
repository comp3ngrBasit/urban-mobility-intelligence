"""Model 1 (congestion classification) and Model 2 (segment speed regression) on the
Chicago Traffic Tracker archive, trained per forecast horizon.

The tracker is archived, so the "forecast" rows saved for the dashboard are predictions
for the bins *after the archive's last timestamp* — they are labelled as such in the API."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import get_settings, model_config
from src.common.logging import get_logger
from src.features.safety import crash_features
from src.features.traffic import (build_traffic_features, congestion_label, feature_columns,
                                  free_flow_speed, speed_profile)
from src.geospatial.road_features import build_segment_road_features
from src.models.registry import latest_dir
from src.models.tabular import TaskSpec, chronological_split, train_task
from src.preprocessing.cleaners import traffic_grid

log = get_logger(__name__)


def _load():
    p = get_settings().processed_dir
    traffic = pd.read_parquet(p / "chicago_traffic.parquet")
    segs = pd.read_parquet(p / "chicago_segments.parquet")
    weather = pd.read_parquet(p / "weather_hourly.parquet") if (p / "weather_hourly.parquet").exists() else None
    crashes = pd.read_parquet(p / "chicago_crashes.parquet") if (p / "chicago_crashes.parquet").exists() else None
    return traffic, segs, weather, crashes


def prepare(traffic: pd.DataFrame, segs: pd.DataFrame, crashes: pd.DataFrame | None):
    cfg = model_config()
    grid = traffic_grid(traffic, cfg["congestion"]["bin_minutes"])
    _, _, _, (t1, _) = chronological_split(grid, "ts", cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    train_part = grid[grid["ts"] < t1]
    ff = free_flow_speed(train_part)
    prof = speed_profile(train_part)
    static = build_segment_road_features(segs)
    if crashes is not None:
        static = static.merge(crash_features(segs, crashes, cutoff=t1), on="segment_id", how="left")
    return grid, ff, prof, static, t1


def _extend_future(grid: pd.DataFrame, horizon_min: int, bin_min: int) -> pd.DataFrame:
    last = grid["ts"].max()
    future = pd.date_range(last + pd.Timedelta(minutes=bin_min), last + pd.Timedelta(minutes=horizon_min),
                           freq=f"{bin_min}min")
    fut = pd.MultiIndex.from_product([grid["segment_id"].unique(), future],
                                     names=["segment_id", "ts"]).to_frame(index=False)
    return pd.concat([grid, fut], ignore_index=True)


def train_all(horizons: list[int] | None = None) -> dict:
    cfg = model_config()
    traffic, segs, weather, crashes = _load()
    grid, ff, prof, static, t1 = prepare(traffic, segs, crashes)
    last_obs = grid.dropna(subset=["speed"])["ts"].max()
    summary = {}
    for H in horizons or cfg["speed"]["horizons_minutes"]:
        ext = _extend_future(grid, H, cfg["congestion"]["bin_minutes"])
        feats = build_traffic_features(ext, H, free_flow=ff, profile=prof, segments=segs,
                                       weather=weather, static_features=static)
        hist = feats[feats["ts"] <= last_obs]
        forecast_rows = feats[feats["ts"] > last_obs]
        fcols = feature_columns(hist)
        meta = {"source": "chicago_traffic (4g9f-3jbs, archived)", "horizon_min": H,
                "free_flow_fit_end": str(t1), "data_end": str(last_obs)}

        speed_spec = TaskSpec(
            task=f"speed_h{H}", kind="reg", target="speed", features=fcols, ts_col="ts",
            algorithms=["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest"],
            baselines={"persistence": lambda d: d["speed_at_issue"].to_numpy(),
                       "historical_profile": lambda d: d["hist_profile_speed"].to_numpy()},
            error_groups=["segment_id", "hour", "dow", "rain_intensity", "weather_category"],
            extra_config={"horizon_min": H})
        summary[f"speed_h{H}"] = train_task(hist, speed_spec, meta)
        _add_intervals_and_forecast(f"speed_h{H}", hist, forecast_rows, fcols)

        labels = cfg["congestion"]["labels"]
        cong_spec = TaskSpec(
            task=f"congestion_h{H}", kind="clf", target="congestion_level", features=fcols, ts_col="ts",
            labels=labels, algorithms=["lightgbm", "xgboost", "catboost", "random_forest"],
            baselines={"persistence": lambda d: congestion_label(d["congestion_ratio_at_issue"]).astype(str).to_numpy(),
                       "historical_profile": lambda d: congestion_label(
                           d["hist_profile_speed"] / d["free_flow_speed"]).astype(str).to_numpy()},
            error_groups=["segment_id", "hour", "dow", "rain_intensity", "weather_category"],
            extra_config={"horizon_min": H})
        summary[f"congestion_h{H}"] = train_task(hist, cong_spec, meta)
        _save_forecast_classes(f"congestion_h{H}", forecast_rows, fcols)

    ff.rename("free_flow_speed").reset_index().to_parquet(get_settings().processed_dir / "chicago_free_flow.parquet")
    return summary


def _add_intervals_and_forecast(task: str, hist: pd.DataFrame, forecast_rows: pd.DataFrame,
                                fcols: list[str]) -> None:
    """80% prediction interval from LightGBM quantile models (q10/q90), empirical test
    coverage, and point+interval predictions for the post-archive forecast rows."""
    import joblib
    import lightgbm as lgb

    cfg = model_config()
    d = latest_dir(task)
    df = hist.dropna(subset=["speed"]).sort_values("ts")
    tr, va, te, _ = chronological_split(df, "ts", cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    fit = pd.concat([tr, va]).iloc[-2_000_000:]
    X = lambda f: f[fcols].apply(pd.to_numeric, errors="coerce").astype("float32")  # noqa: E731
    qs = {}
    for a in (0.1, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=500, learning_rate=0.05,
                              num_leaves=63, verbose=-1, random_state=42)
        m.fit(X(fit), fit["speed"])
        qs[a] = m
        joblib.dump(m, d / f"quantile_{int(a * 100)}.joblib")
    lo, hi = qs[0.1].predict(X(te)), qs[0.9].predict(X(te))
    cover = float(np.mean((te["speed"].to_numpy() >= lo) & (te["speed"].to_numpy() <= hi)))
    (d / "intervals.json").write_text(json.dumps({"nominal": 0.8, "empirical_test_coverage": cover,
                                                  "mean_width": float(np.mean(hi - lo))}))
    tp = pd.read_parquet(d / "test_predictions.parquet")
    tp["lower"], tp["upper"] = lo, hi
    tp.to_parquet(d / "test_predictions.parquet", index=False)
    if len(forecast_rows):
        model = joblib.load(d / "model.joblib")
        fr = forecast_rows[["segment_id", "ts"]].copy()
        fr["predicted"] = model.predict(X(forecast_rows))
        fr["lower"], fr["upper"] = qs[0.1].predict(X(forecast_rows)), qs[0.9].predict(X(forecast_rows))
        fr.to_parquet(d / "forecast.parquet", index=False)


def _save_forecast_classes(task: str, forecast_rows: pd.DataFrame, fcols: list[str]) -> None:
    import joblib

    d = latest_dir(task)
    if not len(forecast_rows):
        return
    cfgm = json.loads((d / "feature_config.json").read_text())
    model = joblib.load(d / "model.joblib")
    X = forecast_rows[fcols].apply(pd.to_numeric, errors="coerce").astype("float32")
    proba = model.predict_proba(X)
    fr = forecast_rows[["segment_id", "ts"]].copy()
    fr["predicted"] = np.array(cfgm["labels"])[proba.argmax(1)]
    fr["confidence"] = proba.max(1)
    fr.to_parquet(d / "forecast.parquet", index=False)
