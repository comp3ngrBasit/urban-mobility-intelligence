"""Model pipeline tests on SYNTHETIC fixtures: they verify that training, evaluation,
registration and prediction run end-to-end — not how good the models are."""
import json

import numpy as np
import pandas as pd

from src.models.registry import latest_dir, load_json, load_model
from src.models.tabular import chronological_split


def test_chronological_split_is_ordered():
    df = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=100, freq="h"), "y": range(100)})
    tr, va, te, _ = chronological_split(df, "ts", 0.7, 0.15)
    assert tr.ts.max() < va.ts.min() and va.ts.max() < te.ts.min()


def test_traffic_models_train_and_register(trained):
    for task in ("speed_h60", "congestion_h60"):
        d = latest_dir(task)
        assert (d / "model.joblib").exists() and (d / "test_predictions.parquet").exists()
        m = json.loads((d / "metrics.json").read_text())
        assert m["best_model"] in m["models"] and "persistence" in m["baselines"]
    assert (latest_dir("speed_h60") / "forecast.parquet").exists()
    cov = load_json("speed_h60", "intervals")
    assert 0 <= cov["empirical_test_coverage"] <= 1


def test_travel_time_model(trained):
    res = load_json("travel_time", "metrics")
    assert res["models"][res["best_model"]]["test"]["mae"] > 0
    assert load_model("travel_time") is not None
    assert set(res["baselines"]) == {"distance_over_hourly_city_speed", "od_historical_median"}


def test_demand_models(trained):
    assert load_json("demand_h1", "metrics")["best_model"] in ("lightgbm", "xgboost")
    fc = pd.read_parquet(latest_dir("demand_h1") / "forecast.parquet")
    assert (fc["predicted"] >= 0).all()


def test_hotspots_and_anomalies(trained):
    from src.common.config import get_settings
    d = get_settings().analytics_dir
    h = json.loads((d / "accident_hotspots.json").read_text())
    assert h["n_crashes"] > 0 and "I" in h["morans_i"] and len(h["dbscan"]) > 0
    a = json.loads((d / "anomaly_summary.json").read_text())
    assert "chicago" in a and 0 <= a["chicago"]["rate"] <= 1


def test_eda_outputs(trained):
    from src.common.config import get_settings
    d = get_settings().analytics_dir
    for name in ("mobility", "traffic", "accidents", "weather_impact"):
        assert (d / f"{name}.json").exists(), name
    w = json.loads((d / "weather_impact.json").read_text())
    assert "chicago_speed" in w
