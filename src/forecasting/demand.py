"""Model 4 — mobility demand forecasting (NYC TLC pickups per taxi zone per hour).

Horizons 1h / 6h / 24h, compared on the same chronological test period:
* Seasonal-naive baseline (same hour last week) and same-hour-yesterday baseline
* Global direct gradient-boosting models (LightGBM, XGBoost) over all zones
* Per-zone SARIMA and Prophet on the busiest zones (rolling-origin evaluation)
* Optional LSTM (configs/models.yaml: demand.enable_lstm; requires torch)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.common.config import get_settings, model_config
from src.common.logging import get_logger
from src.evaluation.metrics import interval_coverage, regression_metrics
from src.features.demand import build_demand_features, complete_panel, demand_feature_columns
from src.models.registry import latest_dir, new_version_dir, save_version
from src.models.tabular import TaskSpec, chronological_split, train_task

log = get_logger(__name__)


def zone_adjacency() -> dict[int, list[int]] | None:
    p = get_settings().external_dir / "taxi_zones.parquet"
    if not p.exists():
        return None
    import geopandas as gpd
    z = gpd.read_parquet(p).to_crs(2263)
    z["geometry"] = z.buffer(50)
    j = gpd.sjoin(z[["LocationID", "geometry"]], z[["LocationID", "geometry"]], predicate="intersects")
    j = j[j["LocationID_left"] != j["LocationID_right"]]
    return j.groupby("LocationID_left")["LocationID_right"].apply(lambda s: sorted(set(s))).to_dict()


def _load():
    p = get_settings().processed_dir
    pickups = pd.read_parquet(p / "tlc_pickups_zone_hour.parquet")
    weather = pd.read_parquet(p / "weather_hourly.parquet") if (p / "weather_hourly.parquet").exists() else None
    return complete_panel(pickups), weather


def train_ml(horizons: list[int] | None = None) -> dict:
    cfg = model_config()["demand"]
    panel, weather = _load()
    adj = zone_adjacency()
    last = panel["ts"].max()
    out = {}
    for H in horizons or cfg["horizons_hours"]:
        future = pd.MultiIndex.from_product(
            [panel["zone"].unique(), pd.date_range(last + pd.Timedelta(hours=1), last + pd.Timedelta(hours=H), freq="h")],
            names=["zone", "ts"]).to_frame(index=False)
        ext = pd.concat([panel, future], ignore_index=True)
        feats = build_demand_features(ext, H, weather, adj)
        hist, fut = feats[feats["ts"] <= last], feats[feats["ts"] > last]
        hist = hist.dropna(subset=["lag_168h"])            # need one full week of history
        fcols = demand_feature_columns(hist)
        spec = TaskSpec(
            task=f"demand_h{H}", kind="reg", target="pickups", features=fcols, ts_col="ts",
            algorithms=["lightgbm", "xgboost"], target_transform="log1p",
            baselines={"seasonal_naive_week": lambda d: d["lag_168h"].to_numpy(),
                       "same_hour_yesterday": lambda d: d["lag_24h"].to_numpy()},
            error_groups=["zone", "hour", "dow"], extra_config={"horizon_h": H})
        out[f"demand_h{H}"] = train_task(hist, spec, {"source": "NYC TLC pickups (zone-hour)",
                                                      "data_end": str(last)})
        _intervals_and_forecast(f"demand_h{H}", hist, fut, fcols)
    return out


def _intervals_and_forecast(task, hist, fut, fcols):
    import joblib
    import lightgbm as lgb

    mc = model_config()
    d = latest_dir(task)
    tr, va, te, _ = chronological_split(hist.sort_values("ts"), "ts", mc["split"]["train_frac"], mc["split"]["valid_frac"])
    fit = pd.concat([tr, va])
    X = lambda f: f[fcols].astype("float32")  # noqa: E731
    qm = {}
    for a in (0.1, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.05,
                              num_leaves=63, verbose=-1, random_state=42)
        m.fit(X(fit), fit["pickups"])
        qm[a] = m
        joblib.dump(m, d / f"quantile_{int(a * 100)}.joblib")
    lo, hi = qm[0.1].predict(X(te)).clip(0), qm[0.9].predict(X(te)).clip(0)
    (d / "intervals.json").write_text(json.dumps({
        "nominal": 0.8, "empirical_test_coverage": interval_coverage(te["pickups"], lo, hi)}))
    tp = pd.read_parquet(d / "test_predictions.parquet")
    tp["lower"], tp["upper"] = lo, hi
    tp.to_parquet(d / "test_predictions.parquet", index=False)
    model = joblib.load(d / "model.joblib")
    fr = fut[["zone", "ts"]].copy()
    fr["predicted"] = np.expm1(model.predict(X(fut))).clip(0)
    fr["lower"], fr["upper"] = qm[0.1].predict(X(fut)).clip(0), qm[0.9].predict(X(fut)).clip(0)
    fr.to_parquet(d / "forecast.parquet", index=False)


def train_classical(horizons: list[int] | None = None) -> dict:
    """SARIMA (statsmodels) and Prophet per zone on the busiest zones. Rolling-origin
    evaluation over the test period: refit-free forecast updating every 24h."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    mc = model_config()
    cfg = mc["demand"]
    horizons = horizons or cfg["horizons_hours"]
    panel, _ = _load()
    top = panel.groupby("zone")["pickups"].sum().nlargest(cfg["top_zones_classical"]).index.tolist()
    res: dict = {"zones": top, "sarima": {f"h{h}": [] for h in horizons},
                 "prophet": {f"h{h}": [] for h in horizons},
                 "seasonal_naive": {f"h{h}": [] for h in horizons}}
    preds = []
    for z in top:
        s = panel[panel["zone"] == z].set_index("ts")["pickups"].asfreq("h", fill_value=0)
        n = len(s)
        cut = int(n * (mc["split"]["train_frac"] + mc["split"]["valid_frac"]))
        train = s.iloc[:cut]
        origins = range(cut, n - max(horizons), 24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sar = SARIMAX(np.log1p(train.iloc[-24 * 28:]), order=(2, 0, 1), seasonal_order=(1, 1, 1, 24),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        prophet_model = _fit_prophet(train)
        for H in horizons:
            y_true, y_sar, y_pro, y_sn = [], [], [], []
            for o in origins:
                target_ts = s.index[o + H - 1]
                y_true.append(s.iloc[o + H - 1])
                upd = sar.apply(np.log1p(s.iloc[o - 24 * 28:o]))
                y_sar.append(float(np.expm1(upd.forecast(H).iloc[-1])))
                y_sn.append(s.iloc[o + H - 1 - 168])
                if prophet_model is not None:
                    y_pro.append(float(prophet_model.predict(pd.DataFrame({"ds": [target_ts]}))["yhat"].iloc[0]))
                preds.append({"zone": z, "horizon_h": H, "ts": target_ts, "actual": y_true[-1],
                              "sarima": y_sar[-1], "prophet": y_pro[-1] if y_pro else None,
                              "seasonal_naive": y_sn[-1]})
            res["sarima"][f"h{H}"].append({"zone": int(z), **regression_metrics(y_true, y_sar)})
            res["seasonal_naive"][f"h{H}"].append({"zone": int(z), **regression_metrics(y_true, y_sn)})
            if y_pro:
                res["prophet"][f"h{H}"].append({"zone": int(z), **regression_metrics(y_true, np.clip(y_pro, 0, None))})
    pred_df = pd.DataFrame(preds)
    agg = {}
    for model in ("sarima", "prophet", "seasonal_naive"):
        for H in horizons:
            sub = pred_df[pred_df["horizon_h"] == H].dropna(subset=[model])
            if len(sub):
                agg[f"{model}_h{H}"] = regression_metrics(sub["actual"], sub[model].clip(lower=0))
    res["aggregate"] = agg
    d = new_version_dir("demand_classical")
    save_version(d, metrics=res, feature_config={"models": ["SARIMA(2,0,1)(1,1,1,24) on log1p", "Prophet"],
                                                 "evaluation": "rolling origin every 24h over test period"},
                 metadata={"zones": top}, frames={"test_predictions": pred_df})
    return res


def _fit_prophet(train: pd.Series):
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("prophet not installed; skipping")
        return None
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.add_country_holidays("US")
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    return m


def train_lstm() -> dict | None:
    """Optional sequence model. Skipped (and reported as such) unless enabled + torch installed."""
    if not model_config()["demand"].get("enable_lstm"):
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("enable_lstm is true but torch is not installed; skipping LSTM")
        return None
    from src.forecasting.lstm import train_lstm as _train
    return _train()
