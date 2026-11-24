"""Traffic Prediction: segment speed/congestion, zone demand and travel time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from backend.app.core.errors import NotReady
from backend.app.schemas.models import TravelTimeRequest, TravelTimeResponse
from backend.app.services import store

router = APIRouter(prefix="/api/predict", tags=["prediction"])


def _horizons(prefix: str) -> list[int]:
    return sorted(int(t.split("_h")[1]) for t in store.list_tasks() if t.startswith(prefix + "_h"))


@router.get("/options")
def options():
    segs = store.records(store.processed("chicago_segments.parquet")[
        ["segment_id", "street", "direction", "from_street", "to_street"]]) \
        if store.processed_exists("chicago_segments.parquet") else []
    zones = []
    zl = store.external("taxi_zone_lookup.csv")
    if zl:
        zones = store.records(pd.read_csv(zl).rename(columns={"LocationID": "zone", "Zone": "zone_name",
                                                              "Borough": "borough"}))
    return {"speed_horizons_min": _horizons("speed"), "congestion_horizons_min": _horizons("congestion"),
            "demand_horizons_h": _horizons("demand"), "segments": segs, "zones": zones,
            "travel_time_available": "travel_time" in store.list_tasks()}


def _forecast_at_horizon(task: str) -> tuple[pd.DataFrame, str]:
    fc = store.model_frame(task, "forecast")
    if fc is None or fc.empty:
        raise NotReady(f"{task} forecast", "retrain to produce forecast rows")
    t = fc["ts"].max()
    return fc[fc["ts"] == t], str(t)


@router.get("/segments")
def segments_forecast(horizon: int = Query(60)):
    sp, t = _forecast_at_horizon(f"speed_h{horizon}")
    out = sp.rename(columns={"predicted": "predicted_speed", "lower": "speed_lower", "upper": "speed_upper"})
    try:
        cg, _ = _forecast_at_horizon(f"congestion_h{horizon}")
        out = out.merge(cg.rename(columns={"predicted": "predicted_level"})[["segment_id", "predicted_level", "confidence"]],
                        on="segment_id", how="left")
    except NotReady:
        pass
    segs = store.processed("chicago_segments.parquet")
    out = out.merge(segs, on="segment_id", how="left")
    meta = store.model_json(f"speed_h{horizon}", "metadata") or {}
    return {"target_time": t, "horizon_min": horizon, "issued_at": meta.get("data_end"),
            "note": "Forecast for the period immediately after the archived Chicago data ends "
                    "(no actuals exist to compare against for these rows).",
            "segments": store.records(out.round(4))}


@router.get("/segment/{segment_id}")
def segment_detail(segment_id: int, horizon: int = Query(60)):
    task = f"speed_h{horizon}"
    tp = store.model_frame(task, "test_predictions")
    d = tp[tp["segment_id"] == segment_id].sort_values("ts")
    if d.empty:
        raise NotReady(f"segment {segment_id}", "segment not present in the test period")
    res = {"segment_id": segment_id, "horizon_min": horizon,
           "speed": store.records(d[["ts", "actual", "predicted"] + [c for c in ("lower", "upper") if c in d]].round(3)),
           "intervals": store.model_json(task, "intervals")}
    m = store.model_json(task, "metrics")
    from src.evaluation.metrics import regression_metrics
    res["segment_metrics"] = regression_metrics(d["actual"], d["predicted"])
    res["overall_test_metrics"] = m["models"][m["best_model"]]["test"]
    ctp = store.model_frame(f"congestion_h{horizon}", "test_predictions") \
        if f"congestion_h{horizon}" in store.list_tasks() else None
    if ctp is not None:
        c = ctp[ctp["segment_id"] == segment_id].sort_values("ts")
        res["congestion"] = store.records(c[["ts", "actual", "predicted"]])
    fc = store.model_frame(task, "forecast")
    if fc is not None:
        res["forecast"] = store.records(fc[fc["segment_id"] == segment_id].round(3))
    return res


@router.get("/zones")
def zones_forecast(horizon: int = Query(1)):
    fc, t = _forecast_at_horizon(f"demand_h{horizon}")
    return {"target_time": t, "horizon_h": horizon, "zones": store.records(fc.round(2))}


@router.get("/zone/{zone}")
def zone_detail(zone: int, horizon: int = Query(1), days: int = Query(14, le=90)):
    task = f"demand_h{horizon}"
    tp = store.model_frame(task, "test_predictions")
    d = tp[tp["zone"] == zone].sort_values("ts")
    if d.empty:
        raise NotReady(f"zone {zone}", "zone not present in the test period")
    d = d[d["ts"] >= d["ts"].max() - pd.Timedelta(days=days)]
    fc = store.model_frame(task, "forecast")
    from src.evaluation.metrics import regression_metrics
    return {"zone": zone, "horizon_h": horizon,
            "series": store.records(d[["ts", "actual", "predicted"] + [c for c in ("lower", "upper") if c in d]].round(2)),
            "zone_metrics": regression_metrics(d["actual"], d["predicted"]),
            "intervals": store.model_json(task, "intervals"),
            "forecast": store.records(fc[fc["zone"] == zone].round(2)) if fc is not None else []}


@router.post("/travel-time", response_model=TravelTimeResponse)
def travel_time(req: TravelTimeRequest):
    from src.features.travel_time import build_trip_features

    task = "travel_time"
    model = store.model_object(task)
    if model is None:
        raise NotReady(task, "python -m src.pipeline train --model travel_time")
    d = store.model_dir(task)
    hist = {k: pd.read_parquet(d / f"history_{k}.parquet") for k in ("zone_hour", "od", "city_hour")}
    fcfg = store.model_json(task, "feature_config")
    dist, src_ = req.distance_mi, "user"
    if dist is None:
        od = store.processed("tlc_od_hour.parquet")
        m = od[(od["pu"] == req.pu_zone) & (od["do"] == req.do_zone)]
        if m.empty:
            raise NotReady("distance", "no historical trips for this OD pair; please provide distance_mi")
        dist, src_ = float(np.average(m["median_distance_mi"], weights=m["trips"])), "historical_od_median"
    dep = pd.Timestamp(req.departure).tz_localize(None)
    trip = pd.DataFrame({"pickup": [dep], "pu": [req.pu_zone], "do": [req.do_zone], "distance_mi": [dist]})
    weather = _weather_for("nyc", dep)
    X = build_trip_features(trip, hist, weather)
    for f in fcfg["features"]:
        if f not in X.columns:
            X[f] = np.nan
    eta = float(model.predict(X[fcfg["features"]].apply(pd.to_numeric, errors="coerce").astype("float32"))[0])
    metrics = store.model_json(task, "metrics")
    wx = None
    if weather is not None:
        cols = [c for c in ("temperature_2m", "precipitation", "wind_speed_10m") if c in X.columns]
        wx = {c: (None if pd.isna(X[c].iloc[0]) else float(X[c].iloc[0])) for c in cols}
    return TravelTimeResponse(
        eta_minutes=round(eta, 2),
        baseline_eta_minutes=None if pd.isna(X["naive_eta_min"].iloc[0]) else round(float(X["naive_eta_min"].iloc[0]), 2),
        distance_mi=round(dist, 3), distance_source=src_, weather_used=wx,
        model=metrics["best_model"], model_version=d.name,
        test_mae_minutes=metrics["models"][metrics["best_model"]]["test"]["mae"],
        note="Estimated NYC taxi journey time between TLC zones (not general road traffic).")


def _weather_for(city: str, ts: pd.Timestamp) -> pd.DataFrame | None:
    """Archive weather if the departure is in the past, Open-Meteo forecast if future."""
    from src.common.config import get_settings
    frames = []
    if store.processed_exists("weather_hourly.parquet"):
        frames.append(store.processed("weather_hourly.parquet"))
    fp = get_settings().raw_dir / "weather" / f"forecast_{city}.parquet"
    if fp.exists():
        frames.append(pd.read_parquet(fp))
    if not frames:
        return None
    w = pd.concat(frames, ignore_index=True)
    w["time"] = pd.to_datetime(w["time"])
    return w[(w["city"] == city) & (w["time"] == ts.floor("h"))].drop_duplicates("time", keep="last")
