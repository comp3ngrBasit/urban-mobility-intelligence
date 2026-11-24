"""SYNTHETIC TEST FIXTURES — used only to exercise code paths in the test-suite.

These frames mimic the *schemas* of the real sources (column names, types, units) so the
pipeline can be tested offline. They are never written to data/ in a real run, never shown
in the dashboard and never used to report model performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


def chicago_traffic_raw(n_segments: int = 12, days: int = 21, end="2026-05-31") -> pd.DataFrame:
    times = pd.date_range(end=pd.Timestamp(end), periods=days * 144, freq="10min")
    rows = []
    for s in range(n_segments):
        lat0, lon0 = 41.85 + 0.01 * (s % 4), -87.68 + 0.012 * (s // 4)
        base = RNG.uniform(22, 32)
        hod = np.asarray(times.hour + times.minute / 60)
        rush = np.exp(-((hod - 8) ** 2) / 2) + np.exp(-((hod - 17.5) ** 2) / 2)
        wk = np.asarray(times.dayofweek < 5, float)
        speed = base - 12 * rush * wk - 3 * rush * (1 - wk) + RNG.normal(0, 2.5, len(times))
        speed = np.clip(np.asarray(speed), 2, 45).round(0)
        speed[RNG.random(len(times)) < 0.05] = -1
        rows.append(pd.DataFrame({
            "time": times.strftime("%Y-%m-%dT%H:%M:%S.000"), "segment_id": str(1000 + s),
            "speed": speed.astype(str), "street": f"Street {s}", "direction": "EB",
            "from_street": "A", "to_street": "B", "length": "0.5",
            "bus_count": RNG.integers(0, 6, len(times)).astype(str),
            "message_count": RNG.integers(0, 20, len(times)).astype(str),
            "start_latitude": str(lat0), "start_longitude": str(lon0),
            "end_latitude": str(lat0 + 0.004), "end_longitude": str(lon0 + 0.006)}))
    return pd.concat(rows, ignore_index=True)


def crashes_raw(n: int = 3000) -> pd.DataFrame:
    centers = np.array([[41.88, -87.63], [41.95, -87.66], [41.80, -87.70]])
    c = centers[RNG.integers(0, 3, n)]
    lat = c[:, 0] + RNG.normal(0, 0.006, n)
    lon = c[:, 1] + RNG.normal(0, 0.006, n)
    dates = pd.Timestamp("2025-06-01") + pd.to_timedelta(RNG.integers(0, 365 * 24 * 60, n), unit="min")
    inj = RNG.poisson(0.3, n)
    return pd.DataFrame({
        "crash_record_id": [f"r{i}" for i in range(n)], "crash_date": dates.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "posted_speed_limit": "30", "weather_condition": RNG.choice(["CLEAR", "RAIN", "SNOW"], n, p=[.8, .15, .05]),
        "lighting_condition": RNG.choice(["DAYLIGHT", "DARKNESS, LIGHTED ROAD"], n),
        "roadway_surface_cond": RNG.choice(["DRY", "WET"], n), "first_crash_type": "REAR END",
        "trafficway_type": "NOT DIVIDED", "prim_contributory_cause": RNG.choice(["SPEEDING", "UNABLE TO DETERMINE"], n),
        "most_severe_injury": "NO INDICATION OF INJURY", "injuries_total": inj.astype(str),
        "injuries_fatal": (RNG.random(n) < 0.003).astype(int).astype(str),
        "injuries_incapacitating": (RNG.random(n) < 0.02).astype(int).astype(str),
        "num_units": "2", "latitude": lat.astype(str), "longitude": lon.astype(str)})


def weather_raw(city: str, start: str, end: str) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"time": t, "city": city,
                         "temperature_2m": 10 + 8 * np.sin(2 * np.pi * t.hour / 24) + RNG.normal(0, 2, len(t)),
                         "precipitation": np.where(RNG.random(len(t)) < 0.1, RNG.exponential(1.5, len(t)), 0),
                         "rain": 0.0, "snowfall": 0.0, "wind_speed_10m": RNG.gamma(2, 6, len(t)),
                         "relative_humidity_2m": RNG.uniform(40, 95, len(t)),
                         "weather_code": RNG.choice([0, 1, 3, 61, 63], len(t))})


def tlc_yellow(n: int = 40_000, month: str = "2026-08") -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01")
    hours = RNG.integers(0, 24 * 28 * 60, n)
    pickup = start + pd.to_timedelta(hours, unit="min")
    dist = RNG.gamma(2, 1.2, n)
    hod = pickup.hour
    mph = np.clip(14 - 5 * ((hod >= 7) & (hod <= 19)) + RNG.normal(0, 2, n), 3, 40)
    dur = dist / mph * 60 + RNG.exponential(2, n)
    return pd.DataFrame({"tpep_pickup_datetime": pickup,
                         "tpep_dropoff_datetime": pickup + pd.to_timedelta(dur, unit="min"),
                         "passenger_count": 1.0, "trip_distance": dist,
                         "PULocationID": RNG.choice(np.arange(100, 115), n).astype("int32"),
                         "DOLocationID": RNG.choice(np.arange(100, 115), n).astype("int32"),
                         "fare_amount": 10.0, "total_amount": 15.0})


def dot_speeds_raw(n_links: int = 20, hours: int = 72) -> pd.DataFrame:
    t = pd.date_range(end="2026-11-24 12:00", periods=hours * 12, freq="5min")
    rows = []
    for i in range(n_links):
        sp = np.clip(30 + RNG.normal(0, 6, len(t)), 1, 60)
        rows.append(pd.DataFrame({
            "id": str(i), "speed": sp.round(2).astype(str), "travel_time": (RNG.uniform(60, 300, len(t))).astype(str),
            "status": "0", "data_as_of": t.strftime("%Y-%m-%dT%H:%M:%S.000"), "link_id": str(4000 + i),
            "link_points": f"40.7{i:02d},-73.99 40.7{i:02d}1,-73.98 40.7{i:02d}2,-73.97",
            "borough": "Manhattan", "link_name": f"LINK {i}"}))
    return pd.concat(rows, ignore_index=True)
