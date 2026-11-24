"""Leakage tests: features for target time t at horizon H must not change when data
after the issue time T = t - H is altered."""
import numpy as np
import pandas as pd

from src.features.demand import build_demand_features, complete_panel
from src.features.temporal import add_temporal
from src.features.traffic import HIST_FEATURES, build_traffic_features, free_flow_speed, speed_profile


def _grid():
    ts = pd.date_range("2026-03-01", periods=96 * 9, freq="15min")
    rng = np.random.default_rng(1)
    rows = [pd.DataFrame({"segment_id": s, "ts": ts, "speed": rng.uniform(10, 30, len(ts))}) for s in (1, 2)]
    return pd.concat(rows, ignore_index=True)


SEGS = pd.DataFrame({"segment_id": [1, 2], "start_latitude": [41.88, 41.881], "start_longitude": [-87.63, -87.631],
                     "end_latitude": [41.882, 41.883], "end_longitude": [-87.628, -87.629]})


def test_traffic_features_do_not_leak_future():
    g = _grid()
    ff, prof = free_flow_speed(g), speed_profile(g)
    H = 60
    target = g["ts"].iloc[96 * 8]
    a = build_traffic_features(g, H, free_flow=ff, profile=prof, segments=SEGS)
    g2 = g.copy()
    g2.loc[g2["ts"] > target - pd.Timedelta(minutes=H), "speed"] = 999.0   # corrupt post-issue data
    b = build_traffic_features(g2, H, free_flow=ff, profile=prof, segments=SEGS)
    cols = [c for c in HIST_FEATURES if c not in ("hist_profile_speed", "free_flow_speed")]
    ra = a[(a.ts == target) & (a.segment_id == 1)][cols].reset_index(drop=True)
    rb = b[(b.ts == target) & (b.segment_id == 1)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(ra, rb)


def test_speed_at_issue_equals_value_h_before():
    g = _grid()
    f = build_traffic_features(g, 60, free_flow=free_flow_speed(g), profile=speed_profile(g), segments=SEGS)
    s1 = g[g.segment_id == 1].set_index("ts")["speed"]
    row = f[(f.segment_id == 1)].iloc[500]
    assert row["speed_at_issue"] == s1.loc[row["ts"] - pd.Timedelta(minutes=60)]


def test_demand_lags_start_at_horizon():
    ts = pd.date_range("2026-01-01", periods=24 * 10, freq="h")
    panel = complete_panel(pd.DataFrame({"ts": ts, "zone": 1, "pickups": np.arange(len(ts))}))
    for H in (1, 6, 24):
        f = build_demand_features(panel, H).set_index("ts")
        t = ts[200]
        assert f.loc[t, "lag_at_issue"] == f.loc[t - pd.Timedelta(hours=H), "pickups"]
        assert f.loc[t, "lag_24h"] == f.loc[t - pd.Timedelta(hours=max(24, H)), "pickups"]


def test_temporal_flags():
    df = add_temporal(pd.DataFrame({"ts": pd.to_datetime(["2026-07-04 08:00", "2026-07-07 08:00",
                                                          "2026-07-07 03:00"])}), "ts")
    assert df["is_holiday"].tolist() == [1, 0, 0]
    assert df["is_peak"].tolist() == [0, 1, 0]
    assert np.allclose(df["tod_sin"] ** 2 + df["tod_cos"] ** 2, 1)
