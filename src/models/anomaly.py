"""Model 6 — traffic anomaly detection.

Three complementary detectors on Chicago segment speeds (archived) and NYC DOT speeds (live):
* Robust z-score of speed vs. the segment's own hour×weekday profile (median / MAD)
* Isolation Forest on [speed, deviation from profile, rolling change]
* Local Outlier Factor on the same features
A record is flagged when the robust z-score crosses the threshold OR both ML detectors agree.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.common.config import get_settings, model_config


def detect(df: pd.DataFrame, id_col: str, ts_col: str, value_col: str = "speed") -> pd.DataFrame:
    cfg = model_config()["anomaly"]
    d = df.dropna(subset=[value_col]).copy()
    d["hour"] = d[ts_col].dt.hour
    d["dow"] = d[ts_col].dt.dayofweek
    grp = d.groupby([id_col, "dow", "hour"])[value_col]
    d["profile_median"] = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s))))
    d["robust_z"] = 0.6745 * (d[value_col] - d["profile_median"]) / mad.replace(0, np.nan)
    d = d.sort_values([id_col, ts_col])
    d["delta"] = d.groupby(id_col)[value_col].diff()
    feats = d[[value_col, "robust_z", "delta"]].fillna(0).clip(-50, 50)
    X = StandardScaler().fit_transform(feats)
    iso = IsolationForest(contamination=cfg["contamination"], random_state=42, n_jobs=-1).fit(X)
    d["iso_score"] = -iso.score_samples(X)
    d["iso_flag"] = iso.predict(X) == -1
    sample = np.random.default_rng(42).choice(len(X), size=min(len(X), 200_000), replace=False)
    lof = LocalOutlierFactor(n_neighbors=35, contamination=cfg["contamination"], novelty=True).fit(X[sample])
    d["lof_flag"] = lof.predict(X) == -1
    d["z_flag"] = d["robust_z"].abs() >= cfg["robust_z_threshold"]
    d["is_anomaly"] = d["z_flag"] | (d["iso_flag"] & d["lof_flag"])
    d["anomaly_type"] = np.where(~d["is_anomaly"], None,
                                 np.where(d[value_col] < d["profile_median"], "unusually slow", "unusually fast"))
    return d


def run() -> dict:
    p = get_settings().processed_dir
    out_dir = get_settings().analytics_dir
    summary = {}
    if (p / "chicago_traffic.parquet").exists():
        t = pd.read_parquet(p / "chicago_traffic.parquet", columns=["segment_id", "time", "speed"])
        last = t["time"].max()
        recent = t[t["time"] >= last - pd.Timedelta(days=28)]
        a = detect(recent, "segment_id", "time")
        an = a[a["is_anomaly"]]
        an.to_parquet(out_dir / "anomalies_chicago.parquet", index=False)
        summary["chicago"] = {"rows_scored": int(len(a)), "anomalies": int(len(an)),
                              "rate": float(len(an) / max(len(a), 1)), "window_end": str(last),
                              "agreement_iso_lof": float((a["iso_flag"] & a["lof_flag"]).sum() /
                                                         max(a["iso_flag"].sum(), 1))}
    if (p / "nyc_dot_speeds.parquet").exists():
        n = pd.read_parquet(p / "nyc_dot_speeds.parquet", columns=["id", "data_as_of", "speed"])
        if len(n) > 500:
            a = detect(n, "id", "data_as_of")
            an = a[a["is_anomaly"]]
            an.to_parquet(out_dir / "anomalies_nyc.parquet", index=False)
            summary["nyc"] = {"rows_scored": int(len(a)), "anomalies": int(len(an)),
                              "rate": float(len(an) / max(len(a), 1)),
                              "window_end": str(n["data_as_of"].max())}
    (out_dir / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
