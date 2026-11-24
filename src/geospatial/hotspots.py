"""Model 5 — accident hotspot detection.

Combines density-based clustering (DBSCAN, HDBSCAN), partitioning (K-Means), Kernel
Density Estimation and statistically tested hotspots (Getis-Ord Gi* with FDR) on H3 cells.
Only the Gi* result is described as "statistically significant"; clusters are descriptive."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.neighbors import KernelDensity

from src.common.config import get_settings, model_config, sources
from src.common.logging import get_logger
from src.geospatial.h3utils import cell_center, to_cells
from src.geospatial.spatial_stats import global_morans_i, local_clusters

log = get_logger(__name__)
EARTH_R = 6_371_000.0


def _summarise_clusters(df: pd.DataFrame, labels: np.ndarray, name: str) -> list[dict]:
    d = df.assign(cluster=labels)
    d = d[d["cluster"] >= 0]
    if d.empty:
        return []
    spec = dict(
        crashes=("crash_record_id", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"),
        serious_or_fatal=("severity", lambda s: int(s.isin(["Serious", "Fatal"]).sum())),
        injuries=("injuries_total", "sum"),
    )
    if "prim_contributory_cause" in d.columns:
        spec["top_cause"] = ("prim_contributory_cause",
                             lambda s: s.mode().iat[0] if s.notna().any() else None)
    agg = d.groupby("cluster").agg(**spec).reset_index().sort_values("crashes", ascending=False)
    agg["method"] = name
    agg["serious_rate"] = agg["serious_or_fatal"] / agg["crashes"]
    return agg.to_dict("records")


def detect_hotspots(crashes: pd.DataFrame) -> dict:
    cfg = model_config()["hotspots"]
    res = sources()["geo"]["h3_resolution"]
    df = crashes[crashes["has_valid_location"]].copy()
    if len(df) < 100:
        raise ValueError("too few geolocated crashes for hotspot detection")
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    result: dict = {"n_crashes": int(len(df)),
                    "period": [str(df["crash_date"].min()), str(df["crash_date"].max())]}

    # DBSCAN on haversine distance (ball tree) — eps in metres converted to radians
    db = DBSCAN(eps=cfg["dbscan_eps_m"] / EARTH_R, min_samples=cfg["dbscan_min_samples"],
                metric="haversine", algorithm="ball_tree").fit(coords)
    result["dbscan"] = _summarise_clusters(df, db.labels_, "DBSCAN")[:100]

    # HDBSCAN: subsample for tractability on very large inputs
    sample = df.sample(min(len(df), 60_000), random_state=42)
    hdb = HDBSCAN(min_cluster_size=cfg["hdbscan_min_cluster_size"], metric="haversine").fit(
        np.radians(sample[["latitude", "longitude"]].to_numpy()))
    result["hdbscan"] = _summarise_clusters(sample, hdb.labels_, "HDBSCAN")[:100]

    km = KMeans(n_clusters=cfg["kmeans_k"], n_init=10, random_state=42).fit(
        df[["latitude", "longitude"]].to_numpy())
    result["kmeans"] = _summarise_clusters(df, km.labels_, "KMeans")

    # KDE surface on a regular grid (for heatmap rendering)
    kde = KernelDensity(bandwidth=cfg["kde_bandwidth_m"] / EARTH_R, metric="haversine",
                        kernel="gaussian").fit(np.radians(sample[["latitude", "longitude"]].to_numpy()))
    b = sources()["geo"]["bbox"]["chicago"]
    lats = np.linspace(b["min_lat"], b["max_lat"], 80)
    lons = np.linspace(b["min_lon"], b["max_lon"], 60)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    dens = np.exp(kde.score_samples(np.radians(np.c_[LA.ravel(), LO.ravel()])))
    result["kde"] = {"lats": lats.round(5).tolist(), "lons": lons.round(5).tolist(),
                     "density": (dens / dens.max()).reshape(LA.shape).round(4).tolist()}

    # H3 counts -> global & local spatial autocorrelation
    df["h3"] = to_cells(df["latitude"], df["longitude"], res)
    counts = df.groupby("h3").agg(crashes=("crash_record_id", "size"),
                                  serious=("severity", lambda s: int(s.isin(["Serious", "Fatal"]).sum())))
    result["morans_i"] = global_morans_i(counts["crashes"])
    local = local_clusters(counts["crashes"], alpha=cfg["gi_alpha"])
    local = local.merge(counts, left_on="h3", right_index=True)
    centers = local["h3"].map(cell_center)
    local["lat"] = [c[0] for c in centers]
    local["lon"] = [c[1] for c in centers]
    result["h3_cells"] = local.round(5).to_dict("records")
    result["significant_hotspots"] = int((local["gi_class"] == "Hotspot").sum())
    return result


def run(crashes: pd.DataFrame | None = None) -> dict:
    if crashes is None:
        crashes = pd.read_parquet(get_settings().processed_dir / "chicago_crashes.parquet")
    out = detect_hotspots(crashes)
    dest = get_settings().analytics_dir / "accident_hotspots.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, default=str))
    log.info("hotspots: %d significant Gi* cells, Moran's I=%.3f (p=%.3f)",
             out["significant_hotspots"], out["morans_i"]["I"], out["morans_i"]["p_value"])
    return out
