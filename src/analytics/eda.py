"""Exploratory & statistical analysis. Produces the JSON documents behind the dashboard
(artifacts/analytics/*.json). Every number is computed from processed data at run time.

Each section answers one of the brief's business questions; the mapping is noted inline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analytics.stats import anova, chi_square, correlation, decompose, mean_ci, two_group
from src.common.config import get_settings, model_config, sources
from src.common.logging import get_logger
from src.features.temporal import add_temporal
from src.features.traffic import congestion_label
from src.geospatial.h3utils import cell_center, to_cells
from src.geospatial.spatial_stats import global_morans_i, local_clusters
from src.preprocessing.cleaners import traffic_grid

log = get_logger(__name__)


def _p(name):
    return get_settings().processed_dir / name


def _write(name: str, obj) -> None:
    d = get_settings().analytics_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, default=str))
    log.info("wrote analytics/%s.json", name)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ------------------------------------------------------------------- NYC taxi mobility
def mobility() -> dict:
    """Q6 peak mobility hours, Q8 weekday/weekend/holiday patterns, OD analysis, clustering."""
    pick = pd.read_parquet(_p("tlc_pickups_zone_hour.parquet"))
    drop = pd.read_parquet(_p("tlc_dropoffs_zone_hour.parquet"))
    od = pd.read_parquet(_p("tlc_od_hour.parquet"))
    lookup_p = get_settings().external_dir / "taxi_zone_lookup.csv"
    lookup = pd.read_csv(lookup_p) if lookup_p.exists() else pd.DataFrame(columns=["LocationID", "Borough", "Zone"])
    lookup = lookup.rename(columns={"LocationID": "zone", "Borough": "borough", "Zone": "zone_name"})

    city = pick.groupby("ts")["pickups"].sum().asfreq("h", fill_value=0)
    ct = add_temporal(city.rename("pickups").reset_index(), "ts", subdiv="NY")
    hourly = []
    for daytype, sub in (("weekday", ct[(ct.is_weekend == 0) & (ct.is_holiday == 0)]),
                         ("weekend", ct[ct.is_weekend == 1])):
        for h, g in sub.groupby("hour"):
            hourly.append({"daytype": daytype, "hour": int(h), **mean_ci(g["pickups"])})
    heat = ct.pivot_table(index="dow", columns="hour", values="pickups", aggfunc="mean")

    pz = pick.merge(lookup, on="zone", how="left")
    pz["hour"] = pz["ts"].dt.hour
    borough_peak = (pz.groupby(["borough", "hour"])["pickups"].sum().reset_index()
                    .sort_values("pickups").groupby("borough").tail(1)
                    .rename(columns={"hour": "peak_hour", "pickups": "pickups_at_peak"}))
    zone_tot = (pick.groupby("zone")["pickups"].sum().to_frame()
                .join(drop.groupby("zone")["dropoffs"].sum(), how="outer").fillna(0).reset_index()
                .merge(lookup, on="zone", how="left"))
    days = max((pick["ts"].max() - pick["ts"].min()).days, 1)
    zone_tot["pickups_per_day"] = zone_tot["pickups"] / days
    zone_tot["net_flow_per_day"] = (zone_tot["dropoffs"] - zone_tot["pickups"]) / days

    od_tot = od.groupby(["pu", "do"]).agg(trips=("trips", "sum"),
                                          median_duration_min=("median_duration_min", "median"),
                                          median_distance_mi=("median_distance_mi", "median")).reset_index()
    od_top = od_tot[od_tot["pu"] != od_tot["do"]].nlargest(200, "trips")
    od_top = od_top.merge(lookup[["zone", "zone_name", "borough"]].add_prefix("pu_"), left_on="pu",
                          right_on="pu_zone", how="left") \
                   .merge(lookup[["zone", "zone_name", "borough"]].add_prefix("do_"), left_on="do",
                          right_on="do_zone", how="left")

    # Demand clustering: K-Means on each zone's normalised 24h weekday profile
    from sklearn.cluster import KMeans
    prof = pz[pz["ts"].dt.dayofweek < 5].pivot_table(index="zone", columns="hour", values="pickups",
                                                     aggfunc="mean").fillna(0)
    prof = prof[prof.sum(axis=1) > 24]  # ignore near-empty zones
    norm = prof.div(prof.sum(axis=1), axis=0)
    k = min(5, len(norm))
    clusters = []
    if k >= 2:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(norm)
        cl = pd.DataFrame({"zone": norm.index, "cluster": km.labels_}).merge(lookup, on="zone", how="left")
        for c in range(k):
            centroid = km.cluster_centers_[c]
            clusters.append({"cluster": c, "n_zones": int((km.labels_ == c).sum()),
                             "peak_hour": int(np.argmax(centroid)), "profile": centroid.round(4).tolist(),
                             "example_zones": cl[cl.cluster == c]["zone_name"].dropna().head(6).tolist()})
        zone_tot = zone_tot.merge(cl[["zone", "cluster"]], on="zone", how="left")

    weekend_test = two_group(ct.loc[(ct.is_weekend == 0) & (ct.is_holiday == 0), "pickups"],
                             ct.loc[ct.is_weekend == 1, "pickups"], ("weekday", "weekend"))
    holiday_test = two_group(ct.loc[ct.is_holiday == 1, "pickups"],
                             ct.loc[(ct.is_holiday == 0) & (ct.is_weekend == 0), "pickups"],
                             ("holiday", "regular_weekday"))
    daily = city.resample("D").sum()
    out = {
        "scope_note": "NYC TLC yellow/green (and HVFHV if ingested) trips. Taxi/FHV demand only — "
                      "not a measure of total road traffic.",
        "period": [str(pick["ts"].min()), str(pick["ts"].max())],
        "total_trips": int(pick["pickups"].sum()),
        "hourly_profile": hourly,
        "dow_hour_heatmap": {"z": heat.round(1).values.tolist(), "hours": heat.columns.tolist(),
                             "dows": heat.index.tolist()},
        "borough_peak_hours": _records(borough_peak),
        "zones": _records(zone_tot),
        "od_top": _records(od_top),
        "demand_clusters": clusters,
        "weekday_vs_weekend": weekend_test,
        "holiday_vs_weekday": holiday_test,
        "daily": {"ts": [str(t.date()) for t in daily.index], "pickups": daily.astype(int).tolist()},
        "decomposition_weekly": decompose(city.iloc[-24 * 7 * 6:].astype(float), 168),
    }
    _write("mobility", out)
    return out


# ------------------------------------------------------------------- Chicago traffic
def traffic() -> dict:
    """Q1 where/when congestion occurs, Q7 recurring bottlenecks, spatial autocorrelation."""
    t = pd.read_parquet(_p("chicago_traffic.parquet"))
    segs = pd.read_parquet(_p("chicago_segments.parquet"))
    grid = traffic_grid(t, model_config()["congestion"]["bin_minutes"]).dropna(subset=["speed"])
    ffp = _p("chicago_free_flow.parquet")
    if ffp.exists():
        ff = pd.read_parquet(ffp).set_index("segment_id")["free_flow_speed"]
    else:
        from src.features.traffic import free_flow_speed
        ff = free_flow_speed(grid)
    grid = grid.join(ff, on="segment_id")
    grid["ratio"] = grid["speed"] / grid["free_flow_speed"]
    grid["level"] = congestion_label(grid["ratio"]).astype(str)
    grid = add_temporal(grid, "ts", subdiv="IL")

    hourly = [{"daytype": dt, "hour": int(h), **mean_ci(g.groupby("ts")["speed"].mean())}
              for dt, sub in (("weekday", grid[grid.is_weekend == 0]), ("weekend", grid[grid.is_weekend == 1]))
              for h, g in sub.groupby("hour")]
    dist = grid["level"].value_counts(normalize=True).round(4).to_dict()
    heat = grid.pivot_table(index="dow", columns="hour", values="ratio", aggfunc="mean")

    peak = grid[grid.is_peak == 1].copy()
    peak["congested"] = peak["level"].isin(["High", "Severe"])
    bott = peak.groupby("segment_id").agg(
        peak_bins=("level", "size"), congested_share=("congested", "mean"),
        mean_peak_speed=("speed", "mean")).reset_index()
    days_c = peak[peak["congested"]].groupby("segment_id")["ts"].apply(lambda s: s.dt.date.nunique())
    bott["days_congested"] = bott["segment_id"].map(days_c).fillna(0)
    ndays = peak["ts"].dt.date.nunique()
    bott["recurrence_pct_days"] = bott["days_congested"] / max(ndays, 1) * 100
    bott = bott.merge(segs, on="segment_id", how="left").join(ff, on="segment_id")
    bott = bott[bott["peak_bins"] >= 20].sort_values("congested_share", ascending=False)

    seg_stats = grid.groupby("segment_id").agg(mean_speed=("speed", "mean"), mean_ratio=("ratio", "mean"),
                                               p_congested=("level", lambda s: float(s.isin(["High", "Severe"]).mean()))
                                               ).reset_index().merge(segs, on="segment_id", how="left")
    seg_stats["mid_lat"] = (seg_stats.start_latitude + seg_stats.end_latitude) / 2
    seg_stats["mid_lon"] = (seg_stats.start_longitude + seg_stats.end_longitude) / 2
    seg_stats["h3"] = to_cells(seg_stats["mid_lat"], seg_stats["mid_lon"], sources()["geo"]["h3_resolution"])
    cell = seg_stats.groupby("h3")["p_congested"].mean()
    spatial = {}
    if len(cell) >= 20:
        spatial["morans_i"] = global_morans_i(cell)
        lc = local_clusters(cell)
        c = lc["h3"].map(cell_center)
        lc["lat"], lc["lon"] = [x[0] for x in c], [x[1] for x in c]
        spatial["local"] = _records(lc.round(5))

    city_speed = grid.groupby("ts")["speed"].mean()
    out = {
        "scope_note": "Chicago Traffic Tracker (CTA-bus GPS speed estimates on ~300 miles of arterials). "
                      "ARCHIVED dataset — historical analysis only.",
        "period": [str(grid["ts"].min()), str(grid["ts"].max())],
        "segments": int(grid["segment_id"].nunique()),
        "observations": int(len(grid)),
        "mean_speed": mean_ci(grid["speed"]),
        "hourly_profile": hourly,
        "congestion_distribution": dist,
        "dow_hour_ratio_heatmap": {"z": heat.round(3).values.tolist(), "hours": heat.columns.tolist(),
                                   "dows": heat.index.tolist()},
        "bottlenecks": _records(bott.head(100).round(4)),
        "segment_stats": _records(seg_stats.round(5)),
        "spatial": spatial,
        "weekday_vs_weekend_speed": two_group(city_speed[city_speed.index.dayofweek < 5],
                                              city_speed[city_speed.index.dayofweek >= 5],
                                              ("weekday", "weekend")),
        "decomposition_daily": decompose(city_speed.resample("h").mean().iloc[-24 * 28:], 24),
    }
    _write("traffic", out)
    return out


# ------------------------------------------------------------------- Chicago crashes
def accidents() -> dict:
    """Q4 accident risk areas: temporal trends, severity, causes, conditions."""
    c = pd.read_parquet(_p("chicago_crashes.parquet"))
    monthly = c.set_index("crash_date").resample("MS").agg(
        crashes=("crash_record_id", "size"),
        serious=("severity", lambda s: int(s.isin(["Serious", "Fatal"]).sum())),
        fatal=("severity", lambda s: int((s == "Fatal").sum())))
    heat = c.pivot_table(index="dow", columns="hour", values="crash_record_id", aggfunc="size").fillna(0)
    ndays = max(c["date"].nunique(), 1)
    sev = c["severity"].value_counts().reindex(["No injury", "Minor", "Serious", "Fatal"]).fillna(0)
    causes = (c.groupby("prim_contributory_cause").agg(crashes=("crash_record_id", "size"),
                                                        serious_rate=("severity", lambda s: float(s.isin(["Serious", "Fatal"]).mean())))
              .sort_values("crashes", ascending=False).head(15).reset_index())
    light = pd.crosstab(c["lighting_condition"], c["severity"].isin(["Serious", "Fatal"]))
    surface = (c.groupby("roadway_surface_cond").agg(crashes=("crash_record_id", "size"),
                                                     serious_rate=("severity", lambda s: float(s.isin(["Serious", "Fatal"]).mean())))
               .sort_values("crashes", ascending=False).reset_index())
    out = {
        "scope_note": "Chicago Traffic Crashes (85ca-t3if): police-reported crashes on city streets.",
        "period": [str(c["crash_date"].min()), str(c["crash_date"].max())],
        "total_crashes": int(len(c)),
        "crashes_per_day": mean_ci(c.groupby("date").size()),
        "severity_counts": {k: int(v) for k, v in sev.items()},
        "injuries_total": int(c["injuries_total"].fillna(0).sum()),
        "fatalities": int(c["injuries_fatal"].fillna(0).sum()),
        "monthly": _records(monthly.reset_index().rename(columns={"crash_date": "month"})),
        "dow_hour_heatmap": {"z": (heat / (ndays / 7)).round(3).values.tolist(),
                             "hours": heat.columns.tolist(), "dows": heat.index.tolist(),
                             "unit": "crashes per hour-slot per week"},
        "top_causes": _records(causes),
        "lighting_vs_serious": chi_square(light),
        "surface_conditions": _records(surface),
    }
    _write("accidents", out)
    return out


# ------------------------------------------------------------------- weather impact
def weather_impact() -> dict:
    """Q5 how weather affects traffic, demand and crashes."""
    w = pd.read_parquet(_p("weather_hourly.parquet"))
    res: dict = {"note": "Hourly Open-Meteo reanalysis joined on local hour. Correlation is not causation; "
                         "hour-of-day is controlled for by comparing within peak hours where stated."}
    if _p("chicago_traffic.parquet").exists():
        t = pd.read_parquet(_p("chicago_traffic.parquet"), columns=["time", "speed"])
        sp = t.dropna().set_index("time")["speed"].resample("h").mean().rename("speed").reset_index()
        m = sp.merge(w[w.city == "chicago"], left_on="time", right_on="time")
        m["hour"] = m["time"].dt.hour
        m["peak"] = m["hour"].isin([7, 8, 16, 17, 18]) & (m["time"].dt.dayofweek < 5)
        # deseasonalise: deviation from the hour×weekday mean
        m["speed_anom"] = m["speed"] - m.groupby([m["time"].dt.dayofweek, "hour"])["speed"].transform("mean")
        res["chicago_speed"] = {
            "precip_vs_speed_anomaly": correlation(m["precipitation"], m["speed_anom"]),
            "temp_vs_speed_anomaly": correlation(m["temperature_2m"], m["speed_anom"]),
            "wind_vs_speed_anomaly": correlation(m["wind_speed_10m"], m["speed_anom"]),
            "rain_intensity_anova_peak": anova({k: g["speed"] for k, g in m[m.peak].groupby("rain_intensity")}),
            "by_category": _records(m.groupby("weather_category")["speed_anom"].agg(["mean", "count"]).reset_index()),
            "scatter": _records(m[["precipitation", "speed_anom", "temperature_2m"]].dropna().sample(
                min(3000, len(m)), random_state=1).round(3)),
        }
    if _p("tlc_pickups_zone_hour.parquet").exists():
        p = pd.read_parquet(_p("tlc_pickups_zone_hour.parquet")).groupby("ts")["pickups"].sum().reset_index()
        m = p.merge(w[w.city == "nyc"], left_on="ts", right_on="time")
        m["anom"] = m["pickups"] - m.groupby([m["ts"].dt.dayofweek, m["ts"].dt.hour])["pickups"].transform("mean")
        res["nyc_demand"] = {
            "precip_vs_demand_anomaly": correlation(m["precipitation"], m["anom"]),
            "temp_vs_demand_anomaly": correlation(m["temperature_2m"], m["anom"]),
            "rain_vs_dry": two_group(m.loc[m.precipitation > 0.5, "anom"], m.loc[m.precipitation == 0, "anom"],
                                     ("rain", "dry")),
        }
    if _p("chicago_crashes.parquet").exists():
        c = pd.read_parquet(_p("chicago_crashes.parquet"))
        hourly = c.set_index("crash_date").resample("h").size().rename("crashes").reset_index()
        m = hourly.merge(w[w.city == "chicago"], left_on="crash_date", right_on="time")
        m["anom"] = m["crashes"] - m.groupby([m["time"].dt.dayofweek, m["time"].dt.hour])["crashes"].transform("mean")
        sev_by_wx = pd.crosstab(c["weather_condition"], c["severity"].isin(["Serious", "Fatal"]))
        res["chicago_crashes"] = {
            "precip_vs_crash_anomaly": correlation(m["precipitation"], m["anom"]),
            "rain_vs_dry_crash_rate": two_group(m.loc[m.precipitation > 0.5, "crashes"],
                                                m.loc[m.precipitation == 0, "crashes"], ("rain", "dry")),
            "snow_vs_none": two_group(m.loc[m.snowfall > 0, "crashes"], m.loc[m.snowfall == 0, "crashes"],
                                      ("snow", "no_snow")) if "snowfall" in m else None,
            "reported_weather_vs_serious": chi_square(sev_by_wx),
            "by_reported_weather": _records(c.groupby("weather_condition").agg(
                crashes=("crash_record_id", "size"),
                serious_rate=("severity", lambda s: float(s.isin(["Serious", "Fatal"]).mean())))
                .sort_values("crashes", ascending=False).reset_index()),
        }
    _write("weather_impact", res)
    return res


# ------------------------------------------------------------------- infrastructure need
def infrastructure_priority() -> dict:
    """Q10: Chicago H3 cells ranked by a transparent composite of crash burden (serious
    crashes per year) and recurring congestion (share of peak bins High/Severe). Both are
    standardised (z-scores); the index is their mean. Weights are equal and documented."""
    hot_p = get_settings().analytics_dir / "accident_hotspots.json"
    tr_p = get_settings().analytics_dir / "traffic.json"
    if not (hot_p.exists() and tr_p.exists()):
        return {}
    hot = pd.DataFrame(json.loads(hot_p.read_text())["h3_cells"])
    seg = pd.DataFrame(json.loads(tr_p.read_text())["segment_stats"])
    cong = seg.groupby("h3")["p_congested"].mean().rename("congestion_share")
    df = hot.set_index("h3")[["crashes", "serious", "gi_class", "lat", "lon"]].join(cong, how="inner")
    if len(df) < 5:
        return {}
    z = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    df["z_serious"] = z(df["serious"])
    df["z_congestion"] = z(df["congestion_share"])
    df["priority_index"] = (df["z_serious"] + df["z_congestion"]) / 2
    out = {"method": "mean of z(serious crashes) and z(peak congestion share) per H3 res-8 cell; "
                     "cells must contain at least one tracker segment",
           "cells": _records(df.sort_values("priority_index", ascending=False).reset_index().round(4))}
    _write("infrastructure_priority", out)
    return out


def run_all() -> None:
    for fn in (mobility, traffic, accidents, weather_impact, infrastructure_priority):
        try:
            fn()
        except FileNotFoundError as exc:
            log.warning("%s skipped: %s", fn.__name__, exc)
