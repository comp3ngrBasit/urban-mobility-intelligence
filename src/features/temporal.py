"""Calendar features. These describe the *target* timestamp and are known in advance,
so they are leakage-free for any horizon."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

PEAK_AM = range(7, 10)    # 07:00-09:59
PEAK_PM = range(16, 19)   # 16:00-18:59


def add_temporal(df: pd.DataFrame, ts_col: str = "ts", subdiv: str | None = None,
                 prefix: str = "") -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    out = df.copy()
    years = range(int(t.dt.year.min()), int(t.dt.year.max()) + 1) if len(t) else []
    hol = holidays.US(subdiv=subdiv, years=years)
    hour = t.dt.hour
    out[f"{prefix}hour"] = hour
    out[f"{prefix}minute_of_day"] = hour * 60 + t.dt.minute
    out[f"{prefix}dow"] = t.dt.dayofweek
    out[f"{prefix}day"] = t.dt.day
    out[f"{prefix}week"] = t.dt.isocalendar().week.astype(int)
    out[f"{prefix}month"] = t.dt.month
    out[f"{prefix}is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    out[f"{prefix}is_holiday"] = t.dt.date.map(lambda d: d in hol).astype(int)
    out[f"{prefix}is_peak"] = (hour.isin(PEAK_AM) | hour.isin(PEAK_PM)).astype(int) * \
        (1 - out[f"{prefix}is_weekend"]) * (1 - out[f"{prefix}is_holiday"])
    mod = out[f"{prefix}minute_of_day"]
    out[f"{prefix}tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    out[f"{prefix}tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    out[f"{prefix}dow_sin"] = np.sin(2 * np.pi * out[f"{prefix}dow"] / 7)
    out[f"{prefix}dow_cos"] = np.cos(2 * np.pi * out[f"{prefix}dow"] / 7)
    out[f"{prefix}month_sin"] = np.sin(2 * np.pi * (out[f"{prefix}month"] - 1) / 12)
    out[f"{prefix}month_cos"] = np.cos(2 * np.pi * (out[f"{prefix}month"] - 1) / 12)
    return out


TEMPORAL_FEATURES = ["hour", "dow", "month", "is_weekend", "is_holiday", "is_peak",
                     "tod_sin", "tod_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
