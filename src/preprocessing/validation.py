"""Data validation: schema, types, ranges, coordinates, duplicates, missingness and
outliers. Every cleaner calls `validate()` and the report is persisted for the
Data & Pipeline Monitoring page."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import get_settings
from src.common.runs import utcnow_iso


@dataclass
class ColumnRule:
    dtype: str                        # "datetime" | "float" | "int" | "str"
    required: bool = True
    min: float | None = None
    max: float | None = None


@dataclass
class Schema:
    name: str
    columns: dict[str, ColumnRule]
    key: list[str] = field(default_factory=list)
    lat_col: str | None = None
    lon_col: str | None = None
    bbox: dict[str, float] | None = None
    outlier_cols: list[str] = field(default_factory=list)


class SchemaError(ValueError):
    pass


def coerce(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Coerce declared columns to their types (invalid values -> NaN/NaT)."""
    missing = [c for c, r in schema.columns.items() if r.required and c not in df.columns]
    if missing:
        raise SchemaError(f"{schema.name}: required columns missing: {missing}")
    out = df.copy()
    for col, rule in schema.columns.items():
        if col not in out.columns:
            continue
        if rule.dtype == "datetime":
            out[col] = pd.to_datetime(out[col], errors="coerce")
            if getattr(out[col].dt, "tz", None) is not None:   # standardise to naive local time
                out[col] = out[col].dt.tz_localize(None)
        elif rule.dtype in ("float", "int"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif rule.dtype == "str":
            out[col] = out[col].astype("string")
    return out


def iqr_outliers(s: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def validate(df: pd.DataFrame, schema: Schema) -> dict[str, Any]:
    n = len(df)
    report: dict[str, Any] = {"dataset": schema.name, "checked_at": utcnow_iso(), "rows": n,
                              "columns": {}, "issues": []}
    for col, rule in schema.columns.items():
        if col not in df.columns:
            continue
        s = df[col]
        info: dict[str, Any] = {"missing_pct": round(float(s.isna().mean() * 100), 3) if n else 0.0}
        if rule.min is not None or rule.max is not None:
            num = pd.to_numeric(s, errors="coerce")
            bad = ((num < rule.min) if rule.min is not None else False) | \
                  ((num > rule.max) if rule.max is not None else False)
            info["out_of_range"] = int(np.sum(bad))
        report["columns"][col] = info
        if info["missing_pct"] > 20:
            report["issues"].append(f"{col}: {info['missing_pct']}% missing")
    if schema.key and all(k in df.columns for k in schema.key):
        dups = int(df.duplicated(subset=schema.key).sum())
        report["duplicates"] = dups
        if dups:
            report["issues"].append(f"{dups} duplicate rows on key {schema.key}")
    if schema.lat_col and schema.bbox and schema.lat_col in df.columns:
        lat, lon = df[schema.lat_col], df[schema.lon_col]
        b = schema.bbox
        outside = ~(lat.between(b["min_lat"], b["max_lat"]) & lon.between(b["min_lon"], b["max_lon"]))
        report["invalid_coordinates"] = int((outside & lat.notna()).sum())
        report["missing_coordinates"] = int(lat.isna().sum())
    report["outliers"] = {c: int(iqr_outliers(df[c].dropna()).sum())
                          for c in schema.outlier_cols if c in df.columns and df[c].notna().any()}
    return report


def save_report(report: dict[str, Any]) -> None:
    d = get_settings().analytics_dir / "quality"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{report['dataset']}.json").write_text(json.dumps(report, indent=2, default=str))
