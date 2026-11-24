import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaners import CRASH_SCHEMA, crash_severity, traffic_grid, weather_category
from src.preprocessing.validation import SchemaError, coerce, iqr_outliers, validate


def test_schema_rejects_missing_required_column():
    with pytest.raises(SchemaError):
        coerce(pd.DataFrame({"crash_date": ["2026-01-01"]}), CRASH_SCHEMA)


def test_validation_counts_duplicates_and_bad_coordinates():
    df = pd.DataFrame({"crash_record_id": ["a", "a", "b"], "crash_date": pd.to_datetime(["2026-01-01"] * 3),
                       "latitude": [41.9, 41.9, 10.0], "longitude": [-87.6, -87.6, -87.6]})
    CRASH_SCHEMA.bbox = {"min_lat": 41.6, "max_lat": 42.1, "min_lon": -88, "max_lon": -87.5}
    rep = validate(df, CRASH_SCHEMA)
    assert rep["duplicates"] == 1
    assert rep["invalid_coordinates"] == 1


def test_iqr_outliers_flags_extremes():
    s = pd.Series(list(range(100)) + [10_000])
    assert iqr_outliers(s).sum() == 1


def test_severity_ordering():
    df = pd.DataFrame({"injuries_fatal": [1, 0, 0, 0], "injuries_incapacitating": [0, 1, 0, 0],
                       "injuries_total": [1, 1, 2, 0]})
    assert crash_severity(df).tolist() == ["Fatal", "Serious", "Minor", "No injury"]


def test_weather_category_codes():
    assert weather_category(pd.Series([0, 3, 45, 61, 73, 95])).tolist() == \
        ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Thunderstorm"]


def test_traffic_grid_does_not_forward_fill():
    df = pd.DataFrame({"segment_id": [1, 1], "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
                       "speed": [20.0, 30.0], "bus_count": [1, 1]})
    g = traffic_grid(df, 15)
    assert len(g) == 5 and g["speed"].isna().sum() == 3


def test_cleaners_produce_processed_tables(populated):
    from src.common.config import get_settings
    p = get_settings().processed_dir
    t = pd.read_parquet(p / "chicago_traffic.parquet")
    assert t["speed"].min() >= 0 or np.isnan(t["speed"].min())   # -1 sentinels converted to NaN
    assert (p / "tlc_pickups_zone_hour.parquet").exists()
    assert (p / "nyc_dot_links.parquet").exists()
    assert (get_settings().analytics_dir / "quality" / "chicago_traffic.json").exists()
