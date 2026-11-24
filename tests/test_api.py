"""Backend endpoint tests (TestClient) against artifacts produced from SYNTHETIC fixtures."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained):
    from backend.main import create_app
    return TestClient(create_app())


@pytest.fixture(scope="module")
def empty_client(tmp_path_factory):
    """App pointed at an empty data dir -> structured 404 'not_ready', never fake data."""
    import os
    from src.common import config
    base = tmp_path_factory.mktemp("empty")
    old = os.environ.get("DATA_DIR"), os.environ.get("ARTIFACTS_DIR")
    os.environ["DATA_DIR"], os.environ["ARTIFACTS_DIR"] = str(base / "d"), str(base / "a")
    config.get_settings.cache_clear()
    from backend.main import create_app
    yield TestClient(create_app())
    os.environ["DATA_DIR"], os.environ["ARTIFACTS_DIR"] = old
    config.get_settings.cache_clear()


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and "speed_h60" in r["model_tasks"]


@pytest.mark.parametrize("path", [
    "/api/overview", "/api/live/nyc", "/api/live/chicago", "/api/live/anomalies?city=chicago",
    "/api/predict/options", "/api/predict/segments?horizon=60", "/api/predict/zones?horizon=1",
    "/api/mobility", "/api/mobility/od?min_trips=1", "/api/mobility/decomposition",
    "/api/traffic", "/api/accidents/summary", "/api/accidents/hotspots?method=gi", "/api/accidents/hotspots?method=dbscan",
    "/api/accidents/hotspots?method=kde", "/api/accidents/points?severity=Minor&hour_from=7&hour_to=9",
    "/api/weather/impact", "/api/weather/predictions-by-condition?horizon=60",
    "/api/geo/chicago-segments", "/api/geo/h3?layer=crashes", "/api/models", "/api/models/speed_h60",
    "/api/models/congestion_h60/predictions?sample=100", "/api/pipeline/status", "/api/pipeline/quality",
])
def test_endpoints_ok(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.text[:300])


def test_live_freshness_is_labelled(client):
    r = client.get("/api/live/nyc").json()
    assert r["freshness"]["mode"] in ("live", "latest_available")
    assert r["freshness"]["data_as_of"]
    c = client.get("/api/live/chicago").json()
    assert c["freshness"]["mode"] == "historical"


def test_segment_detail_and_filter(client):
    seg = client.get("/api/predict/segments?horizon=60").json()["segments"][0]["segment_id"]
    d = client.get(f"/api/predict/segment/{seg}?horizon=60").json()
    assert d["speed"] and {"actual", "predicted", "lower", "upper"} <= set(d["speed"][0])
    pts = client.get("/api/accidents/points?severity=Minor").json()
    assert all(p["severity"] == "Minor" for p in pts["points"])


def test_travel_time_prediction(client):
    r = client.post("/api/predict/travel-time",
                    json={"pu_zone": 100, "do_zone": 105, "departure": "2026-08-20T08:30:00"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eta_minutes"] > 0 and body["distance_source"] == "historical_od_median"


def test_travel_time_validation(client):
    assert client.post("/api/predict/travel-time",
                       json={"pu_zone": 999, "do_zone": 1, "departure": "2026-08-20T08:30:00"}).status_code == 422


def test_not_ready_when_no_data(empty_client):
    r = empty_client.get("/api/mobility")
    assert r.status_code == 404 and r.json()["detail"]["error"] == "not_ready"
    o = empty_client.get("/api/overview").json()
    assert o["mobility"] is None and o["accidents"] is None


def test_safe_json_maps_nan_to_null():
    from backend.app.core.responses import SafeJSONResponse
    import numpy as np
    body = SafeJSONResponse({"a": float("nan"), "b": [np.float64("inf"), 1.5], "c": np.int64(3)}).body
    assert body == b'{"a":null,"b":[null,1.5],"c":3}'
