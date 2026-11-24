# API examples

Interactive OpenAPI docs: `http://localhost:8000/docs`.

```bash
# health & monitoring
curl -s localhost:8000/api/health | jq
curl -s localhost:8000/api/pipeline/status | jq '.sources[] | {source,last_status,data_max_ts}'
curl -s "localhost:8000/api/pipeline/runs?source=nyc_dot_speeds&limit=5" | jq

# live (NYC DOT) — note freshness.mode: live | latest_available
curl -s localhost:8000/api/live/nyc | jq '.freshness, .summary'
curl -s "localhost:8000/api/live/nyc/4616/history?hours=24" | jq '.series | length'
curl -s localhost:8000/api/live/chicago | jq '.freshness.mode'      # always "historical"
curl -s "localhost:8000/api/live/anomalies?city=nyc&limit=20" | jq

# predictions
curl -s localhost:8000/api/predict/options | jq '{speed_horizons_min, demand_horizons_h}'
curl -s "localhost:8000/api/predict/segments?horizon=60" | jq '.target_time, .segments[0]'
curl -s "localhost:8000/api/predict/segment/1309?horizon=60" | jq '.segment_metrics, .intervals'
curl -s "localhost:8000/api/predict/zone/161?horizon=6" | jq '.zone_metrics'
curl -s -X POST localhost:8000/api/predict/travel-time \
  -H 'Content-Type: application/json' \
  -d '{"pu_zone":161,"do_zone":132,"departure":"2026-12-01T17:30:00"}' | jq

# analytics
curl -s "localhost:8000/api/mobility?borough=Manhattan" | jq '.weekday_vs_weekend'
curl -s "localhost:8000/api/mobility/od?hour=8&min_trips=100&limit=20" | jq
curl -s localhost:8000/api/traffic | jq '.bottlenecks[:5]'
curl -s "localhost:8000/api/accidents/hotspots?method=gi" | jq '.morans_i, .significant_hotspots'
curl -s "localhost:8000/api/accidents/points?severity=Fatal&weather=RAIN" | jq '.total_matching'
curl -s localhost:8000/api/weather/impact | jq '.chicago_speed.precip_vs_speed_anomaly'
curl -s "localhost:8000/api/weather/predictions-by-condition?horizon=60" | jq

# geo layers (GeoJSON)
curl -s "localhost:8000/api/geo/road-network?city=chicago&min_rank=4" | jq '.features | length'
curl -s "localhost:8000/api/geo/h3?layer=priority" | jq '.features[0].properties'

# models
curl -s localhost:8000/api/models | jq '.[] | {task,best_model,best_test}'
curl -s localhost:8000/api/models/congestion_h60 | jq '.metrics.models[.metrics.best_model].test.confusion_matrix'
```

Response when a dataset or model has not been produced yet:

```json
{"detail": {"error": "not_ready", "what": "mobility", "hint": "python -m src.pipeline analyze"}}
```
