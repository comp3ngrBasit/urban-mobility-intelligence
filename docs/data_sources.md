# Data sources

Verified 2026-11-24 against each publisher's metadata API. Re-verify at any time with
`python -m src.pipeline verify` (results appear on the Data & Pipeline page).

| Source | Endpoint | Status at verification | Update cadence | Used for |
|---|---|---|---|---|
| NYC TLC trip records | `https://d37ci6vzurychx.cloudfront.net/trip-data/{yellow,green,fhvhv}_tripdata_YYYY-MM.parquet` | Published through 2026-09 | Monthly, ~2-month lag | Demand, OD, travel time |
| NYC TLC zones | `/misc/taxi_zone_lookup.csv`, `/misc/taxi_zones.zip` | Available | Static | Zone names, boundaries, adjacency |
| NYC DOT Traffic Speeds | `data.cityofnewyork.us/resource/i4gi-tjb9.json` | Updating | Several times per minute | **Live** speeds, anomalies |
| Chicago Traffic Tracker – historical by segment | `data.cityofchicago.org/resource/4g9f-3jbs.json` | **No longer updated** (last rows ≈ 2026-05-31) | None (archive) | Speed & congestion models (historical) |
| Chicago Traffic Tracker – current by segment | `…/n4j6-wkkf.json` | **Deprecated** (last rows 2026-04-30); data fully contained in 4g9f-3jbs | None | Not used for modelling |
| Chicago Traffic Crashes | `data.cityofchicago.org/resource/85ca-t3if.json` | Updating | Daily | Hotspots, severity, weather–crash analysis, safety features |
| OpenStreetMap | OSMnx (`graph_from_place`), Overpass `overpass-api.de/api/interpreter` | Available | Continuous | Road class, intersections, connectivity, bridges/tunnels/signals |
| Open-Meteo | `archive-api.open-meteo.com/v1/archive`, `api.open-meteo.com/v1/forecast` | Available | Archive ≈5-day lag; forecast hourly | Weather features & impact analysis |

## Important differences between sources

* **Different phenomena.** TLC = taxi/for-hire demand (a sample of trips, not traffic volume).
  DOT/Tracker = speed estimates on specific links. Crashes = police reports. They are analysed
  separately and never merged into a single "traffic" measure.
* **Different geographies.** NYC data uses TLC zones (polygons) and DOT links (polylines);
  Chicago uses tracker segments (from/to points) and crash points. H3 res-8 cells are used for
  spatial statistics within each city.
* **Different clocks.** All sources publish local wall-clock time; weather is requested in the
  matching IANA timezone so hourly joins align. DST-ambiguous hours are left as published.
* **Visibility** is only available from the Open-Meteo *forecast* API, not the archive, so it
  is not used as a training feature.
* **Speed = -1** in the Chicago tracker means "no estimate" (too few bus GPS pings) and is
  converted to missing, not zero.

## Fallbacks

| Failure | Behaviour |
|---|---|
| API unreachable / 5xx | Retries with exponential backoff (honours `Retry-After`), then the run is recorded as failed; previous partitions remain and are served with their real timestamp. |
| Column removed from a Socrata schema | Detected by `verify_columns`; the column is dropped with a warning and the run continues. |
| TLC month not yet published | 404 is treated as "not published", not an error. |
| OSM / Overpass unavailable | Road-network features are left empty (models handle missing values); map layer shows an empty state. |
| No data at all | API returns `404 {"error": "not_ready", "hint": "<command>"}`; the UI shows the command to run. No placeholder numbers are ever produced. |
