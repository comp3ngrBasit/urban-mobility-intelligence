# Architecture

## System architecture

```mermaid
flowchart LR
  subgraph Sources
    TLC[NYC TLC trip Parquet<br/>monthly]
    DOT[NYC DOT Traffic Speeds<br/>i4gi-tjb9 · live]
    CTT[Chicago Traffic Tracker<br/>4g9f-3jbs · archived]
    CRA[Chicago Crashes<br/>85ca-t3if · daily]
    OSM[OpenStreetMap<br/>OSMnx + Overpass]
    OM[Open-Meteo<br/>archive + forecast]
  end
  subgraph Pipeline["src/ (Python)"]
    ING[ingestion<br/>retry · paginate · watermark]
    RAW[(data/raw<br/>Parquet partitions)]
    CLN[preprocessing<br/>validate · clean]
    PRO[(data/processed)]
    FE[features]
    ML[models · forecasting<br/>Optuna · TS-CV · SHAP]
    AN[analytics · geospatial<br/>stats · Moran · Gi*]
    ART[(artifacts/<br/>models + analytics)]
  end
  PG[(PostgreSQL + PostGIS<br/>historical store)]
  MLF[MLflow]
  API[FastAPI]
  UI[React + MapLibre + Plotly]
  SCH[APScheduler]

  Sources --> ING --> RAW --> CLN --> PRO --> FE --> ML --> ART
  PRO --> AN --> ART
  PRO --> PG
  ML --> MLF
  ART --> API
  PRO --> API
  API --> UI
  SCH -. triggers .-> ING
```

## Data pipeline

```mermaid
flowchart TB
  V[verify: endpoint, schema,<br/>publisher freshness] --> I
  I[ingest: incremental window<br/>from watermark] -->|atomic write| R[raw partition]
  I -->|run record| L[(_runs.jsonl ledger)]
  R --> C[coerce types → validate → clean]
  C -->|quality report| Q[(analytics/quality/*.json)]
  C --> P[processed tables]
  P --> T[train] --> A[analyze]
  P --> D[(PostGIS upsert)]
  L --> M[Monitoring page /<br/>API freshness labels]
```

Failure isolation: every source runs in its own guarded stage. A failed pull is recorded in
the ledger and leaves the previous partitions untouched, so the API keeps serving the last
successful data with its real `data_as_of` timestamp and a "latest available" label.

## ML workflow

```mermaid
flowchart LR
  F[feature table<br/>sorted by time] --> S{chronological split}
  S --> TR[train 70%]
  S --> VA[valid 15%]
  S --> TE[test 15%]
  TR --> O[Optuna per algorithm<br/>score on valid]
  O --> CV[TimeSeriesSplit CV<br/>stability on train]
  O --> SEL[select best by VALID score]
  TR & VA --> RF[refit train+valid]
  RF --> EV[evaluate once on TEST]
  TE --> EV
  TE --> BL[baselines on same rows]
  EV --> X[SHAP · importance · errors · intervals]
  X --> REG[(artifacts/models/task/version)]
  REG --> MLF[MLflow run]
```

Leakage controls:
* every lag/rolling feature is shifted by the forecast horizon (tested in `tests/test_features.py`);
* free-flow speed, speed profiles, crash-exposure and OD/zone history are fit on the training period only;
* weather is joined at the forecast *issue* time, not the target time;
* model selection uses the validation period; the test period is scored once.
