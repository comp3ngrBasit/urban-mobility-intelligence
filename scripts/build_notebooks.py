"""Generate the analysis notebooks (committed without outputs; execute them after the
pipeline has ingested real data: `jupyter nbconvert --execute --inplace notebooks/*.ipynb`)."""
from pathlib import Path

import nbformat as nbf

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
import plotly.express as px
from src.common.config import get_settings
S = get_settings(); P = S.processed_dir; A = S.analytics_dir"""

NB = {
 "01_data_collection": [
  ("md", "# 01 · Data collection\nVerifies every source (endpoint, schema, publisher freshness), then runs the incremental ingestion. Re-running only fetches new data (watermarks)."),
  ("code", SETUP),
  ("code", "from src.ingestion.verify import verify_all\npd.DataFrame(verify_all())[['source','status','dataset_id','rows_updated_at','deprecated','is_live','last_observation']]"),
  ("code", "from src import pipeline\npipeline.ingest()   # or pipeline.ingest('nyc_dot') for a single source"),
  ("code", "from src.common.runs import read_runs\npd.DataFrame(read_runs())[['source','status','records','data_min_ts','data_max_ts','duration_s','error']].tail(20)"),
 ],
 "02_eda": [
  ("md", "# 02 · Exploratory data analysis\nCleaning + validation, then temporal distributions, correlations, hypothesis tests and decomposition. All figures read the processed tables."),
  ("code", SETUP),
  ("code", "from src import pipeline\npipeline.clean()\nfor p in sorted((A/'quality').glob('*.json')):\n    r = json.loads(p.read_text()); print(r['dataset'], r['rows'], '->', r.get('rows_after_cleaning'), r.get('issues'))"),
  ("code", "from src.analytics import eda\nmob = eda.mobility(); acc = eda.accidents(); wx = eda.weather_impact()"),
  ("code", "h = pd.DataFrame(mob['hourly_profile'])\npx.line(h, x='hour', y='mean', color='daytype', error_y=h['hi']-h['mean'], title='NYC pickups per hour (95% CI)')"),
  ("code", "pd.DataFrame(acc['monthly']).plot(x='month', y=['crashes','serious'], secondary_y='serious', title='Chicago crashes per month')"),
  ("code", "pd.json_normalize(wx, sep='.').T.rename(columns={0:'value'}).filter(like='pearson', axis=0)"),
  ("code", "print('weekday vs weekend demand:', mob['weekday_vs_weekend'])\nprint('holiday vs weekday:', mob['holiday_vs_weekday'])"),
 ],
 "03_geospatial_analysis": [
  ("md", "# 03 · Geospatial analysis\nH3 aggregation, Global Moran's I, LISA, Getis-Ord Gi* (FDR-controlled), density clustering and KDE."),
  ("code", SETUP),
  ("code", "from src.geospatial import hotspots\nh = hotspots.run()\nh['morans_i'], h['significant_hotspots']"),
  ("code", "cells = pd.DataFrame(h['h3_cells'])\npx.scatter_mapbox(cells, lat='lat', lon='lon', color='gi_class', size='crashes', mapbox_style='carto-darkmatter', zoom=9.5, title='Gi* crash hotspots')"),
  ("code", "from src.analytics import eda\ntr = eda.traffic(); tr['spatial'].get('morans_i')"),
  ("code", "pd.DataFrame(tr['bottlenecks']).head(15)[['segment_id','street','from_street','to_street','congested_share','recurrence_pct_days']]"),
  ("code", "import folium\nfrom src.geospatial.h3utils import cell_polygon\nm = folium.Map([41.86,-87.68], zoom_start=11, tiles='cartodbdark_matter')\nfor r in cells[cells.gi_class=='Hotspot'].itertuples():\n    folium.GeoJson(cell_polygon(r.h3).__geo_interface__, style_function=lambda _: {'color':'#F2545B','weight':1}).add_to(m)\nm"),
 ],
 "04_feature_engineering": [
  ("md", "# 04 · Feature engineering\nHorizon-aware traffic features (every lag shifted by the horizon), H3 neighbour speeds, OSM road attributes, issue-time weather and train-period-only safety features."),
  ("code", SETUP),
  ("code", "from src.models.traffic import _load, prepare\nfrom src.features.traffic import build_traffic_features, feature_columns\ntraffic, segs, weather, crashes = _load()\ngrid, ff, prof, static, t1 = prepare(traffic, segs, crashes)\nprint('train cutoff for free-flow/profile/safety:', t1)"),
  ("code", "F = build_traffic_features(grid, 60, free_flow=ff, profile=prof, segments=segs, weather=weather, static_features=static)\ncols = feature_columns(F); len(cols), cols"),
  ("code", "F[cols].describe().T[['count','mean','std','min','max']]"),
  ("code", "F[cols + ['speed']].corr(method='spearman')['speed'].sort_values()"),
 ],
 "05_model_training": [
  ("md", "# 05 · Model training\nChronological split, Optuna tuning on validation, time-series CV, refit on train+valid, single evaluation on test. Runs are logged to MLflow."),
  ("code", SETUP),
  ("code", "from src.models import traffic, travel_time\nfrom src.forecasting import demand\nres_traffic = traffic.train_all()"),
  ("code", "res_tt = travel_time.train()\nres_dem = demand.train_ml()\nres_cls = demand.train_classical()"),
  ("code", "from src.models import anomaly\nanomaly.run()"),
 ],
 "06_model_evaluation": [
  ("md", "# 06 · Model evaluation\nReads the registered model versions: test metrics vs baselines, interval coverage, SHAP and error analysis."),
  ("code", SETUP),
  ("code", "from src.models.registry import load_json\nrows = []\nfor d in sorted(S.models_dir.iterdir()):\n    m = load_json(d.name, 'metrics')\n    if m and 'best_model' in m:\n        t = m['models'][m['best_model']]['test']\n        rows.append({'task': d.name, 'best': m['best_model'], **{k: t.get(k) for k in ('mae','rmse','r2','f1_macro','accuracy')},\n                     **{f'baseline_{b}': (v.get('mae') or v.get('f1_macro')) for b, v in m['baselines'].items()}})\npd.DataFrame(rows)"),
  ("code", "shap = load_json('speed_h60', 'shap')\npd.DataFrame(shap['mean_abs_shap']).head(15).plot.barh(x='feature', y='value', title='mean |SHAP| · speed +60 min')"),
  ("code", "load_json('speed_h60', 'intervals'), load_json('demand_h1', 'intervals')"),
  ("code", "err = load_json('speed_h60', 'errors')\npd.DataFrame(err['hour']).sort_values('hour').plot(x='hour', y=['mae','bias'], title='Error by hour')"),
 ],
}

out = Path(__file__).resolve().parents[1] / "notebooks"
out.mkdir(exist_ok=True)
for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, out / f"{name}.ipynb")
    print("wrote", name)
