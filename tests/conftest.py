"""Test configuration: every test runs against an isolated temp data/artifacts dir with a
fast model config, populated with SYNTHETIC fixtures (see tests/fixtures.py)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("umi")
    cfg_dir = base / "configs"
    cfg_dir.mkdir()
    shutil.copy(ROOT / "configs" / "sources.yaml", cfg_dir / "sources.yaml")
    mc = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    mc.update(optuna_trials=2, cv_folds=2, shap_sample=200)
    mc["speed"]["horizons_minutes"] = [15, 60]
    mc["demand"]["horizons_hours"] = [1, 24]
    mc["demand"]["top_zones_classical"] = 2
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(mc))

    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(base / "data"))
    mp.setenv("ARTIFACTS_DIR", str(base / "artifacts"))
    mp.setenv("CONFIGS_DIR", str(cfg_dir))
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{base / 'mlflow.db'}")
    mp.delenv("DATABASE_URL", raising=False)
    from src.common import config
    config.get_settings.cache_clear()
    config.sources.cache_clear()
    config.model_config.cache_clear()
    config.get_settings().ensure_dirs()
    yield base
    mp.undo()
    config.get_settings.cache_clear()
    config.sources.cache_clear()
    config.model_config.cache_clear()


@pytest.fixture(scope="session")
def populated(env):
    """Write synthetic raw partitions and run the cleaners once for the whole session."""
    from src.common.config import get_settings
    from src.preprocessing import cleaners
    from tests import fixtures as fx

    raw = get_settings().raw_dir
    for sub in ("chicago_traffic", "chicago_crashes", "nyc_dot_speeds", "nyc_tlc", "weather"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    fx.chicago_traffic_raw().to_parquet(raw / "chicago_traffic" / "segments_test.parquet")
    fx.crashes_raw().to_parquet(raw / "chicago_crashes" / "crashes_test.parquet")
    fx.dot_speeds_raw().to_parquet(raw / "nyc_dot_speeds" / "speeds_test.parquet")
    fx.tlc_yellow().to_parquet(raw / "nyc_tlc" / "yellow_2026-08.parquet")
    import pandas as pd
    pd.concat([fx.weather_raw("chicago", "2025-05-01", "2026-06-02"),
               fx.weather_raw("nyc", "2026-07-20", "2026-09-02")]).to_parquet(raw / "weather" / "archive_test.parquet")
    cleaners.clean_crashes()
    cleaners.clean_traffic()
    cleaners.clean_dot_speeds()
    cleaners.clean_tlc()
    cleaners.clean_weather()
    return env
