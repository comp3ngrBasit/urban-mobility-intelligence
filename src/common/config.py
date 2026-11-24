"""Central configuration: environment settings + YAML source/model registries."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    socrata_app_token: str | None = None
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    configs_dir: Path = REPO_ROOT / "configs"
    database_url: str | None = None
    mlflow_tracking_uri: str = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    api_cors_origins: str = "http://localhost:5173,http://localhost:8080"
    log_level: str = "INFO"
    http_timeout_s: int = 60
    http_max_retries: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def analytics_dir(self) -> Path:
        return self.artifacts_dir / "analytics"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.external_dir,
                  self.analytics_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir = Path(s.data_dir).resolve()
    s.artifacts_dir = Path(s.artifacts_dir).resolve()
    return s


def _load_yaml(name: str) -> dict[str, Any]:
    path = get_settings().configs_dir / name
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache
def sources() -> dict[str, Any]:
    return _load_yaml("sources.yaml")


@lru_cache
def model_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")
