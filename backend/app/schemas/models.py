"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Freshness(BaseModel):
    source: str
    data_as_of: str | None = Field(None, description="Latest observation timestamp in the data")
    mode: Literal["live", "latest_available", "historical", "unavailable"]
    note: str | None = None
    last_ingestion: str | None = None
    last_ingestion_status: str | None = None


class TravelTimeRequest(BaseModel):
    pu_zone: int = Field(..., ge=1, le=263, description="TLC pickup zone (LocationID)")
    do_zone: int = Field(..., ge=1, le=263, description="TLC drop-off zone (LocationID)")
    departure: datetime
    distance_mi: float | None = Field(None, gt=0, lt=100,
                                      description="Trip distance; defaults to historical OD median")


class TravelTimeResponse(BaseModel):
    eta_minutes: float
    baseline_eta_minutes: float | None
    distance_mi: float
    distance_source: Literal["user", "historical_od_median", "unavailable"]
    weather_used: dict | None
    model: str
    model_version: str
    test_mae_minutes: float | None
    note: str


class Health(BaseModel):
    status: str
    processed_tables: list[str]
    model_tasks: list[str]
    database: str
