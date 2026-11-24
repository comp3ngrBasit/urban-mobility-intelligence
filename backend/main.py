"""FastAPI application entry point.

    uvicorn backend.main:app --reload        (run from the repository root)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from backend.app.api import analytics, geo, live, models, predict, system  # noqa: E402
from backend.app.core.responses import SafeJSONResponse  # noqa: E402
from src.common.config import get_settings  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title="Urban Mobility & Traffic Intelligence API",
        version="1.0.0",
        default_response_class=SafeJSONResponse,
        description="NYC taxi demand, NYC DOT live speeds, Chicago arterial traffic (archived), "
                    "Chicago crashes, OSM and Open-Meteo. Every response is computed from ingested data.",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(CORSMiddleware, allow_origins=get_settings().api_cors_origins.split(","),
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    for r in (system.router, live.router, predict.router, analytics.router, geo.router, models.router):
        app.include_router(r)
    return app


app = create_app()
