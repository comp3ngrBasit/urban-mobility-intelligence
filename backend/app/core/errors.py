"""API error types. A missing artifact is a normal state (pipeline not run yet), so it
returns 404 with a machine-readable hint the frontend turns into an empty state."""
from __future__ import annotations

from fastapi import HTTPException


class NotReady(HTTPException):
    def __init__(self, what: str, hint: str):
        super().__init__(status_code=404, detail={"error": "not_ready", "what": what, "hint": hint})
