"""Structured logging used by pipelines, scheduler and API."""
from __future__ import annotations

import logging
import sys

from src.common.config import get_settings

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
        logger.setLevel(get_settings().log_level.upper())
        logger.propagate = False
    return logger
