"""Automated scheduled refresh (APScheduler). Cron expressions come from .env.

    python scripts/scheduler.py

Live NYC DOT speeds refresh every 15 min; crashes and weather daily; TLC monthly
(followed by clean + analyze + retrain)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402

from src import pipeline  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.preprocessing import cleaners  # noqa: E402

log = get_logger("scheduler")


def live_refresh():
    pipeline.ingest("nyc_dot")
    pipeline._safe("clean_dot", cleaners.clean_dot_speeds)
    from src.models import anomaly
    pipeline._safe("anomaly", anomaly.run)


def daily_refresh():
    pipeline.ingest("chicago_crashes")
    pipeline.ingest("weather")
    pipeline.ingest("weather_forecast")
    pipeline._safe("clean_crashes", cleaners.clean_crashes)
    pipeline._safe("clean_weather", cleaners.clean_weather)
    pipeline.analyze()


def monthly_refresh():
    pipeline.ingest("tlc_trips")
    pipeline.clean()
    pipeline.train()
    pipeline.analyze()


def main():
    s = BlockingScheduler()
    s.add_job(live_refresh, CronTrigger.from_crontab(os.getenv("SCHEDULE_NYC_DOT", "*/15 * * * *")),
              id="live", max_instances=1, coalesce=True)
    s.add_job(daily_refresh, CronTrigger.from_crontab(os.getenv("SCHEDULE_CRASHES", "0 3 * * *")),
              id="daily", max_instances=1, coalesce=True)
    s.add_job(monthly_refresh, CronTrigger.from_crontab(os.getenv("SCHEDULE_TLC", "0 4 5 * *")),
              id="monthly", max_instances=1, coalesce=True)
    log.info("scheduler started: %s", [j.id for j in s.get_jobs()])
    live_refresh()
    s.start()


if __name__ == "__main__":
    main()
