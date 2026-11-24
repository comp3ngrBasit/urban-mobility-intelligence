"""PostGIS schema (SQLAlchemy 2.0 + GeoAlchemy2) for the historical store."""
from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChicagoSegment(Base):
    __tablename__ = "chicago_segments"
    segment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    street: Mapped[str | None] = mapped_column(String(120))
    direction: Mapped[str | None] = mapped_column(String(8))
    from_street: Mapped[str | None] = mapped_column(String(120))
    to_street: Mapped[str | None] = mapped_column(String(120))
    length: Mapped[float | None] = mapped_column(Float)
    geom = mapped_column(Geometry("LINESTRING", srid=4326, spatial_index=True))


class ChicagoSpeed(Base):
    __tablename__ = "chicago_speeds"
    segment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    speed: Mapped[float | None] = mapped_column(Float)
    bus_count: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("ix_chicago_speeds_time", "time"),)


class Crash(Base):
    __tablename__ = "chicago_crashes"
    crash_record_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    crash_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    severity: Mapped[str | None] = mapped_column(String(16))
    weather_condition: Mapped[str | None] = mapped_column(String(64))
    roadway_surface_cond: Mapped[str | None] = mapped_column(String(64))
    prim_contributory_cause: Mapped[str | None] = mapped_column(String(160))
    injuries_total: Mapped[float | None] = mapped_column(Float)
    geom = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))


class DotSpeed(Base):
    __tablename__ = "nyc_dot_speeds"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    data_as_of: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    speed: Mapped[float | None] = mapped_column(Float)
    travel_time: Mapped[float | None] = mapped_column(Float)
    borough: Mapped[str | None] = mapped_column(String(32))
    link_name: Mapped[str | None] = mapped_column(String(200))


class ZoneDemand(Base):
    __tablename__ = "tlc_pickups_zone_hour"
    zone: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    pickups: Mapped[int] = mapped_column(Integer)


class TaxiZone(Base):
    __tablename__ = "taxi_zones"
    location_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone: Mapped[str | None] = mapped_column(String(120))
    borough: Mapped[str | None] = mapped_column(String(40))
    geom = mapped_column(Geometry("MULTIPOLYGON", srid=4326, spatial_index=True))


class WeatherHourly(Base):
    __tablename__ = "weather_hourly"
    city: Mapped[str] = mapped_column(String(16), primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    temperature_2m: Mapped[float | None] = mapped_column(Float)
    precipitation: Mapped[float | None] = mapped_column(Float)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float)
    relative_humidity_2m: Mapped[float | None] = mapped_column(Float)
    weather_category: Mapped[str | None] = mapped_column(String(16))
