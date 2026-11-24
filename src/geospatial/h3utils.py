"""H3 helpers (h3-py v4 API)."""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def to_cells(lat: pd.Series, lon: pd.Series, res: int) -> pd.Series:
    out = np.full(len(lat), None, dtype=object)
    la, lo = lat.to_numpy(), lon.to_numpy()
    ok = ~(np.isnan(la) | np.isnan(lo))
    out[ok] = [h3.latlng_to_cell(a, b, res) for a, b in zip(la[ok], lo[ok])]
    return pd.Series(out, index=lat.index, dtype="object")


def cell_polygon(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def cell_center(cell: str) -> tuple[float, float]:
    return h3.cell_to_latlng(cell)


def neighbors(cell: str, k: int = 1) -> set[str]:
    return set(h3.grid_disk(cell, k)) - {cell}
