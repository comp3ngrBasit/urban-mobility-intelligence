"""Raw-zone persistence: partitioned Parquet files written atomically, never overwritten
in place, so the last successful pull always survives a failed refresh."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import get_settings


def raw_path(source: str, name: str) -> Path:
    p = get_settings().raw_dir / source / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def list_partitions(source: str, pattern: str = "*.parquet") -> list[Path]:
    root = get_settings().raw_dir / source
    return sorted(root.glob(pattern)) if root.exists() else []
