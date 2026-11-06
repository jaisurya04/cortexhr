"""Small I/O helpers: read data with validation, write evidence files."""
from __future__ import annotations
import datetime as _dt
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import CSV

log = logging.getLogger("cortexhr.phase3")


def read_csv(name: str) -> pd.DataFrame:
    p = CSV / (name if name.endswith(".csv") else name + ".csv")
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    log.info("loaded %s %s", p.name, df.shape)
    return df


def _clean(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date)): return str(o)
    if isinstance(o, Path): return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_clean), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
