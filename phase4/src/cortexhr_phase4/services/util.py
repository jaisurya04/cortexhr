"""JSON-safety helpers."""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def jsonable(obj):
    """Recursively convert numpy / pandas values into plain JSON types (NaN and inf become ``None``)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj
