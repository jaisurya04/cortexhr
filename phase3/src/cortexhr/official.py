"""Loads the OFFICIAL starter helper (05_Starter/cortexhr_helpers.py, supplied by the course, kept unmodified).

Official definitions used as the headline in Tasks 4-5:
  binary_report : top 10% = int(n * 0.10) rows by score; precision/recall/F1 at that top group
  risk_bands    : High = score >= 90th percentile, Medium = score >= 70th percentile, Low = rest
  catch_rate    : precision / recall of anomaly flags vs truth ids
"""
from __future__ import annotations
import importlib.util
from . import config as C

_PATH = C.project_root() / "05_Starter" / "cortexhr_helpers.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("cortexhr_helpers", _PATH)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.DATA = C.project_root() / "03_Data" / "csv"        # the starter's relative path assumes another folder layout
    return mod


H = load_helpers()
