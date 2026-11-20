"""First thing to run on a new machine: are the pinned libraries installed and do they work together?

    python phase4/scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import sys

PINS = {"pandas": "3.0.2", "numpy": "2.4.4", "sklearn": "1.8.0", "scipy": "1.17.1", "shap": "0.54.1", "statsmodels": "0.14.6", "fastapi": "0.141.1",
        "pydantic": "2.13.5", "uvicorn": "0.52.4", "streamlit": "1.66.0", "pypdf": "5.9.0", "docx": "1.2.0", "multipart": "0.0.20", "dotenv": "1.1.1"}


def main() -> int:
    bad = 0
    print(f"python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        bad += 1
        print("  DIFF Python runtime: use Python 3.12 for the pinned release environment")
    for mod, want in PINS.items():
        try:
            imported = importlib.import_module(mod)
            have = getattr(imported, "__version__", "?")
            if mod == "dotenv":
                from importlib.metadata import version
                have = version("python-dotenv")
            flag = "ok " if have == want else "DIFF"
            bad += have != want
            print(f"  {flag} {mod:12s} installed {have:10s} pinned {want}")
        except ImportError:
            bad += 1
            print(f"  MISSING {mod:12s} pinned {want}   -> pip install -r phase4/requirements.txt")
    try:                                              # the improved forecasting model must actually fit
        import numpy as np, pandas as pd
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = pd.Series(np.random.default_rng(0).normal(size=60), index=pd.date_range("2020-01-01", periods=60, freq="MS"))
        res = SARIMAX(y, order=(1, 0, 0), trend="c").fit(disp=False)
        print(f"  ok  SARIMAX smoke fit (ar.L1 = {float(res.params['ar.L1']):.2f})")
    except Exception as exc:
        bad += 1
        print(f"  FAIL SARIMAX smoke fit: {type(exc).__name__}: {exc}")
    print("environment OK" if not bad else f"{bad} problem(s): fix them before running the forecast, API or console")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
