"""Shared test helpers. Tests are plain functions with asserts: run them with ``python tests/run_all.py`` (no pytest needed)
or with ``pytest phase4/tests`` (same files)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


class Skip(Exception):
    """Raised by :func:`skip`; the runner counts it as SKIP and prints the reason (never as a pass)."""


def skip(reason: str) -> None:
    try:
        import pytest
    except ImportError:
        raise Skip(reason)
    pytest.skip(reason)


def need(*modules: str) -> None:
    """Skip the calling test when a library is not installed in this environment."""
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            skip(f"{m} is not installed in this environment")


_REG = None


def registry():
    """One shared, fully loaded Registry (loading takes a few seconds)."""
    global _REG
    if _REG is None:
        from cortexhr_phase4.services.registry import Registry
        _REG = Registry().load_all()
    return _REG


def csv_dir() -> Path:
    from cortexhr_phase4.core.settings import load_settings
    return load_settings().data_dir
