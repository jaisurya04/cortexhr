"""Load config/phase3_config.yaml once; fail loudly if a key is missing."""
from __future__ import annotations
from functools import lru_cache
import yaml
from .paths import ROOT


@lru_cache(maxsize=1)
def load_config() -> dict:
    p = ROOT / "config" / "phase3_config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def seed() -> int:
    return int(load_config()["seed"])
