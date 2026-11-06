"""Project paths. ROOT is the phase3/ folder, wherever the repository is cloned."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "03_Data" / "csv"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
METADATA = ARTIFACTS / "metadata"
PREDICTIONS = ARTIFACTS / "predictions"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
VAULT = ROOT / "04_Validation_Vault"


def task_dir(n: int) -> Path:
    d = REPORTS / f"task{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs() -> None:
    for d in (MODELS, METADATA, PREDICTIONS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
