"""Train and save every Phase 2 model into models/ (Task 8).

    python scripts/train_phase2_models.py

Saves  models/payroll_detector_v2.joblib, models/attendance_detector_v2.joblib, models/chatbot_intent_model_v1.joblib
and models/model_registry.json (version, rows, library version, SHA-256 of each file).
No label column is read by any detector.  The chatbot is trained on the ORIGINAL 858 messages only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortexhr.phase2 import service  # noqa: E402

if __name__ == "__main__":
    registry = service.train_all()
    print(json.dumps(registry, indent=2))
