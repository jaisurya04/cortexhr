"""Plain-Python service layer behind the Phase 2 API.

Everything the endpoints do lives here as ordinary functions (dict in, dict out) so it can be tested without a web
server.  ``05_Starter/api_skeleton/app/main.py`` only adds validation (Pydantic) and routing.

Model lifecycle (Task 8, "load once, version everything")
---------------------------------------------------------
``scripts/train_phase2_models.py`` trains the detectors and the chatbot and saves them in ``models/`` together with
``model_registry.json`` (version, training rows, library version, file hash).  The API loads them ONCE at start-up.
If a file is missing, or was saved with an incompatible library version, the loader retrains from the CSVs and
saves again, so a fresh clone still starts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from . import attendance as att
from . import payroll as pay

log = logging.getLogger("cortexhr.phase2")

ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = ROOT / "03_Data" / "csv"
MODELS_DIR = ROOT / "models"
REGISTRY = "model_registry.json"
PAYROLL_FILE = "payroll_detector_v2.joblib"
ATTENDANCE_FILE = "attendance_detector_v2.joblib"
CHATBOT_FILE = "chatbot_intent_model_v1.joblib"           # trained on the ORIGINAL 858 messages
CHATBOT_VERSION = "chatbot-tfidf-lr-v1"
CHATBOT_EXPANDED_VERSION = "chatbot-tfidf-lr-expanded-dev"
MAX_BATCH_ROWS = 500

PAYROLL_FIELDS = pay.NUMERIC_BASE + ["payroll_item_id", "run_id", "employee_id", "month", "department_id",
                                     "role_id", "variable_pay_type", "overtime_eligible", "job_level"]


# --------------------------------------------------------------------------- registry
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_registry(models_dir: Path = MODELS_DIR) -> dict:
    p = Path(models_dir) / REGISTRY
    return json.loads(p.read_text()) if p.exists() else {}


def _register(models_dir: Path, name: str, path: Path, **info) -> None:
    reg = read_registry(models_dir)
    reg[name] = {"file": path.name, "sha256": _sha256(path), "sklearn_version": sklearn.__version__,
                 "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
    (Path(models_dir) / REGISTRY).write_text(json.dumps(reg, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- train / load
def train_payroll(csv_dir: Path = CSV_DIR, models_dir: Path = MODELS_DIR) -> pay.PayrollDetector:
    data = pay.load_payroll(Path(csv_dir) / "payroll_monthly_features.csv")
    det = pay.fit_isolation_forest(data, contamination=0.03, random_state=42)
    path = det.save(Path(models_dir) / PAYROLL_FILE)
    _register(models_dir, "payroll", path, version=det.version, trained_rows=det.trained_rows,
              contamination=det.contamination, training_data="payroll_monthly_features.csv", labels_used=False)
    return det


def train_attendance(csv_dir: Path = CSV_DIR, models_dir: Path = MODELS_DIR) -> att.AttendanceDetector:
    data = pd.read_csv(Path(csv_dir) / "attendance_monthly_features.csv")
    det = att.fit_attendance_detector(data, contamination=0.03, random_state=42)
    path = det.save(Path(models_dir) / ATTENDANCE_FILE)
    _register(models_dir, "attendance", path, version=det.version, trained_rows=det.trained_rows,
              contamination=det.contamination, training_data="attendance_monthly_features.csv", labels_used=False)
    return det


def train_chatbot(csv_dir: Path = CSV_DIR, models_dir: Path = MODELS_DIR):
    from .chatbot import train_intent_model
    model = train_intent_model(Path(csv_dir) / "chatbot_nlu_training.csv")
    path = Path(models_dir) / CHATBOT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    _register(models_dir, "chatbot", path, version=CHATBOT_VERSION, trained_rows=858,
              training_data="chatbot_nlu_training.csv (original, no test-set-driven additions)")
    return model


def _load_or_train(path: Path, loader, trainer, label: str):
    try:
        if path.exists():
            return loader(path)
    except Exception as exc:                      # unpickle or version problem: retrain instead of crashing
        log.warning("Could not load %s (%s: %s); retraining.", label, type(exc).__name__, exc)
    log.warning("%s not found or unusable at %s; training now.", label, path)
    return trainer()


def load_payroll_detector(models_dir: Path = MODELS_DIR, csv_dir: Path = CSV_DIR) -> pay.PayrollDetector:
    return _load_or_train(Path(models_dir) / PAYROLL_FILE, pay.PayrollDetector.load,
                          lambda: train_payroll(csv_dir, models_dir), "payroll detector")


def load_attendance_detector(models_dir: Path = MODELS_DIR, csv_dir: Path = CSV_DIR) -> att.AttendanceDetector:
    return _load_or_train(Path(models_dir) / ATTENDANCE_FILE, att.AttendanceDetector.load,
                          lambda: train_attendance(csv_dir, models_dir), "attendance detector")


def chatbot_variant() -> str:
    """'original' (default, honest 81% model) or 'expanded' (development-only, tuned while looking at the test set)."""
    return os.getenv("CORTEXHR_CHATBOT_TRAINING", "original").strip().lower()


def load_chatbot_model(models_dir: Path = MODELS_DIR, csv_dir: Path = CSV_DIR):
    from .chatbot import train_intent_model
    if chatbot_variant() == "expanded":
        return train_intent_model(Path(csv_dir) / "chatbot_nlu_training_expanded.csv"), CHATBOT_EXPANDED_VERSION
    return _load_or_train(Path(models_dir) / CHATBOT_FILE, joblib.load, lambda: train_chatbot(csv_dir, models_dir),
                          "chatbot model"), CHATBOT_VERSION


def train_all(csv_dir: Path = CSV_DIR, models_dir: Path = MODELS_DIR) -> dict:
    train_payroll(csv_dir, models_dir); train_attendance(csv_dir, models_dir); train_chatbot(csv_dir, models_dir)
    return read_registry(models_dir)


# --------------------------------------------------------------------------- scoring
def _clean(records: pd.DataFrame) -> list[dict]:
    out = []
    for rec in records.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT:
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def score_payroll_rows(rows: list[dict], detector: pay.PayrollDetector, source: str = "api_payroll_request") -> list[dict]:
    """Score one or many payslips with the loaded detector.  Output follows the official sample columns,
    ranked flagged-first then by money at risk."""
    frame = pd.DataFrame(rows)
    frame["month"] = [pay.parse_month(m) for m in frame["month"]]
    for col, default in (("department_id", "API"), ("role_id", "API"), ("variable_pay_type", "Other"),
                         ("overtime_eligible", "UNKNOWN"), ("job_level", 1), ("payroll_item_id", "API-ROW")):
        if col not in frame:
            frame[col] = default
    scored, _ = pay.score_payroll(frame, model=detector)
    official = pay.to_official_format(scored, source_dataset=source, result_prefix="API-PAY-")
    return _clean(official)


def score_attendance_rows(rows: list[dict], detector: att.AttendanceDetector) -> list[dict]:
    frame = pd.DataFrame(rows)
    if "record_id" not in frame:
        frame["record_id"] = [f"API-ATT-{i + 1}" for i in range(len(frame))]
    if "month" in frame:
        frame["month"] = [pay.parse_month(m) if m else pd.NaT for m in frame["month"]]
    scored = att.score_attendance(frame, detector)
    official = att.to_official_format(scored, result_prefix="API-ATT-")
    official["source_dataset"] = "api_attendance_request"
    return _clean(official)


# --------------------------------------------------------------------------- chatbot aggregate view
_FLAG_CACHE: dict = {}


def payroll_flag_summary(month: str) -> dict:
    """Aggregate view for the chatbot's ``payroll_flags`` intent (HR/ADMIN only): counts and top reason, never
    individual pay.  Uses the frozen payroll predictions; scores the file once if they do not exist."""
    if "frame" not in _FLAG_CACHE:
        frozen = ROOT / "reports" / "phase2" / "payroll_predictions_frozen.csv"
        if frozen.exists():
            frame = pd.read_csv(frozen)
        else:
            data = pay.load_payroll(CSV_DIR / "payroll_monthly_features.csv")
            frame, _ = pay.score_payroll(data, model=load_payroll_detector())
        frame["_period"] = pd.to_datetime(frame["month"], errors="coerce").dt.strftime("%Y-%m")
        _FLAG_CACHE["frame"] = frame
    frame = _FLAG_CACHE["frame"]
    in_month = frame[frame["_period"] == month]
    if in_month.empty:
        return {"count": None}
    flagged = in_month[in_month["flagged"].astype(str).str.lower().isin(["true", "1", "y"])]
    if flagged.empty:
        return {"count": 0, "top_reason": "none", "money_at_risk": 0.0}
    top = flagged["reason_code"].str.split("|").str[0].value_counts().index[0]
    return {"count": int(len(flagged)), "top_reason": top, "money_at_risk": float(flagged["money_at_risk"].sum())}


def attendance_flag_from_summary(summary: dict) -> dict:
    """Screen ONE month of the signed-in user's own attendance summary (as the mock service returns it)."""
    if "attendance_detector" not in _FLAG_CACHE:
        _FLAG_CACHE["attendance_detector"] = load_attendance_detector()
    row = {k: summary.get(k) for k in ("employee_id", "scheduled_days", "present_days", "paid_leave_days", "unpaid_absence_days",
                                        "late_arrivals", "wfh_days") if summary.get(k) is not None}
    row.setdefault("avg_hours_worked", None); row.setdefault("overtime_hours", 0.0)
    scored = att.score_attendance(pd.DataFrame([row]), _FLAG_CACHE["attendance_detector"])
    first = scored.iloc[0]
    return {"flagged": bool(first.flagged), "reason_code": first.reason_code, "reason_text": first.reason_text}
