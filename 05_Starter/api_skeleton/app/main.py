"""CortexHR Phase 2 API: payroll and attendance anomaly scoring plus the HR chatbot.

Run from the repository root:
    uvicorn app.main:app --app-dir 05_Starter/api_skeleton --reload

Design notes
------------
* Models are loaded ONCE at start-up (``lifespan``) from ``models/`` and kept in memory.  If a file is missing the
  service layer retrains and saves it, so a fresh clone still starts.
* All scoring logic lives in ``cortexhr.phase2.service`` (plain functions, unit-tested without a web server).  This
  file only validates input and routes.
* Bad input returns HTTP 422 with Pydantic's field-level message.  Unknown fields are rejected (``extra='forbid'``).
* Payroll and attendance responses use exactly the columns of ``sample_anomaly_results``.
* Synthetic HR data only.  Flags are review candidates, never proof of fraud and never grounds for an
  employment decision on their own.
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "03_Data" / "csv"
sys.path.insert(0, str(ROOT / "src"))

from cortexhr.phase2 import service  # noqa: E402
from cortexhr.phase2.chatbot import HRChatbot  # noqa: E402
from cortexhr.phase2.payroll import parse_month  # noqa: E402

log = logging.getLogger("cortexhr.api")
API_VERSION = "2.1.0"
MAX_BATCH = service.MAX_BATCH_ROWS
MONTH_PATTERN = r"^\d{4}-\d{2}(-\d{2})?$"

_STATE: dict = {}          # loaded once; read-only after start-up (chat sessions live inside the chatbot object)


# --------------------------------------------------------------------------- model access (load once)
def get_payroll_detector():
    if "payroll" not in _STATE:
        _STATE["payroll"] = service.load_payroll_detector()
    return _STATE["payroll"]


def get_attendance_detector():
    if "attendance" not in _STATE:
        _STATE["attendance"] = service.load_attendance_detector()
    return _STATE["attendance"]


def get_chatbot():
    if "chatbot" not in _STATE:
        model, version = service.load_chatbot_model()
        _STATE["chatbot_version"] = version
        _STATE["chatbot"] = HRChatbot(
            model, CSV / "chatbot_responses.csv", CSV / "hr_faq.csv",
            users_csv=ROOT / "05_Starter" / "mock_services" / "mock_data" / "demo_users.csv")
    return _STATE["chatbot"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_payroll_detector(); get_attendance_detector(); get_chatbot()
    log.info("Models loaded once at start-up: %s", {k: type(v).__name__ for k, v in _STATE.items()})
    yield
    _STATE.clear()


app = FastAPI(title="CortexHR Phase 2 API", version=API_VERSION, lifespan=lifespan,
              description="Training/demo API on synthetic HR data. Never use for real employment decisions.")


# --------------------------------------------------------------------------- request schemas
class AttendanceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_id: str = Field(min_length=1, max_length=80)
    month: Optional[str] = Field(default=None, pattern=MONTH_PATTERN, description="YYYY-MM or YYYY-MM-DD")
    record_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    scheduled_days: int = Field(ge=0, le=31)
    present_days: int = Field(ge=0, le=31)
    wfh_days: int = Field(default=0, ge=0, le=31)
    late_arrivals: int = Field(default=0, ge=0, le=31)
    avg_hours_worked: float = Field(ge=0, le=24, allow_inf_nan=False)
    overtime_hours: float = Field(default=0, ge=0, le=200, allow_inf_nan=False)
    # Optional extras the detector understands; when omitted they are derived or set to a neutral zero.
    paid_leave_days: Optional[float] = Field(default=None, ge=0, le=31, allow_inf_nan=False)
    half_days_unpaid: Optional[float] = Field(default=None, ge=0, le=31, allow_inf_nan=False)
    unpaid_absence_days: Optional[float] = Field(default=None, ge=0, le=31, allow_inf_nan=False)
    longest_unpaid_streak_days: Optional[float] = Field(default=None, ge=0, le=31, allow_inf_nan=False)

    @field_validator("month")
    @classmethod
    def _month_ok(cls, v):
        if v is not None:
            parse_month(v)          # raises ValueError -> HTTP 422
        return v

    @model_validator(mode="after")
    def _relations(self):
        if self.present_days > self.scheduled_days:
            raise ValueError("present_days cannot exceed scheduled_days")
        if self.wfh_days > self.present_days:
            raise ValueError("wfh_days cannot exceed present_days (work-from-home days are present days)")
        if self.paid_leave_days is not None and self.paid_leave_days > self.scheduled_days:
            raise ValueError("paid_leave_days cannot exceed scheduled_days")
        if self.unpaid_absence_days is not None and self.unpaid_absence_days > self.scheduled_days:
            raise ValueError("unpaid_absence_days cannot exceed scheduled_days")
        if (self.longest_unpaid_streak_days is not None and self.unpaid_absence_days is not None
                and self.longest_unpaid_streak_days > self.unpaid_absence_days):
            raise ValueError("longest_unpaid_streak_days cannot exceed unpaid_absence_days")
        return self


class PayrollIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payroll_item_id: str = Field(default="API-ROW", min_length=1, max_length=80)
    run_id: str = Field(min_length=1, max_length=40)
    employee_id: str = Field(min_length=1, max_length=80)
    month: str = Field(pattern=MONTH_PATTERN, description="YYYY-MM or YYYY-MM-DD")
    monthly_fixed_gross: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    payable_days: float = Field(ge=0, le=31, allow_inf_nan=False)
    lop_days: float = Field(ge=0, le=31, allow_inf_nan=False)
    overtime_hours_paid: float = Field(ge=0, le=744, allow_inf_nan=False)
    overtime_pay: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    variable_pay: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    gross_pay: float = Field(ge=-1e9, le=1e9, allow_inf_nan=False)
    pf_employee: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    esi_employee: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    professional_tax: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    tds: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    other_deductions: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    total_deductions: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    net_pay: float = Field(ge=-1e9, le=1e9, allow_inf_nan=False)
    department_id: str = Field(default="API", max_length=40)
    role_id: str = Field(default="API", max_length=40)
    job_level: int = Field(default=1, ge=1, le=20)
    overtime_eligible: Optional[str] = Field(default=None, pattern="^[YN]$", description="Y or N; leave out if unknown")
    variable_pay_type: str = Field(default="Other", max_length=40)

    @field_validator("month")
    @classmethod
    def _month_ok(cls, v):
        parse_month(v)
        return v

    @model_validator(mode="after")
    def _days(self):
        if self.payable_days + self.lop_days > 31:
            raise ValueError("payable_days + lop_days cannot exceed 31")
        return self


class PayrollBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PayrollIn] = Field(min_length=1, max_length=MAX_BATCH,
                                  description=f"1 to {MAX_BATCH} payslips per call; duplicates inside the batch are detected")


class AttendanceBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[AttendanceIn] = Field(min_length=1, max_length=MAX_BATCH)


class ChatIn(BaseModel):
    """``user_role`` is accepted for backwards compatibility but IGNORED: the role comes from the demo identity
    directory (demo_users.csv), so a client cannot promote itself to HR by sending a field."""
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=120)
    user_role: str = Field(default="EMPLOYEE", max_length=40)
    demo_user: str = Field(default="EMP-0002", max_length=80)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v):
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


# --------------------------------------------------------------------------- endpoints
def _dump(model: BaseModel) -> dict:
    return model.model_dump(exclude_none=True)


@app.get("/v1/health")
def health():
    registry = service.read_registry()
    loaded = {k: (k in _STATE) for k in ("payroll", "attendance", "chatbot")}
    return {"status": "ok", "service": "cortexhr-phase2", "version": API_VERSION, "models_loaded": loaded,
            "model_versions": {k: v.get("version") for k, v in registry.items()}}


@app.get("/v1/models")
def models():
    return service.read_registry()


@app.post("/v1/anomaly/payroll")
def anomaly_payroll(row: PayrollIn):
    try:
        return service.score_payroll_rows([_dump(row)], get_payroll_detector())[0]
    except ValueError as exc:
        raise HTTPException(422, f"Invalid payroll row: {exc}")
    except Exception as exc:
        log.exception("payroll scoring failed")
        raise HTTPException(500, f"Payroll scoring failed: {type(exc).__name__}")


@app.post("/v1/anomaly/payroll/batch")
def anomaly_payroll_batch(batch: PayrollBatchIn):
    try:
        results = service.score_payroll_rows([_dump(r) for r in batch.rows], get_payroll_detector())
    except ValueError as exc:
        raise HTTPException(422, f"Invalid payroll batch: {exc}")
    except Exception as exc:
        log.exception("payroll batch scoring failed")
        raise HTTPException(500, f"Payroll batch scoring failed: {type(exc).__name__}")
    return {"count": len(results), "flagged_count": sum(r["flagged"] == "Y" for r in results),
            "max_batch_rows": MAX_BATCH, "results": results}


@app.post("/v1/anomaly/attendance")
def anomaly_attendance(row: AttendanceIn):
    try:
        return service.score_attendance_rows([_dump(row)], get_attendance_detector())[0]
    except ValueError as exc:
        raise HTTPException(422, f"Invalid attendance row: {exc}")
    except Exception as exc:
        log.exception("attendance scoring failed")
        raise HTTPException(500, f"Attendance scoring failed: {type(exc).__name__}")


@app.post("/v1/anomaly/attendance/batch")
def anomaly_attendance_batch(batch: AttendanceBatchIn):
    try:
        results = service.score_attendance_rows([_dump(r) for r in batch.rows], get_attendance_detector())
    except ValueError as exc:
        raise HTTPException(422, f"Invalid attendance batch: {exc}")
    except Exception as exc:
        log.exception("attendance batch scoring failed")
        raise HTTPException(500, f"Attendance batch scoring failed: {type(exc).__name__}")
    return {"count": len(results), "flagged_count": sum(r["flagged"] == "Y" for r in results),
            "max_batch_rows": MAX_BATCH, "results": results}


@app.post("/v1/chat")
def chat(msg: ChatIn):
    try:
        bot = get_chatbot()
        reply = bot.reply(msg.message.strip(), msg.session_id, msg.user_role, msg.demo_user)
        reply["model_version"] = _STATE.get("chatbot_version", service.CHATBOT_VERSION)
        return reply
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(500, f"Chat processing failed safely: {type(exc).__name__}")
