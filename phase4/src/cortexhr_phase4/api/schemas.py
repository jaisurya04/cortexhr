"""Pydantic request and response contracts of the /v1 API (pydantic v2)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MONTH_PATTERN = r"^\d{4}-\d{2}(-\d{2})?$"
MAX_BATCH = 500
_OUT = ConfigDict(extra="allow")                 # responses may carry extra, documented fields
_IN = ConfigDict(extra="forbid")                 # requests never accept unknown fields


# ------------------------------------------------------------------ requests
class AttritionIn(BaseModel):
    model_config = _IN
    employee_id: str = Field(min_length=1, max_length=40, examples=["EMP-0002"])


class PayrollIn(BaseModel):
    model_config = _IN
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
    overtime_eligible: Optional[str] = Field(default=None, pattern="^[YN]$")
    variable_pay_type: str = Field(default="Other", max_length=40)

    @model_validator(mode="after")
    def _days(self) -> "PayrollIn":
        if self.payable_days + self.lop_days > 31:
            raise ValueError("payable_days + lop_days cannot exceed 31")
        return self


class PayrollBatchIn(BaseModel):
    model_config = _IN
    rows: list[PayrollIn] = Field(min_length=1, max_length=MAX_BATCH)


class AttendanceIn(BaseModel):
    model_config = _IN
    employee_id: str = Field(min_length=1, max_length=80)
    month: Optional[str] = Field(default=None, pattern=MONTH_PATTERN)
    record_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    scheduled_days: int = Field(ge=0, le=31)
    present_days: int = Field(ge=0, le=31)
    wfh_days: int = Field(default=0, ge=0, le=31)
    late_arrivals: int = Field(default=0, ge=0, le=31)
    avg_hours_worked: float = Field(ge=0, le=24, allow_inf_nan=False)
    overtime_hours: float = Field(default=0, ge=0, le=200, allow_inf_nan=False)

    @model_validator(mode="after")
    def _relations(self) -> "AttendanceIn":
        if self.present_days > self.scheduled_days:
            raise ValueError("present_days cannot exceed scheduled_days")
        if self.wfh_days > self.present_days:
            raise ValueError("wfh_days cannot exceed present_days")
        return self


class AttendanceBatchIn(BaseModel):
    model_config = _IN
    rows: list[AttendanceIn] = Field(min_length=1, max_length=MAX_BATCH)


class ChatIn(BaseModel):
    model_config = _IN
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=120)
    demo_user: str = Field(default="EMP-0002", max_length=80)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class JobIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: str
    required_skills: str = Field(description="';'-separated skills")
    min_experience_years: float = Field(ge=0, le=40)
    description_text: Optional[str] = None
    nice_to_have_skills: Optional[str] = None


class ResumeRankIn(BaseModel):
    """Extra fields are passed through on purpose: the ranker rejects protected fields (name, gender, age, ...) with an explicit message."""
    model_config = ConfigDict(extra="allow")
    job_id: Optional[str] = Field(default=None, examples=["J01"])
    job: Optional[JobIn] = None
    resume_ids: Optional[list[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class PayBandIn(BaseModel):
    model_config = _IN
    department_id: str
    role_id: str
    job_level: int = Field(ge=1, le=10)
    band_mid_2026: float = Field(gt=0, le=1e8, description="role band mid-point, INR per year")
    market_index: float = Field(gt=0, le=10)
    prior_experience_years: float = Field(ge=0, le=50)
    skills_count: int = Field(ge=0, le=50)
    work_location: str
    highest_education: str


class ScenarioEvent(BaseModel):
    model_config = _IN
    effect: Literal["hiring_freeze", "hiring_boost", "policy_change"]
    start: str = Field(pattern=r"^\d{4}-\d{2}(-\d{2})?$")
    end: str = Field(pattern=r"^\d{4}-\d{2}(-\d{2})?$")
    department_id: str = "ALL"
    name: str = "scenario"


class ScenarioIn(BaseModel):
    model_config = _IN
    scenarios: list[ScenarioEvent] = Field(min_length=1, max_length=5)
    departments: Optional[list[str]] = None


# ------------------------------------------------------------------ responses (extra fields allowed)
class HealthOut(BaseModel):
    model_config = _OUT
    status: str
    service: str
    version: str
    uptime_s: float
    models: dict[str, str]
    notice: str


class ModelsOut(BaseModel):
    model_config = _OUT
    version: str
    models: dict[str, Any]


class AttritionOut(BaseModel):
    model_config = _OUT
    employee_id: str
    department_id: str
    as_of_date: str
    risk_score_12m: float
    risk_band: str
    top_reasons: list[str]
    model_version: str
    source: str
    notice: str


class BatchOut(BaseModel):
    model_config = _OUT
    count: int
    flagged_count: int
    results: list[dict[str, Any]]
    model_version: str
    notice: str


class PayBandOut(BaseModel):
    model_config = _OUT
    recommended_low: float
    recommended_mid: float
    recommended_high: float
    currency: str
    model_version: str
    notice: str


class StatusOut(BaseModel):
    model_config = _OUT
    status: str
    notice: str


class ErrorOut(BaseModel):
    error: str
    details: list[str] = []


class AuthLoginIn(BaseModel):
    model_config = _IN
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class AuthRegisterIn(BaseModel):
    model_config = _IN
    username: str = Field(min_length=3, max_length=120)
    display_name: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=12, max_length=256)


class AuthApproveIn(BaseModel):
    model_config = _IN
    username: str = Field(min_length=3, max_length=120)
    role: Literal["hr_manager", "recruiter", "analyst"]
