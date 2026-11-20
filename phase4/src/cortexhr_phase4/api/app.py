"""The unified, versioned CortexHR API (FastAPI). Every route lives under /v1.

Start locally from the repository root::

    PYTHONPATH=phase4/src uvicorn cortexhr_phase4.api.app:app --reload

All settings come from environment variables (see phase4/.env.example). Models are loaded once at start-up.
"""
from __future__ import annotations

import hmac
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.logging import get_logger
from ..core.security import load_users, verify_password, issue_token, verify_token, revoke_token, role_allows, secret_is_strong
from ..core.settings import API_VERSION, Settings, load_settings
from ..services import handlers as h
from ..services.errors import ApiError
from ..services.registry import Registry
from . import schemas as s

DESCRIPTION = ("One service, many models: attrition risk, payroll and attendance anomalies, HR chatbot, resume ranking, pay-band advice, "
               "headcount forecast and monitoring. Synthetic HR data. Every output is AI-generated and needs human review.")


def _reg(request: Request) -> Registry:
    return request.app.state.registry


def _auth(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional shared key: enforced only when CORTEXHR_API_KEY is set. /v1/health stays open for uptime checks."""
    key = request.app.state.settings.api_key
    if key and not (x_api_key and hmac.compare_digest(x_api_key.encode(), key.encode())):
        raise ApiError(401, "unauthorized", "missing or wrong X-API-Key header")


def _principal(request: Request, authorization: Optional[str] = Header(default=None)) -> dict:
    """Authenticate every protected API request and enforce the caller's role."""
    cfg = request.app.state.settings
    if not cfg.auth_secret:
        if cfg.env.lower() == "production":
            raise ApiError(503, "auth_not_configured", "Set CORTEXHR_AUTH_SECRET and CORTEXHR_AUTH_USERS before production use.")
        principal = {"sub": "local-demo", "role": "admin"}
    else:
        if not secret_is_strong(cfg.auth_secret) or not request.app.state.auth_users:
            raise ApiError(503, "auth_misconfigured", request.app.state.auth_config_error or "Use a signing secret of at least 32 bytes and configure at least one valid PBKDF2 user.")
        scheme, _, bearer_value = (authorization or "").partition(" ")
        principal = verify_token(bearer_value, cfg.auth_secret) if scheme.lower() == "bearer" and bearer_value else None
        if not principal:
            raise ApiError(401, "unauthorized", "A valid Bearer token is required. Please sign in again.")
    if not role_allows(principal["role"], request.url.path):
        raise ApiError(403, "forbidden", f"Role '{principal['role']}' cannot access this resource.")
    request.state.principal = principal
    return principal


def build_router() -> tuple[APIRouter, APIRouter]:
    open_r, r = APIRouter(prefix="/v1"), APIRouter(prefix="/v1", dependencies=[Depends(_auth), Depends(_principal)])

    @open_r.get("/health", response_model=s.HealthOut, tags=["service"])
    def health(reg: Registry = Depends(_reg)):
        return h.health(reg)

    @open_r.post("/auth/login", tags=["authentication"])
    def login(body: s.AuthLoginIn, request: Request):
        cfg = request.app.state.settings
        if not cfg.auth_secret:
            raise ApiError(503, "auth_not_configured", "Authentication is not configured. Set CORTEXHR_AUTH_SECRET and CORTEXHR_AUTH_USERS.")
        if not secret_is_strong(cfg.auth_secret) or not request.app.state.auth_users:
            raise ApiError(503, "auth_misconfigured", request.app.state.auth_config_error or "Use a signing secret of at least 32 bytes and configure at least one valid PBKDF2 user.")
        record = request.app.state.auth_users.get(body.username.strip().lower())
        if not record or not verify_password(body.password, record["password_hash"]):
            # Same response for unknown user and wrong password; do not reveal account existence.
            raise ApiError(401, "invalid_credentials", "Username or password is incorrect.")
        signed_value = issue_token(record["username"], record["role"], cfg.auth_secret)
        return {"access_token": signed_value, "token_type": "bearer", "expires_in": 28800,
                "user": {"username": record["username"], "display_name": record["display_name"], "role": record["role"]}}

    @r.post("/auth/logout", tags=["authentication"])
    def logout(request: Request, authorization: Optional[str] = Header(default=None)):
        scheme, _, bearer_value = (authorization or "").partition(" ")
        if scheme.lower() == "bearer" and bearer_value:
            revoke_token(bearer_value)
        return {"status": "signed_out", "notice": "This token is revoked for the lifetime of the current API process."}

    @r.get("/models", response_model=s.ModelsOut, tags=["service"])
    def models(reg: Registry = Depends(_reg)):
        return h.models(reg)

    @r.get("/overview", tags=["dashboard"])
    def overview(reg: Registry = Depends(_reg)):
        return h.overview(reg)

    @r.post("/attrition/predict", response_model=s.AttritionOut, tags=["attrition"])
    def attrition(body: s.AttritionIn, reg: Registry = Depends(_reg)):
        return h.attrition_predict(reg, body.employee_id)

    @r.get("/attrition/employees", tags=["attrition"])
    def attrition_employees(band: Optional[str] = None, department_id: Optional[str] = None, limit: int = Query(25, ge=1, le=500), reg: Registry = Depends(_reg)):
        return h.attrition_list(reg, band, department_id, limit)

    @r.post("/anomaly/payroll", response_model=s.BatchOut, tags=["anomaly"])
    def payroll(body: s.PayrollIn, reg: Registry = Depends(_reg)):
        return h.payroll(reg, [body.model_dump(exclude_none=True)])

    @r.post("/anomaly/payroll/batch", response_model=s.BatchOut, tags=["anomaly"])
    def payroll_batch(body: s.PayrollBatchIn, reg: Registry = Depends(_reg)):
        return h.payroll(reg, [x.model_dump(exclude_none=True) for x in body.rows])

    @r.post("/anomaly/attendance", response_model=s.BatchOut, tags=["anomaly"])
    def attendance(body: s.AttendanceIn, reg: Registry = Depends(_reg)):
        return h.attendance(reg, [body.model_dump(exclude_none=True)])

    @r.post("/anomaly/attendance/batch", response_model=s.BatchOut, tags=["anomaly"])
    def attendance_batch(body: s.AttendanceBatchIn, reg: Registry = Depends(_reg)):
        return h.attendance(reg, [x.model_dump(exclude_none=True) for x in body.rows])

    @r.post("/chat", tags=["chatbot"])
    def chat(body: s.ChatIn, reg: Registry = Depends(_reg)):
        return h.chat(reg, body.message, body.session_id, body.demo_user)

    @r.post("/resume/rank", tags=["resume"])
    def resume_rank(body: s.ResumeRankIn, reg: Registry = Depends(_reg)):
        return h.resume_rank(reg, body.model_dump(exclude_none=True))

    @r.post("/resume/upload", tags=["resume"])
    async def resume_upload(file: UploadFile = File(...), job_id: str = Form(...), top_k: int = Form(5), reg: Registry = Depends(_reg)):
        if not 1 <= top_k <= 20:
            raise ApiError(422, "validation", "top_k must be between 1 and 20")
        content = await file.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise ApiError(413, "file_too_large", "Resume must be no larger than 8 MB.")
        return h.resume_upload_rank(reg, file.filename or "resume", content, job_id, top_k)

    @r.post("/payband/recommend", response_model=s.PayBandOut, tags=["pay band"])
    def payband(body: s.PayBandIn, reg: Registry = Depends(_reg)):
        return h.payband(reg, body.model_dump())

    @r.get("/payband/options", tags=["pay band"])
    def payband_options(reg: Registry = Depends(_reg)):
        return h.payband_options(reg)

    @r.get("/resume/jobs", tags=["resume"])
    def resume_jobs(reg: Registry = Depends(_reg)):
        return h.resume_jobs(reg)

    @r.get("/performance", tags=["service"])
    def performance(reg: Registry = Depends(_reg)):
        return h.performance(reg)

    @r.get("/forecast", tags=["forecast"])
    def forecast(metric: str = Query("headcount_end", pattern="^(headcount_end|exits|attrition_rate_monthly_pct)$"),
                 department_id: str = Query("ALL", max_length=10), reg: Registry = Depends(_reg)):
        return h.forecast(reg, metric, department_id)

    @r.post("/forecast/scenario", tags=["forecast"])
    def forecast_scenario(body: s.ScenarioIn, reg: Registry = Depends(_reg)):
        return h.forecast_scenario(reg, [x.model_dump() for x in body.scenarios], body.departments)

    @r.get("/monitoring", response_model=s.StatusOut, tags=["monitoring"])
    def monitoring(refresh: bool = False, reg: Registry = Depends(_reg)):
        return h.monitoring(reg, refresh)

    return open_r, r


def create_app(settings: Settings | None = None, registry: Registry | None = None) -> FastAPI:
    """Application factory (tests pass a prepared ``registry``)."""
    cfg = settings or load_settings()
    log = get_logger("cortexhr.api", cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = registry or Registry(cfg).load_all()           # models are loaded ONCE, here
        log.info("startup version=%s ready=%s models=%s", API_VERSION, app.state.registry.ready(), {k: v.status for k, v in app.state.registry.info.items()})
        yield

    app = FastAPI(title="CortexHR API", version=API_VERSION, description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = cfg
    try:
        app.state.auth_users = load_users(cfg.auth_users)
        app.state.auth_config_error = None
    except RuntimeError as exc:
        app.state.auth_users = {}
        app.state.auth_config_error = str(exc)
    if registry is not None:
        app.state.registry = registry
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.allowed_origins), allow_methods=["GET", "POST"], allow_headers=["*"])
    for router in build_router():
        app.include_router(router)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid, t0 = request.headers.get("x-request-id") or uuid.uuid4().hex[:12], time.perf_counter()
        request.state.rid = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info("request_id=%s method=%s path=%s status=%s ms=%.1f", rid, request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
        return response

    def _err(status: int, error: str, details: list[str]) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": error, "details": details})

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return _err(exc.status, exc.error, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return _err(422, "validation", [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return _err(exc.status_code, {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error"), [str(exc.detail)])

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        rid = getattr(request.state, "rid", "-")
        log.exception("request_id=%s unhandled=%s", rid, type(exc).__name__)
        return _err(500, "internal_error", [f"unexpected error ({type(exc).__name__}); quote request id {rid} when reporting it"])

    return app


app = create_app()
