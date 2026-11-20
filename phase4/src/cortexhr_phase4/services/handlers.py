"""Framework-free request handlers: plain dict in, plain dict out, :class:`ApiError` for failures.

The FastAPI layer only validates the shape of the request (Pydantic) and calls these functions, so every behaviour is unit-tested
without a web server.
"""
from __future__ import annotations

import json
import math
import threading
import time
import io
import uuid
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.settings import API_VERSION, PHASE3
from ..forecasting import data as fdata
from ..forecasting import events as fevents
from ..forecasting import models as fmodels
from ..forecasting import pipeline as fpipe
from ..monitoring import monitor as mon
from .errors import ApiError
from .registry import FALLBACK, LOADED, Registry
from .util import jsonable

HUMAN_REVIEW = "AI-generated, for human review. This is decision support, not a decision about any person."
_RESUME_LOCK = threading.RLock()


def _need(reg: Registry, name: str) -> object:
    info = reg.info.get(name)
    if info is None or info.status not in (LOADED, FALLBACK) or name not in reg.obj and name != "attrition":
        raise ApiError(503, "model_unavailable", f"model '{name}' is not available: {info.detail if info else 'not loaded'}")
    return reg.obj.get(name)


# ------------------------------------------------------------------------------------------------ health / models
def health(reg: Registry) -> dict:
    states = {k: v.status for k, v in reg.info.items()}
    ok = reg.ready() and all(s == LOADED for s in states.values())
    return {"status": "ok" if ok else ("degraded" if reg.ready() else "unavailable"), "service": "cortexhr", "version": API_VERSION,
            "uptime_s": reg.uptime_s(), "models": states, "notice": HUMAN_REVIEW}


def models(reg: Registry) -> dict:
    return {"version": API_VERSION, "models": reg.models()}


# ------------------------------------------------------------------------------------------------ attrition
def attrition_predict(reg: Registry, employee_id: str) -> dict:
    _need(reg, "attrition")
    saved: pd.DataFrame = reg.obj["attrition_saved"]
    if employee_id not in saved.index:
        raise ApiError(404, "unknown_employee", f"{employee_id!r} is not in the 2026 scoring population (430 active employees)")
    row = saved.loc[employee_id]
    score, source = float(row.risk_score_12m), "saved_phase3_scores"
    live = reg.obj.get("attrition_live")
    if live is not None:
        ids = pd.read_csv(reg.settings.data_dir / "attrition_scoring_2026.csv").employee_id.tolist()
        score, source = float(live.predict_proba(reg.obj["attrition_X"].iloc[[ids.index(employee_id)]])[:, 1][0]), "live_model"
    return {"employee_id": employee_id, "department_id": row.department_id, "as_of_date": row.as_of_date, "risk_score_12m": round(score, 3), "risk_band": row.risk_band,
            "top_reasons": [row.top_reason_1, row.top_reason_2, row.top_reason_3], "model_version": row.model_version, "source": source, "notice": HUMAN_REVIEW}


def attrition_list(reg: Registry, band: str | None = None, department_id: str | None = None, limit: int = 25) -> dict:
    _need(reg, "attrition")
    d = reg.obj["attrition_saved"].reset_index()
    if band:
        if band not in ("High", "Medium", "Low"):
            raise ApiError(422, "validation", "band must be High, Medium or Low")
        d = d[d.risk_band == band]
    if department_id:
        d = d[d.department_id == department_id]
    d = d.sort_values("risk_score_12m", ascending=False)
    total = reg.obj["attrition_saved"]
    return {"count": int(len(d)), "population": int(len(total)), "bands": total.risk_band.value_counts().to_dict(),
            "mean_risk": float(total.risk_score_12m.mean()), "employees": jsonable(d.head(limit)[["employee_id", "department_id", "risk_score_12m", "risk_band", "top_reason_1", "top_reason_2", "top_reason_3"]]),
            "notice": HUMAN_REVIEW}


# ------------------------------------------------------------------------------------------------ anomalies
def _score_rows(kind: str, reg: Registry, rows: list[dict]) -> list[dict]:
    from cortexhr.phase2 import service
    det = _need(reg, kind)
    try:
        return jsonable(service.score_payroll_rows(rows, det) if kind == "payroll" else service.score_attendance_rows(rows, det))
    except ValueError as exc:
        raise ApiError(422, "validation", f"invalid {kind} row: {exc}")


def payroll(reg: Registry, rows: list[dict]) -> dict:
    res = _score_rows("payroll", reg, rows)
    return {"count": len(res), "flagged_count": sum(r.get("flagged") == "Y" for r in res), "results": res, "model_version": reg.info["payroll"].version,
            "notice": "Flags are review candidates, never proof of error or fraud. " + HUMAN_REVIEW}


def attendance(reg: Registry, rows: list[dict]) -> dict:
    res = _score_rows("attendance", reg, rows)
    return {"count": len(res), "flagged_count": sum(r.get("flagged") == "Y" for r in res), "results": res, "model_version": reg.info["attendance"].version, "notice": HUMAN_REVIEW}


# ------------------------------------------------------------------------------------------------ chat / resume / pay band
def _chat_analytics_answer(reg: Registry, bot: object, text: str, session_id: str) -> dict | None:
    """Answer supported workforce analytics intents from the supplied Phase 3 data and Phase 4 model outputs.

    Aggregate-only responses are intentional: the assistant never returns employee-level records.
    None means the question should be routed to the original Phase 2 HR policy chatbot.
    """
    data_dir = reg.settings.data_dir

    def read_csv(name: str) -> pd.DataFrame | None:
        path = data_dir / name
        try:
            return pd.read_csv(path) if path.is_file() else None
        except Exception:
            return None

    intent, answer = None, None
    # Attrition risk is the saved, explainable Phase 3 model output.
    if any(k in text for k in ("attrition", "turnover", "leave risk", "leaving risk")):
        saved = reg.obj.get("attrition_saved")
        if saved is not None and len(saved):
            if any(k in text for k in ("department", "highest", "which team")):
                grouped = saved.groupby("department_id").risk_score_12m.agg(["mean", "count"]).sort_values("mean", ascending=False)
                top = grouped.iloc[0]
                intent = "attrition_department_risk"
                answer = (f"Department {grouped.index[0]} has the highest mean predicted 12-month attrition risk "
                          f"({top['mean'] * 100:.1f}%) across {int(top['count'])} scored employees. This is a review signal, not a prediction about any one person.")
            elif any(k in text for k in ("factor", "reason", "why")):
                reasons = pd.concat([saved[c].dropna().astype(str) for c in ("top_reason_1", "top_reason_2", "top_reason_3")]).value_counts().head(3)
                intent = "attrition_risk_factors"
                answer = "The most frequently recorded contributing factors in the saved Phase 3 explanations are: " + "; ".join(reasons.index.tolist()) + ". These are associations used by the model, not causes or certainties."
            elif any(k in text for k in ("trend", "over time", "monthly")):
                hist = read_csv("monthly_headcount_history.csv")
                if hist is not None and {"month", "voluntary_exits", "headcount_end"}.issubset(hist.columns):
                    company = hist[hist.department_id == "ALL"].sort_values("month").tail(6)
                    answer = "Recent monthly attrition trend (voluntary exits / ending headcount): " + ", ".join(f"{r.month[:7]}: {int(r.voluntary_exits)}/{int(r.headcount_end)}" for r in company.itertuples()) + "."
                    intent = "attrition_trend"
            else:
                high = int((saved.risk_band.astype(str).str.casefold() == "high").sum())
                intent, answer = "attrition_high_risk_count", f"{high} of {len(saved)} employees in the saved Phase 3 scoring population are in the High risk band. HR review is required; this must not be used as an automatic employment decision."

    # Attendance analytics use the Phase 3 monthly feature table and saved anomaly evidence.
    if answer is None and any(k in text for k in ("attendance", "absence", "absent", "late arrival")):
        att = read_csv("attendance_monthly_features.csv")
        anomalies = read_csv("sample_anomaly_results.csv")
        if att is not None and len(att):
            if any(k in text for k in ("anomal", "irregular")):
                if anomalies is not None and {"source_dataset", "flagged"}.issubset(anomalies.columns):
                    subset = anomalies[(anomalies.source_dataset == "attendance_monthly_features") & (anomalies.flagged.astype(str).str.upper() == "Y")]
                    intent, answer = "attendance_anomalies", f"The saved anomaly evaluation contains {len(subset)} flagged attendance records. These are review candidates, not proof of misconduct."
                else:
                    intent, answer = "attendance_anomalies_unavailable", "Saved attendance anomaly results are not available in this data package, so I cannot report an anomaly count."
            elif any(k in text for k in ("absence", "absent")):
                total = int(pd.to_numeric(att.get("unpaid_absence_days", 0), errors="coerce").fillna(0).sum())
                intent, answer = "attendance_absences", f"The supplied monthly attendance dataset records {total:,} unpaid absence days across {len(att):,} employee-month records."
            elif any(k in text for k in ("department", "team", "issue")):
                grp = att.groupby("department_id").attendance_rate.mean().sort_values()
                intent, answer = "attendance_department", f"Department {grp.index[0]} has the lowest mean attendance rate in the supplied monthly records ({grp.iloc[0] * 100:.1f}%). Compare coverage and period before drawing conclusions."
            else:
                rate = pd.to_numeric(att.attendance_rate, errors="coerce").dropna().mean()
                intent, answer = "attendance_rate", f"The mean attendance rate across {len(att):,} supplied employee-month records is {rate * 100:.1f}%. This is a dataset-wide average, not a live today metric."

    # Payroll figures are aggregate-only and based on the latest supplied payroll month.
    if answer is None and any(k in text for k in ("payroll", "salary", "salaries", "compensation", "pay anomaly")):
        pay = read_csv("payroll_monthly_features.csv")
        anomalies = read_csv("sample_anomaly_results.csv")
        if any(k in text for k in ("anomal", "irregular")):
            if anomalies is not None and {"source_dataset", "flagged"}.issubset(anomalies.columns):
                subset = anomalies[(anomalies.source_dataset == "payroll_monthly_features") & (anomalies.flagged.astype(str).str.upper() == "Y")]
                intent, answer = "payroll_anomalies", f"The saved anomaly evaluation contains {len(subset)} flagged payroll records. Flags require authorized payroll review; they are not proof of error or fraud."
            else:
                intent, answer = "payroll_anomalies_unavailable", "Saved payroll anomaly results are not available, so I cannot report an anomaly count."
        elif pay is not None and len(pay):
            latest = str(pay.month.astype(str).max())
            period = pay[pay.month.astype(str) == latest]
            if any(k in text for k in ("department", "average by", "by team")):
                means = period.groupby("department_id").gross_pay.mean().sort_values(ascending=False)
                intent, answer = "payroll_department_average", "Average gross monthly pay by department for " + latest[:7] + ": " + "; ".join(f"{k}: INR {v:,.0f}" for k, v in means.items()) + "."
            else:
                total = float(pd.to_numeric(period.gross_pay, errors="coerce").fillna(0).sum())
                intent, answer = "payroll_total", f"Total gross payroll in the latest supplied payroll month ({latest[:7]}) is INR {total:,.0f} across {len(period):,} payslip records."

    # Recruitment counts are aggregate and do not reveal candidate identities.
    if answer is None and any(k in text for k in ("candidate", "recruitment", "resume", "resumes", "screened")):
        resumes, labels = read_csv("resumes.csv"), read_csv("resume_job_labels.csv")
        if resumes is not None and labels is not None:
            screened = int(resumes.resume_id.nunique())
            shortlisted = int(labels.loc[labels.shortlist.astype(str).str.upper() == "Y", "resume_id"].nunique())
            intent, answer = "recruitment_screening_summary", f"The supplied recruitment dataset contains {screened} unique resumes and {shortlisted} candidates with at least one saved shortlist label. These are historical dataset counts, not live applicant totals."

    # Forecasts are read from the already-loaded Phase 4 forecast artifact.
    if answer is None and any(k in text for k in ("forecast", "predicted workforce", "workforce change", "future headcount")):
        forecast = reg.obj.get("forecast", {}).get("forecast")
        if forecast is not None and len(forecast):
            rows = forecast[(forecast.metric == "headcount_end") & (forecast.department_id == "ALL")].sort_values("forecast_month")
            if len(rows):
                first, last = rows.iloc[0], rows.iloc[-1]
                intent, answer = "workforce_forecast", (f"The saved forecast estimates company headcount moving from {first.forecast_value:.0f} in "
                    f"{str(first.forecast_month)[:7]} to {last.forecast_value:.0f} in {str(last.forecast_month)[:7]}. "
                    f"The final 80% prediction interval is {last.lower_80:.0f}–{last.upper_80:.0f}. These are forecasts, not actual results.")

    if answer is None:
        return None
    state = bot._session(session_id)
    result = jsonable(bot._out(intent, answer, state, 0.99, False))
    result["model_version"] = reg.info["chatbot"].version
    result["notice"] = HUMAN_REVIEW
    return result


def chat(reg: Registry, message: str, session_id: str, demo_user: str = "EMP-0002") -> dict:
    bot = _need(reg, "chatbot")
    if not message.strip():
        raise ApiError(422, "validation", "message must not be blank")

    analytics_reply = _chat_analytics_answer(reg, bot, message.casefold(), session_id)
    if analytics_reply is not None:
        return analytics_reply

    # Phase 4 aggregate headcount handling. No individual employee data is exposed.
    text = message.casefold()
    people_terms = ("employee", "employees", "staff", "workforce")
    count_terms = ("how many", "number of", "count", "total", "headcount")
    active_terms = ("active", "currently working", "currently employed", "working now")

    if any(term in text for term in people_terms) and any(term in text for term in count_terms):
        employees = getattr(bot, "employees", None)
        if employees is None:
            answer = "The employee directory is not available, so I cannot calculate headcount."
        elif any(term in text for term in active_terms):
            active_col = next((c for c in employees.columns if str(c).casefold() in ("is_active", "active")), None)
            status_col = next((c for c in employees.columns if str(c).casefold() in ("status", "employment_status", "employee_status")), None)
            if active_col:
                active_mask = employees[active_col].astype(str).str.strip().str.casefold().isin(("true", "1", "yes", "y", "active"))
                answer = f"There are {int(active_mask.sum())} active employees in the employee directory."
            elif status_col:
                active_mask = employees[status_col].astype(str).str.strip().str.casefold().isin(("active", "employed", "current"))
                answer = f"There are {int(active_mask.sum())} active employees in the employee directory."
            else:
                answer = f"The employee directory contains {len(employees):,} employee records. However, it has no employment-status field, so I cannot verify how many are currently active."
        else:
            answer = f"The employee directory currently contains {len(employees):,} employee records."

        state = bot._session(session_id)
        reply = bot._out("employee_headcount", answer, state, 1.0, False)
        reply = jsonable(reply)
        reply["model_version"] = reg.info["chatbot"].version
        reply["notice"] = HUMAN_REVIEW
        return reply

    reply = bot.reply(message.strip(), session_id, "EMPLOYEE", demo_user)       # role always comes from the identity directory
    reply = jsonable(reply)
    reply["model_version"] = reg.info["chatbot"].version
    reply["notice"] = HUMAN_REVIEW
    return reply


def resume_rank(reg: Registry, payload: dict) -> dict:
    from cortexhr.phase3.resume import api as rapi
    ranker = _need(reg, "resume")
    try:
        with _RESUME_LOCK:
            out = jsonable(rapi.handle_rank_request(payload, ranker))
        out["notice"] = "Orders resumes to help a recruiter read them; never use it to reject anyone automatically. " + HUMAN_REVIEW
        return out
    except rapi.ApiError as exc:
        raise ApiError(exc.status, exc.error, exc.details)


def resume_upload_rank(reg: Registry, filename: str, content: bytes, job_id: str, top_k: int = 5) -> dict:
    """Extract a PDF/DOCX in memory, add a masked temporary candidate, and rank it.

    Uploaded bytes and extracted text are not persisted. The in-memory ranking
    corpus is rebuilt so the uploaded candidate uses the same TF-IDF/skill engine;
    this upload-time score is therefore not directly comparable to saved benchmark
    scores. Restarting the API clears uploaded candidates.
    """
    from pathlib import Path
    from cortexhr.phase3.resume.scoring import ResumeRanker
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ApiError(415, "unsupported_file", "Upload a PDF or DOCX resume only.")
    if not content or len(content) > 8 * 1024 * 1024:
        raise ApiError(413, "file_too_large", "Resume must be non-empty and no larger than 8 MB.")
    try:
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError("File signature is not a PDF.")
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content), strict=True)
            if len(reader.pages) > 20:
                raise ValueError("PDF must contain no more than 20 pages.")
            extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            if not content.startswith(b"PK"):
                raise ValueError("File signature is not a DOCX document.")
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                if len(infos) > 500 or sum(x.file_size for x in infos) > 25 * 1024 * 1024:
                    raise ValueError("DOCX expands beyond the safe document limits.")
                if any(x.file_size > 0 and x.compress_size == 0 for x in infos):
                    raise ValueError("DOCX contains an invalid compressed entry.")
                if any(x.compress_size and x.file_size / x.compress_size > 150 for x in infos):
                    raise ValueError("DOCX compression ratio exceeds the safe limit.")
            from docx import Document
            doc = Document(io.BytesIO(content))
            extracted = "\n".join(p.text for p in doc.paragraphs)
            for table in doc.tables:
                extracted += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(422, "resume_parse_error", f"Could not safely read this resume ({type(exc).__name__}). Export it as a text-based PDF or DOCX and try again.")
    extracted = extracted.strip()
    if len(extracted) < 80:
        raise ApiError(422, "resume_text_missing", "No usable text was extracted. Scanned/image-only PDFs need OCR before upload.")
    if len(extracted) > 100_000:
        extracted = extracted[:100_000]
    # Redact common direct identifiers before the candidate enters the ranking corpus.
    from cortexhr.phase3.resume import text as resume_text
    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", extracted)
    email = email_match.group(0) if email_match else None
    lines = [line.strip() for line in extracted.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    role_terms = {"engineer", "developer", "manager", "analyst", "consultant", "architect", "designer", "intern", "scientist", "lead"}
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", first)
    candidate_name = first if 1 < len(tokens) <= 4 and len(first) <= 64 and not (set(t.lower() for t in tokens) & role_terms) and "@" not in first else None
    known_cities = ("Bengaluru", "Chennai", "Hyderabad", "Pune", "Delhi NCR", "Mumbai", "Coimbatore", "Erode", "Kochi")
    city = next((c for c in known_cities if re.search(rf"\b{re.escape(c)}\b", extracted, re.I)), None)
    extracted = re.sub(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)", " ", extracted)
    # Remove explicitly labelled demographic / personal-detail lines when present.
    extracted = re.sub(r"(?im)^\s*(?:gender|sex|age|date of birth|dob|pronouns?|marital status)\s*[:\-].*$", " ", extracted)
    extracted = re.sub(r"(?im)^\s*(?:he/him|she/her|they/them)(?:\s*[/|,].*)?$", " ", extracted)
    extracted = resume_text.mask_identifiers(extracted, candidate_name, email, city, tier=True, strip_emails=True)
    with _RESUME_LOCK:
        ranker = _need(reg, "resume")
        if job_id not in ranker.jobs.index:
            raise ApiError(404, "unknown_job_id", f"Unknown job_id {job_id!r}.")
        # Refit a private in-memory copy; never mutate the registry model mid-request.
        source_rows = ranker.resumes.reset_index(drop=True)
        required_columns = [c for c in ("resume_id", "resume_text", "candidate_name", "email", "city") if c in source_rows.columns]
        all_rows = source_rows[required_columns].copy()
        upload_id = "UPLOAD-" + uuid.uuid4().hex[:12].upper()
        row = {c: "" for c in required_columns}
        row.update({"resume_id": upload_id, "resume_text": extracted})
        all_rows.loc[len(all_rows)] = row
        fresh = ResumeRanker(weights=dict(ranker.weights), mask=True, ngram_range=ranker.ngram_range, mask_parts=ranker.mask_parts)
        fresh.fit(all_rows, ranker.jobs.reset_index(drop=True))
        reg.obj["resume"] = fresh
        from cortexhr.phase3.resume import api as rapi
        try:
            out = rapi.handle_rank_request({"job_id": job_id, "resume_ids": [upload_id], "top_k": 1}, fresh)
        except rapi.ApiError as exc:
            raise ApiError(exc.status, exc.error, exc.details)
    out = jsonable(out)
    out["uploaded_resume_id"] = upload_id
    out["file_name"] = Path(filename).name
    out["retention"] = "In-memory only; not written to disk. Cleared when the API process restarts."
    out["score_comparability"] = "Upload-time corpus refit; do not compare this score directly with the saved offline benchmark metrics."
    out["notice"] = "Resume text is processed in memory and protected identifiers are masked before ranking. Human review is mandatory; never auto-reject. " + HUMAN_REVIEW
    return out


def payband(reg: Registry, payload: dict) -> dict:
    from cortexhr.phase3.salary import model as S
    ctx = _need(reg, "payband")
    cats, rng, bundle = ctx["categories"], ctx["ranges"], ctx["bundle"]
    problems = [f"{c} must be one of the known values ({len(v)} allowed), got {payload.get(c)!r}" for c, v in cats.items() if payload.get(c) not in v]
    for c, (lo, hi) in rng.items():
        v = payload.get(c)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or v < 0:
            problems.append(f"{c} must be a non-negative number")
        elif not lo * 0.5 <= v <= hi * 2:
            problems.append(f"{c}={v} is far outside the training range {lo:g} to {hi:g}")
    if problems:
        raise ApiError(422, "validation", problems)
    df = S.add_features(pd.DataFrame([payload]))
    band = S.recommend_band(bundle["model"], df, bundle["band_multipliers"], bundle["features"]).iloc[0]
    return {"recommended_low": float(band.recommended_low), "recommended_mid": float(band.recommended_mid), "recommended_high": float(band.recommended_high),
            "currency": "INR per year (fixed)", "inputs_used": bundle["features"], "never_used": ["gender", "age", "career_gap_months", "hire date", "source"],
            "model_version": bundle["version"], "notice": "A range for a manager and HR to discuss, never a final offer. " + HUMAN_REVIEW}


# ------------------------------------------------------------------------------------------------ forecast
def _series_points(s: pd.Series) -> list[dict]:
    return [{"month": m.strftime("%Y-%m-%d"), "value": float(v)} for m, v in s.items()]


def forecast(reg: Registry, metric: str = "headcount_end", department_id: str = "ALL") -> dict:
    ctx = _need(reg, "forecast")
    f: pd.DataFrame = ctx["forecast"]
    sel = f[(f.metric == metric) & (f.department_id == department_id)]
    if sel.empty:
        raise ApiError(404, "unknown_series", f"no forecast for metric={metric!r}, department_id={department_id!r}; available: {sorted(set(zip(f.metric, f.department_id)))[:6]}...")
    hist = fdata.load_history(reg.settings.data_dir)
    if metric == "attrition_rate_monthly_pct":
        s = fdata.attrition_rate_pct(fdata.get_series(hist, "ALL", "exits"), fdata.get_series(hist, "ALL", "headcount_end"))
    else:
        s = fdata.get_series(hist, department_id, metric)
    cmp = ctx["comparison"]
    acc = cmp[(cmp.metric == ("exits" if metric == "attrition_rate_monthly_pct" else metric)) & (cmp.department_id == department_id) & (cmp.scope != "failed")]
    ev = fevents.event_table(fdata.load_events(reg.settings.data_dir))
    ev = ev[ev.department_id.isin(["ALL", department_id]) | (department_id == "ALL")]
    return {"metric": metric, "department_id": department_id, "trained_through": sel.trained_through.iloc[0], "model_version": sel.model_version.iloc[0], "method": sel.method.iloc[0],
            "history": _series_points(s.tail(60)), "forecast": jsonable(sel[["forecast_month", "forecast_value", "lower_80", "upper_80"]]),
            "events": jsonable(ev[["event_id", "event_name", "effect", "department_id", "start_date", "end_date"]].astype(str)),
            "accuracy": jsonable(acc.drop(columns=["note"], errors="ignore")), "explanation": jsonable(ctx["explanation"]) if department_id == "ALL" else None,
            "improved_model_status": ctx["meta"].get("improved_model_status"),
            "notice": "Forecast with an 80% interval, tested on past data. " + HUMAN_REVIEW}


def forecast_scenario(reg: Registry, scenarios: list[dict], departments: list[str] | None = None) -> dict:
    try:
        sc = [fevents.Scenario(**s) for s in scenarios]
        out = fpipe.forecast_only(reg.settings.data_dir, sc, departments)
    except fmodels.ModelUnavailable as exc:
        raise ApiError(503, "model_unavailable", str(exc))
    except (ValueError, fpipe.ForecastError, TypeError) as exc:
        raise ApiError(422, "validation", str(exc))
    return {"scenarios": scenarios, "forecast": jsonable(out), "notice": "What-if forecast, not a plan. " + HUMAN_REVIEW}


# ------------------------------------------------------------------------------------------------ monitoring
_MON_LOCK = threading.Lock()
_MON_LAST_RUN = {"at": 0.0}
MIN_REFRESH_SECONDS = 60          # refresh requests inside this window reuse the last run (the check costs seconds of CPU)


def monitoring(reg: Registry, refresh: bool = False) -> dict:
    """Latest monitoring report. ``refresh`` re-runs the checks, but at most once per minute and never in parallel: found by the
    100-user load test (100 simultaneous refreshes took ~50 s each and could expose a half-written file)."""
    s = reg.settings
    latest = s.monitoring_dir / "monitoring_latest.json"
    refreshed = False
    if refresh or not latest.exists():
        with _MON_LOCK:                                              # single flight: waiters reuse the result of the first caller
            fresh_enough = latest.exists() and time.time() - _MON_LAST_RUN["at"] < MIN_REFRESH_SECONDS
            if not latest.exists() or (refresh and not fresh_enough):
                report = mon.run_monitor(s)
                try:
                    mon.persist(report, s)
                except OSError:                                      # read-only deployment disk: still return the fresh report
                    pass
                _MON_LAST_RUN["at"] = time.time()
                refreshed = True
            else:
                report = json.loads(latest.read_text(encoding="utf-8"))
    else:
        report = json.loads(latest.read_text(encoding="utf-8"))
    hist = mon.read_history(s)[-30:]
    return {"status": report["status"], "created_utc": report["created_utc"], "coverage": report["coverage"], "not_checked": report["not_checked"], "explanation": report["explanation"],
            "refreshed": refreshed, "report": jsonable(report), "history": hist, "notice": "Monitoring shows that checks passed or failed, not that the system is fair. " + HUMAN_REVIEW}


# ------------------------------------------------------------------------------------------------ overview (dashboard data)
def overview(reg: Registry) -> dict:
    """Every dashboard number comes from the project's own data or model outputs; a value that cannot be computed is ``None``."""
    d = reg.settings.data_dir
    hist = fdata.load_history(d)
    hc, ex, hires = (fdata.get_series(hist, "ALL", c) for c in ("headcount_end", "exits", "hires"))
    rate = fdata.attrition_rate_pct(ex, hc)
    saved = reg.obj["attrition_saved"] if "attrition_saved" in reg.obj else None
    jobs = pd.read_csv(d / "job_descriptions.csv")
    scoring = pd.read_csv(d / "attrition_scoring_2026.csv")
    sal = pd.read_csv(d / "salary_training.csv")
    meta = json.loads((PHASE3 / "artifacts" / "metadata" / "salary_band_v1_metadata.json").read_text(encoding="utf-8"))
    tm = meta["test_metrics"]
    final_name, base_name = meta["model"], "baseline: average offered salary by role"
    f = reg.obj["forecast"]["explanation"] if "forecast" in reg.obj else {}
    mon_path = reg.settings.monitoring_dir / "monitoring_latest.json"
    mrep = json.loads(mon_path.read_text(encoding="utf-8")) if mon_path.exists() else None
    bands = saved.risk_band.value_counts() if saved is not None else pd.Series(dtype=int)
    last6 = hc.tail(7)
    names = hist.drop_duplicates("department_id").set_index("department_id").department_name
    last_m = hist.month.max()
    depts = [{"id": r.department_id, "name": r.department_name, "headcount": int(r.headcount_end), "as_of": last_m.strftime("%Y-%m")}
             for r in hist[(hist.month == last_m) & (hist.department_id != "ALL")].itertuples()]
    series = [{"month": m.strftime("%Y-%m"), "label": m.strftime("%b"), "employees": float(hc[m]), "exits": float(ex[m]) if m in ex.index else 0.0} for m in hc.tail(13).index]
    hire_periods = {}
    for months in (3, 6, 12):
        current_hires = int(hires.tail(months).sum())
        previous_hires = int(hires.iloc[-2 * months:-months].sum())
        hire_periods[str(months)] = {
            "hires": current_hires,
            "previous": previous_hires,
            "change_pct": float((current_hires / previous_hires - 1) * 100)
            if previous_hires else None,
        }

    hires6 = hire_periods["6"]["hires"]
    hires_prev = hire_periods["6"]["previous"]
    where = scoring.work_location.value_counts()
    # Approximate city-centre coordinates for the categorical work_location values in the supplied synthetic workforce data.
    city_centres = {"Bengaluru": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707), "Hyderabad": (17.3850, 78.4867),
                    "Pune": (18.5204, 73.8567), "Delhi NCR": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
                    "Coimbatore": (11.0168, 76.9558), "Erode": (11.3410, 77.7172), "Kochi": (9.9312, 76.2673)}
    location_counts = [{"name": str(k), "employees": int(v), "lat": city_centres[str(k)][0], "lon": city_centres[str(k)][1], "coordinate_type": "approximate city centre"}
                       for k, v in where.items() if str(k) in city_centres]
    hiring = {"hires_last_6m": hires6, "periods": hire_periods, "hires_previous_6m": hires_prev, "change_pct": float((hires6 / hires_prev - 1) * 100) if hires_prev else None,
              "as_of": hc.index[-1].strftime("%Y-%m"), "locations": int(where.size), "active_employees": int(where.sum()),
              "top_locations": location_counts[:3], "location_counts": location_counts,
              "source": "hires: monthly_headcount_history (company total); map: work_location of active employees (attrition_scoring_2026); city-centre coordinates are approximate"}
    out = {
        "hiring": hiring,
        "kpis": {
            "total_employees": {"value": float(hc.iloc[-1]), "as_of": hc.index[-1].strftime("%Y-%m"), "change_pct_6m": float((hc.iloc[-1] / hc.iloc[-7] - 1) * 100), "spark": hc.tail(12).tolist()},
            "attrition_risk": {"value": float(saved.risk_score_12m.mean() * 100) if saved is not None else None, "unit": "mean predicted 12-month risk, %", "change_pct": None,
                               "spark": rate.tail(12).round(2).tolist(), "spark_label": "monthly attrition rate, last 12 months"},
            "open_positions": {"value": int(jobs.openings.sum()), "jobs": int(len(jobs)), "change_pct": None, "spark": hires.tail(12).tolist(), "spark_label": "hires per month, last 12 months"},
            "avg_salary": {"value": float(scoring.annual_fixed_pay.mean()), "unit": "INR per year, active employees", "change_pct": None,
                           "spark": sal.groupby("hire_year").offered_annual_fixed.mean().round(0).tolist(), "spark_label": "average offer by hire year"},
        },
        "departments": depts,
        "workforce": {"series": series, "months": [m.strftime("%b") for m in last6.index[1:]], "employees": last6.iloc[1:].tolist(), "exits": ex.reindex(last6.index[1:]).tolist(),
                      "labels": [m.strftime("%Y-%m") for m in last6.index[1:]]},
        "attrition": {"high": int(bands.get("High", 0)), "medium": int(bands.get("Medium", 0)), "low": int(bands.get("Low", 0)), "population": int(len(saved)) if saved is not None else 0,
                      "mean_risk_pct": float(saved.risk_score_12m.mean() * 100) if saved is not None else None},
        "insights": {"headcount_growth_pct_6m": f.get("growth_pct_6m"), "high_risk_count": int(bands.get("High", 0)),
                     "monitoring_status": mrep["status"] if mrep else None, "monitoring_explanation": mrep["explanation"] if mrep else None,
                     "monitoring_summary": monitoring_summary(mrep) if mrep else None,
                     "monitoring_headline": monitoring_headline(mrep) if mrep else None},
        "model_performance": {"model": final_name, "mae": tm[final_name]["MAE_rupees"], "r2": tm[final_name]["R2"], "coverage": meta["band_coverage_test"], "baseline_mae": tm[base_name]["MAE_rupees"]},
        "activity": [a for a in (_activity(reg.settings.forecast_dir / "forecast_run.json", "Forecast report generated"),
                                 _activity(mon_path, "Monitoring check completed"), {"text": f"API v{API_VERSION} started", "uptime_s": reg.uptime_s()}) if a],
        "notifications": notifications(reg, mrep),
        "version": API_VERSION, "notice": HUMAN_REVIEW,
    }
    return jsonable(out)


def monitoring_summary(rep: dict) -> str:
    """One sentence for the dashboard: what the latest monitoring run found (never 'fair', only 'no gap above the thresholds')."""
    c, parts = rep["components"], []
    drift = c.get("drift", {}).get("table", [])
    bad = sorted((r for r in drift if r["status"] != "OK"), key=lambda r: -r["psi"])
    if bad:
        parts.append(f"Input drift {c['drift']['status']}: {bad[0]['feature']} PSI {bad[0]['psi']:.2f}" + (f" and {len(bad) - 1} more" if len(bad) > 1 else "") + ".")
    else:
        parts.append("No input drift above the WATCH line.")
    fair = [c.get("twin_resume_gap", {}).get("status"), *[g.get("status") for g in c.get("pay_gap", {}).get("groups", {}).values() if g.get("status") != "SKIPPED"]]
    parts.append("Fairness re-checks found no gap above the thresholds (not proof of fairness)." if all(f == "OK" for f in fair) else "A fairness re-check needs review.")
    return f"Monitoring {rep['status']}. " + " ".join(parts)


def monitoring_headline(rep: dict) -> str:
    """Short form (two lines of the Key Insights card)."""
    drift = sorted((r for r in rep["components"].get("drift", {}).get("table", []) if r["status"] != "OK"), key=lambda r: -r["psi"])
    if drift:
        return f"Monitoring {rep['status']}: {drift[0]['feature']} drifted (PSI {drift[0]['psi']:.2f})."
    return f"Monitoring {rep['status']}: no input drift above the WATCH line."


def notifications(reg: Registry, rep: dict | None) -> list[dict]:
    out = []
    if rep:
        for r in rep["components"].get("drift", {}).get("table", []):
            if r["status"] != "OK":
                out.append({"level": r["status"], "text": f"Input drift: {r['feature']} PSI {r['psi']:.2f}"})
        for name, comp in rep["components"].items():
            if name != "drift" and comp.get("status") not in ("OK", None) and comp.get("status") != "SKIPPED":
                out.append({"level": comp["status"], "text": f"Fairness re-check '{name}' needs review"})
        for k in rep.get("not_checked", []):
            out.append({"level": "INFO", "text": f"Not checked: {k} (restricted file not available)"})
    else:
        out.append({"level": "INFO", "text": "Monitoring has not run yet"})
    for name, info in reg.info.items():
        if info.status != LOADED:
            out.append({"level": "WATCH" if info.status == "unavailable" else "INFO", "text": f"Model '{name}': {info.status}. {info.detail}"[:160]})
    return out


def _activity(path: Path, text: str) -> dict | None:
    if not path.exists():
        return None
    created = json.loads(path.read_text(encoding="utf-8")).get("created_utc")
    return {"text": text, "created_utc": created} if created else None


# ------------------------------------------------------------------------------------------------ option lists and evaluation tables
def payband_options(reg: Registry) -> dict:
    from . import samples
    ctx = _need(reg, "payband")
    return {"categories": ctx["categories"], "ranges": ctx["ranges"], "example": samples.payband_row(reg.settings.data_dir)}


def resume_jobs(reg: Registry) -> dict:
    jobs = pd.read_csv(reg.settings.data_dir / "job_descriptions.csv")
    cols = ["job_id", "job_title", "department_id", "job_level", "min_experience_years", "required_skills", "nice_to_have_skills", "openings"]
    return {"jobs": jsonable(jobs[cols])}


def performance(reg: Registry) -> dict:
    """Evaluation tables behind the Model Performance page (all read from the saved Phase 3 / Phase 4 evaluation files)."""
    r3, fdir = PHASE3 / "reports", reg.settings.forecast_dir
    rd = lambda p: pd.read_csv(p)
    out = {"attrition": {"cv": jsonable(rd(r3 / "task13" / "cv_summary_with_lightgbm.csv")[["model", "roc_auc_mean", "pr_auc_mean", "brier_mean", "roc_auc_sd"]]),
                         "paired_differences": jsonable(rd(r3 / "task13" / "paired_differences.csv")),
                         "verdict": "Gain over the simple baseline is NOT proven; the baseline is not beaten (Phase 3 Task 13)."},
           "resume": {"ranking": jsonable(rd(r3 / "task11" / "ranking_quality_summary.csv")), "per_job": jsonable(rd(r3 / "task11" / "ranking_quality_per_job.csv"))},
           "fairness_phase3": {"twin_before_after": jsonable(rd(r3 / "task12" / "before_after_gap.csv")), "pay_gap_career_gap": jsonable(rd(r3 / "task15" / "gap_bin_gaps.csv"))},
           "payband": {"test_metrics": jsonable(rd(r3 / "task15" / "test_metrics.csv"))}}
    cmp_path = fdir / "model_comparison.csv"
    out["forecast"] = {"comparison": jsonable(rd(cmp_path)[lambda d: (d.department_id == "ALL") & (d.scope != "failed")]), "meta": jsonable(reg.obj["forecast"]["meta"]) if "forecast" in reg.obj else None} if cmp_path.exists() else None
    out["notice"] = "Numbers are measured on held-out or out-of-fold data. " + HUMAN_REVIEW
    return out
