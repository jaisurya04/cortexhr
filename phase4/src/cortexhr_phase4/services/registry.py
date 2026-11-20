"""Load every model ONCE at start-up and keep it in memory; report honestly what is loaded, what fell back and what is missing.

Phase 1-3 artifacts are read-only inputs. Nothing here writes to a Phase 1-3 folder (Phase 2's own loader is NOT used because it
retrains and saves when a file is missing).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from ..core import legacy
from ..core.logging import get_logger
from ..core.settings import API_VERSION, PHASE2, PHASE3, Settings, load_settings

LOADED, FALLBACK, UNAVAILABLE = "loaded", "fallback", "unavailable"


@dataclass
class ModelInfo:
    name: str
    status: str
    version: str | None = None
    source: str = ""
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "version": self.version, "source": self.source, "detail": self.detail}


@dataclass
class Registry:
    settings: Settings = field(default_factory=load_settings)
    started_at: float = field(default_factory=time.time)
    info: dict[str, ModelInfo] = field(default_factory=dict)
    obj: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.log = get_logger("cortexhr.registry", self.settings.log_level)

    # ------------------------------------------------------------------ loading
    def load_all(self) -> "Registry":
        legacy.bootstrap()
        for name, fn in (("attrition", self._attrition), ("payroll", self._payroll), ("attendance", self._attendance), ("chatbot", self._chatbot),
                         ("resume", self._resume), ("payband", self._payband), ("forecast", self._forecast)):
            t = time.time()
            try:
                fn()
            except Exception as exc:                       # one broken model must not stop the others from serving
                self.info[name] = ModelInfo(name, UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}")
                self.log.error("model=%s status=unavailable error=%s", name, exc)
            else:
                self.log.info("model=%s status=%s seconds=%.2f", name, self.info[name].status, time.time() - t)
        return self

    def _phase2_registry(self) -> dict:
        p = PHASE2 / "models" / "model_registry.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _load_phase2(self, key: str):
        """Load a Phase 2 joblib read-only and verify its SHA-256 against Phase 2's own model_registry.json (provenance check)."""
        entry = self._phase2_registry().get(key, {})
        path = PHASE2 / "models" / entry.get("file", f"{key}.joblib")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("sha256") and digest != entry["sha256"]:
            raise RuntimeError(f"{path.name} does not match the SHA-256 recorded in phase2/models/model_registry.json")
        return joblib.load(path), entry, f"sha256 verified against phase2/models/model_registry.json ({digest[:12]})"

    def _payroll(self) -> None:
        self.obj["payroll"], e, note = self._load_phase2("payroll")
        self.info["payroll"] = ModelInfo("payroll", LOADED, e.get("version", "payroll-iforest-rules-v2"), f"phase2/models/{e.get('file')}", note)

    def _attendance(self) -> None:
        self.obj["attendance"], e, note = self._load_phase2("attendance")
        self.info["attendance"] = ModelInfo("attendance", LOADED, e.get("version", "attendance-iforest-rules-v2"), f"phase2/models/{e.get('file')}", note)

    def _chatbot(self) -> None:
        from cortexhr.phase2 import service
        from cortexhr.phase2.chatbot import HRChatbot
        model, entry, note = self._load_phase2("chatbot")
        csv = PHASE2 / "03_Data" / "csv"
        self.obj["chatbot"] = HRChatbot(model, csv / "chatbot_responses.csv", csv / "hr_faq.csv",
                                        users_csv=PHASE2 / "05_Starter" / "mock_services" / "mock_data" / "demo_users.csv")
        self.info["chatbot"] = ModelInfo("chatbot", LOADED, entry.get("version", service.CHATBOT_VERSION), f"phase2/models/{entry.get('file')}", note)

    def _resume(self) -> None:
        self.obj["resume"] = joblib.load(PHASE3 / "artifacts" / "models" / "resume_ranker_v2.joblib")
        meta = json.loads((PHASE3 / "artifacts" / "metadata" / "resume_ranker_v2_metadata.json").read_text(encoding="utf-8"))
        self.info["resume"] = ModelInfo("resume", LOADED, meta.get("engine_version") or meta.get("model_version") or "resume-ranker-v2", "phase3/artifacts/models/resume_ranker_v2.joblib")

    def _payband(self) -> None:
        bundle = joblib.load(PHASE3 / "artifacts" / "models" / "salary_band_v1.joblib")
        train = pd.read_csv(self.settings.data_dir / "salary_training.csv")
        self.obj["payband"] = {"bundle": bundle, "categories": {c: sorted(train[c].dropna().unique().tolist()) for c in ("department_id", "role_id", "work_location", "highest_education")},
                               "ranges": {c: (float(train[c].min()), float(train[c].max())) for c in ("job_level", "band_mid_2026", "market_index", "prior_experience_years", "skills_count")}}
        self.info["payband"] = ModelInfo("payband", LOADED, bundle["version"], "phase3/artifacts/models/salary_band_v1.joblib")

    def _attrition(self) -> None:
        """Saved Phase 3 scores (real model output) are always loaded; the live model is attempted and cross-checked against them."""
        saved = pd.read_csv(PHASE3 / "reports" / "task14" / "attrition_risk_2026.csv")
        self.obj["attrition_saved"] = saved.set_index("employee_id")
        detail = ""
        try:
            from cortexhr.phase3.attrition import v2
            model = joblib.load(PHASE3 / "artifacts" / "models" / "attrition_v2.joblib")
            _, X, _ = v2.load_scoring()
            live = pd.Series(model.predict_proba(X)[:, 1], index=pd.read_csv(self.settings.data_dir / "attrition_scoring_2026.csv").employee_id.to_numpy())
            diff = float((live.round(3) - self.obj["attrition_saved"].risk_score_12m.reindex(live.index)).abs().max())
            if diff > 1e-3:
                raise RuntimeError(f"live scores differ from the saved Phase 3 scores by {diff:.4f}")
            self.obj["attrition_live"], self.obj["attrition_X"] = model, X.reset_index(drop=True)
            self.info["attrition"] = ModelInfo("attrition", LOADED, "attrition-v2", "phase3/artifacts/models/attrition_v2.joblib", "live model; matches saved Phase 3 scores")
            return
        except Exception as exc:
            detail = f"live model not used ({type(exc).__name__}: {str(exc)[:120]}); serving saved Phase 3 scores of the same model"
        self.info["attrition"] = ModelInfo("attrition", FALLBACK, "attrition-v2", "phase3/reports/task14/attrition_risk_2026.csv", detail)

    def _forecast(self) -> None:
        from ..forecasting import pipeline
        f = self.settings.forecast_dir / "forecast_next_6_months.csv"
        if f.exists():
            self.obj["forecast"] = {"forecast": pd.read_csv(f), "explanation": json.loads((self.settings.forecast_dir / "forecast_explanation.json").read_text(encoding="utf-8")),
                                    "meta": json.loads((self.settings.forecast_dir / "forecast_run.json").read_text(encoding="utf-8")),
                                    "comparison": pd.read_csv(self.settings.forecast_dir / "model_comparison.csv"), "source": "saved run"}
            meta = self.obj["forecast"]["meta"]
        else:
            run = pipeline.run_forecast(self.settings.data_dir, rolling_step=3)
            self.obj["forecast"] = {"forecast": run.forecast, "explanation": run.explanation, "meta": run.meta, "comparison": run.comparison, "source": "computed at start-up (not saved)"}
            meta = run.meta
        st = LOADED if meta.get("improved_model_status") == "run" else FALLBACK
        self.info["forecast"] = ModelInfo("forecast", st, meta.get("model_version"), self.obj["forecast"]["source"],
                                          "" if st == LOADED else f"improved model: {meta.get('improved_model_status')}; baseline forecasts served")

    # ------------------------------------------------------------------ views
    def models(self) -> dict:
        return {k: v.as_dict() for k, v in self.info.items()}

    def uptime_s(self) -> float:
        return round(time.time() - self.started_at, 1)

    def ready(self) -> bool:
        return all(v.status in (LOADED, FALLBACK) for v in self.info.values()) and len(self.info) == 7
