"""CortexHR API skeleton. Every endpoint is a promise: replace the 501 with your real model call.
Response shapes copy the sample_* sheets. Run:  uvicorn app.main:app --reload   then open /docs"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
app = FastAPI(title="CortexHR API", version="0.1.0")
class RiskOut(BaseModel):
    employee_id: str; risk_score_12m: float = Field(ge=0, le=1); risk_band: str; top_reason_1: str; top_reason_2: str; top_reason_3: str; model_version: str
class AttendanceIn(BaseModel):
    employee_id: str; scheduled_days: int; present_days: int; wfh_days: int = 0; late_arrivals: int = 0; avg_hours_worked: float; overtime_hours: float = 0
class AnomalyOut(BaseModel):
    anomaly_score: float; flagged: bool; reason_code: str; reason_text: str; money_at_risk: float | None = None; model_version: str
class ChatIn(BaseModel):
    message: str; session_id: str; user_role: str = "EMPLOYEE"
class ChatOut(BaseModel):
    intent: str; reply: str; slots: dict = {}
@app.get("/v1/health")
def health(): return {"status": "ok", "service": "cortexhr"}
@app.get("/v1/attrition/risk/{employee_id}", response_model=RiskOut)
def attrition(employee_id: str): raise HTTPException(501, "CX-14: load your saved model and explain the score")
@app.post("/v1/anomaly/attendance", response_model=AnomalyOut)
def anomaly_attendance(row: AttendanceIn): raise HTTPException(501, "CX-08: call your attendance detector")
@app.post("/v1/chat", response_model=ChatOut)
def chat(msg: ChatIn): raise HTTPException(501, "CX-10: classify intent, fill slots, call the mock services")
