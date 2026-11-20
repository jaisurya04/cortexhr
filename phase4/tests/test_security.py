from __future__ import annotations
from dataclasses import replace
import json

from cortexhr_phase4.core.security import hash_password, verify_password, issue_token, verify_token, role_allows


def test_password_hash_is_salted_and_verifies():
    a = hash_password("correct-horse-battery-staple")
    b = hash_password("correct-horse-battery-staple")
    assert a != b
    assert verify_password("correct-horse-battery-staple", a)
    assert not verify_password("wrong-password", a)


def test_token_signature_expiry_and_role():
    signed_value = issue_token("recruiter1", "recruiter", "test-secret-for-unit-tests-at-least-32-bytes", ttl_s=60, now=100)
    assert verify_token(signed_value, "test-secret-for-unit-tests-at-least-32-bytes", now=120)["sub"] == "recruiter1"
    assert verify_token(signed_value, "wrong-secret", now=120) is None
    assert verify_token(signed_value, "test-secret-for-unit-tests-at-least-32-bytes", now=160) is None


def test_role_matrix_is_deny_by_default():
    assert role_allows("recruiter", "/v1/resume/jobs")
    assert not role_allows("recruiter", "/v1/monitoring")
    assert role_allows("analyst", "/v1/monitoring")
    assert role_allows("analyst", "/v1/chat")
    assert not role_allows("analyst", "/v1/resume/rank")
    assert not role_allows("unknown", "/v1/overview")


def test_login_and_backend_role_enforcement():
    from fastapi.testclient import TestClient
    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    from _common import registry
    record = {"recruiter1": {"display_name": "Recruiter One", "role": "recruiter", "password_hash": hash_password("correct-horse-battery-staple")}}
    cfg = replace(load_settings(), env="development", api_key=None, **{"auth_secret": "unit-test-signing-secret-at-least-32-bytes-long", "auth_users": json.dumps(record)})
    c = TestClient(create_app(cfg, registry()))
    assert c.get("/v1/resume/jobs").status_code == 401
    login = c.post("/v1/auth/login", json={"username": "recruiter1", "password": "correct-horse-battery-staple"})
    assert login.status_code == 200
    signed_value = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {signed_value}"}
    assert c.get("/v1/resume/jobs", headers=headers).status_code == 200
    assert c.get("/v1/monitoring", headers=headers).status_code == 403
    assert c.post("/v1/auth/logout", headers=headers, json={}).status_code == 200
    assert c.get("/v1/resume/jobs", headers=headers).status_code == 401
    assert c.post("/v1/auth/login", json={"username": "recruiter1", "password": "bad-password"}).status_code == 401


def test_resume_docx_upload_is_masked_and_memory_only():
    from io import BytesIO
    from docx import Document
    from fastapi.testclient import TestClient
    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    from _common import registry
    users = {
        "recruiter1": {"display_name": "Recruiter One", "role": "recruiter", "password_hash": hash_password("correct-horse-battery-staple")}
    }
    cfg = replace(load_settings(), env="development", api_key=None, **{"auth_secret": "upload-test-signing-secret-at-least-32-bytes-long", "auth_users": json.dumps(users)})
    c = TestClient(create_app(cfg, registry()))
    login = c.post("/v1/auth/login", json={"username": "recruiter1", "password": "correct-horse-battery-staple"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    doc = Document()
    doc.add_paragraph("Aarav Kumar")
    doc.add_paragraph("aarav.kumar@example.com Â· +91 98765 43210 Â· Coimbatore")
    doc.add_paragraph("Gender: Female | Age: 31 | Date of Birth: 1995-03-11")
    doc.add_paragraph("SUMMARY\nExperienced software engineer with Python, Java, SQL, AWS and Docker.")
    doc.add_paragraph("EXPERIENCE\nSoftware Engineer, Example Systems (4.5 yrs) developed Python APIs, SQL data pipelines, machine learning services, cloud automation, CI/CD and distributed systems.")
    doc.add_paragraph("SKILLS\nPython; Java; SQL; AWS; Docker; Kubernetes; machine learning; REST API")
    stream = BytesIO(); doc.save(stream)
    response = c.post("/v1/resume/upload", headers=headers, data={"job_id": "J01", "top_k": "5"},
                      files={"file": ("candidate.docx", stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["uploaded_resume_id"].startswith("UPLOAD-")
    assert payload["retention"].startswith("In-memory only")
    assert "not compare" in payload["score_comparability"].lower()
    assert len(payload["candidates"]) == 1
    assert "candidate_name" not in payload["candidates"][0]
    stored_text = registry().obj["resume"].resumes.loc[payload["uploaded_resume_id"], "resume_text"]
    assert "Aarav" not in stored_text and "aarav.kumar@example.com" not in stored_text and "98765" not in stored_text
    assert "Coimbatore" not in stored_text
    assert "Gender:" not in stored_text and "Age:" not in stored_text and "1995-03-11" not in stored_text


def test_production_fails_closed_without_auth_configuration():
    from fastapi.testclient import TestClient
    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    from _common import registry
    cfg = replace(load_settings(), env="production", api_key=None, **{"auth_secret": None, "auth_users": ""})
    c = TestClient(create_app(cfg, registry()))
    assert c.get("/v1/health").status_code == 200
    denied = c.get("/v1/models")
    assert denied.status_code == 503 and denied.json()["error"] == "auth_not_configured"


def test_short_signing_secret_fails_closed():
    from fastapi.testclient import TestClient
    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    from _common import registry
    cfg = replace(load_settings(), env="production", api_key=None, **{"auth_secret": "too-short", "auth_users": "{}"})
    c = TestClient(create_app(cfg, registry()))
    assert c.get("/v1/models").status_code == 503
    login = c.post("/v1/auth/login", json={"username": "admin", "password": "irrelevant"})
    assert login.status_code == 503 and login.json()["error"] == "auth_misconfigured"


def test_auth_me_returns_verified_identity_and_rejects_revoked_session():
    from fastapi.testclient import TestClient
    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    from _common import registry
    record = {"admin1": {"display_name": "Admin One", "role": "admin", "password_hash": hash_password("correct-horse-battery-staple")}}
    cfg = replace(load_settings(), env="development", api_key=None, **{"auth_secret": "auth-me-test-secret-at-least-32-bytes-long", "auth_users": json.dumps(record)})
    c = TestClient(create_app(cfg, registry()))
    assert c.get("/v1/auth/me").status_code == 401
    login = c.post("/v1/auth/login", json={"username": "admin1", "password": "correct-horse-battery-staple"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    identity = c.get("/v1/auth/me", headers=headers)
    assert identity.status_code == 200
    assert identity.json()["user"] == {"username": "admin1", "display_name": "Admin One", "role": "admin"}
    assert c.post("/v1/auth/logout", headers=headers, json={}).status_code == 200
    assert c.get("/v1/auth/me", headers=headers).status_code == 401
