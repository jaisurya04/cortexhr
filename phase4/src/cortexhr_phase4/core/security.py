"""Small, dependency-free authentication primitives for the CortexHR API.

Passwords are stored as PBKDF2-SHA256 hashes, never plaintext. Bearer tokens are
HMAC-signed, short-lived and contain only a username, role and expiry.
"""
from __future__ import annotations
import base64, hashlib, hmac, json, os, secrets, time, threading
from typing import Any

ROLES = {"admin", "hr_manager", "recruiter", "analyst"}
ITERATIONS = 310_000
_REVOKED: dict[str, int] = {}
_REVOCATION_LOCK = threading.Lock()


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256": return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)).hex()
        return hmac.compare_digest(actual, digest_hex)
    except (ValueError, TypeError):
        return False


def load_users(raw: str | None = None) -> dict[str, dict[str, str]]:
    raw = os.getenv("CORTEXHR_AUTH_USERS", "") if raw is None else raw
    if not raw.strip(): return {}
    try: parsed = json.loads(raw)
    except json.JSONDecodeError as exc: raise RuntimeError("CORTEXHR_AUTH_USERS must be valid JSON.") from exc
    if not isinstance(parsed, dict): raise RuntimeError("CORTEXHR_AUTH_USERS must be a JSON object keyed by username.")
    users: dict[str, dict[str, str]] = {}
    for username, record in parsed.items():
        if not isinstance(record, dict): continue
        role = str(record.get("role", "")).lower()
        password_hash = str(record.get("password_hash", ""))
        if role in ROLES and password_hash.startswith("pbkdf2_sha256$"):
            users[str(username).strip().lower()] = {"username": str(username), "role": role, "password_hash": password_hash,
                "display_name": str(record.get("display_name") or username)}
    return users


def secret_is_strong(secret: str | None) -> bool:
    return bool(secret and len(secret.encode("utf-8")) >= 32)


def issue_token(username: str, role: str, secret: str, ttl_s: int = 8 * 3600, now: int | None = None) -> str:
    if not secret_is_strong(secret): raise ValueError("CORTEXHR_AUTH_SECRET must contain at least 32 bytes.")
    if role not in ROLES: raise ValueError("Unknown role")
    payload = {"sub": username, "role": role, "iat": int(time.time() if now is None else now), "exp": int(time.time() if now is None else now) + ttl_s}
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    sig = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return body.decode() + "." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def revoke_token(token: str) -> None:
    """Revoke a token for the lifetime of this API process (entries expire with the token)."""
    fingerprint = hashlib.sha256(token.encode()).hexdigest()
    expiry = int(time.time()) + 8 * 3600
    try:
        body_s = token.split(".", 1)[0]
        payload = json.loads(base64.urlsafe_b64decode(body_s + "=" * (-len(body_s) % 4)))
        expiry = int(payload.get("exp", expiry))
    except Exception:
        pass
    with _REVOCATION_LOCK:
        now = int(time.time())
        for key in [key for key, until in _REVOKED.items() if until <= now]:
            _REVOKED.pop(key, None)
        _REVOKED[fingerprint] = expiry


def verify_token(token: str, secret: str, now: int | None = None) -> dict[str, Any] | None:
    if not secret_is_strong(secret): return None
    try:
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        current = int(time.time() if now is None else now)
        with _REVOCATION_LOCK:
            for key in [key for key, until in _REVOKED.items() if until <= current]:
                _REVOKED.pop(key, None)
            if fingerprint in _REVOKED: return None
        body_s, sig_s = token.split(".", 1)
        body = body_s.encode()
        sig = base64.urlsafe_b64decode(sig_s + "=" * (-len(sig_s) % 4))
        expected = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected): return None
        payload = json.loads(base64.urlsafe_b64decode(body_s + "=" * (-len(body_s) % 4)))
        current = int(time.time() if now is None else now)
        if payload.get("exp", 0) <= current or payload.get("role") not in ROLES: return None
        return payload
    except Exception:
        return None


# API-level authorization is deliberately deny-by-default for unknown routes.
ROLE_PREFIXES = {
    "admin": ("/v1/",),
    "hr_manager": ("/v1/",),
    "analyst": ("/v1/overview", "/v1/models", "/v1/performance", "/v1/attrition", "/v1/anomaly", "/v1/forecast", "/v1/monitoring", "/v1/chat"),
    "recruiter": ("/v1/overview", "/v1/resume", "/v1/chat", "/v1/payband"),
}

def role_allows(role: str, path: str) -> bool:
    if path == "/v1/auth/logout": return role in ROLES
    return any(path == prefix or path.startswith(prefix) for prefix in ROLE_PREFIXES.get(role, ()))
