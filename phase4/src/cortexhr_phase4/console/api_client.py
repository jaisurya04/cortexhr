"""Tiny HTTP client for the CortexHR /v1 API (standard library only). Every call returns an :class:`ApiResult`; nothing raises."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ApiResult:
    ok: bool
    status: int                       # HTTP status, 0 when the API could not be reached
    data: dict = field(default_factory=dict)
    error: str = ""
    details: list[str] = field(default_factory=list)

    def message(self, api_url: str = "") -> str:
        """A friendly sentence for the user interface."""
        if self.ok:
            return ""
        if self.status == 0:
            return f"Cannot reach the CortexHR API at {api_url or 'the configured address'}. Is it running? ({self.error})"
        if self.status == 401:
            return "Your session is invalid or expired, or the API key is incorrect. Please sign in again and check CORTEXHR_API_KEY."
        if self.status == 403:
            return "Your account role does not have permission to access this feature. Contact your CortexHR administrator if this is unexpected."
        if self.status == 503:
            return "A model this feature needs is not available right now. " + " ".join(self.details)
        if self.status in (404, 422):
            return "The request was not accepted: " + "; ".join(self.details or [self.error])
        return f"The API reported a problem ({self.status} {self.error}). " + " ".join(self.details)


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 20.0) -> None:
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self.bearer_token: str | None = None

    def request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> ApiResult:
        url = self.base_url + path + (("?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})) if params else "")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return ApiResult(True, r.status, json.loads(r.read().decode() or "{}"))
        except urllib.error.HTTPError as exc:
            try:
                js = json.loads(exc.read().decode())
            except Exception:
                js = {}
            return ApiResult(False, exc.code, error=str(js.get("error", exc.reason)), details=[str(d) for d in js.get("details", [])])
        except Exception as exc:                                   # connection refused, DNS, timeout, bad JSON ...
            return ApiResult(False, 0, error=f"{type(exc).__name__}: {exc}"[:200])


    def upload(self, path: str, filename: str, content: bytes, content_type: str, fields: dict[str, str]) -> ApiResult:
        """Send a bounded multipart upload without writing the file to disk."""
        import uuid
        boundary = "----CortexHR" + uuid.uuid4().hex
        chunks = []
        for key, value in fields.items():
            chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
        safe_name = filename.replace("\"", "") .replace("\r", "").replace("\n", "")
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\nContent-Type: {content_type}\r\n\r\n").encode())
        chunks.extend([content, f"\r\n--{boundary}--\r\n".encode()])
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
        if self.api_key: headers["X-API-Key"] = self.api_key
        if self.bearer_token: headers["Authorization"] = f"Bearer {self.bearer_token}"
        req = urllib.request.Request(self.base_url + path, method="POST", data=b"".join(chunks), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return ApiResult(True, r.status, json.loads(r.read().decode() or "{}"))
        except urllib.error.HTTPError as exc:
            try: js = json.loads(exc.read().decode())
            except Exception: js = {}
            return ApiResult(False, exc.code, error=str(js.get("error", exc.reason)), details=[str(d) for d in js.get("details", [])])
        except Exception as exc:
            return ApiResult(False, 0, error=f"{type(exc).__name__}: {exc}"[:200])

    def get(self, path: str, **params) -> ApiResult:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: dict) -> ApiResult:
        return self.request("POST", path, body=body)
