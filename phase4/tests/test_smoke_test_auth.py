from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import smoke_test  # noqa: E402


def test_smoke_headers_include_api_key_and_bearer_token():
    headers = smoke_test.request_headers("api-secret", "signed-token")
    assert headers["X-API-Key"] == "api-secret"
    assert headers["Authorization"] == "Bearer signed-token"
    assert headers["Content-Type"] == "application/json"


def test_smoke_headers_support_local_demo_without_credentials():
    assert smoke_test.request_headers() == {"Content-Type": "application/json"}
