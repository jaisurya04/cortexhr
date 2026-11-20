"""One error type for every service; the API maps it to an HTTP status and a consistent JSON body."""
from __future__ import annotations


class ApiError(Exception):
    """``status`` is the HTTP status, ``error`` a short machine-readable code, ``details`` human-readable specifics."""

    def __init__(self, status: int, error: str, details=None):
        super().__init__(error)
        self.status, self.error, self.details = status, error, ([details] if isinstance(details, str) else list(details or []))

    def body(self) -> dict:
        return {"error": self.error, "details": self.details}
