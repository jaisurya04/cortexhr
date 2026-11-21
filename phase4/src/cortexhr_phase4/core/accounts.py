"""PostgreSQL-backed organization accounts; public registrations require admin approval."""
from __future__ import annotations

import os


class AccountExistsError(Exception):
    pass


def enabled() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _connect():
    import psycopg
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(url)


def initialize() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cortexhr_accounts (
                username TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'active', 'rejected')),
                role TEXT NULL CHECK (role IS NULL OR role IN
                    ('hr_manager', 'recruiter', 'analyst')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                approved_at TIMESTAMPTZ NULL
            )
        """)


def register(username: str, display_name: str, password_hash: str) -> None:
    import psycopg
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO cortexhr_accounts
                   (username, display_name, password_hash, status)
                   VALUES (%s, %s, %s, 'pending')""",
                (username, display_name, password_hash),
            )
    except psycopg.errors.UniqueViolation as exc:
        raise AccountExistsError from exc


def get_user(username: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """SELECT username, display_name, password_hash, status, role
               FROM cortexhr_accounts WHERE username = %s""",
            (username,),
        ).fetchone()
    if not row:
        return None
    return dict(zip(("username", "display_name", "password_hash", "status", "role"), row))


def pending() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT username, display_name, created_at::text
               FROM cortexhr_accounts WHERE status = 'pending'
               ORDER BY created_at ASC"""
        ).fetchall()
    return [
        {"username": row[0], "display_name": row[1], "created_at": row[2]}
        for row in rows
    ]


def approve(username: str, role: str) -> bool:
    if role not in {"hr_manager", "recruiter", "analyst"}:
        raise ValueError("Invalid approval role")
    with _connect() as conn:
        cur = conn.execute(
            """UPDATE cortexhr_accounts
               SET status = 'active', role = %s, approved_at = NOW()
               WHERE username = %s AND status = 'pending'""",
            (role, username),
        )
        return cur.rowcount == 1
