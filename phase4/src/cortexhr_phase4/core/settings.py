"""Central configuration. Every deployable setting comes from an environment variable; no secret is stored in the repo."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PHASE4_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PHASE4_ROOT.parent
PHASE1, PHASE2, PHASE3 = REPO_ROOT / "phase1", REPO_ROOT / "phase2", REPO_ROOT / "phase3"
API_VERSION = "1.0.0"


def _bool(name: str, default: bool, environ=None) -> bool:
    raw = (os.environ if environ is None else environ).get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings (read once from the environment)."""

    env: str = "development"
    log_level: str = "INFO"
    api_url: str = "http://127.0.0.1:8000"          # where the console finds the live API
    api_key: str | None = None                       # optional shared key (header X-API-Key)
    auth_secret: str | None = None                   # HMAC signing secret; required for production auth
    auth_users: str = ""                              # JSON user directory; contains PBKDF2 password hashes only
    allowed_origins: tuple[str, ...] = ("*",)
    data_dir: Path = PHASE3 / "03_Data" / "csv"
    output_dir: Path = PHASE4_ROOT / "outputs"
    vault_dir: Path | None = None                    # restricted CX-12 / CX-15 files; never committed
    seed: int = 42
    request_timeout_s: float = 20.0
    show_welcome: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def forecast_dir(self) -> Path:
        return self.output_dir / "forecast"

    @property
    def monitoring_dir(self) -> Path:
        return self.output_dir / "monitoring"


def load_settings(environ: dict | None = None) -> Settings:
    """Build :class:`Settings` from ``environ`` (defaults to process env + repository .env)."""
    if environ is None:
        # Local .env is optional and never overrides explicitly exported variables.
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO_ROOT / ".env", override=False)
        except ImportError:
            pass
        e = os.environ
    else:
        e = environ
    default_vault = PHASE3 / "04_Validation_Vault"
    vault = e.get("CORTEXHR_VAULT_DIR") or (str(default_vault) if default_vault.exists() else None)
    origins = tuple(o.strip() for o in e.get("CORTEXHR_ALLOWED_ORIGINS", "*").split(",") if o.strip()) or ("*",)
    return Settings(
        env=e.get("CORTEXHR_ENV", "development"),
        log_level=e.get("CORTEXHR_LOG_LEVEL", "INFO").upper(),
        api_url=e.get("CORTEXHR_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        api_key=e.get("CORTEXHR_API_KEY") or None,
        auth_secret=e.get("CORTEXHR_AUTH_SECRET") or None,
        auth_users=e.get("CORTEXHR_AUTH_USERS", ""),
        allowed_origins=origins,
        data_dir=Path(e.get("CORTEXHR_DATA_DIR", str(PHASE3 / "03_Data" / "csv"))),
        output_dir=Path(e.get("CORTEXHR_OUTPUT_DIR", str(PHASE4_ROOT / "outputs"))),
        vault_dir=Path(vault) if vault else None,
        seed=int(e.get("CORTEXHR_SEED", "42")),
        request_timeout_s=float(e.get("CORTEXHR_TIMEOUT_S", "20")),
        show_welcome=_bool("CORTEXHR_SHOW_WELCOME", True, e),
    )
