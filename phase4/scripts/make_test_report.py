"""Run the whole Phase 4 suite and save docs/test_report.md with the real output.   python scripts/make_test_report.py"""
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    t0 = datetime.now(timezone.utc)
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "run_all.py")], capture_output=True, text=True, cwd=ROOT)
    body = "\n".join(l for l in (out.stdout + out.stderr).splitlines() if "level=INFO" not in l and "127.0.0.1 - -" not in l)
    libs = {m: ("installed" if importlib.util.find_spec(m) else "NOT installed") for m in ("fastapi", "pydantic", "statsmodels", "streamlit", "httpx", "pytest", "playwright", "flask", "shap")}
    try:
        import pandas, sklearn
        v = f"pandas {pandas.__version__}, scikit-learn {sklearn.__version__}"
    except Exception:
        v = "?"
    md = (f"# Phase 4 test report\n\nRun at {t0.strftime('%Y-%m-%d %H:%M UTC')} with `python tests/run_all.py` on Python {platform.python_version()}, {v}.\n\n"
          f"Libraries in this environment: {', '.join(f'{k} {s}' for k, s in libs.items())}.\n\n"
          "**PASS** = executed and passed. **SKIP** = NOT executed (reason shown); skipped tests are not claimed as passed. The last lines prove that no Phase 1-3 file changed during the run.\n\n"
          f"```\n{body}\n```\n\nExit code: {out.returncode} (0 = no failures and no Phase 1-3 file changed).\n")
    (ROOT / "docs" / "test_report.md").write_text(md, encoding="utf-8")
    print(body[-700:])
    return out.returncode


if __name__ == "__main__":
    sys.exit(main())
