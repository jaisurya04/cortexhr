#!/usr/bin/env sh
# Start the API and console from the repository root using the pinned Python 3.12 environment.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/phase4/src"
python phase4/scripts/check_environment.py
uvicorn cortexhr_phase4.api.app:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup() { kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
python - <<'PYWAIT'
import time, urllib.request
url = "http://127.0.0.1:8000/v1/health"
for attempt in range(30):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200: break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("CortexHR API did not become healthy within 30 seconds")
PYWAIT
python phase4/scripts/smoke_test.py --base-url http://127.0.0.1:8000
cd phase4
CORTEXHR_API_URL=http://127.0.0.1:8000 streamlit run src/cortexhr_phase4/console/app.py
