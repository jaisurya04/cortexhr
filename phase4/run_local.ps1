# Start CortexHR AI locally (run from repository root with the pinned Python 3.12 environment active).
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\phase4\src"
python phase4\scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed. Install the pinned requirements before continuing." }
Start-Process powershell -ArgumentList '-NoExit','-Command',"`$env:PYTHONPATH='$PWD\phase4\src'; uvicorn cortexhr_phase4.api.app:app --port 8000"
Start-Sleep -Seconds 8
python phase4\scripts\smoke_test.py --base-url http://127.0.0.1:8000
if ($LASTEXITCODE -ne 0) { throw "API smoke test failed. Do not launch the console until the API is healthy." }
Set-Location phase4
$env:CORTEXHR_API_URL = "http://127.0.0.1:8000"
streamlit run src\cortexhr_phase4\console\app.py
