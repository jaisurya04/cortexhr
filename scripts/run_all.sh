#!/usr/bin/env bash
# Reproduces every report, table, figure, model and notebook output. Run from the repo root.
set -euo pipefail
python scripts/run_audit.py
python scripts/run_pipeline.py
python scripts/train_baselines.py
python scripts/evaluate_models.py
python scripts/detect_attendance_anomalies.py
python scripts/build_notebooks.py && python scripts/execute_notebooks.py   # or: jupyter nbconvert --execute --inplace 05_Starter/notebooks/*.ipynb
python tests/run_all.py                                                    # or: pytest
