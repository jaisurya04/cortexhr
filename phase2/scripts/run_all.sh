#!/usr/bin/env bash
# Reproduces every report, table, figure, model and notebook output. Run from the repo root.
set -euo pipefail
python scripts/run_audit.py
python scripts/run_pipeline.py
python scripts/train_baselines.py
python scripts/evaluate_models.py
python scripts/detect_attendance_anomalies.py          # measures against hidden labels only if 04_Validation_Vault/unlock_CX-05_* exists
[ -f 04_Validation_Vault/unlock_CX-01_answer_key_planted_dirt.csv ] && python scripts/compare_vault_cx01.py || true
python scripts/build_notebooks.py && python scripts/build_master_notebook.py
python scripts/execute_notebooks.py                    # or: jupyter nbconvert --execute --inplace 05_Starter/notebooks/*.ipynb
python scripts/make_docs.py
python tests/run_all.py                                # or: pytest
