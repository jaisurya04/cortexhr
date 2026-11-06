# Phase 3 quality gate

Generated 2026-11-05 16:24 UTC by `python scripts/run_quality_gate.py`.

**Tests:** 72 passed, 0 failed.

| Test module | passed | failed |
|---|---|---|
| test_attrition_v2 | 10 | 0 |
| test_data_validation | 8 | 0 |
| test_outputs | 8 | 0 |
| test_resume_fairness | 7 | 0 |
| test_resume_ranking | 12 | 0 |
| test_risk_scoring | 7 | 0 |
| test_salary_model | 10 | 0 |
| test_shap_explanations | 10 | 0 |

**Requirement traceability:** 51 of 51 requirements complete (see `requirements_traceability.md`).

**Gate:** GREEN: every test passes and every tracked requirement has its notebook, code, output and test.

**Additional evidence / limitations:** LightGBM candidate was trained and evaluated separately on the same Task 3 folds; real SHAP PermutationExplainer (shap 0.54.1) was run on all 430 employees (additivity error 5.6e-16). The authorized CX-12 and CX-15 files were supplied and validated locally; Task 12 and Task 15 were rerun with protected-group measurements. Task 13 saved group counts match the supplied vault, but its full retraining/tuning rerun exceeded the 10-minute execution limit. Restricted files remain excluded from the ZIP.