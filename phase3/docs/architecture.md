# Architecture

```
phase3/
├── config/phase3_config.yaml          every number (seed, weights, thresholds, success criteria) in one place
├── 03_Data/                           programme data (read only)
├── 05_Starter/cortexhr_helpers.py     programme helpers: risk_bands(), group_gap()
├── src/cortexhr/                      Phase 1 and 2 code reused unchanged (pipeline, loaders, folds)
│   └── phase3/
│       ├── common/        paths, config, io, validators
│       ├── resume/        text, skills, experience, scoring (TF-IDF + skills + experience), metrics, analysis, api
│       ├── fairness/      twin-pair audit
│       ├── attrition/     v2 candidates, tuning, calibration, training run
│       ├── explanations/  groups, Monte Carlo Shapley, reason lookup, risk report
│       └── salary/        regression models, bands, group gaps
├── notebooks/             15 notebooks, each with the 38-stage headings, executed with real outputs
├── scripts/               notebook builders, executor, docs generator
├── artifacts/             models, metadata (versions, checksums), predictions
├── reports/               figures, tables, one folder per task
├── tests/                 8 test modules + a runner that needs no pytest
└── docs/                  this folder
```

**Data flow.** Task 11 builds a scoring engine (v1 text as given, v2 masked). Task 12 audits v1 and v2 with twin resumes and keeps v2. Task 13 builds attrition-v2 on the Phase 1 pipeline and folds. Task 14 scores 430 employees with v2 and explains each. Task 15 is independent of 11 to 14 but reuses the fairness method of Task 12.

**Reuse of earlier phases.** `cortexhr.features.build_pipeline`, `cortexhr.runs.load_model_frame`, `cortexhr.modeling.model_zoo` and the cut-off-safe monthly join come from Phase 1 and are imported, not copied. Phase 1 and Phase 2 model files are never overwritten (Phase 3 artifacts have new names in this folder).

**Why plain functions and no web server.** The resume-ranking API contract is implemented as a pure function (`resume/api.py`) that validates input and returns the response object, so it is fully tested without FastAPI. It can be mounted on the Phase 2 service unchanged.
