# Architecture (Phase 1)

```mermaid
flowchart LR
  X[CortexHR_Dataset.xlsx<br/>03_Data] --> L[data_loading<br/>clean_raw, dedupe, join_monthly_features<br/>cutoff = observation_date]
  L --> A[audit.py<br/>quality, leakage proof] --> R1[reports/data_audit_report.md]
  L --> F[features.py build_pipeline<br/>RawCleaner -> Derive -> ColumnTransformer]
  F --> M[modeling.py<br/>stratified split, 5-fold, Dummy/LR/Tree/RF/SMOTE]
  M --> E[evaluation.py + official cortexhr_helpers<br/>binary_report, risk_bands]
  E --> R2[reports: metrics, calibration, bands, manager_summary]
  X --> T[attendance.py<br/>IsolationForest + rules] --> R3[attendance flags, catch_rate vs hidden labels]
  V[(04_Validation_Vault<br/>restricted, git-ignored)] -. measure only .-> R3
  V -. compare only, after audit .-> A
```

Rules the design enforces: nothing learned from the test rows (pipeline fitted on training rows only), no post-outcome columns (`config.FORBIDDEN_AS_INPUT`), monthly features only from complete months before the snapshot, vault files never used as inputs.
