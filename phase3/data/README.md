# data/

Phase 3 reads its datasets from `03_Data/csv/` (the same files the programme supplies; nothing is copied or changed).

| File | Rows | Used for |
|---|---|---|
| resumes.csv | 185 | Task 11 and 12 (160 originals + 25 twins) |
| job_descriptions.csv | 10 | Task 11 |
| resume_job_labels.csv | 555 | Task 11 evaluation (human judgement; labels never used to fit anything) |
| resume_bias_pairs.csv | 25 | Task 12 twin audit |
| attrition_dataset.csv / CortexHR_Dataset.xlsx | 2,389 | Task 13 (through the Phase 1 loader, with monthly aggregates) |
| baseline_benchmarks.csv | 14 | Task 13 reference numbers |
| attrition_scoring_2026.csv | 430 | Task 14 |
| sample_attrition_risk.csv | 430 | Task 14 output format |
| salary_training.csv | 2,934 | Task 15 |
| data_dictionary.csv | | column meanings |

**Restricted files (never committed):** `04_Validation_Vault/unlock_CX-12_resume_protected_attrs.csv` and `unlock_CX-15_protected_attributes.csv` hold gender, age band and college tier. They are used only after the scores exist, to measure gaps. If they are absent the notebooks print that the group measurement was skipped; everything else runs.
