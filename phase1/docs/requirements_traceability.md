| 5.5 | Note on hard anomaly types | `reports/attendance_anomaly_report.md` + notebook 04 note: LATE_SPIKE recall 0.20; paid-leave false positives | `attendance_by_anomaly_type.csv` | Done (rewrite in your own words) |

Status: **Done** = implemented, run and checked · **Partial** = done with a stated gap · **Not done** = could not be done here (reason given). Nothing is marked Done unless it was run.

## Day 1: workspace and audit
| # | Requirement | Implementation | Test / output | Status |
|---|---|---|---|---|
| 1.1 | Python 3.11+, Git, venv, install requirements | `requirements.txt`, `pyproject.toml`, README quick start | built/tested on Python 3.12.3 (3.11 not available) | Partial |
| 1.2 | Private GitHub repo `cortexhr`, first commit | local git repo + commits; push commands in README | `git log` | Partial: GitHub repo needs your account; cannot be created here |
| 1.3 | Copy 03_Data and 05_Starter, open 01_data_audit.ipynb | `03_Data/CortexHR_Dataset.xlsx` copied from Drive; `05_Starter/notebooks/01_data_audit.ipynb` written | executed, 0 errors | Partial: starter notebook, `cortexhr_helpers.py`, `requirements.txt`, `csv/` were NOT in Drive; own versions written |
| 1.4 | Load `attrition_dataset` (2,389 rows), shape/types/5 rows | `data_loading.load_attrition`, notebook 01 | `test_raw_shape_and_target` | Done |
| 1.5 | One row and target sentence | notebook 01, `reports/data_audit_report.md` s1 | | Done |
| 1.6 | Missing, duplicates, impossible values | `audit.missingness/duplicates/impossible_values` | tables/missingness, impossible_values | Done (no `age` column exists, so age<18 untestable; stated) |
| 1.7 | IQR and Z-score outliers, "candidates not errors" | `audit.iqr_outliers/zscore_outliers`, boxplot | `test_outlier_helpers_synthetic` | Done |
| 1.8 | Leaver share 14.6%, always-stay ~85% | `audit.class_balance`, class_balance.png | 14.61% / 85.39% | Done |
| 1.9 | Leakage columns with proof | `audit.leakage_review/missingness_indicator_power/single_column_power` | `test_audit_has_at_least_eight_findings_and_leakage_proof`, leakage_evidence.png | Done |
| 1.10 | Audit table >= 8 distinct problems | `audit.build_audit_table` (17 findings), audit_table.png | same test | Done |
| 1.11 | Vault only after audit | commit `0e1d5ce` before opening; `reports/vault_comparison_cx01.md` | git log order | Partial: only `unlock_CX-01_answer_key_notes.csv` existed in Drive; `answer_key_planted_dirt` (row-level) missing |
| 1.12 | Missed problems and why | vault comparison (salary typos missed, planted vs structural blanks) | | Done |
| 1.13 | Evidence: audit table screenshot, class balance chart | `figures/audit_table.png`, `figures/class_balance.png` | | Done |
| 1.14 | Reference links (scikit-learn, pandas, JupyterLab GitHub) | see README references | | Done |

## Day 2: pipeline
| # | Requirement | Implementation | Test / output | Status |
|---|---|---|---|---|
| 2.1 | Drop / keep / transform per audit | `audit.column_roles` -> `tables/column_roles.csv` | | Done |
| 2.2 | Exclude ids, dates, leakage; target never in X | `config.FORBIDDEN_AS_INPUT`, `features.assert_safe_input_columns`, `make_xy` | `test_forbidden_columns_rejected`, `test_target_and_leakage_never_in_X_or_output` | Done |
| 2.3 | Inspect attendance/payroll keys, duplicates | join report, `dedupe_payroll` (15 duplicate payroll rows) | `test_payroll_duplicates_dropped` | Done |
| 2.4 | Employee-level aggregates (last 3 months), cutoff-safe | `data_loading.aggregate_attendance/aggregate_payroll`, `FEATURE_DOCS` | `test_join_cardinality_and_cutoff_no_future` | Done |
| 2.5 | Join with validated cardinality, report unmatched/dropped | `join_monthly_features` (`validate="one_to_one"`), `artifacts/metadata/baseline_run.json` | 383 matched, 2,000 unmatched, 0 dropped | Done |
| 2.6 | ColumnTransformer + Pipeline, `build_pipeline()` in features.py | `features.build_pipeline` | smoke tests | Done |
| 2.7 | Fit on training only; dept-relative pay without leakage | `Derive.fit`, Pipeline inside CV | `test_fit_on_training_only_dept_medians` | Done |
| 2.8 | Unseen categories, missing values | `OneHotEncoder(handle_unknown="infrequent_if_exist")`, imputers | `test_three_row_smoke_unseen_and_missing` | Done |
| 2.9 | Three-row smoke test | `features.representative_rows`, notebook 02 | same test | Done |
| 2.10 | Three-sentence training-only rationale | notebook 02 | | Done |
| 2.11 | Sensible new feature | `pay_vs_dept_median`, `raise_stagnation_ratio`, `never_promoted` | `docs/feature_definitions.md` | Done |
| 2.12 | Save fitted pipeline + metadata | `artifacts/models/feature_pipeline.joblib`, `artifacts/metadata/feature_pipeline.json` | | Done |

## Day 3: baselines
| # | Requirement | Implementation | Test / output | Status |
|---|---|---|---|---|
| 3.1 | Stratified 80/20, fixed seed; test untouched | `modeling.make_split` | `test_split_stratified_reproducible_and_target_separate` | Done |
| 3.2 | Stratified 5-fold, same folds for all models | `make_folds` (explicit list) | `test_same_folds_for_every_model` | Done |
| 3.3 | Dummy majority + LR balanced (+ tree, RF) | `model_zoo` | `test_baselines_sane_no_leakage` | Done |
| 3.4 | Out-of-fold probabilities via cross_val_predict | `modeling.oof_scores` | `tables/oof_scores_train.csv` | Done |
| 3.5 | ROC-AUC, precision, recall, F1, confusion matrix; accuracy supplementary | `evaluation.binary_report` | baseline_fold_metrics.csv | Done |
| 3.6 | Compare with `baseline_benchmarks` | `runs.compare_with_benchmark` | baseline_comparison.csv: LR 0.698 vs 0.713 | Done |
| 3.7 | >0.90 triggers leakage investigation | `runs.leakage_guard` + leaky demo (AUC 1.00 flagged) | | Done |
| 3.8 | Tree train vs CV; LR trade-off explained | baseline_model_comparison.md | tree 1.00 vs 0.55 | Done |
| 3.9 | Zero-division policy, no blanket warning suppression | `zero_division=0`, `precision_defined`, warnings captured and listed (none) | `test_binary_report_zero_division_and_top10` | Done |
| 3.10 | 5-sentence note, comparison table | baseline_model_comparison.md | | Done |
| 3.11 | Record config, seed, folds, versions | `artifacts/metadata/baseline_run.json` | | Done |
| 3.12 | SMOTE / imbalance handling (extra) | `SimpleSMOTE`, `oof_scores_smote`, imbalance_handling_comparison.csv | `test_smote_only_adds_minority_and_is_reproducible` | Done (own implementation) |

## Day 4: evaluation
| # | Requirement | Implementation | Test / output | Status |
|---|---|---|---|---|
| 4.1 | binary_report, ROC-AUC + PR-AUC explained | `evaluation.binary_report`, model_evaluation_report.md | | Done (own `binary_report`; official helper missing) |
| 4.2 | Top-10% precision/recall, ties documented | `top_fraction_mask`, `rank_order` | `test_binary_report_zero_division_and_top10` | Done |
| 4.3 | Calibration curve; no test-label fitting | `Recalibrator` fitted on training OOF only; calibration_curve.png | `test_recalibrator_monotone` | Done |
| 4.4 | Risk bands 10/20/70, deterministic ties | `evaluation.risk_bands`, risk_band_counts.csv | `test_risk_bands_10_20_70_and_ties_deterministic` | Done |
| 4.5 | Leaver capture in High band; lift chart | `band_summary`, `gains_table`, gains_lift_chart.png | `test_band_summary_and_gains` | Done |
| 4.6 | One-page manager note incl. limits + human oversight | `reports/manager_summary.md` | | Done |
| 4.7 | Reproducible from one notebook | `03_model_evaluation.ipynb` | executed, 0 errors | Done |

## Day 5: anomalies
| # | Requirement | Implementation | Test / output | Status |
|---|---|---|---|---|
| 5.1 | IsolationForest, ~3% contamination, no labels | `attendance.detect` | `test_attendance_never_uses_labels` | Done |
| 5.2 | 1-2 simple rules combined with model | zero-attendance, implausible-hours | `test_attendance_rules_and_detector` | Done |
| 5.3 | Rank flags for limited reviewers | `priority_score`, `review_rank`, `review_queue` | attendance_review_queue_top100.csv | Done |
| 5.4 | Measure with catch_rate vs hidden labels; compare 0.40 / 0.59 | `attendance.measure`, `evaluation.catch_rate`; auto-runs when labels file is in the vault | `test_catch_rate_math` | **Not done**: labels file not supplied (monthly precision/recall pending) |
| 5.5 | Note on hard anomaly types | report gives hypotheses only | | **Not done** until labels supplied |
| 5.6 | Labels never used for training | flags saved before labels are read | | Done |
| 5.7 | Supplementary measured proxy | labelled daily sample: combined P 0.87 / R 0.78 | attendance_daily_proxy_eval.csv | Done (different table; labelled as supplementary) |

## Environment
| Docker + docker-compose | files provided | not run (no Docker in sandbox) | Partial |
| pytest | tests are pytest-style, run with fallback runner | pytest not installable offline | Partial |

## Workflow coverage (all five notebooks)
| Stage | Where |
|---|---|
| 1 Data Collection ... 12 Final Conclusions and Recommendations | every notebook in `05_Starter/notebooks/` has all 12 numbered stage headings with real code and output |
| Confusion matrices | Stage 9 of notebooks 01, 01b, 02, 03 and 04 (plots via `workflow.plot_confusion`) |
| Hyperparameter tuning | Stage 10 (LR C grid, tree depth, aggregation window, threshold/calibrator choice, contamination sensitivity); tuned on training folds only |
| Model interpretation | Stage 11 (coefficients, permutation importance on validation rows, band profiles, anomaly explanations) |
| Notebook for Task 2 | `01b_feature_pipeline.ipynb` (Day 2 had no separate notebook name in the brief) |

## Scripts (equivalent to the notebooks)
| Task | Script | Writes (reports/) |
|---|---|---|
| 1 Audit | `scripts/run_audit.py` | audit tables, class balance, EDA, cleaning report, leak-vs-safe proof models, confusion matrices, C tuning, coefficients, permutation importance, audit report |
| 2 Pipeline | `scripts/run_pipeline.py` | join report, 3-row smoke-test output and checks, fitted pipeline, feature definitions, with/without monthly comparison, window tuning, derived-feature coefficients, pay-quartile table |
| 3 Baselines | `scripts/train_baselines.py` | comparison vs benchmark, confusion counts and matrices, LR/tree tuning, SMOTE comparison, interpretation, 5-sentence note |
| 4 Evaluation | `scripts/evaluate_models.py` | metrics, calibration, bands, lift, confusion matrices, threshold sweep, calibrator comparison, band profile, interpretation, manager note |
| 5 Anomalies | `scripts/detect_attendance_anomalies.py` | flags (saved before labels), review queue, audit checks, EDA, temporal stability, contamination sensitivity, flagged-vs-normal interpretation, confusion matrices, measured metrics when labels exist |
`tests/test_script_outputs.py` fails if any of these files is missing.

## Added after the official starter files, templates and vault were supplied
| Item | Where |
|---|---|
| Official `cortexhr_helpers.py` (binary_report, risk_bands, catch_rate, group_gap, psi) used unmodified | `05_Starter/cortexhr_helpers.py`, `src/cortexhr/official.py`, all notebooks, `evaluation.py` |
| `03_Data/csv/` exports of every sheet (derived from the workbook) so the official `load()` works | `03_Data/csv/` |
| CX-02 level-ups (raise recency buckets, overtime flag) | `features.Derive`, `tests/test_features.py` |
| Playbook Git habits (one branch per task, PR text, daily log, weekly demo) | `git log --graph`, `.github/PULL_REQUEST_TEMPLATE.md`, `docs/daily_log.md`, `docs/weekly_demo_week1.md` |
| README per template, architecture diagram, model card draft | `README.md`, `docs/architecture.md`, `docs/model_cards/attrition_baseline.md` |
| 37-point classification workflow | `05_complete_classification_workflow.ipynb`, `docs/workflow_coverage.md`, `tests/test_notebooks.py` |
| Still not done by me | GitHub push, your own explanations in the demo, Docker run, XGBoost/LightGBM/CatBoost/optuna cells (guarded, need installs) |
