# Experiment log (generated from the saved tables)

| Area | Experiments | Selection rule | Outcome | File |
|---|---|---|---|---|
| Resume score weights | 286 weightings (step 0.1) | weights fixed a priori; grid is a sensitivity analysis; leave-one-job-out check | NDCG@5 ranges 0.351 to 0.964; configured weights {'tfidf': 0.3, 'required_skill': 0.4, 'nice_skill': 0.1, 'experience': 0.2} give 0.923 | `reports/task11/weight_experiments.csv` |
| Masking ablation | 5 variants | measure which identifier causes the twin gap | none (v1): mean gap 0.000126, name + e-mail only: mean gap 0.000000, college-tier phrase only: mean gap 0.000127, city only: mean gap 0.000108, everything (v2): mean gap 0.000000 | `reports/task12/ablation_masking.csv` |
| Attrition gradient boosting | 12 settings, 5 folds each | highest mean CV PR-AUC on the training part | selected `HGB-02`; top-5 settings span 0.007 PR-AUC (fold sd about 0.043) | `reports/task13/tuning_log.csv` |
| Attrition calibration | 3 methods | lowest out-of-fold Brier | chosen sigmoid: none Brier 0.1895 ECE 0.2521, sigmoid Brier 0.1144 ECE 0.0137, isotonic Brier 0.1153 ECE 0.0144 | `reports/task13/calibration_comparison.csv` |
| Attrition v2 vs baseline | 5 paired folds, bootstrap | gain real only if the 95% interval is above 0 | Gain is NOT proven: the 95% interval of the paired fold difference includes 0 (or is below it), so the simpler baseline is not beaten. | `reports/task13/paired_differences.csv` |
| Explanations | 120 permutations (2 independent halves) | split-half agreement | phi correlation 0.971; additivity error 2.1e-16 | `reports/task14/explanation_run.json` |
| Salary gradient boosting | 6 settings, 5 folds each | lowest CV MAE | selected `S01` | `reports/task15/tuning_log.csv` |

**Official LightGBM candidate (separate challenger).** Best of three predeclared settings on the same five Task 3 folds: CV ROC-AUC 0.692, PR-AUC 0.319; calibrated holdout ROC-AUC 0.696, PR-AUC 0.360. It did not beat the existing LR + HGB blend and is stored separately as `artifacts/models/attrition_lightgbm_candidate.joblib`. Details: `reports/task13/lightgbm_tuning_log.csv`.
