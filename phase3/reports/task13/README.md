# Task 13: attrition model v2
Notebooks 07 to 09. Same folds as Task 3. Candidates: logistic regression (baseline), random forest, histogram gradient boosting, LR + gradient boosting blend (v2).
**CV ROC-AUC:** baseline 0.709 +- 0.021; gradient boosting alone 0.695; attrition-v2 0.716 +- 0.017.
**Verdict:** Gain is NOT proven: the 95% interval of the paired fold difference includes 0 (or is below it), so the simpler baseline is not beaten. Mean paired difference +0.0070, 95% interval [-0.0022, +0.0162].
**Calibration:** sigmoid; test ECE 0.028, Brier 0.113 (uncalibrated ECE 0.252).
**Saved:** `artifacts/models/attrition_v2.joblib` with metadata and checksum. Model card: `docs/model_cards/attrition_v2.md`.

**Official LightGBM candidate (separate challenger).** Best of three predeclared settings on the same five Task 3 folds: CV ROC-AUC 0.692, PR-AUC 0.319; calibrated holdout ROC-AUC 0.696, PR-AUC 0.360. It did not beat the existing LR + HGB blend and is stored separately as `artifacts/models/attrition_lightgbm_candidate.joblib`. Details: `reports/task13/lightgbm_tuning_log.csv`.
