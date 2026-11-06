# Phase 3 final code audit status

Generated: 2026-11-05

## Changes implemented in this audit

- Added a real LightGBM classifier using the Phase 1 preprocessing pipeline.
- Added a three-configuration LightGBM tuning/evaluation workflow using the same five Task 3 folds, with a separate calibrated holdout evaluation.
- Saved a separate LightGBM challenger artifact and metadata; the existing production attrition-v2 artifact was not silently replaced because LightGBM did not establish a clear improvement.
- Added combined CV and holdout comparison tables alongside the original baseline/HGB results.
- Implemented the previously unfinished SHAP PermutationExplainer cross-check. Categorical inputs are encoded for the SHAP masker and decoded before model prediction, preserving the model's original input representation.
- Ran the real SHAP library (PermutationExplainer) on all 430 scoring employees with a 50-row background; aggregated feature-level values to the existing HR explanation groups and saved both outputs.
- Updated Task 13 and Task 14 notebook builders/notebooks, traceability, limitations, model card, task reports, quality-gate instructions and the 3-page Phase 3 PDF.
- Added automated tests for LightGBM candidate construction, saved candidate artifact/checksum, real SHAP execution and saved SHAP outputs.

## Verified results

- LightGBM: 3 predeclared configurations x 5 Task 3 folds (15 fold-result rows).
- Best LightGBM CV ROC-AUC: 0.692233; PR-AUC: 0.319219.
- Calibrated LightGBM holdout ROC-AUC: 0.695683; PR-AUC: 0.359738; Brier: 0.114869.
- All 15 notebooks pass nbformat validation; all code cells have execution counts and no stored error outputs.
- The notebook workflow test verifies that each Task 11–15 notebook group covers stages 1 through 38.
- Full automated test suite: 72 passed, 0 failed.
- Quality gate: GREEN, 51/51 tracked requirements have their referenced notebooks, code, outputs and tests.

## Restricted validation files received and checked (2026-11-05)

The user supplied both authorized files. They were copied into the local `04_Validation_Vault/` folder for validation only; they are deliberately excluded from the deliverable ZIP.

- CX-12 resume attributes: 185 rows, 185 unique resume IDs; exact 185/185 match to the project resume table; no nulls or duplicate rows/keys.
- CX-15 protected attributes: 5,317 rows split into 2,383 `attrition_snapshot` rows and 2,934 `salary_hire` rows. Each record type has unique person references; 100% join coverage against the corresponding attrition and salary source IDs; no nulls or duplicate rows.
- Task 11 notebook 04 was re-executed under the current environment and the resume-ranker artifact/metadata were regenerated, resolving an older pandas-pickle compatibility issue in the saved artifact.
- Task 12 notebooks 05 and 06 were re-executed with CX-12 present. Protected-group views and twin-level attribute interpretation were generated.
- Task 15 end-to-end model/fairness workflow was re-run with CX-15 present; notebooks 12, 13 and 14 were re-executed. Gender, age-band and career-gap tables were regenerated. Test band coverage: 0.8024.
- Task 13's full end-to-end training/tuning function was attempted but exceeded the 10-minute execution limit. Its already-saved group fairness tables were retained; their gender and age group counts exactly match the supplied CX-15 attrition-snapshot distributions. This is a data-join validation, not a claim that the full Task 13 model was freshly retrained in this pass.

**Security:** The two restricted CSV files are not included in the project ZIP and must remain outside version control. Keep them in the local vault only when an authorized fairness rerun is needed.

## Execution scope limitation

The complete 15-notebook clean-kernel rebuild was not rerun in this audit. Notebooks 04, 05 and 06 (Tasks 11 and 12), and 12, 13 and 14 (Task 15) were re-executed successfully; notebook 15 was also re-executed successfully after regenerating the resume-ranker artifact under the current environment. Notebooks 01-03 and 07-11 retain their previously verified saved outputs. Task 13's full training/tuning function exceeded the 10-minute execution limit, so its prior saved model metrics and group tables were retained; the group counts were cross-checked against the supplied protected data. The new LightGBM audit and SHAP cross-check functions were executed independently, and Task 14 integration was smoke-tested on all 430 scoring rows with a reduced permutation count. The final Task 14 results come from the full run of `scripts/run_shap_full430.py` (all 430 employees).

## Reproduction commands

From `cortexhr/phase3/`:

```bash
python tests/run_all.py
python scripts/run_quality_gate.py
python -c "from cortexhr.phase3.attrition.train import run_task13; run_task13(write=True)"
```

For the separately versioned official LightGBM candidate:

```bash
python -c "from cortexhr.phase3.attrition.train import run_lightgbm_audit; run_lightgbm_audit(write=True)"
```

For the full Task 14 explanation run, use notebook 10 or call `run_task14(write=True)` after confirming the authorized local data and optional SHAP dependency are available.

## Reviewer-facing decision notes (also answered in the notebooks)

- **Task 12 — equal accuracy and trade-off:** Equal group accuracy does not guarantee equal treatment of comparable individuals; a protected detail can still shift an individual's score or shortlist outcome. If masking reduced ranking quality materially, the hiring-process owner, HR and legal should weigh the documented trade-off; the developer should not ship a screener that reacts to protected details merely to gain accuracy.
- **Task 13 — ship or not:** Do not ship v2 as a proven upgrade on a small, noisy gain. Keep the simpler logistic baseline for decisions and v2 as a challenger until an independent evaluation shows a confidence interval above zero and the gain justifies added complexity.
- **Task 14 — tenure wording:** Describe short tenure as a historical pattern, not a personal failing or a prediction about an individual. For a new joiner, use “A short time at the company so far” and state that it is not a judgement of their performance.
- **Task 15 — historical pay gap:** Do not train the model to reproduce historical underpayment. Exclude gender and career-gap proxy inputs, measure residual differences, use job-related pay bands, and audit/rebase historical offers with HR before real deployment.

## Task 14 explanation-method disclosure

The final Task 14 report (`attrition_risk_2026.csv`, 430 employees) takes its top-3 reasons from the real `shap` library (`PermutationExplainer`, shap 0.54.1, all 430 employees, 50-row background, 23 HR-readable groups; sensitive-proxy groups are never shown as reasons). No Monte Carlo or other home-made estimator remains in the code. Figure 4 in the report is the SHAP summary.

## Submission documentation corrections

- README installation instructions: `requirements.txt` now includes `shap==0.54.1` (required); `requirements-optional.txt` holds only `lightgbm`.
- The current audit container already had scikit-learn 1.8.0 and LightGBM 4.6.0 installed, and the full test suite passed 72/72. The installed pandas 2.2.3, NumPy 2.3.5 and SciPy 1.17.0 do not match the exact pins in `requirements.txt` (pandas 3.0.2, NumPy 2.4.4, SciPy 1.17.1). A fresh `pip install -r requirements.txt` was attempted but could not reach the package index because this execution environment has no DNS/network access; therefore a clean-machine installation with every pinned version was not verified here.
- The report states that the 430 final reasons come from the shap library. The figure and caption match.
- The PDF now describes the saved notebook outputs and 38-stage coverage without claiming that every notebook was clean-kernel re-executed in this final audit.

## Native SHAP on all 430 employees

Executed on the author's machine (Windows, project `.venv`): shap 0.54.1 PermutationExplainer on all 430 employees (50-row background): additivity error 5.6e-16. rows = 430 confirmed in the run summary. `requirements.txt` pins the same version (0.54.1).
