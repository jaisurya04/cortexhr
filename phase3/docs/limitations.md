# Limitations (written down, not hidden)

**Tooling.** The core production paths remain runnable without optional packages, but the audited LightGBM challenger and native-SHAP cross-check are separate outputs that do depend on their libraries. With `lightgbm==4.6.0`, Task 13 trains and compares a LightGBM challenger on the same Task 3 folds; it did not replace the production LR + HGB blend. With `shap==0.50.0`, Task 14 runs native `PermutationExplainer` on a reproducible 10-employee sample and saves feature-level and grouped cross-check outputs. The full 430-employee explanations remain grouped Monte Carlo Shapley. Install `requirements-optional.txt` to reproduce both audit outputs.

**Task 11.** 10 jobs and 555 human labels are a small basis; some jobs have fewer than 5 relevant candidates, which caps recall@5. Skills outside the vocabulary are invisible. Years are read from text and capped. Resumes with missing sections contain fewer skills than the structured list, and no text method can recover them. The score ranks; it does not predict hiring.

**Task 12.** 25 twin pairs vary a gender-coded name (and in 13 pairs the college tier). A zero gap after masking is guaranteed by construction (the scoring text becomes identical); it proves the leak is closed, not that the screener is fair in every respect. Writing style, hobbies, career gaps, language and employer names were not varied.

**Task 13.** About 350 leavers in 2,383 rows; five folds give wide uncertainty. The measured gain of v2 over the baseline is not proven. Monthly aggregates cover only recent months. 77% of the 430 scored employees were also in the training data at an earlier snapshot.

**Task 14.** Reasons are associations in historical data, not causes. The order of the three reasons is the least stable part (two independent runs agree on all three in about half of the employees). Groups that can proxy for protected traits are hidden from the reasons but still influence the score. Never use for dismissal, pay or promotion decisions.

**Task 15.** The target is historical offers and may contain past bias; a band describes what was offered, not what is fair. 2026 bands are applied to older hires through a market index. No performance, negotiation or internal-equity data. The third gender group is small (60 hires). Advice for a manager and HR only.

**Restricted files.** `unlock_CX-12` and `unlock_CX-15` are used for measurement only and are intentionally excluded from the repository and deliverable ZIP. Keep authorized copies in `04_Validation_Vault/` locally to reproduce protected-group measurements. They were supplied and validated during the 2026-11-05 audit; Task 12 and Task 15 were rerun with them. Task 13 saved group tables were checked against their exact group counts, but a full Task 13 retraining/rerun exceeded the execution time limit in that audit.


**New optional-library verification (2026-11-05).** In the audit environment, LightGBM 4.6.0 and SHAP 0.50.0 were available. A real LightGBM candidate was trained on the same five Task 3 folds and compared with the existing models; it was not a clear improvement and remains a separately versioned challenger. A real SHAP PermutationExplainer cross-check was executed on 10 scoring rows; its feature-level values were aggregated to the existing 23 HR groups for comparison. The main grouped Monte Carlo explanation remains the full-population method.

**Native SHAP on all 430 (2026-11-06).** shap 0.54.1 PermutationExplainer on all 430 employees (50-row background): additivity error 5.6e-16; mean top-3 overlap with the grouped Monte Carlo reasons 0.82; all three reasons identical for 50.0% of employees; same #1 reason for 68.1%; median per-person correlation of group contributions 0.954. Agreement is good but not perfect: the 430-row report was not regenerated from SHAP, and the two methods give different top-3 reasons for about half of the employees, so reasons are drivers, not certainty.
