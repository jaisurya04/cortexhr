# Methodology

## Task 11: resume ranking
Score = 0.30 x text similarity + 0.40 x required-skill coverage + 0.10 x nice-to-have coverage + 0.20 x experience fit, each part scaled to 0..1.
- **Text similarity:** TF-IDF (1 to 2 word n-grams, sublinear tf) fitted on resumes and job texts only; cosine similarity, scaled per job with the min and max over the corpus so jobs are comparable. Multi-word skills are kept as single tokens.
- **Skills:** dictionary matching, longest phrase first, matched text consumed so "customer communication" does not also count as "communication"; a small synonym map (MS Excel to excel, sklearn to scikit-learn, ...); exact and synonym matches are both reported.
- **Experience:** years per role read from the text ("(2.4 yrs)"); duplicate roles ignored; relevant years = roles whose title shares at least half of the job title's core words; score = half total years, half relevant years, each capped at the job minimum so more years never beat the requirement.
- **Weights** were fixed a priori. A 286-point grid (`reports/task11/weight_experiments.csv`) and a leave-one-job-out check test the choice; the labels are used only to measure.
- **Ties** are broken by a neutral hash of (job, resume), not by id order.
- **Protected information** is removed before scoring: name, e-mail, city and the "Tier N institution" phrase (`resume/text.py`). `target_job_family` is never used (it names the answer).

## Task 12: fairness audit
Counterfactual twins: score both twins on every job they are labelled for; gap = twin minus original. Ablation of the masking switches shows the source (name and e-mail). Group view uses the supplied `group_gap()` on the full pool with attributes from the restricted file.

## Task 13: attrition v2
Phase 1 frame, split and folds (80/20, 5 stratified folds, seed 42). Candidates: logistic regression (baseline), random forest, histogram gradient boosting (default and tuned over 16 logged settings, chosen by CV PR-AUC on the training part), and an equal-weight blend of logistic regression and tuned gradient boosting (v2). Gain is called real only if the bootstrap 95% interval of the paired fold difference is above zero. Calibration (none, sigmoid, isotonic) is learned inside folds and chosen by out-of-fold Brier score. The test set is used once.

## Task 14: explanations
Monte Carlo permutation Shapley values over 23 groups of the 43 inputs (120 permutations, each also reversed, 200-row background), in probability units. Efficiency is exact; stability between two independent halves is reported. Reasons are the three groups that raised the score most, excluding groups that can proxy for protected traits (career history, education, location), turned into sentences by a lookup (`explanations/reasons.py`). Bands come from the supplied `risk_bands()`.

## Task 15: pay bands
Log-salary regression on job-related inputs. Baselines: average by role, and role average adjusted for market index. Ridge and tuned gradient boosting compared by CV MAE. Band = predicted mid times the 10th and 90th percentile of out-of-fold log error. Fairness: group gaps by gender and career gap (restricted file, measurement only) for a model with and without `career_gap_months`.
