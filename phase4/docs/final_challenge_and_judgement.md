# Day 20 - Judgement and Final Challenge (YOUR words are required here)

## Think for yourself: "Which model would you not release, if you were the manager? Write down why."
Draft answer from the evidence (edit it into your own judgement):

> I would not release **attrition v2** as it stands. Its cross-validated ROC-AUC (0.716) is not shown to beat the simple logistic baseline (0.709); the paired fold difference includes zero. At an AUC near 0.70 it misses most leavers and flags many stayers. And the latest monitoring run shows that one of its two most important inputs, `months_since_last_raise`, collapsed in the 2026 population (PSI 3.8), so the 2026 scores rest on a distribution the model never saw. I would release the pay-band advisor and the resume ranker (with a recruiter always reading the resume) and the payroll anomaly detector (as a review queue), keep attrition scores for internal exploration only, and re-evaluate after HR explains the raise-date change.

## Final Challenge (step 6: improve one thing on its own branch, with no guide, and describe what you decided and why)
This is meant to be **your own** unguided work, so nothing was done for you. Candidates the evidence supports (pick one):
1. Calibrate the forecast intervals (rolling coverage is about 68% against an 80% target): try horizon-specific widening and show the new coverage.
2. Make the attrition model robust to the collapsed raise feature (for example, cap or bucket it) and show the effect on CV AUC and PSI.
3. Add per-department exit forecasts and show whether they beat the baseline.
4. Replace link-navigation in the console with a session-preserving navigation.

Write here: the branch name, what you changed, the before/after numbers, and why you decided it.

* Branch:
* What I changed:
* Evidence (numbers):
* Why I decided this:
