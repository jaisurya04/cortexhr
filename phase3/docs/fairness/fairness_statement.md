# Fairness statement: CortexHR resume screener (Task 12)

**What was tested.** 25 twin resume pairs (75 resume-job comparisons) that are identical in skills and experience and differ only in a gender-coded name (25 pairs) and, in 13 of them, also the college-tier phrase. Both twins were scored by the screener on every job they were labelled for.

**What was found (before the fix).** The first screener read the resume text as given. The score of the twin differed from the original in 74 of 75 comparisons (mean absolute gap 0.000126, maximum 0.001619 on a 0 to 1 scale). The mean signed gap was -0.000031: no systematic direction detected (the interval includes zero). The score was reacting to the candidate's name and e-mail even if it did not favour one side. Human reviewers scored all twins identically: True.

**What was changed.** Name, e-mail, city and the college-tier phrase are removed before scoring. The ablation shows the whole gap came from name and e-mail words.

**After the fix.** Mean absolute gap 0.000000, maximum 0.000000, non-zero in 0 comparisons. Ranking quality moved from NDCG@5 0.926 to 0.923 (change -0.003) and precision@5 from 0.960 to 0.940.

**Equal accuracy is not equal treatment:** two groups can have the same average accuracy while an individual still receives a different score or crosses a shortlist threshold because of a protected detail; accuracy compares aggregate outcomes, while treatment asks whether comparable individuals are handled consistently.

**What remains.** (1) A zero gap on 25 twin pairs is guaranteed by construction once the identical text is masked; it proves the leak is closed, not that the screener is fair in every respect. (2) Other proxies were not tested: writing style, hobbies, career gaps, language, employer names, degree names. (3) Group gaps in the full pool of 185 resumes still exist (average group gap by college tier 0.0412 before and 0.0329 after masking; by gender 0.0130 before and 0.0114 after (best minus worst group mean score, 0 to 1 scale)); they reflect real differences in skills and experience as well as any bias, so they need review with the recruiter, not a mechanical fix. (4) Ten jobs and 555 labels are a small basis. (5) A human must review every candidate.

**Before real use.** Test with more pairs and more attributes (age-coded years, career gaps, region), re-test after every change to the text processing, monitor shortlist rates by group, and keep a documented human override.
