# Model Card: CortexHR resume ranker v2 (masked)   (CX-20)

**Purpose** orders resumes for one job so a recruiter reads the most relevant first.  ·  **Not for** rejecting anyone automatically; jobs or skills outside its vocabulary; as a probability of success.

**Data** 185 resumes, 10 job descriptions, 555 human relevance labels (`resume_job_labels.csv`). Name, e-mail, city and college-tier text are masked before scoring. Evidence: `phase3/reports/task11/`, `task12/`.

**Method** TF-IDF similarity + required and nice-to-have skill coverage + capped experience, weights fixed in advance and checked with 286 alternative weightings and leave-one-job-out; neutral hash tie-break.

**Metrics**

| engine | precision@5 | recall@5 | hit_rate@5 | ndcg@5 | mrr | average_precision | r_precision |
|---|---|---|---|---|---|---|---|
| final engine (v2, masked) | 0.94 | 0.549 | 1.0 | 0.923 | 1.0 | 0.96 | 0.882 |
| random ordering | 0.166 | 0.089 | 0.592 | 0.266 | 0.355 | 0.223 | 0.168 |

Weakest job J02 has only 3 relevant candidates, so its precision is unstable.

**Explanations** Every score is built from four numbers; example top candidate for J01: `RES-0138` with score 0.825.

**Fairness** Twin-resume test (25 pairs, 75 comparisons): score gap before masking and after:

| measure | before (v1) | after (v2) | change | improvement_% |
|---|---|---|---|---|
| comparisons | 75.0 | 75.0 | 0.0 | 0.0 |
| pairs | 25.0 | 25.0 | 0.0 | 0.0 |
| mean_abs_gap | 0.0 | 0.0 | -0.0 | 100.0 |
| median_abs_gap | 0.0 | 0.0 | -0.0 | 100.0 |
| max_abs_gap | 0.002 | 0.0 | -0.002 | 100.0 |
| mean_signed_gap(twin-original) | -0.0 | 0.0 | 0.0 | 100.0 |
| comparisons_with_nonzero_gap | 74.0 | 0.0 | -74.0 | 100.0 |
| comparisons_above_tolerance | 2.0 | 0.0 | -2.0 | 100.0 |
| mean_abs_relative_gap_pct | 0.069 | 0.0 | -0.069 | 100.0 |
| share_twin_ranked_higher | 0.48 | 0.427 | -0.053 | 11.111 |
| mean_abs_rank_difference | 1.0 | 1.0 | 0.0 | 0.0 |

The gap is zero after masking **by construction**; it shows that name, e-mail, city and tier text no longer move the score, not that the system is fair. Other proxies were not tested.

**Limits & risks** Ten jobs and 555 labels are a small basis. Re-test fairness after every change.

**Version & owner** `resume-ranker-v2-masked` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
