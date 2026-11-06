# Task 11: resume ranking
Engine resume-ranker-v2-masked (TF-IDF + skills + experience, identifiers masked). Notebooks 01 to 04.
**Quality vs human labels (555 pairs, 10 jobs):** precision@5 0.940, recall@5 0.549, hit rate@5 1.000, NDCG@5 0.923, MRR 1.000 (random: precision@5 0.166, NDCG@5 0.266).
**Weakest job:** J02 (precision@5 0.400, 3 relevant candidates, so precision@5 cannot exceed 0.60).
**Misranks:** 5 shown with root cause in `misranking_analysis.csv`.
**Files:** ranking_quality_per_job.csv, ranking_quality_summary.csv, misranking_analysis.csv, top5_per_job_all_resumes.csv, weight_experiments.csv. API contract: `docs/api_notes_resume_ranking.md`.