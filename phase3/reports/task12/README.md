# Task 12: fairness audit of the resume screener
25 twin pairs, 75 comparisons. Notebooks 05 and 06; statement in `docs/fairness/fairness_statement.md`.
**Before (text as given):** non-zero gap in 74 of 75 comparisons; mean absolute gap 0.000126, median 0.000028, maximum 0.001619.
**After (identifiers masked):** mean 0.000000, maximum 0.000000. Ranking quality NDCG@5 0.926 to 0.923, precision@5 0.960 to 0.940.
**Source of the gap (ablation):** none (v1) -> 0.000126; name + e-mail only -> 0.000000; college-tier phrase only -> 0.000127; city only -> 0.000108; everything (v2) -> 0.000000.
**Remaining:** a zero gap is by construction; other proxies untested (see statement).