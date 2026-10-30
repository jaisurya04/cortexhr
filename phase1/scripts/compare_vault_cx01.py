"""CX-01 vault comparison at ROW level. Run ONLY after the audit is finished and committed.
Needs 04_Validation_Vault/unlock_CX-01_answer_key_planted_dirt.csv (restricted, git-ignored)."""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from cortexhr import config as C, data_loading as D, audit as A, reporting as R

VAULT = C.project_root() / "04_Validation_Vault" / "unlock_CX-01_answer_key_planted_dirt.csv"

def main():
    if not VAULT.exists(): print("planted-dirt file not found:", VAULT); return
    raw = D.load_attrition(); key = pd.read_csv(VAULT)
    ids = lambda mask: set(raw.loc[mask, "employee_id"])
    dups = A.duplicates(raw)
    mine = {"MISSING_VALUE": None, "INCONSISTENT_CATEGORY": ids(raw.work_location != D.canonical_city(raw.work_location)),
            "DUPLICATE_ROW": set(dups["ids_affected"]), "NEGATIVE_TENURE": ids(raw.tenure_months < 0),
            "IMPOSSIBLE_VALUE": ids(raw.attendance_rate_3m > 1),
            "SALARY_TYPO_OUTLIER": set(A.pay_consistency_outliers(raw).employee_id)}
    # MISSING_VALUE: planted detail names a column; caught when that cell is blank (my missingness table listed both columns)
    blank_hit = []
    for r in key[key.issue_type == "MISSING_VALUE"].itertuples():
        col = next((c for c in raw.columns if c in str(r.detail)), None)
        blank_hit.append(bool(col and raw.loc[raw.employee_id == r.employee_id, col].isna().any()))
    orig_audit_caught = {"SALARY_TYPO_OUTLIER": False}      # the ORIGINAL audit (commit 0e1d5ce) missed these; the consistency check was added after the vault comparison
    rows = []
    for t, g in key.groupby("issue_type"):
        planted = set(g.employee_id)
        if t == "MISSING_VALUE": caught = sum(blank_hit)
        else: caught = len(planted & mine[t])
        rows.append({"issue_type": t, "planted_rows": len(g), "found_by_current_audit_code": caught, "recall": round(caught / len(g), 3),
                     "found_in_ORIGINAL_audit": ("no (missed)" if t in orig_audit_caught else "yes")})
    tab = pd.DataFrame(rows); R.save_table(tab, "vault_cx01_rowlevel_comparison")
    extra = {"wfh_share_3m outside 0-1": int(((raw.wfh_share_3m > 1) | (raw.wfh_share_3m < 0)).sum()),
             "total_experience below tenure/negative": int((raw.total_experience_years < raw.tenure_months / 12 - 0.05).sum()),
             "leavers who had already resigned before the snapshot": int((raw.resignation_date <= raw.observation_date).sum())}
    print(tab.to_string(index=False)); print("found but NOT in the planted list:", extra)
    txt = f"""# Vault comparison for CX-01 (opened AFTER the audit was committed)

**Sequence proof.** git commit `0e1d5ce` ("Day 1: data audit (completed BEFORE opening 04_Validation_Vault)") holds the finished audit table. Only then were the vault files opened: first `unlock_CX-01_answer_key_notes.csv`, later (once supplied) the row-level `unlock_CX-01_answer_key_planted_dirt.csv`. The answers were never used as model inputs.

## Row-level result (planted dirt vs my audit code)
{R.md_table(tab)}

*Found by current audit code* = the same row ids are flagged by the audit/cleaning functions in `src/cortexhr/audit.py` and `data_loading.py`.
Every planted issue is caught at row level **except the 3 salary typos, which my original audit missed**.

## Counts from the notes file
| Vault statement | My original audit | Verdict |
|---|---|---|
| Leakage: exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount | all five flagged CONFIRMED with proof | Caught |
| (not listed) voluntary_left_within_12m | flagged as derived from the exit event | Extra catch |
| dept_attrition_rate_prior_12m, comp_ratio, months_since_last_raise, attendance windows are legitimate | classified legitimate (best single-column AUC 0.60) | Agree |
| 6 duplicates / 13 inconsistent categories / 4 negative tenure / 2 impossible values | all found | Caught |
| 70 missing values (15 highest_education + 55 commute_distance_km) | reported, but lumped with structural blanks | Partly missed (did not separate accidental from structural) |
| 3 salary typo outliers | called "real senior pay, keep" | **MISSED** |

## What I missed and why
1. **Salary typos (3 rows).** I looked at `annual_fixed_pay` on its own scale and decided every extreme was real senior pay. The check that finds them compares two related columns: `annual_fixed_pay / comp_ratio` is the role mid-point and should be near constant inside a role; rows 15, 217 and 345 are about 11x their role (a monthly-vs-annual style typo). *Lesson:* test consistency between related columns, not just each column alone. *Fix:* `audit.pay_consistency_outliers` and the pipeline blanks pay above 3x the training role mid-point (`features.Derive`).
2. **Planted vs structural blanks.** I counted blanks but did not ask which are errors and which carry information (never promoted, no survey reply). *Lesson:* classify missingness before choosing a treatment.

## Found by me, not in the planted list
{R.md_table(pd.DataFrame(list(extra.items()), columns=['check', 'rows']))}

These come from generator noise or table design rather than the planted list, but they still change how features must be cleaned and cut off.
"""
    (C.reports_dir() / "vault_comparison_cx01.md").write_text(txt)

if __name__ == "__main__":
    main()
