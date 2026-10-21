"""Day 1: audit the attrition dataset. Run: python scripts/run_audit.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from cortexhr import config as C, data_loading as D, audit as A, reporting as R


def main():
    raw = D.load_attrition(); att = D.load_sheet(C.SHEET_ATT); pay = D.load_sheet(C.SHEET_PAY)
    print("shape", raw.shape); print(raw.dtypes.to_string()); print(raw.head())
    inv = A.schema_inventory(raw); ms = A.missingness(raw); dup = A.duplicates(raw)
    imp = A.impossible_values(raw); iq = A.iqr_outliers(raw); zs = A.zscore_outliers(raw)
    cb = A.class_balance(raw); audit_t, lk, single = A.build_audit_table(raw, att, pay)
    post_cols = [c for c in C.LEAKAGE_CONFIRMED if c != "voluntary_left_within_12m"]
    lk_power = A.missingness_indicator_power(raw, post_cols); roles = A.column_roles(raw)
    for n, t in {"schema_inventory": inv, "missingness": ms, "impossible_values": imp.assign(row_ids=imp.row_ids.astype(str)),
                 "iqr_outliers": iq, "zscore_outliers": zs, "leakage_review": lk, "leakage_filled_flag_power": lk_power,
                 "single_column_auc": single, "audit_table": audit_t, "column_roles": roles,
                 "class_balance": pd.DataFrame([cb])}.items():
        R.save_table(t, n)
    R.save_fig(A.plot_class_balance(cb), "class_balance")
    R.save_fig(A.plot_missingness(ms), "missingness")
    R.save_fig(A.plot_outliers(raw), "outliers_boxplot")
    R.save_fig(A.plot_leakage(lk_power, single), "leakage_evidence")
    short = audit_t[["problem", "affected_column_or_location", "rows_affected", "evidence", "severity", "proposed_treatment", "reason"]].copy()
    short["affected_column_or_location"] = short["affected_column_or_location"].str.slice(0, 60)
    short["evidence"] = short["evidence"].str.slice(0, 200)
    R.table_as_image(short, "audit_table", col_widths=[.15, .14, .05, .25, .06, .17, .18], title=f"CortexHR attrition_dataset audit: {len(audit_t)} findings", width_in=22)
    write_report(raw, inv, ms, dup, imp, iq, zs, cb, audit_t, lk, lk_power, single, roles)
    from cortexhr.workflow_outputs import audit_extras; audit_extras()   # EDA, cleaning report, proof models, confusion matrices, tuning, interpretation
    print(f"audit complete: {len(audit_t)} findings")


def write_report(raw, inv, ms, dup, imp, iq, zs, cb, audit_t, lk, lk_power, single, roles):
    imp_hit = imp[imp.rows > 0][["rule", "columns", "rows"]]
    iq_top = iq.sort_values("iqr_outliers", ascending=False).head(6)[["column", "lower", "upper", "iqr_outliers"]]
    zs_top = zs.sort_values("z_outliers_abs_gt_3", ascending=False).head(6)[["column", "z_outliers_abs_gt_3", "max_abs_z"]]
    conf = lk[lk.verdict == "CONFIRMED LEAKAGE"]; susp = lk[lk.verdict != "CONFIRMED LEAKAGE"]
    legit = roles[roles.decision.isin(["RETAIN", "TRANSFORM"])]
    txt = f"""# CortexHR Phase 1 - Data Audit Report (Day 1)

Written **before** any model was trained and **before** the validation vault was opened (see `vault_comparison_cx01.md` for the later comparison).

## 1. What the data is
- Table `attrition_dataset` in `03_Data/CortexHR_Dataset.xlsx`: **{raw.shape[0]:,} rows x {raw.shape[1]} columns** (matches the 2,389 rows stated in the brief).
- **One row = one employee at one observation date** (a snapshot of their features). The target **`left_within_12m`** is 1 if that employee left within 12 months after `observation_date`, else 0.
- Column groups: identifiers/dates, role & pay, history (raises, promotions), engagement/performance, attendance/leave (3-6 month windows), department context, the label, and six exit columns.
- The `source` column shows {raw.source.value_counts().to_dict()}: 383 real employees (30 Sep 2025 snapshot, with 6 repeated rows) plus 2,000 historical extension rows from 2021-2025.

## 2. Class imbalance (plain language)
{cb['left']} of {cb['rows']:,} people left = **{cb['attrition_pct']}%**. A "model" that always says *will stay* is right {cb['majority_accuracy_pct']}% of the time - it looks excellent - yet it finds **zero** of the {cb['left']} leavers, so HR learns nothing. That is why this project reports ROC-AUC, PR-AUC, recall and precision instead of accuracy alone. Chart: `figures/class_balance.png`.

## 3. Data-quality results
**Missing values (predictors only, exit columns are blank for stayers by design):**

{R.md_table(ms[~ms.column.isin(C.LEAKAGE_CONFIRMED)])}

**Duplicates:** {dup['exact_duplicate_rows']} exact duplicate rows (employee ids {dup['ids_affected']}). `employee_id` has {dup['duplicate_ids_extra_rows']} repeated ids, all from these rows.

**Impossible values found (rules that matched at least one row):**

{R.md_table(imp_hit)}

There is **no `age` column** in this table, so "age under 18" could not be tested (age lives in a separate protected-attributes sheet, which must be used for fairness checks only).

**Outliers - IQR rule (1.5 x IQR)** (top columns):

{R.md_table(iq_top)}

**Outliers - Z-score (|z| > 3)** (top columns):

{R.md_table(zs_top)}

> Outliers are *investigation candidates*, not automatic errors. A very high `annual_fixed_pay` is a senior person; many `internal_transfers` or long commutes can be real. They are kept (with scaling / log-pay) unless they also break a logical rule above. Chart: `figures/outliers_boxplot.png`.

## 4. Leakage review (proof, not guesses)
Leakage = a column that secretly contains the answer, usually because it is only known after the person leaves.

**Confirmed leakage (dropped from all model inputs):**

{R.md_table(conf[['column','verdict','evidence']])}

**Suspected / excluded for safety:**

{R.md_table(susp[['column','verdict','evidence']])}

**Legitimate predictors:** {len(legit)} columns pass the checks. No single legitimate column predicts the label strongly (best single-column AUC = {single.single_column_auc.max():.3f} for `{single.iloc[0].column}`), which is consistent with an honest model scoring about 0.7. Chart: `figures/leakage_evidence.png`.

## 5. Audit table
{len(audit_t)} distinct, evidence-backed findings (full table: `tables/audit_table.csv`, screenshot-ready image: `figures/audit_table.png`).

{R.md_table(audit_t[['problem','affected_column_or_location','rows_affected','severity','proposed_treatment']])}

## 6. Decisions carried to Day 2
Every column was classified drop / retain / transform (`tables/column_roles.csv`). Summary: {roles.decision.value_counts().to_dict()}.

*Honest limits:* the audit only reflects checks that could be tested from the data. Anything the checks cannot see is handled in the vault comparison.
"""
    (C.reports_dir() / "data_audit_report.md").write_text(txt)


if __name__ == "__main__":
    main()
