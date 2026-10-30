"""Task 6 and 7: payroll features, rules, detector, reasons, money at risk, official format."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from _helpers import CSV, clean_row, payroll_df, payroll_scored
from cortexhr.phase2 import payroll as P


def test_audit_and_features_are_finite():
    df = payroll_df()
    audit = P.audit_payroll(df)
    assert audit["rows"] == 7355 and audit["duplicate_employee_run_pairs"] == 15
    x = P.engineer_payroll_features(df)
    block = x[["gross_to_fixed_ratio", "deduction_ratio", "net_to_gross_ratio", "gross_vs_expected_ratio"]]
    assert np.isfinite(block.to_numpy()).all()


def test_expected_gross_formula_explains_almost_every_row():
    df = payroll_df()
    off = ((df.gross_pay - P.expected_gross(df)).abs() > np.maximum(1.0, 0.002 * P.expected_gross(df).abs())).sum()
    assert off <= 40, f"{off} rows deviate from the expected-gross formula"


def test_each_rule_fires_on_its_own_error():
    scored, det = payroll_scored()
    base = clean_row(0)
    cases = {
        "NEGATIVE_NET_PAY": dict(net_pay=-100.0),
        "ZERO_PAY": dict(gross_pay=0.0, net_pay=0.0, total_deductions=0.0),
        "GROSS_ABOVE_NORM": dict(gross_pay=float(base.gross_pay.iloc[0]) * 10),
        "GROSS_BELOW_EXPECTED": dict(gross_pay=float(base.gross_pay.iloc[0]) * 0.88),
        "NET_PAY_MISMATCH": dict(net_pay=float(base.net_pay.iloc[0]) * 0.9),
        "OVERTIME_ABOVE_CAP": dict(overtime_hours_paid=70.0),
        "DEDUCTIONS_EXCEED_GROSS": dict(total_deductions=float(base.gross_pay.iloc[0]) * 1.5),
    }
    for code, change in cases.items():
        row = base.copy()
        for k, v in change.items():
            row[k] = v
        flags = P.payroll_rule_flags(row, reference=det.reference)
        assert bool(flags[code].iloc[0]), f"{code} did not fire for {change}"
    assert not P.payroll_rule_flags(base, reference=det.reference).rule_flagged.iloc[0]


def test_pf_not_deducted_uses_employee_history():
    scored, det = payroll_scored()
    stats = det.reference.emp_stats
    emp = stats[stats.pf_paid_share >= 0.95].index[0]
    row = clean_row(1); row["employee_id"] = emp; row["pf_employee"] = 0.0
    assert bool(P.payroll_rule_flags(row, reference=det.reference).PF_NOT_DEDUCTED.iloc[0])
    row["employee_id"] = "EMP-UNKNOWN"          # no history -> the rule must stay quiet, not guess
    assert not bool(P.payroll_rule_flags(row, reference=det.reference).PF_NOT_DEDUCTED.iloc[0])


def test_single_row_score_is_not_collapsed_to_zero():
    """Regression for the old min-max-on-one-row bug: a broken row must score high, a clean row low."""
    scored, det = payroll_scored()
    bad = clean_row(2); bad["net_pay"] = -500.0
    bad_out, _ = P.score_payroll(bad, model=det)
    assert bad_out.anomaly_score.iloc[0] >= 0.9 and bool(bad_out.flagged.iloc[0])
    good = clean_row(3)
    good_out, _ = P.score_payroll(good, model=det)
    assert good_out.anomaly_score.iloc[0] < 0.6 and not bool(good_out.flagged.iloc[0])


def test_scoring_one_row_equals_scoring_it_inside_a_batch():
    scored, det = payroll_scored()
    rows = pd.concat([clean_row(4), clean_row(5), clean_row(6)])
    batch, _ = P.score_payroll(rows, model=det)
    for _, r in rows.iterrows():
        single, _ = P.score_payroll(pd.DataFrame([r]), model=det)
        in_batch = batch[batch.payroll_item_id == r.payroll_item_id].anomaly_score.iloc[0]
        assert abs(single.anomaly_score.iloc[0] - in_batch) < 1e-9


def test_reason_text_is_specific_and_never_the_old_generic_fallback():
    scored, det = payroll_scored()
    assert "validation rules triggered" not in " ".join(scored.reason_text)
    ok = scored[~scored.flagged]
    assert set(ok.reason_code) == {P.NOT_FLAGGED} and set(ok.reason_text) == {"No issue found."}
    row = clean_row(7); row["net_pay"] = -9000.0
    out, _ = P.score_payroll(row, model=det)
    codes = out.reason_code.iloc[0].split("|")
    assert "NEGATIVE_NET_PAY" in codes and "NET_PAY_MISMATCH" in codes
    text = out.reason_text.iloc[0]
    assert "below zero" in text and "arithmetic" in text            # one sentence per code, joined
    flagged = scored[scored.flagged]
    assert (flagged.reason_text.str.len() > 30).all() and flagged.reason_code.ne(P.NOT_FLAGGED).all()


def test_money_at_risk_is_the_gap_not_just_net_pay():
    scored, det = payroll_scored()
    row = clean_row(8); gross = float(row.gross_pay.iloc[0]); expected = float(P.expected_gross(row).iloc[0])
    row["gross_pay"] = gross * 10
    out, _ = P.score_payroll(row, model=det)
    assert abs(out.money_at_risk.iloc[0] - (gross * 10 - expected)) < 5, "gross-above-norm exposure is gross minus expected"
    row2 = clean_row(9); row2["gross_pay"] = 0.0; row2["net_pay"] = 0.0; row2["total_deductions"] = 0.0
    out2, _ = P.score_payroll(row2, model=det)
    assert out2.money_at_risk.iloc[0] > 0, "zero pay still has money at risk: what should have been paid"
    flagged = scored[scored.flagged]
    assert flagged.money_at_risk.is_monotonic_decreasing, "flags must be ranked by money at risk"
    assert (scored[~scored.flagged].money_at_risk == 0).all()


def test_duplicates_work_for_one_api_row_and_in_batch():
    scored, det = payroll_scored()
    df = payroll_df()
    existing = df.iloc[[10]].copy()
    again, _ = P.score_payroll(existing, model=det)
    assert "DUPLICATE_PAYMENT" not in again.reason_code.iloc[0], "re-scoring a known row is not a duplicate"
    copy = existing.copy(); copy["payroll_item_id"] = "API-COPY"
    out, _ = P.score_payroll(copy, model=det)
    assert "DUPLICATE_PAYMENT" in out.reason_code.iloc[0] and out.money_at_risk.iloc[0] >= abs(float(copy.net_pay.iloc[0])) - 1
    new1 = clean_row(10); new2 = new1.copy(); new2["payroll_item_id"] = "TEST-10b"
    batch, _ = P.score_payroll(pd.concat([new1, new2]), model=det)
    assert batch.DUPLICATE_PAYMENT.all()
    assert sorted(batch.money_at_risk)[0] == 0 or sorted(batch.money_at_risk)[0] < sorted(batch.money_at_risk)[1]


def test_official_format_matches_the_sample_sheet_exactly():
    scored, det = payroll_scored()
    sample = pd.read_csv(CSV / "sample_anomaly_results.csv")
    out = P.to_official_format(scored)
    assert list(out.columns) == list(sample.columns)
    assert set(out.flagged) <= {"Y", "N"} and out.result_id.is_unique
    y = out[out.flagged == "Y"]
    assert (out.flagged.iloc[:len(y)] == "Y").all(), "flagged rows must come first"
    assert y.money_at_risk.is_monotonic_decreasing
    assert y.result_id.iloc[0] == "P2-00001"
    assert pd.to_numeric(out.money_at_risk).notna().all()


def test_detector_roundtrip_and_no_labels():
    scored, det = payroll_scored()
    with tempfile.TemporaryDirectory() as d:
        path = det.save(Path(d) / "m.joblib")
        loaded = P.PayrollDetector.load(path)
        rows = clean_row(11)
        a, _ = P.score_payroll(rows, model=det); b, _ = P.score_payroll(rows, model=loaded)
        assert abs(a.anomaly_score.iloc[0] - b.anomaly_score.iloc[0]) < 1e-12
    import inspect
    assert not {"labels", "y", "target", "is_error", "error_type"} & set(inspect.signature(P.fit_isolation_forest).parameters)
    assert not any(c in P.FEATURE_COLS for c in ("is_error", "error_type", "is_anomaly", "label"))
    df = payroll_df().head(300)
    P.fit_isolation_forest(df.drop(columns=[c for c in ("is_error", "error_type") if c in df.columns]))


def test_evaluate_by_type_on_a_SYNTHETIC_injection_sanity_check():
    """Sanity check of the detector and of evaluate_by_type on errors injected here.  NOT the CX-06 vault result."""
    scored, det = payroll_scored()
    df = payroll_df()
    ok_ids = scored[~scored.flagged].payroll_item_id.tolist()
    pf_emps = set(det.reference.emp_stats[det.reference.emp_stats.pf_paid_share >= 0.95].index)
    pool = df[df.payroll_item_id.isin(ok_ids) & df.employee_id.isin(pf_emps) & (df.pf_employee > 0) & (df.gross_pay > 20000)]
    pool = pool.sample(frac=1.0, random_state=1).reset_index(drop=True)
    n, labels, frames, i = 10, [], [], 0

    def take():
        nonlocal i
        r = pool.iloc[i:i + n].copy(); i += n
        return r
    base = df.copy(); idx = base.set_index("payroll_item_id")

    def mutate(kind, rows, fn):
        for pid in rows.payroll_item_id:
            fn(idx.loc[pid:pid] if False else None, pid)
            labels.append({"payroll_item_id": pid, "error_type": kind, "is_error": 1})
    work = df.set_index("payroll_item_id")
    for pid in take().payroll_item_id:
        work.loc[pid, "net_pay"] = -abs(work.loc[pid, "net_pay"]) * 0.05 - 1; labels.append({"payroll_item_id": pid, "error_type": "negative_net", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, ["gross_pay", "net_pay", "total_deductions"]] = 0.0; labels.append({"payroll_item_id": pid, "error_type": "zero_pay", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, "gross_pay"] *= 10; labels.append({"payroll_item_id": pid, "error_type": "gross_x10", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, "gross_pay"] *= 0.88; labels.append({"payroll_item_id": pid, "error_type": "gross_low", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, "overtime_hours_paid"] = 70.0; labels.append({"payroll_item_id": pid, "error_type": "ot_cap", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, "pf_employee"] = 0.0; labels.append({"payroll_item_id": pid, "error_type": "pf_missing", "is_error": 1})
    for pid in take().payroll_item_id:
        work.loc[pid, "net_pay"] *= 0.9; labels.append({"payroll_item_id": pid, "error_type": "net_mismatch", "is_error": 1})
    dups = take().copy(); dups["payroll_item_id"] = ["DUP-%d" % k for k in range(len(dups))]
    for pid in dups.payroll_item_id:
        labels.append({"payroll_item_id": pid, "error_type": "duplicate", "is_error": 1})
    data = pd.concat([work.reset_index(), dups], ignore_index=True)
    pred, _ = P.score_payroll(data, model=det)
    result = P.evaluate_by_type(pred, pd.DataFrame(labels), label_col="is_error")
    by = result.set_index("error_type")
    assert set(by.index) == {"negative_net", "zero_pay", "gross_x10", "gross_low", "ot_cap", "pf_missing", "net_mismatch", "duplicate", "overall"}
    assert (by.drop(index="overall").recall == 1.0).all(), by[["support", "tp", "recall"]]
    assert by.loc["overall", "support"] == 80 and by.loc["overall", "rules_recall"] == 1.0
    assert (by.drop(index="overall").support == 10).all()


def test_parse_month():
    assert P.parse_month("2026-09") == pd.Timestamp("2026-09-01")
    assert P.parse_month("2026-09-15") == pd.Timestamp("2026-09-01")
    for bad in ("2026-13", "September", "2026/09", "", "26-09"):
        try:
            P.parse_month(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


def test_model_only_flags_name_the_unusual_figures_instead_of_one_generic_sentence():
    """Task 7: 'the model said so' is not a reason. Model-only flags must quote the figures that look odd."""
    scored, _ = payroll_scored()
    m = scored[scored.flagged & (scored.reason_code == P.MODEL_CODE)]
    assert len(m) > 100
    assert m.reason_text.str.contains(r"\(typical |most payslips have none", regex=True).mean() > 0.95
    assert m.reason_text.nunique() > 50, "reasons must differ from row to row"
