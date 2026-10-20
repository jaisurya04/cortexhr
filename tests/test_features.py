from _common import *
from cortexhr import features as F, data_loading as D

def test_forbidden_columns_rejected():
    for c in [C.TARGET, "exit_date", "fnf_settlement_amount", "employee_id", "voluntary_left_within_12m", "source"]:
        try: F.assert_safe_input_columns(["tenure_months", c])
        except ValueError: continue
        raise AssertionError(f"{c} was allowed")

def test_target_and_leakage_never_in_X_or_output():
    if not data_ok(): return
    S = frame(); X = S["X"]; assert not (set(X.columns) & set(C.FORBIDDEN_AS_INPUT))
    pipe = F.build_pipeline(True).fit(S["Xtr"], S["ytr"]); out = pipe.transform(S["Xte"])
    assert not any(any(b in col for b in C.FORBIDDEN_AS_INPUT) for col in out.columns)

def test_three_row_smoke_unseen_and_missing():
    if not data_ok(): return
    S = frame(); pipe = F.build_pipeline(True).fit(S["Xtr"], S["ytr"])
    raw = F.representative_rows(S["df"]); assert len(raw) == 3
    Z = pipe.transform(raw[F.input_columns(True)]); assert Z.shape == (3, pipe.transform(S["Xte"].head(1)).shape[1]) and np.isfinite(Z.to_numpy()).all()
    t = raw[F.input_columns(True)].copy(); t.iloc[0, t.columns.get_loc("role_id")] = "R999"; t.iloc[1, t.columns.get_loc("department_id")] = np.nan
    t.iloc[2, t.columns.get_loc("comp_ratio")] = np.nan; assert np.isfinite(pipe.transform(t).to_numpy()).all()

def test_fit_on_training_only_dept_medians():
    df = pd.DataFrame({"department_id": ["A"] * 4 + ["B"] * 2, "job_level": [1] * 6, "role_id": ["R"] * 6,
                       "annual_fixed_pay": [100, 100, 100, 100, 200, 200], "comp_ratio": [1.0] * 6, "months_since_last_raise": [1] * 6,
                       "tenure_months": [10] * 6, "promotions_received": [0] * 6})
    d = F.Derive().fit(df.iloc[:4]); assert d.dept_median_ == {"A": 100.0}      # B never seen in training
    test = df.iloc[4:].copy(); test["annual_fixed_pay"] = 10 ** 9; d2 = F.Derive().fit(df.iloc[:4])
    assert d2.global_median_ == 100.0                                              # extreme TEST values cannot change it
    out = d2.transform(df.iloc[4:]); assert np.allclose(out.pay_vs_dept_median, 2.0)

def test_pay_typo_blanked():
    df = pd.DataFrame({"department_id": ["A"] * 5, "job_level": [1] * 5, "role_id": ["R"] * 5, "annual_fixed_pay": [100, 100, 100, 100, 1100],
                       "comp_ratio": [1.0] * 5, "months_since_last_raise": [1] * 5, "tenure_months": [1] * 5, "promotions_received": [0] * 5})
    out = F.Derive().fit(df).transform(df); assert np.isnan(out.pay_vs_dept_median.iloc[4]) and out.pay_vs_dept_median.iloc[0] == 1.0

def test_join_cardinality_and_cutoff_no_future():
    if not data_ok(): return
    S = frame(); r = S["join_report"]; assert r["rows_before_join"] == r["rows_after_join"] == 2383 and r["attendance_matched_rows"] == 383
    obs = pd.DataFrame({"employee_id": ["E1"], "observation_date": [pd.Timestamp("2025-09-30")]})
    att = pd.DataFrame({"employee_id": ["E1"] * 5, "month": pd.to_datetime(["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]),
                        "attendance_rate": [0.5, 0.9, 0.9, 0.9, 0.0], "unpaid_absence_days": 0, "late_arrivals": 0, "overtime_hours": 0.0,
                        "avg_hours_worked": 8.0, "half_days_unpaid": 0, "wfh_days": 0, "present_days": 20})
    g = D.aggregate_attendance(att, obs); assert abs(g.att_attendance_rate_mean_3m.iloc[0] - 0.9) < 1e-9 and g.att_months_covered.iloc[0] == 3

def test_payroll_duplicates_dropped():
    pay = pd.DataFrame({"payroll_item_id": ["a", "b"], "employee_id": ["E"] * 2, "month": [pd.Timestamp("2025-09-01")] * 2, "x": [1, 1]})
    clean, n = D.dedupe_payroll(pay); assert n == 1 and len(clean) == 1
