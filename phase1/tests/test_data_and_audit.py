from _common import *
from cortexhr import data_loading as D, audit as A

def test_raw_shape_and_target():
    if not data_ok(): return
    a = D.load_attrition(); assert a.shape == (2389, 42) and int(a[C.TARGET].sum()) == 349
    assert abs(a[C.TARGET].mean() - 0.146) < 0.001

def test_duplicates_removed_and_cleaning():
    if not data_ok(): return
    a, log = D.prepare_attrition(D.load_attrition()); assert log["exact_duplicates_dropped"] == 6 and len(a) == 2383
    c = D.clean_raw(a); assert (c.tenure_months.dropna() >= 0).all() and c.wfh_share_3m.between(0, 1).all()
    assert c.work_location.nunique() == 9 and c.attendance_rate_3m.max() <= 1

def test_audit_has_at_least_eight_findings_and_leakage_proof():
    if not data_ok(): return
    t, lk, sc = A.build_audit_table(D.load_attrition(), D.load_sheet(C.SHEET_ATT), D.load_sheet(C.SHEET_PAY))
    assert len(t) >= 8 and t.problem.is_unique
    assert set(lk[lk.verdict == "CONFIRMED LEAKAGE"].column) == set(C.LEAKAGE_CONFIRMED)
    assert sc.single_column_auc.max() < 0.9

def test_outlier_helpers_synthetic():
    df = pd.DataFrame({"x": list(range(1, 30)) + [1000]})
    assert A.iqr_outliers(df, ["x"]).iqr_outliers.iloc[0] == 1 and A.zscore_outliers(df, ["x"]).z_outliers_abs_gt_3.iloc[0] >= 1

def test_pay_typo_check_finds_three():
    if not data_ok(): return
    assert len(A.pay_consistency_outliers(D.load_attrition())) == 3
