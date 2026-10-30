# Phase 2 payroll features (Task 6): what each column means and every ratio

## Columns of `payroll_monthly_features.csv`
| Column | Meaning |
|---|---|
| payroll_item_id, run_id, employee_id, month | Which payslip: unique item, payroll run (PR-YYYY-MM), employee and the pay month. Identifiers only; never model inputs. |
| department_id, role_id, job_level, variable_pay_type | Who and what kind of pay (context; not used as model features). |
| monthly_fixed_gross | Full-month fixed pay before loss of pay. |
| payable_days, lop_days | Days paid and loss-of-pay (unpaid) days in the month. |
| overtime_eligible, overtime_hours_paid, overtime_pay | Whether the role gets overtime, hours paid, and the amount (policy: 2x hourly, cap 50 h a month, FAQ-08). |
| variable_pay | Bonus, incentive or other variable amount. |
| gross_pay | Total earnings paid. |
| pf_employee, esi_employee, professional_tax, tds, other_deductions | The deduction components. |
| total_deductions | Should equal the sum of the five components. |
| net_pay | Should equal gross_pay minus total_deductions. |

## Engineered values and why each one makes odd rows stand out
| Value | Formula | Odd row it exposes |
|---|---|---|
| expected_gross | fixed gross x payable_days / days in month + overtime pay + variable pay | The yardstick for the next ratio. Only 35 of 7,355 rows differ by more than Rs 1. |
| gross_vs_expected_ratio | gross_pay / expected_gross | About 10 means an extra digit; about 0.88 means underpayment; 0 means nothing paid. |
| gross_to_fixed_ratio | gross_pay / monthly_fixed_gross | Pay far above or below the contract. |
| deduction_ratio | total_deductions / gross_pay | Deductions out of proportion (for example TDS blown up). |
| net_to_gross_ratio | net_pay / gross_pay | Employee keeps too little or too much. |
| net_identity_gap_ratio | abs(gross - deductions - net) / gross | The arithmetic of the payslip does not add up. |
| variable_to_fixed_ratio | variable_pay / monthly_fixed_gross | Unusually large bonus. |
| lop_day_ratio | lop_days / (payable_days + lop_days) | Many unpaid days. |
| overtime_pay_ratio | overtime_pay / monthly_fixed_gross | Overtime large compared with pay. |
| overtime_over_cap_hours | max(overtime_hours_paid - 50, 0) | Overtime beyond the policy cap. |
| fixed_gross_log, gross_pay_log, net_pay_log | log(1 + amount) | Puts small and large salaries on one scale. |

## Rules (readable, run before the model)
DUPLICATE_PAYMENT, NEGATIVE_NET_PAY, ZERO_PAY, GROSS_ABOVE_NORM, GROSS_BELOW_EXPECTED, NET_PAY_MISMATCH,
DEDUCTION_TOTAL_MISMATCH, DEDUCTIONS_EXCEED_GROSS, OVERTIME_ABOVE_CAP, OVERTIME_NOT_ELIGIBLE, PF_NOT_DEDUCTED,
NEGATIVE_COMPONENT. Each has a one-line meaning in `REASON_GLOSSARY` in `src/cortexhr/phase2/payroll.py`.
The IsolationForest never sees a label; its flags are labelled STATISTICAL_OUTLIER.
