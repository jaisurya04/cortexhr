"""Per-error-type payroll evaluation against the AUTHORISED vault file (Task 6, step 5).

    python scripts/evaluate_cx06.py [path/to/unlock_CX-06.csv]

Run it only after predictions are frozen (reports/phase2/payroll_predictions_frozen.csv).  The file is never used for
training or tuning.  If the file is absent the script says so and exits: it never invents labels.
Column mapping is auto-detected and printed; pass --id/--type/--label to override.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr.phase2.payroll import evaluate_by_type  # noqa: E402


def pick(cols, names):
    low = {c.lower(): c for c in cols}
    return next((low[n] for n in names if n in low), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(ROOT / "04_Validation_Vault" / "unlock_CX-06.csv"))
    ap.add_argument("--id"); ap.add_argument("--type"); ap.add_argument("--label")
    a = ap.parse_args()
    path = Path(a.path)
    if not path.exists():
        print(f"BLOCKED: {path} not found. Obtain the authorised file; do not create or guess labels."); return 2
    labels = pd.read_csv(path)
    idc = a.id or pick(labels.columns, ["payroll_item_id", "record_id", "item_id", "id"])
    typ = a.type or pick(labels.columns, ["error_type", "anomaly_type", "injected_error_type", "type"])
    lab = a.label or pick(labels.columns, ["is_error", "is_anomaly", "injected", "label"])
    print("columns:", labels.columns.tolist(), "-> id:", idc, "| type:", typ, "| label:", lab)
    if not idc or not typ:
        print("Could not map the id/type columns safely. Pass --id and --type."); return 3
    pred = pd.read_csv(ROOT / "reports" / "phase2" / "payroll_predictions_frozen.csv")
    res = evaluate_by_type(pred, labels, id_col=idc, type_col=typ, label_col=lab)
    out = ROOT / "reports" / "phase2" / "payroll_cx06_per_type.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False)); print("saved", out)
    absent = 8 - (len(res) - 1)
    if absent > 0:
        print(f"NOTE: only {len(res) - 1} error types found; report the other {absent} as unavailable, do not omit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
