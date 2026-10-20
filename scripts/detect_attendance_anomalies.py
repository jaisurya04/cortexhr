"""Day 5: attendance anomalies. Run: python scripts/detect_attendance_anomalies.py [--labels path]
Flags are saved BEFORE any label file is read; labels are used for measurement only."""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cortexhr.runs import run_attendance
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", default=None); a = ap.parse_args()
    o = run_attendance(a.labels)
    print("flagged:", int(o["flags"].flagged.sum()), "| labels:", o["labels_path"] or "NOT SUPPLIED -> not measured")
    if o["measured"]: print(o["measured"]["overall"].round(3).to_string())
    print(o["daily_proxy"].round(3).to_string())
