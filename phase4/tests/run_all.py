"""Run every Phase 4 test and print an honest summary:  python phase4/tests/run_all.py [-k substring] [-v]

PASS = executed and passed, FAIL = executed and failed, SKIP = NOT executed (reason shown). Also proves that running the tests
did not change a single file in phase1/, phase2/ or phase3/.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, Skip  # noqa: E402

REPO = ROOT.parent


def manifest() -> dict[str, str]:
    out = {}
    for ph in ("phase1", "phase2", "phase3"):
        for f in sorted((REPO / ph).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                out[str(f.relative_to(REPO))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


try:                                             # pytest's skip outcome, when pytest is installed
    from _pytest.outcomes import Skipped as _PytestSkipped
    SKIPPED = (Skip, _PytestSkipped)
except ImportError:
    SKIPPED = (Skip,)


def main() -> int:
    args = sys.argv[1:]
    pat = args[args.index("-k") + 1] if "-k" in args else ""
    verbose = "-v" in args
    before = manifest()
    res = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    lines, fails, results = [], [], {}
    t_all = time.time()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name in sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))):
            if pat and pat not in f"{path.stem}::{name}":
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                st, note = "PASS", ""
            except SKIPPED as s:             # our Skip, or pytest's Skipped (a BaseException, so `except Exception` would miss it)
                st, note = "SKIP", str(s)
            except AssertionError as e:
                st, note = "FAIL", f"AssertionError: {e}"
                fails.append((f"{path.stem}::{name}", traceback.format_exc()))
            except Exception as e:  # noqa: BLE001
                st, note = "FAIL", f"{type(e).__name__}: {e}"
                fails.append((f"{path.stem}::{name}", traceback.format_exc()))
            res[st] += 1
            results[f"{path.stem}::{name}"] = {"status": st, "note": note, "seconds": round(time.time() - t0, 2)}
            lines.append(f"{st:4s} {path.stem}::{name} ({time.time() - t0:.1f}s){'  - ' + note if note else ''}")
    after = manifest()
    if not pat:                                   # machine-readable record of THIS run; scripts/build_traceability.py computes statuses from it
        import json
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "last_test_run.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "phase1_3_files": len(before), "results": results}, indent=1), encoding="utf-8")
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    print("\n".join(lines))
    for name, tb in fails:
        print(f"\n--- {name}\n{tb}")
    print(f"\nPhase 1-3 files checked: {len(before)}; changed during this run: {len(changed)} {changed[:5] if changed else ''}")
    skipped = [ln for ln in lines if ln.startswith("SKIP")]
    blocked = [ln for ln in skipped if "not installed" in ln or "model_unavailable" in ln]          # could not run because a dependency/model is missing here
    other_skips = [ln for ln in skipped if ln not in blocked]
    users = _user_actions()
    print("\n" + "=" * 78)
    print(f"PASSED   {res['PASS']}   (executed and passed)")
    print(f"FAILED   {res['FAIL']}")
    print(f"SKIPPED  {len(other_skips)}   (not executed; never counted as passed)")
    print(f"BLOCKED  {len(blocked)}   (a dependency could not be installed or a model was unavailable in this environment)")
    for ln in blocked:
        print("           -", ln.split("(")[0].strip()[5:], "|", ln.split("  - ", 1)[-1])
    print(f"USER ACTIONS REMAINING  {len(users)}   (need your account, phone, vault or judgement; see docs/requirements_traceability.csv)")
    for u in users:
        print("           -", u)
    print(f"Phase 1-3 files checked: {len(before)}; changed during this run: {len(changed)} {changed[:5] if changed else ''}   ({time.time() - t_all:.0f}s)")
    ready = not (res["FAIL"] or changed or blocked or other_skips or users or not before)
    print("RELEASE GATE: " + ("all executed checks passed" if ready else "NOT READY - see FAILED / BLOCKED / USER ACTIONS above"))
    print("=" * 78)
    if not before:                                # nothing to compare: the integrity gate was NOT executed, so it must not look like a pass
        print("BLOCKED: no Phase 1-3 files found next to phase4/ - the Phase 1-3 integrity gate and every test that needs their models/data did not run properly.")
        return 2
    return 1 if res["FAIL"] or changed else (2 if blocked else 0)


def _user_actions() -> list[str]:
    import csv
    p = ROOT / "docs" / "requirements_traceability.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    key = "status" if rows and "status" in rows[0] else None
    return [f"{r.get('task_id', r.get('id', '?'))}: {(r.get('requirement') or r.get('requirement (verbatim for the task images)') or '')[:90]}" for r in rows if key and r[key] == "USER ACTION REQUIRED"]


if __name__ == "__main__":
    sys.exit(main())
