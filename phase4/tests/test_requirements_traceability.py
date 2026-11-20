"""The traceability matrix may not point at files or tests that do not exist."""
import csv
import re

from _common import ROOT


def test_every_row_points_to_real_files_and_real_tests():
    rows = list(csv.DictReader((ROOT / "docs" / "requirements_traceability.csv").open(encoding="utf-8")))
    assert len(rows) >= 100
    problems = []
    for r in rows:
        for field in ("implementation", "evidence"):
            for part in [p.strip() for p in r[field].split(";") if p.strip()]:
                path = re.split(r" \(| section", part)[0].replace("*", "").strip()
                if path.startswith(("/v1", "src/cortexhr_phase4/console/app.py (")) or " " in path and "/" not in path:
                    continue
                if not (path.startswith(("src/", "scripts/", "tests/", "docs/", "outputs/", "deploy/", "assets/", "README", "RELEASE", "requirements", ".env", "console/", "services/", "monitoring/", "forecasting/", "core/", "api/")) or path in ("this file",)):
                    continue
                cands = [ROOT / path, ROOT / "src" / "cortexhr_phase4" / path]
                if path.endswith("/"):
                    ok = any(c.is_dir() for c in cands)
                else:
                    ok = any(c.exists() for c in cands) or "this file" in path
                if not ok and "/" in path and path.split("/")[0] in ("outputs",):
                    ok = (ROOT / path.split("/")[0]).exists()
                if not ok:
                    problems.append(f"{r['task_id']}: missing {path}")
        t = r["test"]
        if t:
            f, fn = t.split("::")
            src = (ROOT / f).read_text(encoding="utf-8")
            if f"def {fn}(" not in src:
                problems.append(f"{r['task_id']}: test {t} does not exist")
        assert r["status"] in ("PASS", "PARTIAL", "FAIL", "BLOCKED", "USER ACTION REQUIRED"), r["task_id"]
        for f in ("task_id", "requirement", "implementation", "required_correction", "validation_command", "actual_result", "evidence", "status"):
            assert r[f].strip(), (r["task_id"], f)
    assert not problems, problems[:15]


def test_every_pass_row_names_a_test_that_exists():
    """PASS-by-existence is impossible: a PASS row must reference a real test function. (build_traceability.py additionally computes
    each status from outputs/last_test_run.json, so regenerate the matrix after a test run on your machine: python scripts/build_traceability.py)"""
    rows = list(csv.DictReader((ROOT / "docs" / "requirements_traceability.csv").open(encoding="utf-8")))
    for r in rows:
        if r["status"] == "PASS":
            f, fn = r["test"].split("::")
            assert f"def {fn}(" in (ROOT / f).read_text(encoding="utf-8"), r["task_id"]
