"""Phase 1-3 must be untouched. (run_all.py also hashes them before and after the whole run.)"""
import os
import subprocess
from pathlib import Path

from _common import ROOT, skip

REPO = ROOT.parent
SRC = ROOT / "src" / "cortexhr_phase4"


def test_no_phase4_source_writes_into_phase1_to_3():
    """Static check: no write call in phase4 code targets the PHASE1/2/3 constants."""
    bad = []
    for f in SRC.rglob("*.py"):
        txt = f.read_text(encoding="utf-8")
        for i, line in enumerate(txt.splitlines(), 1):
            if any(w in line for w in ("write_text", "write_bytes", ".to_csv(", "joblib.dump", "open(")) and any(p in line for p in ("PHASE1", "PHASE2", "PHASE3", '"phase1"', '"phase2"', '"phase3"')):
                bad.append(f"{f.name}:{i}: {line.strip()[:100]}")
    assert not bad, bad


def test_phase4_has_no_files_outside_its_folder_and_repo_phases_match_baseline():
    pristine = os.environ.get("CORTEXHR_PRISTINE_REPO")
    if not pristine:
        skip("set CORTEXHR_PRISTINE_REPO to an untouched copy of the repository to compare byte by byte (the git check is in scripts/verify_phase1_3_unchanged.py)")
    import hashlib
    for ph in ("phase1", "phase2", "phase3"):
        a, b = REPO / ph, Path(pristine) / ph
        fa = sorted(x.relative_to(a) for x in a.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
        fb = sorted(x.relative_to(b) for x in b.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
        extra, missing = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
        # FAIL only when a pristine file is missing or its bytes differ. Files that exist only in YOUR copy (local data such as an .xlsx,
        # .gitignore, .pytest_cache from running pytest earlier) are reported, never hidden, but cannot be "changed Phase 1-3 files".
        assert not missing, f"{ph}: files missing from your copy: {[str(x) for x in missing[:8]]}"
        if extra:
            shown = [str(x) for x in extra if not any(part.startswith(".") for part in x.parts)]
            print(f"NOTE {ph}: {len(extra)} files exist only in your copy (hidden/cache files: {len(extra) - len(shown)}); visible ones: {shown[:10]}")
        for rel in fb:
            assert hashlib.sha256((a / rel).read_bytes()).digest() == hashlib.sha256((b / rel).read_bytes()).digest(), f"{ph}/{rel} changed"


def test_secrets_are_not_in_the_repository():
    hits = []
    for f in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.toml")) + list(ROOT.rglob("*.yml")) + list(ROOT.rglob("*.yaml")) + list(ROOT.rglob("*.txt")):
        if "outputs" in f.parts or ".git" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            low = line.lower().replace(" ", "")
            if any(k in low for k in ('api_key=\"', "api_key='", 'secret=\"', "secret='", 'password=\"', "password='", 'token=\"', "token='")) and "os.environ" not in low and "environ" not in low and "example" not in f.name and "s3cret" not in low and "$" not in low:
                hits.append(f"{f.relative_to(ROOT)}:{i}")
    assert not hits, hits


def test_requirements_are_exactly_pinned():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            assert "==" in line, f"not pinned: {line}"
