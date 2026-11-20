"""Prove that Phase 1, 2 and 3 are byte-for-byte what was submitted.

    python phase4/scripts/verify_phase1_3_unchanged.py                 # compares against the git tag phase-3 (default)
    python phase4/scripts/verify_phase1_3_unchanged.py --base <commit-or-tag>
    python phase4/scripts/verify_phase1_3_unchanged.py --pristine <folder with an untouched copy of the repo>   # no git needed

Exit code 0 = unchanged. Anything else prints the files that differ.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PHASES = ("phase1", "phase2", "phase3")


def git_check(base: str) -> int:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if run("rev-parse", "--verify", base).returncode:
        print(f"cannot find git revision {base!r}; use --base or --pristine")
        return 2
    diff = run("diff", "--name-status", base, "--", *PHASES).stdout.strip()
    new = run("ls-files", "--others", "--exclude-standard", "--", *PHASES).stdout.strip()
    if diff or new:
        print("Phase 1-3 differ from", base)
        print(diff or "", new or "", sep="\n")
        return 1
    print(f"OK: phase1, phase2, phase3 are identical to {base} (no modified, deleted or new files).")
    return 0


def tree_check(other: Path) -> int:
    bad = []
    for ph in PHASES:
        a, b = REPO / ph, other / ph
        ha = {p.relative_to(a): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        hb = {p.relative_to(b): hashlib.sha256(p.read_bytes()).hexdigest() for p in b.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        bad += [f"{ph}/{k}" for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)]
    if bad:
        print(f"{len(bad)} file(s) differ:", *bad[:30], sep="\n  ")
        return 1
    print("OK: phase1, phase2, phase3 are byte-for-byte identical to the pristine copy.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="phase-3")
    ap.add_argument("--pristine", type=Path)
    a = ap.parse_args()
    sys.exit(tree_check(a.pristine) if a.pristine else git_check(a.base))
