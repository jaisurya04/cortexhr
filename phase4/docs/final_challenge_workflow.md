# Final Challenge - workflow only (the improvement itself is YOURS; nothing was built for you)

1. Safe starting point: `git switch main && git pull && git switch -c final-challenge/<short-name>` (never work on `main`). Confirm `python tests/run_all.py` is green first so you know what you start from.
2. Pick ONE improvement (candidates with evidence are in `docs/final_challenge_and_judgement.md`).
3. Write the hypothesis and the number you will measure BEFORE you code.
4. Build it without a guide; keep Phase 1-3 files untouched; add a test.
5. Measure before/after on the same split; keep the failures too.
6. Commit on the branch; open a pull request to yourself; do not merge before the evidence is written.

Record template (copy into `docs/final_challenge_and_judgement.md`)
```
Branch:
Hypothesis / metric:
What I decided (and what I rejected):
Before -> after (numbers, command that produced them):
What got worse or stayed unproven:
Why I decided this:
```
Status: **USER ACTION REQUIRED**.
