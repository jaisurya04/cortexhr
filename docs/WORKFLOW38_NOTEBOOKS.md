# 38-stage workflow in every Phase 2 notebook

Every notebook in `05_Starter/notebooks/phase2/` (16 files) now ends with the complete
"End-to-End Data Science Workflow for Classification" (`docs/CLASSIFICATION_WORKFLOW_38.md`):
**39 markdown cells** (1 index + one per section, each listing all of its checklist items) and **39 code cells**
(one setup cell + one evidence cell per section). Original cells are untouched.

| Track | Notebooks |
|-------|-----------|
| Payroll anomaly (binary: payslip error vs normal) | 07, 08, 09, 10, 11, 12, 13, P2_T06, P2_T07, P2_T08 |
| HR chatbot (33-class intent) | 14, 15, 16, P2_MASTER_37, P2_T09, P2_T10 |

Rebuild everything: `python scripts/build_phase2_workflow38.py` (about 15 minutes, runs every cell for real).
Checks: `python tests_phase2/run_all.py` (includes tests that every notebook has all 38 stages and every item).

Rules the new cells follow
* The CX-06 validation vault and the official chatbot test file are never used for a modelling decision.
  Payroll development uses clearly labelled synthetic error injection; the chatbot uses a paraphrase-family-grouped split.
* `models/` and `03_Data/` are never modified; demo artefacts go to `reports/phase2/workflow38/`.
* Where something cannot be done here (CX-06 labels, FastAPI not installed, production approval) the cell says
  BLOCKED / SKIPPED / MANUAL instead of pretending.

Findings the workflow surfaced (all printed in the notebooks)
* Chatbot: ordinary cross-validation gives macro-F1 ~0.99 but paraphrase-family-grouped CV gives ~0.64 (templated near-duplicates leak).
  Sealed test: 81.06% accuracy, which is optimistic because the recipe was developed while looking at that file. 90% is not reached.
* Chatbot: a polite filler prefix ("hello please ... thanks a lot") collapses accuracy; Tanglish works only partly.
* Payroll: the forest alone has weak recall; rules + forest is needed; INFLATED_VARIABLE is only visible to the forest.
* Payroll: a 6x salary payslip is flagged because raw-amount features are in the model (a ratios-only set did better model-only).
