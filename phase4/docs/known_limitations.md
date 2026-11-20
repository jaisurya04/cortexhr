# Known limitations and disclosures

## About this build
1. **Runtime verification:** FastAPI/Pydantic endpoints were executed in this sandbox; the local HTTP smoke test passed 7/10 checks. The original Phase 2 payroll model artifact cannot be unpickled under sandbox pandas 2.2.3 (`StringDtype.__init__` signature mismatch), while the project pins pandas 3.0.2. Streamlit is not installed here, and Chromium could not be launched, so the Streamlit shell and desktop/mobile browser interactions remain unverified. The exact pinned Python 3.12 environment has not been reproduced in this sandbox.
2. **Saved forecast outputs are baseline-only** and say so (`improved_model_status: NOT RUN`). Re-run `scripts/run_forecast.py` after installing requirements; the improved model is then compared with the baseline and selected only if its rolling MAE is at least 1% lower.
3. **Streamlit specifics are unverified** (the Streamlit API changes quickly: tables use the default width and the dashboard uses `components.html`; if 1.66 prints a deprecation notice, switch to the suggested replacement). The theme CSS selectors target Streamlit 1.66; the dashboard canvas is an iframe and does not depend on them. Navigation uses state-preserving Streamlit buttons; the live shell still needs testing on the pinned Streamlit release.
4. **No live deployment, phone test, scheduled run, recorded demo or `v1.0.0` tag** exists yet (see `docs/deployment.md`).
5. **Logo and illustrations are low-resolution crops** of the reference screenshot; replace them with your originals (same file names in `assets/`).
6. Phase 1-3 evaluations are read from their saved files, not re-run (the restricted vault and pinned scikit-learn are required to retrain). The Data Foundation page labels this as saved evidence rather than a fresh audit.
7. Resume PDF/DOCX ingestion is in-memory only, with an 8 MB limit and 20-page PDF limit. Scanned PDFs require OCR. Uploaded candidate scores use a temporary corpus refit and are not comparable to the saved benchmark scores. Upload parsing is not antivirus scanning.
8. The workforce map aggregates active employees by `work_location` and uses approximate city-centre coordinates; it is not office-level mapping or employee tracking.
9. Password authentication uses environment-backed users and HMAC-signed bearer tokens. Token revocation is process-local; enterprise SSO, durable revocation, account lifecycle, and distributed rate limiting require external infrastructure.

## About the models (also in each model card)
* **Attrition v2** is not proven better than the simple logistic baseline (ROC-AUC about 0.70-0.72). The 2026 scoring data show a collapsed `months_since_last_raise` (PSI ALERT): treat 2026 scores with caution until HR explains it.
* **Forecast:** a single hiring freeze (2020) is the only example of that effect; intervals ignore parameter uncertainty; the baseline's "80%" interval covered only about 68% of rolling-test actuals (company headcount), so ranges are optimistic. Department series are small.
* **Resume ranker:** ten jobs and 555 labels; the zero twin gap is by construction after masking and does not prove fairness; other proxies are untested.
* **Pay band:** trained on past offers (which can contain past bias); the gender gap check needs a restricted file that is not in the repository and is therefore reported as *not checked*.
* **Payroll / attendance detectors:** evaluated on synthetically injected errors; about half of attendance flags are false alarms; flags are review candidates only.
* **Chatbot:** 81% held-out accuracy; leave and payslip answers need the Phase 2 mock HR service.

## About the dashboard numbers
Every figure comes from the project's data or model files (cross-checked in `tests/test_services_models.py`). Where the reference screenshot showed a change arrow that the data cannot support (for example "Attrition Risk down 1.2%": there are no earlier risk scores), the card shows a dash instead of an invented value. "Open Positions" is the sum of `openings` over the ten job descriptions.
