# Demo recording checklist (Day 20 - YOU record; nothing here was recorded by the build)

Script: `docs/demo_script_5min.md`. Status: **USER ACTION REQUIRED** (no recording exists).

Before recording
- [ ] API deployed; `python scripts/smoke_test.py --base-url <live API>` is green (save the output)
- [ ] Console deployed with `CORTEXHR_API_URL` = the live API; open it in a fresh tab so the welcome popup plays
- [ ] Monitoring workflow has run at least once; `GET /v1/monitoring` shows the latest status
- [ ] Notifications off, browser zoom 100%, 1080p, microphone tested
- [ ] Re-read the numbers in the script against the live service (they come from the saved offline run)

While recording (<= 5:00)
- [ ] Problem (30 s) - Dashboard + popup (60 s) - Forecast + honest coverage (60 s) - five tabs with the "AI-generated, for human review" label (60 s) - Monitoring ALERT + 4-step plan (60 s) - handover + the model you would not release (30 s)
- [ ] Say what is NOT done or not proven (gender pay gap needs the vault; interval coverage ~68% vs 80% target unless you re-ran with statsmodels)

After recording
- [ ] Upload (YouTube unlisted / Drive "anyone with link"); open the link in a private window and on your phone
- [ ] Paste the link into `docs/final_report.md` and the submission checklist
