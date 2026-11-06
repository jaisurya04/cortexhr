# Task 15: fair pay bands
Notebooks 12 to 14. Model Ridge regression (job-related features), job-related inputs only (career_gap_months excluded, gender and age never inputs).
**Test accuracy:** MAE Rs 53,925, R2 0.983 versus the 'average by role' baseline MAE Rs 115,094, R2 0.933. A role average adjusted for the market index is already close (MAE Rs 55,656): most of the signal is role, level, band and market.
**Band coverage on the test set:** 0.802 (target 0.80).
**Pay ratio by gender (history, model with career gap, chosen model):** Female: 0.9841 / 0.9864 / 0.9885; Male: 0.9979 / 0.9907 / 0.9891; Other: 0.9981 / 0.9879 / 0.9866. Details and bootstrap intervals in notebook 14.
**Limits:** `docs/limitations.md`.