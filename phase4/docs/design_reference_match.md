# Dashboard design-reference match (approved screenshot, 1365x768)

Evidence: `outputs/console/reference_match/` (reference, implementation with popup, popup closed, side-by-side, phone, afternoon/evening/night greetings).
Re-create: `python scripts/render_reference_preview.py --screenshot` then `python scripts/compare_with_reference.py <png> <side_by_side.png>`.

## Measured result (honest)
| Measure | Value | Meaning |
|---|---|---|
| Image size | 1365x768 = reference | same resolution |
| Mean absolute pixel error | 20.1 / 255 | NOT pixel-identical |
| SSIM (grayscale) | 0.56 | layout matches; fine texture does not |

The reference is a rendered mock-up with 3-D glossy objects (sun, clouds, coin, brain), a photo avatar and thousands of
individual glowing particles. Those textures cannot be reproduced exactly with HTML/CSS/SVG, so the pixel metrics stay low even
though every component, position, size, label and value matches. The metrics are published so nobody has to take "looks identical" on trust.

## What matches the reference
Sidebar (167 px, logo, 8 nav items, active "Dashboard" pill, "Build. Predict. Empower." card, v1.0.0 - Production), search bar with "38 K",
sun/bell/avatar profile, "Welcome back, Jaisurya!", tagline, "Last 6 Months" picker, four KPI cards (12,480 / 8.7% / 24 / Rs 8.92L with deltas and
sparklines), Workforce Overview (Jan/Feb/Mar at the reference x positions, 0K-10K axis, Employees + Attrition lines), Attrition Risk Analysis donut
(8.7%, 12.4 / 24.6 / 62.9), Key Insights, Model Performance (Rs 53,925, 0.983, 80.2%, Rs 1,15,094, Ridge Regression), Recent Activity, CortexHR v1.0.0 card,
centred glass popup (465x255 at x=455,y=265) with logo, animated sun + clouds, wink, "Good Morning, Jaisurya!", close button.
The popup is an overlay: after closing it, the whole dashboard is visible (`dashboard_popup_closed_1365x768.png`).
Greeting uses Asia/Kolkata: 05:00-11:59 morning, 12:00-16:59 afternoon, 17:00-20:59 evening, 21:00-04:59 night (screenshots for 14h/19h/23h).

## Known visual differences
- Fonts: the reference uses Playfair/Inter-style faces; this sandbox has no web access, so installed fallbacks (Caladea/Inter) are used. Letter shapes differ slightly.
- Glossy 3-D icons are drawn as flat gradient SVG glyphs; sun/cloud/coin/brain are simplified. The brain and logos are the crops taken from the reference (background keyed out).
- The wave/particle background is generated procedurally, not copied pixel for pixel.
- Avatar is a 44 px crop of the reference photo (`assets/avatar_reference.png`).
- Months Apr-Sep of the chart are hidden behind the popup in the reference; they continue the visible trend (assumption, flagged in `design_reference.py`).

## IMPORTANT: reference numbers are sample values, not real results
12,480 employees, 8.7% attrition, R2 0.983, 80.2% coverage, Rs 1,15,094 baseline MAE etc. are printed in the screenshot. They are used ONLY by
`scripts/render_reference_preview.py` (or `CORTEXHR_DESIGN_REFERENCE=1`) so the design can be compared. The live console/API show numbers from `GET /v1/overview`.
Reproducing e.g. "80.2% coverage" in the live product would be a fabricated claim: the forecast interval evidence in `docs/forecasting_report.md`
reports the honestly measured coverage (about 68% before calibration).
