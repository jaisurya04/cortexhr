# Model Card: CortexHR attendance anomaly detector   (CX-20)

**Purpose** flags unusual attendance months for HR review.  ·  **Not for** performance rating, discipline or any automatic action.

**Data** `attendance_monthly_features.csv`, 7,339 employee-months; unsupervised training, contamination 0.03.

**Method** Isolation Forest on own-history features combined with rules (`phase2/reports/tables/attendance_detector_metrics.csv`).

**Metrics**

Against 150 injected anomalies (`attendance_detector_metrics.csv`):

| detector | flagged | injected | caught | precision | recall |
|---|---|---|---|---|---|
| IsolationForest (raw features only) | 221 | 150 | 88 | 0.398 | 0.587 |
| IsolationForest (+own-history features) | 221 | 150 | 95 | 0.43 | 0.633 |
| Rules only | 94 | 150 | 50 | 0.532 | 0.333 |
| Combined (IF + rules) | 252 | 150 | 126 | 0.5 | 0.84 |
| POST-HOC v2 combined (optimistic) | 278 | 150 | 150 | 0.54 | 1.0 |

The last row is marked POST-HOC and optimistic (tuned after seeing results); use the combined IF + rules row (precision 0.50, recall 0.84) as the honest figure.

**Explanations** Rule-based reasons plus the model score are returned with each result.

**Fairness** Not separately analysed by protected group in Phase 2 beyond the group flag-rate screen; no protected attribute is an input.

**Limits & risks** About half of the flags are false alarms; injected anomalies are simpler than real ones; remote-work patterns can look unusual.

**Version & owner** `attendance-iforest-rules-v2` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
