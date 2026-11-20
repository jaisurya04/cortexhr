# Model Card: CortexHR HR assistant (intent model)   (CX-20)

**Purpose** recognises what an employee is asking about HR (leave, payslip, policy ...) and returns a vetted answer or routes to a service.  ·  **Not for** legal, medical or disciplinary advice; any answer that changes records.

**Data** 858 labelled example sentences across 33 intents (`chatbot_nlu_training.csv`, original set, no test-driven additions) plus an FAQ and response table. Held-out test 132 rows.

**Method** TF-IDF + logistic regression; confidence threshold 0.30 (below it the bot asks for clarification); roles come from the identity directory, never from the message.

**Metrics**

Held-out accuracy **0.811** (25 errors in 132 rows) from `phase2/reports/phase2/chatbot_metric_summary.csv`.

**Explanations** Every reply shows its intent and confidence in the console.

**Fairness** Not tested by demographic group; wording is English only.

**Limits & risks** Questions needing personal records call the Phase 2 mock HR service; without it the assistant says so instead of guessing. Never reveal other people's records (role-based access).

**Version & owner** `chatbot-tfidf-lr-v1` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
