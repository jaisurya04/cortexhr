"""Day 20: every model has a card in the MODEL_CARD_TEMPLATE layout, and no card carries a stale number.

Three independent checks:
1. the cards on disk equal the cards re-generated right now from the saved evaluation files (stale numbers fail);
2. every number of the live ``/v1/performance`` tables appears in the matching card;
3. every number typed by hand into a card's prose (row counts, test-split metrics) is re-derived from its source file.
"""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
from _common import ROOT, registry

from cortexhr_phase4.core.settings import PHASE2, PHASE3
from cortexhr_phase4.services import handlers as h

CARDS = ROOT / "docs" / "model_cards"
CARD_FOR = {"attrition": "attrition_v2", "payroll": "payroll_anomaly", "attendance": "attendance_anomaly", "chatbot": "chatbot_intent",
            "resume": "resume_ranker_v2", "payband": "salary_band_v1", "forecast": "headcount_forecast"}
SECTIONS = ["**Purpose**", "**Not for**", "**Data**", "**Method**", "**Metrics**", "**Explanations**", "**Fairness**", "**Limits & risks**", "**Version & owner**"]


def _text(name: str) -> str:
    return (CARDS / f"{name}.md").read_text(encoding="utf-8")


def _norm(t: str) -> str:
    return re.sub(r"card written \d{4}-\d{2}-\d{2}", "card written DATE", t)


def _build_docs():
    spec = importlib.util.spec_from_file_location("cx_build_docs", ROOT / "scripts" / "build_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_registry_model_has_a_card_in_the_template_layout():
    reg = registry()
    assert set(CARD_FOR) == set(reg.info), "a model without a card (or a card without a model)"
    for model, card in CARD_FOR.items():
        t = _text(card)
        assert t.startswith("# Model Card: ") and "(CX-20)" in t.splitlines()[0], card
        missing = [s for s in SECTIONS if s not in t]
        assert not missing, f"{card}: template sections missing {missing}"
        assert "AI-generated and needs human review" in t, f"{card}: review notice missing"
        version = reg.info[model].version
        if not version and model in {"payroll", "attendance", "chatbot"}:
            version = json.loads((PHASE2 / "models" / "model_registry.json").read_text(encoding="utf-8"))[model]["version"]
        assert version and (f"`{version}`" in t or version in t), f"{card}: version string differs from the verified model metadata ({version})"
        assert len(t.split("**Limits & risks**")[1]) > 80, f"{card}: limits section is empty"


def test_cards_on_disk_equal_cards_regenerated_from_the_evaluation_files():
    fresh = _build_docs().build_cards(registry())
    assert set(fresh) == set(CARD_FOR.values())
    stale = [n for n, t in fresh.items() if _norm(t) != _norm(_text(n))]
    assert not stale, f"stale model cards (run scripts/build_docs.py): {stale}"


def test_live_performance_numbers_appear_in_the_cards():
    perf = h.performance(registry())
    checks = [("attrition_v2", pd.DataFrame(perf["attrition"]["cv"]), ["roc_auc_mean", "pr_auc_mean", "brier_mean"]),
              ("resume_ranker_v2", pd.DataFrame(perf["resume"]["ranking"]), None),
              ("salary_band_v1", pd.DataFrame(perf["payband"]["test_metrics"]), None)]
    for card, df, cols in checks:
        t = _text(card)
        shown = [c for c in df.select_dtypes("number").columns if c in t]          # a card shows the columns it names; every shown column must match the live numbers
        assert cols or shown, f"{card}: none of the /v1/performance columns are in the card"
        for c in (cols or shown):
            for v in df[c].dropna():
                s = str(round(float(v), 3))
                assert s in t or str(round(float(v), 2)) in t or str(round(float(v), 1)) in t, f"{card}: {c}={v} from /v1/performance is not in the card"


def test_hand_typed_numbers_in_cards_are_rederived_from_their_sources():
    csv = PHASE3 / "03_Data" / "csv"
    att = _text("attrition_v2")
    d = pd.read_csv(csv / "attrition_dataset.csv")
    dupes = len(d) - len(d.drop_duplicates(subset=[c for c in d.columns if c != "row_id"]))
    assert f"{len(d) - dupes:,} rows" in att and f"{dupes} exact duplicates" in att, (len(d), dupes)
    tm = pd.read_csv(PHASE3 / "reports" / "task13" / "test_metrics.csv").set_index("model").loc["attrition-v2 (calibrated)"]
    for label, col in (("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier"), ("ECE", "ece")):
        assert f"{label} {tm[col]:.3f}" in att, f"{label} {tm[col]:.3f} not in the attrition card"
    assert f"{len(pd.read_csv(csv / 'resume_job_labels.csv'))} human relevance labels" in _text("resume_ranker_v2")
    assert f"{len(pd.read_csv(csv / 'salary_training.csv')):,} rows" in _text("salary_band_v1")
