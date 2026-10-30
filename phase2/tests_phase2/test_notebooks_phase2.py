"""Phase 2 task notebooks: 12 stages, Think-for-yourself with an answer, executed without errors, and the 37-item master."""
import json, re
from _helpers import ROOT

NB = ROOT / "05_Starter" / "notebooks" / "phase2"
TASKS = ["P2_T06_payroll_outlier_detector", "P2_T07_reason_codes_money_at_risk", "P2_T08_anomaly_api_service", "P2_T09_hr_intent_chatbot", "P2_T10_slots_memory_permissions"]
MASTER = "P2_MASTER_37_chatbot_classification_workflow"

def _nb(name): return json.loads((NB / f"{name}.ipynb").read_text(encoding="utf-8"))
def _md(nb): return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")

def test_phase2_notebooks_executed_without_errors():
    for n in TASKS + [MASTER]:
        nb = _nb(n)
        outs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
        assert not [o for o in outs if o["output_type"] == "error"], n
        assert all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"), n

def test_task_notebooks_have_12_stages_think_for_yourself_and_answer():
    for n in TASKS:
        md = _md(_nb(n))
        assert sorted(map(int, re.findall(r"# Stage (\d+) of 12", md))) == list(range(1, 13)), n
        assert "### Think for yourself" in md and "**Answer.**" in md, n

def test_master_covers_all_37_items_and_12_phases():
    md = _md(_nb(MASTER))
    assert {int(x) for x in re.findall(r"^## (\d+)\. ", md, flags=re.M)} == set(range(1, 38))
    assert len(re.findall(r"# PHASE (\d+):", md)) == 12


# ---------------------------------------------------------------- 38-stage workflow appended to EVERY phase 2 notebook
DOC = ROOT / "docs" / "CLASSIFICATION_WORKFLOW_38.md"

def _doc_sections():
    out = []
    for block in re.split(r"^## ", DOC.read_text(encoding="utf-8"), flags=re.M)[1:]:
        head, *rest = block.splitlines()
        n, title = re.match(r"(\d+)\. (.+)", head).groups()
        out.append((int(n), title.strip(), [re.sub(r"^\d+\. ", "", x) for x in rest if re.match(r"^\d+\. ", x)]))
    return out

def test_workflow_document_has_38_sections():
    secs = _doc_sections()
    assert [n for n, _, _ in secs] == list(range(1, 39)) and sum(len(i) for _, _, i in secs) == 409

def test_every_phase2_notebook_has_all_38_stages_with_every_item_and_30_plus_cells():
    names = sorted(p.stem for p in NB.glob("*.ipynb")); assert len(names) == 16, names
    for name in names:
        nb = _nb(name); cells = nb["cells"]; assert len(cells) >= 30, (name, len(cells))
        wf = [c for c in cells if c.get("metadata", {}).get("wf38")]
        md = "\n".join("".join(c["source"]) for c in wf if c["cell_type"] == "markdown")
        for n, title, items in _doc_sections():
            assert f"## WF38 · Stage {n} — {title}" in md, (name, n)
            for item in items: assert f"- [x] {item}" in md, (name, n, item)
        order = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).lstrip("-\n ").startswith("## WF38 · Stage")]
        assert len(order) == 38 and all(cells[i + 1]["cell_type"] == "code" for i in order), name   # every stage has its evidence cell right after it

def test_wf38_cells_executed_without_errors_and_have_output():
    for p in sorted(NB.glob("*.ipynb")):
        wf = [c for c in json.loads(p.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("wf38")]
        assert len(wf) == 39, (p.stem, len(wf))
        for c in wf:
            assert c.get("execution_count"), p.stem
            assert not [o for o in c.get("outputs", []) if o["output_type"] == "error"], (p.stem, "".join(c["source"])[:60])
        assert sum(bool(c.get("outputs")) for c in wf) >= 36, p.stem
