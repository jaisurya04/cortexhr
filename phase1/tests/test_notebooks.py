from _common import *
import json, re

NB = C.project_root() / "05_Starter" / "notebooks"
TASKS = ["01_data_audit", "01b_feature_pipeline", "02_baselines", "03_model_evaluation", "04_attendance_anomalies"]

def _load(name): return json.loads((NB / f"{name}.ipynb").read_text())
def _outputs(nb): return [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
def _md(nb): return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")

def test_every_notebook_executed_without_errors_or_failed_checks():
    for n in TASKS + ["05_complete_classification_workflow"]:
        nb = _load(n); outs = _outputs(nb)
        assert not [o for o in outs if o["output_type"] == "error"], n
        assert all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"), n
        assert not any("[FAIL]" in "".join(o.get("text", [])) for o in outs if o["output_type"] == "stream"), n

def test_task_notebooks_have_all_12_workflow_stages():
    for n in TASKS:
        heads = re.findall(r"# Stage (\d+) of 12", _md(_load(n))); assert sorted(map(int, heads)) == list(range(1, 13)), (n, heads)

def test_master_notebook_covers_all_37_items_and_12_phases():
    md = _md(_load("05_complete_classification_workflow")); secs = {int(x) for x in re.findall(r"^## (\d+)\. ", md, flags=re.M)}
    assert secs == set(range(1, 38)), sorted(set(range(1, 38)) - secs)
    assert len(re.findall(r"# PHASE (\d+):", md)) == 12

def test_notebooks_use_the_official_helper():
    for n in TASKS + ["05_complete_classification_workflow"]:
        src = "\n".join("".join(c["source"]) for c in _load(n)["cells"] if c["cell_type"] == "code"); assert "cortexhr_helpers" in src, n
