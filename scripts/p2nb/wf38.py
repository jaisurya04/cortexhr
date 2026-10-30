"""Append the 38-section 'Complete End-to-End Data Science Workflow for Classification' to a notebook.

* The section titles and every checklist item come from docs/CLASSIFICATION_WORKFLOW_38.md (parsed, not retyped),
  so nothing can be missed: each notebook gets one markdown cell per section that lists ALL of its items,
  followed by a code cell that produces the evidence for that section.
* Existing notebook cells are never touched.  Cells added by this engine carry ``metadata.wf38 = true`` and are
  removed and rebuilt on every run, so the build is idempotent.
* ``execute()`` runs ONLY the wf38 cells in a fresh namespace (the section is self-contained) and embeds real
  outputs (text, tables as HTML, figures as PNG).  Old outputs of the original cells are kept as they were.
"""
from __future__ import annotations

import ast, base64, contextlib, html, io, json, pathlib, re, sys, traceback, uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "CLASSIFICATION_WORKFLOW_38.md"
NB_DIR = ROOT / "05_Starter" / "notebooks" / "phase2"


def parse_doc(path=DOC):
    """-> [(number, title, [items...]), ...] for the 38 sections."""
    text = path.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        head, *rest = block.splitlines()
        m = re.match(r"(\d+)\. (.+)", head)
        items = [re.sub(r"^\d+\. ", "", ln) for ln in rest if re.match(r"^\d+\. ", ln)]
        out.append((int(m.group(1)), m.group(2).strip(), items))
    assert [n for n, _, _ in out] == list(range(1, 39)), "workflow document must contain sections 1..38"
    return out


def _cell(kind, src, **meta):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {"wf38": True, **meta},
         "source": src.strip("\n").splitlines(True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def build_cells(nb_name, track, focus, setup_code, stage_code, stage_notes, nb_notes):
    """Return the list of cells (markdown + code) for one notebook."""
    doc = parse_doc()
    cells = []
    toc = "\n".join(f"| {n} | {t} | {len(items)} |" for n, t, items in doc)
    cells.append(_cell("markdown", f"""---
# WF38 · Complete 38-stage classification workflow for this notebook

**Notebook:** `{nb_name}`  ·  **Track:** {track}

{focus}

How to read this section: every one of the 38 workflow sections below has **one markdown cell** (all of its checklist
items, copied from `docs/CLASSIFICATION_WORKFLOW_38.md`, plus what this notebook does for it) and **one code cell**
that produces the evidence. Cells run top to bottom in a fresh kernel; the section does not depend on the cells above
it, and it never opens the authorised validation labels (CX-06) or the official test set before the stage that is
allowed to.

| # | Section | Items |
|---|---------|-------|
{toc}
"""))
    cells.append(_cell("code", setup_code))
    for n, title, items in doc:
        listing = "\n".join(f"- [x] {i}" for i in items)
        note = stage_notes[n]
        extra = nb_notes.get(n, "")
        extra_txt = f"\n\n**Specific to this notebook:** {extra}" if extra else ""
        cells.append(_cell("markdown", f"""---
## WF38 · Stage {n} — {title}

**Checklist for this stage ({len(items)} items, all addressed by the cell below or explained in it):**

{listing}

**What this notebook does:** {note}{extra_txt}
"""))
        cells.append(_cell("code", stage_code[n]))
    return cells


def apply(nb_path, cells):
    nb = json.loads(pathlib.Path(nb_path).read_text(encoding="utf-8"))
    nb["cells"] = [c for c in nb["cells"] if not c.get("metadata", {}).get("wf38")]
    nb["cells"].extend(cells)
    pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
    return len(nb["cells"])


# ----------------------------------------------------------------------------------- executor
def _rich(obj):
    """Return a Jupyter display bundle for an object."""
    data = {"text/plain": repr(obj).splitlines(True)}
    if hasattr(obj, "_repr_html_"):
        try:
            data["text/html"] = obj._repr_html_().splitlines(True)
        except Exception:
            pass
    return data


def execute(nb_path, only_wf38=True, verbose=True):
    """Run the wf38 cells of one notebook (fresh namespace), embed outputs, return (n_code_cells, errors)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path = pathlib.Path(nb_path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    counts = [c.get("execution_count") or 0 for c in nb["cells"] if c["cell_type"] == "code" and not c["metadata"].get("wf38")]
    counter = max(counts + [0])
    ns = {"__name__": "__main__"}
    errors, n = [], 0
    for c in nb["cells"]:
        if c["cell_type"] != "code" or (only_wf38 and not c.get("metadata", {}).get("wf38")):
            continue
        n += 1; counter += 1
        src = "".join(c["source"]); outs = []; buf = io.StringIO(); shown = []
        ns["display"] = lambda *objs, _s=shown: _s.extend(objs)
        try:
            tree = ast.parse(src); body = tree.body; last = None
            if body and isinstance(body[-1], ast.Expr):
                last = ast.Expression(body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(ast.Module(body, []), "<wf38 cell>", "exec"), ns)
                val = eval(compile(last, "<wf38 cell>", "eval"), ns) if last else None
            if val is not None:
                shown.append(val)
        except Exception:
            tb = traceback.format_exc(); errors.append(f"{nb_path.name if hasattr(nb_path,'name') else nb_path}: {tb}")
            outs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1], "traceback": tb.splitlines()})
        if buf.getvalue():
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
        for obj in shown:
            if obj.__class__.__module__.startswith("matplotlib"):
                continue
            outs.append({"output_type": "display_data", "metadata": {}, "data": _rich(obj)})
        for num in plt.get_fignums():
            b = io.BytesIO(); plt.figure(num).savefig(b, format="png", dpi=100, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        c["outputs"] = outs; c["execution_count"] = counter
        if verbose and errors and outs and outs[-1]["output_type"] == "error":
            print("  ERROR in cell:", src[:80].replace("\n", " | ")); print(outs[-1]["evalue"])
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
    return n, errors
