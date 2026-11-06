"""Tiny notebook builder: markdown + code cells, the 38-stage workflow headings, and a real executor (no Jupyter needed)."""
from __future__ import annotations
import ast, base64, contextlib, io, json, re, sys, traceback
from pathlib import Path

STAGES = {
 1: ("Business Understanding", "Who needs this result and what decision it supports."),
 2: ("Problem Definition", "The exact prediction, ranking or measurement problem, its inputs and its output."),
 3: ("Objective and Success Criteria", "What number or evidence shows the task is done well, decided before looking at results."),
 4: ("Dataset Discovery", "Which files exist, how they relate, and which are authoritative."),
 5: ("Data Loading", "Load with the project helpers and check the load is complete."),
 6: ("Dataset Structure and Shape", "Rows, columns, keys and grain of each table."),
 7: ("Data Dictionary and Schema Understanding", "What each column means, from the supplied dictionary and from inspection."),
 8: ("Data Type Inspection", "Check every column has the type it should have."),
 9: ("Data Quality Assessment", "A one-page quality profile before any modelling."),
 10: ("Missing Value Analysis", "Where values are missing, how much, and why that may be."),
 11: ("Duplicate Analysis", "Exact and near duplicates of rows, keys and texts."),
 12: ("Outlier Detection and Investigation", "Unusual values: are they errors or real?"),
 13: ("Invalid and Inconsistent Value Detection", "Impossible values, broken links between tables, label contradictions."),
 14: ("Exploratory Data Analysis (EDA)", "Look at the data before modelling."),
 15: ("Univariate Analysis", "One variable at a time."),
 16: ("Bivariate Analysis", "Two variables together, especially against the target or label."),
 17: ("Multivariate Analysis", "Several variables together: correlation, interaction, redundancy."),
 18: ("Target or Label Analysis", "How the target or human label is distributed and how reliable it is."),
 19: ("Data Leakage Investigation", "Anything that carries the answer or protected information into the model."),
 20: ("Feature Engineering", "Build the inputs the model will use from the raw data."),
 21: ("Feature Selection and Relevance Analysis", "Which inputs are allowed, which are relevant, which must never be used."),
 22: ("Data Preprocessing", "Cleaning and encoding, fitted on training data only."),
 23: ("Train, Validation and Test Strategy", "How data is split so no result is measured on data the model has seen."),
 24: ("Baseline Establishment", "The simple method every model must beat."),
 25: ("Model Selection and Justification", "Why these candidate methods."),
 26: ("Model Training", "Fit the candidates."),
 27: ("Cross-Validation", "Repeat the evaluation on several splits to see the spread."),
 28: ("Hyperparameter Tuning", "A small, logged search; no tuning on the test set."),
 29: ("Model Evaluation", "Metrics suited to the problem, against the baseline."),
 30: ("Error and Misclassification Analysis", "Look at the cases the model gets wrong and say why."),
 31: ("Explainability and Feature Importance", "What drives the output, in plain words."),
 32: ("Fairness and Bias Evaluation", "Gaps between groups, measured with the protected attributes only after the fact."),
 33: ("Model Calibration or Prediction Reliability", "Are the numbers trustworthy as numbers?"),
 34: ("Model Comparison and Final Selection", "Choose on evidence, not on the highest score alone."),
 35: ("Final Prediction Generation", "Produce the deliverable."),
 36: ("Output and Business Validation", "Check the output against the required format and read it as the user would."),
 37: ("Model Saving, Versioning and Reproducibility", "Artifacts, metadata, seeds, environment."),
 38: ("Final Findings, Limitations and Conclusion", "What was found, what remains, what must not be done with it."),
}


class NB:
    def __init__(self, title: str, task: str, objective: str, covers: list[int]):
        self.cells = []; self.covers = covers
        self.md(f"# {title}\n\n**Programme:** Aura Workforce by Growmar, AI/ML track, CortexHR **Phase 3: Intelligence**  \n**Task:** {task}  \n**Objective:** {objective}  \n"
                f"**Workflow stages covered in this notebook:** {', '.join(map(str, covers))} of the 38-stage Phase 1/2 standard.  \n"
                "**Rule for every number below:** it is printed by the code in the same notebook; nothing is typed in by hand.")
        self.code("import sys, warnings, pathlib\nwarnings.filterwarnings('ignore')\nROOT = pathlib.Path.cwd().resolve()\n"
                  "while not (ROOT / 'config' / 'phase3_config.yaml').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\n"
                  "import numpy as np, pandas as pd, matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                  "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 80)\n"
                  "from cortexhr.phase3.common.config import load_config\nfrom cortexhr.phase3.common import paths as P\nCFG = load_config(); SEED = CFG['seed']\nprint('project root:', ROOT.name, '| seed:', SEED)")

    def md(self, text: str): self.cells.append(("markdown", text.strip("\n")))
    def code(self, src: str): self.cells.append(("code", src.strip("\n")))

    def stage(self, n: int, note: str = ""):
        t, g = STAGES[n]; self.md(f"## Stage {n}: {t}\n\n*What this stage is for:* {g}\n\n{note}".rstrip())

    def na(self, n: int, reason: str):
        t, g = STAGES[n]; self.md(f"## Stage {n}: {t}\n\n*What this stage is for:* {g}\n\n**Not applicable here, and why:** {reason}")

    def save(self, path: Path):
        nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
        for kind, src in self.cells:
            lines = src.split("\n"); lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            c = {"cell_type": kind, "metadata": {}, "source": lines}
            if kind == "code": c["outputs"] = []; c["execution_count"] = None
            nb["cells"].append(c)
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"); return path


# --------------------------------------------------------------------------- executor
def _fig_outputs():
    import matplotlib.pyplot as plt
    outs = []
    for num in plt.get_fignums():
        buf = io.BytesIO(); plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        outs.append({"output_type": "display_data", "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": ["<Figure>"]}, "metadata": {}}); plt.close(num)
    return outs


def execute_notebook(path: Path, quiet=True) -> dict:
    """Run every code cell top to bottom in one namespace; store stdout, last-expression values and figures as real outputs. Any error aborts."""
    import os
    nb = json.loads(Path(path).read_text(encoding="utf-8")); ns = {"__name__": "__main__"}; count = 0; old = os.getcwd(); os.chdir(Path(path).parent)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            src = "".join(cell["source"]); cell["outputs"] = []; count += 1; cell["execution_count"] = count
            tree = ast.parse(src); last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            body = ast.Module(tree.body[:-1] if last else tree.body, type_ignores=[]); out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    exec(compile(body, str(path), "exec"), ns)
                    val = eval(compile(ast.Expression(last.value), str(path), "eval"), ns) if last else None
            except Exception:
                raise RuntimeError(f"{Path(path).name} cell {count} failed:\n{src[:400]}\n{traceback.format_exc()}")
            if out.getvalue(): cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": out.getvalue().splitlines(True)})
            if val is not None:
                import pandas as _pd
                txt = val.to_string(max_rows=40, max_cols=30) if isinstance(val, _pd.DataFrame) else (val.to_string(max_rows=40) if isinstance(val, _pd.Series) else repr(val))
                cell["outputs"].append({"output_type": "execute_result", "execution_count": count, "data": {"text/plain": txt.splitlines(True)}, "metadata": {}})
            cell["outputs"].extend(_fig_outputs())
    finally:
        os.chdir(old)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"notebook": Path(path).name, "code_cells": count, "images": sum(1 for c in nb["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {}))}
