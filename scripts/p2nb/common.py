"""Tiny notebook DSL (plain JSON, no nbformat needed) shared by the Phase 2 notebook builders."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "05_Starter" / "notebooks" / "phase2"

SETUP = '''import sys, pathlib, json, re, warnings
warnings.filterwarnings('ignore')
ROOT = pathlib.Path.cwd().resolve()
while not (ROOT / 'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / '05_Starter')); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 120)
from cortexhr import config as C
from cortexhr.phase2 import payroll as P, chatbot as CB, service as S
import cortexhr_helpers as H; H.DATA = ROOT / '03_Data' / 'csv'      # the OFFICIAL starter helper (unmodified)
CSV = ROOT / '03_Data' / 'csv'; OUT = ROOT / 'reports' / 'phase2'; MOCK = ROOT / '05_Starter' / 'mock_services' / 'mock_data'
print('Python', sys.version.split()[0], '| project root:', ROOT)'''


class NB:
    def __init__(self): self.cells = []
    def md(self, text): self.cells.append(("markdown", text.strip("\n"))); return self
    def code(self, text): self.cells.append(("code", text.strip("\n"))); return self
    def stage(self, n, title, blurb=""):
        return self.md(f"---\n# Stage {n} of 12: {title}\n{blurb}")
    def think(self, question, code, answer):
        self.md(f"### Think for yourself: {question}")
        if code: self.code(code)
        return self.md(f"**Answer.** {answer}")
    def checklist(self, items_code):
        self.md("### Completion checklist (checked by code)")
        return self.code(items_code)
    def footer(self, evidence, refs, real, connects, say):
        return self.md(f"### Evidence for your report\n{evidence}\nReferences: {refs}\n**How real companies do it:** {real}\n**Connects to the next task:** {connects}\n**You can now say:** *{say}*")
    def save(self, name):
        cells = []
        for kind, src in self.cells:
            lines = src.splitlines(True)
            c = {"cell_type": kind, "metadata": {}, "source": lines}
            if kind == "code": c.update(outputs=[], execution_count=None)
            cells.append(c)
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        return OUT / name


def checks_cell(checks_dict_src):
    """Standard PASS/FAIL/MANUAL printer used by every checklist cell."""
    return checks_dict_src + '''
status = {k: ('MANUAL' if isinstance(v, str) and v == 'MANUAL' else 'PASS' if bool(v) else 'FAIL') for k, v in checks.items()}
for k, v in status.items(): print(f'[{v}]', k)
assert 'FAIL' not in status.values(), 'a checklist item failed'
print('\\nchecklist:', sum(v == 'PASS' for v in status.values()), 'PASS,', sum(v == 'MANUAL' for v in status.values()), 'MANUAL (needs a human or a file only you have)')'''
