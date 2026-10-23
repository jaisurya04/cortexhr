"""Builds 05_complete_classification_workflow.ipynb: every item (1-37) of the supplied 'Complete data science workflow for a
classification problem' as a numbered section with real code, outputs and a requirement-by-requirement coverage table."""
import json, pathlib
OUT = pathlib.Path(__file__).resolve().parents[1] / "05_Starter" / "notebooks"; OUT.mkdir(parents=True, exist_ok=True)
SETUP = r"""import sys, pathlib, inspect, time, warnings, subprocess
ROOT = pathlib.Path.cwd().resolve()
while not (ROOT / 'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / '05_Starter'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.stats as st
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 100)
import cortexhr_helpers as H; H.DATA = ROOT / '03_Data' / 'csv'          # the OFFICIAL starter helper, unmodified
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, GroupKFold, cross_val_predict, cross_validate, GridSearchCV,
                                     RandomizedSearchCV, train_test_split, learning_curve, validation_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, fbeta_score, accuracy_score,
                             balanced_accuracy_score, log_loss, brier_score_loss, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
                             hamming_loss, make_scorer)
from cortexhr import config as C, data_loading as D, audit as A, features as F, modeling as M, evaluation as E, reporting as R, runs, workflow as W
SEED = C.SEED
print('Python', sys.version.split()[0], '| root', ROOT, '| official helper:', H.__file__.split('/')[-2:])"""

def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code": c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
md, code = (lambda s: ("markdown", s)), (lambda s: ("code", s))

def cov(n, title, rows):
    """Section header + requirement coverage table. rows = [(requirement, how it is covered here)]."""
    t = f"---\n## {n}. {title}\n\n| Requirement | How it is covered in this notebook |\n|---|---|\n"
    return md(t + "\n".join(f"| {a} | {b} |" for a, b in rows))
def phase(n, title): return md(f"---\n---\n# PHASE {n}: {title}")

CELLS = [md("""# CortexHR attrition: the complete data-science workflow for a classification problem
Every numbered item of the supplied workflow (1-37) is a section below, with a **coverage table** (requirement -> where/how) followed by real code and outputs on the actual CortexHR attrition data.
Target: `left_within_12m` (1 = left within 12 months). Data: `03_Data/CortexHR_Dataset.xlsx` (fictional). Official helpers: `05_Starter/cortexhr_helpers.py`.
**Rules kept throughout:** the test set is untouched until Section 37; all learned preprocessing is fitted on training rows only; the validation vault is never used as input."""),
code(SETUP)]
exec(open(pathlib.Path(__file__).with_name("master_part1.py")).read())
exec(open(pathlib.Path(__file__).with_name("master_part2.py")).read())
exec(open(pathlib.Path(__file__).with_name("master_part3.py")).read())
(OUT / "05_complete_classification_workflow.ipynb").write_text(json.dumps(nb(CELLS), indent=1)); print("wrote master notebook", len(CELLS), "cells")
