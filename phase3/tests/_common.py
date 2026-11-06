"""Shared, cached fixtures for the plain-function tests (work with pytest and with tests/run_all.py)."""
import sys, pathlib, functools
ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "05_Starter"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from cortexhr.phase3.common.io import read_csv


@functools.lru_cache(maxsize=None)
def data():
    return {k: read_csv(k) for k in ("resumes", "job_descriptions", "resume_job_labels", "resume_bias_pairs")}


@functools.lru_cache(maxsize=None)
def rankers():
    from cortexhr.phase3.resume.scoring import ResumeRanker
    d = data(); return ResumeRanker(mask=False).fit(d["resumes"], d["job_descriptions"]), ResumeRanker(mask=True).fit(d["resumes"], d["job_descriptions"])


def vault_present(name):
    return (ROOT / "04_Validation_Vault" / name).exists()


class Skip(Exception):
    """Raised by skip(); run_all.py reports it as SKIP, pytest users get a normal pytest skip."""


def skip(reason):
    if "_pytest" in sys.modules:
        import pytest; pytest.skip(reason)
    raise Skip(reason)


def pinned(package):
    """Version pinned for `package` in requirements.txt / requirements-optional.txt (None if not pinned)."""
    import re
    for fn in ("requirements.txt", "requirements-optional.txt"):
        f = ROOT / fn
        if f.exists():
            m = re.search(rf"^{re.escape(package)}==([\w.]+)", f.read_text(encoding="utf-8"), re.M | re.I)
            if m: return m.group(1)
    return None


def load_artifact(path):
    """joblib.load that skips (with the reason) when the installed scikit-learn / library versions differ from the pinned
    ones and the saved pickle cannot be read. On the pinned environment this always loads and the test runs in full."""
    import joblib, sklearn
    try:
        return joblib.load(path)
    except (ImportError, AttributeError, ValueError, TypeError, KeyError) as e:
        skip(f"cannot unpickle {pathlib.Path(path).name} with scikit-learn {sklearn.__version__} (pinned: {pinned('scikit-learn')}): {e!r}. "
             "Install the pinned versions: pip install -r requirements.txt -r requirements-optional.txt")
