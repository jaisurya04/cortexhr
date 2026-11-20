"""Read-only bridge to the Phase 1-3 code base.

Phase 2 and Phase 3 both ship a package called ``cortexhr``. Phase 3's copy is imported first and Phase 2's directory is
appended to its ``__path__`` so ``cortexhr.phase2`` resolves too. Nothing here modifies a file outside ``phase4/``.
"""
from __future__ import annotations

import sys
import warnings

from .settings import PHASE2, PHASE3

_DONE = False


def bootstrap() -> None:
    """Make ``cortexhr.phase2`` and ``cortexhr.phase3`` importable (idempotent)."""
    global _DONE
    if _DONE:
        return
    for p in (str(PHASE3 / "src"), str(PHASE3 / "05_Starter")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cortexhr  # noqa: WPS433  (Phase 3 copy)

    extra = str(PHASE2 / "src" / "cortexhr")
    if extra not in cortexhr.__path__:
        cortexhr.__path__.append(extra)
    warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
    _DONE = True
