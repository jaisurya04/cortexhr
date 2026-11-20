"""Structured (key=value) logging used by the API, monitor and console."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger; handlers are attached once."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s", "%Y-%m-%dT%H:%M:%S%z"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(level)
    return log
