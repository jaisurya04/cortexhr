import sys, pathlib, functools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cortexhr import config as C

def data_ok(): return C.data_path().exists()

@functools.lru_cache(maxsize=1)
def frame():
    from cortexhr.runs import load_model_frame
    return load_model_frame(True)
