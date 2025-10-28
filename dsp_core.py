# /mnt/data/dsp_core.py
import numpy as np

def db_to_lin(db: float) -> float:
    return 10.0**(db/20.0)

def oversample_linear(x: np.ndarray, factor: int = 4) -> np.ndarray:
    if factor <= 1: return x
    n = len(x)
    up = np.zeros(n*factor, dtype=x.dtype)
    up[::factor] = x
    for i in range(1, factor):
        up[i::factor] = ( (factor-i)/factor )*x + (i/factor)*np.concatenate((x[1:], x[-1:]))
    return up

def downsample_linear(x: np.ndarray, factor: int = 4) -> np.ndarray:
    if factor <= 1: return x
    return x[::factor]

def brickwall_limit(x: np.ndarray, ceiling_db: float = -0.3) -> np.ndarray:
    c = db_to_lin(ceiling_db)
    return np.clip(x, -c, c)