# /mnt/data/analysis_tools.py
import numpy as np

def measure_lufs(stereo: np.ndarray, sr: int) -> float:
    x = stereo.mean(axis=1) if stereo.ndim == 2 else stereo
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    lufs = 20*np.log10(rms + 1e-12) - 0.691
    return float(lufs)

def measure_true_peak(stereo: np.ndarray, sr: int) -> float:
    x = stereo if stereo.ndim==1 else stereo.max(axis=1)
    peak = np.max(np.abs(x)) + 1e-12
    db = 20*np.log10(peak)
    return float(db)

def measure_correlation(stereo: np.ndarray) -> float:
    if stereo.ndim == 1:
        return 1.0
    L = stereo[:,0] - stereo[:,0].mean()
    R = stereo[:,1] - stereo[:,1].mean()
    num = float(np.sum(L*R))
    den = float(np.sqrt(np.sum(L*L)*np.sum(R*R)) + 1e-12)
    return num/den

def mono_bass_check(stereo: np.ndarray, sr: int, fc: float=150.0) -> float:
    if stereo.ndim==1: return 1.0
    L, R = stereo[:,0], stereo[:,1]
    rc = 2.0 * np.pi * fc / float(sr); a = rc / (rc + 1.0)
    def lp(x):
        y = np.zeros_like(x); acc = 0.0
        for i,xx in enumerate(x): acc = acc + a*(xx-acc); y[i]=acc
        return y
    lL, lR = lp(L), lp(R)
    lL -= lL.mean(); lR -= lR.mean()
    num = float(np.sum(lL*lR)); den = float(np.sqrt(np.sum(lL*lL)*np.sum(lR*lR)) + 1e-12)
    return num/den

def detect_clicks(stereo: np.ndarray, thresh: float=0.95) -> int:
    x = stereo if stereo.ndim==1 else stereo.mean(axis=1)
    dx = np.abs(np.diff(x))
    T = float(thresh)
    return int(np.sum(dx > T))