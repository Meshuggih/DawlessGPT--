# /mnt/data/fx_processors.py
from typing import Optional
import numpy as np
from dsp_core import oversample_linear, downsample_linear, brickwall_limit

class Delay:
    def __init__(self, sample_rate: int = 48000, time_ms: float = 500.0,
                 feedback: float = 0.5, mix: float = 0.4,
                 time: Optional[str] = None, tempo_bpm: Optional[float] = None):
        self.sample_rate = sample_rate
        def _parse_time_str(s: str) -> float:
            s = s.strip()
            dotted = s.endswith('.')
            trip   = s.lower().endswith('t')
            base   = s[:-1] if (dotted or trip) else s
            num, den = base.split('/')
            beats = float(num)/float(den)
            if dotted: beats *= 1.5
            if trip:   beats *= (2.0/3.0)
            return beats
        if tempo_bpm is not None:
            t_str = time or "1/8"
            beats = _parse_time_str(t_str)
            self.delay_s = (60.0/float(tempo_bpm)) * beats
        else:
            self.delay_s = time_ms / 1000.0
        self.feedback = float(feedback)
        self.mix = float(mix)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1: x = x.mean(axis=1)
        n = len(x)
        d = int(round(self.delay_s * self.sample_rate))
        if d <= 0: return x.copy()
        y = np.zeros_like(x)
        buf = np.zeros(d, dtype=x.dtype)
        w = 0
        fb = self.feedback
        for i in range(n):
            delayed = buf[w]
            out = x[i] + delayed * self.mix
            y[i] = out
            buf[w] = x[i] + delayed * fb
            w += 1
            if w >= d: w = 0
        return y

class SchroederReverb:
    def __init__(self, sample_rate: int, room_size: float=0.8, decay: float=0.6, mix: float=1.0):
        self.sr = sample_rate
        self.mix = mix
        self.room_size = room_size
        self.decay = decay
        self.delays = [int(sample_rate*t) for t in (0.0297, 0.0371, 0.0411)]
        self.gains  = [0.805, 0.827, 0.783]
        self.buf = [np.zeros(d) for d in self.delays]
        self.idx = [0]*len(self.delays)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1: x = x.mean(axis=1)
        y = np.zeros_like(x)
        for i, s in enumerate(x):
            acc = 0.0
            for k in range(len(self.delays)):
                j = self.idx[k]
                acc += self.buf[k][j]
                self.buf[k][j] = s + self.buf[k][j]*self.gains[k]
                j += 1
                if j >= self.delays[k]: j = 0
                self.idx[k] = j
            y[i] = s*(1.0-self.mix) + acc*self.mix
        return y

class Compressor:
    def __init__(self, sample_rate: int, threshold_db: float=-24.0, ratio: float=4.0,
                 attack_ms: float=5.0, release_ms: float=80.0):
        self.sr = sample_rate
        self.thr = 10**(threshold_db/20.0)
        self.ratio = ratio
        self.a = np.exp(-1.0/(0.001*attack_ms*sample_rate))
        self.r = np.exp(-1.0/(0.001*release_ms*sample_rate))
        self.env = 0.0

    def process(self, signal: np.ndarray, sidechain_signal: np.ndarray) -> np.ndarray:
        sc = sidechain_signal
        n = min(len(signal), len(sc))
        out = signal.copy()
        for i in range(n):
            x = abs(sc[i])
            if x > self.env:
                self.env = self.env*self.a + (1-self.a)*x
            else:
                self.env = self.env*self.r + (1-self.r)*x
            gain = 1.0
            if self.env > self.thr:
                over = self.env / self.thr
                comp = over**(1.0 - 1.0/self.ratio)
                gain = 1.0/comp
            out[i] *= gain
        return out

class TruePeakLimiter:
    def __init__(self, ceiling_db: float=-0.3, oversample: int=8):
        self.ceiling_db = ceiling_db
        self.os = oversample

    def process(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            sig = signal.astype(np.float32, copy=False)
            up = oversample_linear(sig, self.os)
            lim = brickwall_limit(up, self.ceiling_db)
            return downsample_linear(lim, self.os)
        else:
            left = self.process(signal[:, 0])
            right = self.process(signal[:, 1])
            return np.stack([left, right], axis=1)
