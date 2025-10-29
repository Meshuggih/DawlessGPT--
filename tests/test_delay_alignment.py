import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fx_processors import Delay


def _expected_samples(division: str, bpm: float, sr: int) -> int:
    seconds = Delay.division_to_seconds(division, bpm)
    return int(round(seconds * sr))


def test_delay_alignment_basic():
    sr = 48000
    bpm = 120.0
    divisions = ["1/8", "1/8.", "1/8T", "1/16", "3/16"]
    for div in divisions:
        delay = Delay(sample_rate=sr, tempo_bpm=bpm, time=div, feedback=0.0, mix=1.0)
        impulse = np.zeros(sr // 2, dtype=np.float32)
        impulse[0] = 1.0
        processed = delay.process(impulse)
        wet = processed - impulse
        hit_indices = np.nonzero(np.abs(wet) > 1e-5)[0]
        assert hit_indices.size > 0, f"No delayed tap detected for {div}"
        first_hit = int(hit_indices[0])
        expected = _expected_samples(div, bpm, sr)
        assert math.isclose(
            first_hit,
            expected,
            rel_tol=0.0,
            abs_tol=2,
        ), f"Delay {div} misaligned: got {first_hit}, expected {expected}"
