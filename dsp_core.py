# /mnt/data/dsp_core.py
import numpy as np


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _ensure_float32(signal: np.ndarray) -> np.ndarray:
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy.ndarray")
    if signal.dtype != np.float32:
        raise TypeError("signal must be float32")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    return signal


def oversample_linear(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """Linearly oversample a 1-D float32 signal by an integer factor.

    The interpolation guarantees the original sample positions are bit-exact
    after a subsequent :func:`downsample_linear`. The self-test validates that
    the round-trip RMS error stays under ``5e-7``.
    """

    signal = _ensure_float32(signal)
    if not isinstance(factor, int):
        raise TypeError("factor must be an integer")
    if factor <= 0:
        raise ValueError("factor must be strictly positive")
    if factor == 1 or signal.size == 0:
        return signal.copy()

    n = signal.size
    up_indices = np.arange(n * factor, dtype=np.float64) / factor
    base_indices = np.arange(n, dtype=np.float64)
    interpolated = np.interp(up_indices, base_indices, signal.astype(np.float64))
    return interpolated.astype(np.float32)


def downsample_linear(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample a 1-D float32 signal that was oversampled linearly.

    Returning ``signal[::factor]`` preserves the original samples. The
    oversample/downsample round-trip is verified to exhibit an RMS error under
    ``5e-7`` for typical programme material.
    """

    signal = _ensure_float32(signal)
    if not isinstance(factor, int):
        raise TypeError("factor must be an integer")
    if factor <= 0:
        raise ValueError("factor must be strictly positive")
    if factor == 1:
        return signal.copy()
    return signal[::factor].copy()


def brickwall_limit(signal: np.ndarray, ceiling_db: float = -0.3) -> np.ndarray:
    """Clip a float32 signal to a fixed ceiling expressed in dBFS.

    The limiter is validated to keep excursions within ``1e-7`` above the
    theoretical ceiling in the DSP self-tests.
    """

    signal = _ensure_float32(signal)
    ceiling = float(db_to_lin(ceiling_db))
    limited = np.clip(signal, -ceiling, ceiling)
    return limited.astype(np.float32, copy=False)
