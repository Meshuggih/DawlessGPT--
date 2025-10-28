# /mnt/data/fx_processors.py
from typing import List, Optional

import numpy as np

from dsp_core import brickwall_limit, downsample_linear, oversample_linear


class Delay:
    """Tempo-synchronised feedback delay with a circular buffer."""

    SUPPORTED_BASE_DIVISIONS = ("1/1", "1/2", "1/4", "1/8", "3/16", "1/16", "1/32", "1/64")

    def __init__(
        self,
        sample_rate: int = 48000,
        *,
        tempo_bpm: float = 120.0,
        time: str = "1/8",
        feedback: float = 0.5,
        mix: float = 0.4,
        time_ms: Optional[float] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._buffer = np.zeros(1, dtype=np.float32)
        self._write_idx = 0

        self.feedback = float(feedback)
        self.mix = float(mix)

        self._time_division = str(time)
        self._time_ms = float(time_ms) if time_ms is not None else None
        self.tempo_bpm = float(tempo_bpm)

        self._validate_time(self._time_division)
        self._update_delay_samples()

    # ------------------------------------------------------------------ helpers --
    @classmethod
    def _beats_for_division(cls, division: str) -> float:
        division = division.strip()
        dotted = division.endswith(".")
        triplet = division.lower().endswith("t")
        base = division[:-1] if (dotted or triplet) else division
        if base not in cls.SUPPORTED_BASE_DIVISIONS:
            raise ValueError(
                f"Division temporelle non supportée pour le délai: '{division}'."
            )
        num_str, den_str = base.split("/")
        beats = float(num_str) / float(den_str)
        if dotted:
            beats *= 1.5
        if triplet:
            beats *= 2.0 / 3.0
        return beats

    @classmethod
    def division_to_seconds(cls, division: str, tempo_bpm: float) -> float:
        if tempo_bpm <= 0.0:
            raise ValueError("Le tempo doit être strictement positif pour le délai tempo-sync.")
        beats = cls._beats_for_division(division)
        return (60.0 / float(tempo_bpm)) * beats

    def _validate_time(self, division: str) -> None:
        # Raises when invalid.
        self._beats_for_division(division)

    def set_tempo(self, tempo_bpm: float) -> None:
        self.tempo_bpm = float(tempo_bpm)
        self._update_delay_samples()

    def set_time(self, division: str) -> None:
        self._validate_time(division)
        self._time_division = str(division)
        self._update_delay_samples()

    def set_time_ms(self, time_ms: Optional[float]) -> None:
        self._time_ms = None if time_ms is None else float(time_ms)
        self._update_delay_samples()

    def _update_delay_samples(self) -> None:
        if self._time_ms is not None:
            seconds = max(0.0, self._time_ms / 1000.0)
        else:
            seconds = self.division_to_seconds(self._time_division, self.tempo_bpm)
        delay_samples = int(round(seconds * self.sample_rate))
        self.delay_samples = max(1, delay_samples)
        if self._buffer.size != self.delay_samples:
            self._buffer = np.zeros(self.delay_samples, dtype=np.float32)
            self._write_idx = 0

    # ---------------------------------------------------------------- processing --
    def process(self, x: np.ndarray) -> np.ndarray:
        signal = np.asarray(x, dtype=np.float32)
        if signal.ndim != 1:
            signal = signal.mean(axis=1)
        out = np.zeros_like(signal)

        fb = float(np.clip(self.feedback, -0.999, 0.999))
        mix = float(self.mix)
        buffer = self._buffer
        idx = self._write_idx
        size = self.delay_samples

        for i, sample in enumerate(signal):
            delayed = buffer[idx]
            out[i] = sample + delayed * mix
            buffer[idx] = sample + delayed * fb
            idx += 1
            if idx >= size:
                idx = 0

        self._write_idx = idx
        return out


class SchroederReverb:
    """Classic Schroeder reverb (parallel comb + series all-pass)."""

    SUPPORTED_BASE_DIVISIONS = Delay.SUPPORTED_BASE_DIVISIONS

    def __init__(
        self,
        sample_rate: int,
        *,
        tempo_bpm: float = 120.0,
        room_size: float = 0.8,
        decay: float = 0.6,
        mix: float = 1.0,
        pre_delay: str = "1/64",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.mix = float(mix)
        self.room_size = float(room_size)
        self.decay = float(decay)
        self.tempo_bpm = float(tempo_bpm)
        self.pre_delay_division = str(pre_delay)

        self._pre_delay_buffer = np.zeros(1, dtype=np.float32)
        self._pre_idx = 0

        # Base comb timings (seconds) taken from Schroeder/Moorer style networks.
        base_comb_times = np.array([0.0297, 0.0371, 0.0411, 0.0437], dtype=np.float64)
        self._comb_times = base_comb_times
        self._comb_buffers: List[np.ndarray] = []
        self._comb_indices: List[int] = []

        # Base all-pass timings (seconds).
        self._allpass_times = np.array([0.005, 0.0017], dtype=np.float64)
        self._allpass_buffers: List[np.ndarray] = []
        self._allpass_indices: List[int] = []

        self._configure_filters()

    # ------------------------------------------------------------------ helpers --
    @classmethod
    def _beats_for_division(cls, division: str) -> float:
        return Delay._beats_for_division(division)

    def set_tempo(self, tempo_bpm: float) -> None:
        self.tempo_bpm = float(tempo_bpm)
        self._configure_filters()

    def set_mix(self, mix: float) -> None:
        self.mix = float(mix)

    def set_room_size(self, room_size: float) -> None:
        self.room_size = float(room_size)
        self._configure_filters()

    def set_decay(self, decay: float) -> None:
        self.decay = float(decay)
        self._configure_filters()

    def set_pre_delay(self, division: str) -> None:
        Delay._beats_for_division(division)
        self.pre_delay_division = str(division)
        self._configure_filters()

    def _configure_filters(self) -> None:
        # Pre-delay based on tempo.
        seconds = Delay.division_to_seconds(self.pre_delay_division, self.tempo_bpm)
        samples = max(1, int(round(seconds * self.sample_rate)))
        if self._pre_delay_buffer.size != samples:
            self._pre_delay_buffer = np.zeros(samples, dtype=np.float32)
            self._pre_idx = 0

        # Comb filters scaling.
        scale = np.clip(self.room_size, 0.5, 1.5)
        comb_times = np.maximum(0.005, self._comb_times * scale)
        comb_lengths = np.maximum(1, (comb_times * self.sample_rate).astype(int))
        base_gains = np.array([0.805, 0.827, 0.783, 0.764], dtype=np.float64)
        gain_scale = np.clip(self.decay, 0.1, 1.0)
        self._comb_gains = np.clip(base_gains * gain_scale, 0.2, 0.98)

        if len(self._comb_buffers) != len(comb_lengths):
            self._comb_buffers = []
            self._comb_indices = []
        for i, length in enumerate(comb_lengths):
            if i >= len(self._comb_buffers):
                self._comb_buffers.append(np.zeros(length, dtype=np.float32))
                self._comb_indices.append(0)
            elif self._comb_buffers[i].size != length:
                self._comb_buffers[i] = np.zeros(length, dtype=np.float32)
                self._comb_indices[i] = 0
        self._comb_lengths = comb_lengths

        # All-pass filters scaling (lighter response changes with room size also).
        ap_scale = np.clip(self.room_size, 0.5, 1.5)
        ap_times = np.maximum(0.0007, self._allpass_times * ap_scale)
        ap_lengths = np.maximum(1, (ap_times * self.sample_rate).astype(int))
        self._allpass_gains = np.array([0.7, 0.5], dtype=np.float64)

        if len(self._allpass_buffers) != len(ap_lengths):
            self._allpass_buffers = []
            self._allpass_indices = []
        for i, length in enumerate(ap_lengths):
            if i >= len(self._allpass_buffers):
                self._allpass_buffers.append(np.zeros(length, dtype=np.float32))
                self._allpass_indices.append(0)
            elif self._allpass_buffers[i].size != length:
                self._allpass_buffers[i] = np.zeros(length, dtype=np.float32)
                self._allpass_indices[i] = 0
        self._allpass_lengths = ap_lengths

    # ---------------------------------------------------------------- processing --
    def process(self, x: np.ndarray) -> np.ndarray:
        signal = np.asarray(x, dtype=np.float32)
        if signal.ndim != 1:
            signal = signal.mean(axis=1)
        out = np.zeros_like(signal)

        pre_buffer = self._pre_delay_buffer
        pre_idx = self._pre_idx
        pre_size = pre_buffer.size

        comb_buffers = self._comb_buffers
        comb_indices = self._comb_indices
        comb_lengths = self._comb_lengths
        comb_gains = self._comb_gains

        allpass_buffers = self._allpass_buffers
        allpass_indices = self._allpass_indices
        allpass_lengths = self._allpass_lengths
        allpass_gains = self._allpass_gains

        mix = float(self.mix)

        for n, sample in enumerate(signal):
            # Pre-delay stage
            delayed = pre_buffer[pre_idx]
            pre_buffer[pre_idx] = sample
            pre_idx += 1
            if pre_idx >= pre_size:
                pre_idx = 0

            # Parallel comb filters
            comb_sum = 0.0
            for i in range(len(comb_buffers)):
                idx = comb_indices[i]
                buf = comb_buffers[i]
                comb_out = buf[idx]
                buf[idx] = delayed + comb_out * comb_gains[i]
                idx += 1
                if idx >= comb_lengths[i]:
                    idx = 0
                comb_indices[i] = idx
                comb_sum += comb_out

            stage = comb_sum

            # Series all-pass filters
            for i in range(len(allpass_buffers)):
                idx = allpass_indices[i]
                buf = allpass_buffers[i]
                ap_out = buf[idx]
                gain = allpass_gains[i]
                buf[idx] = stage + ap_out * gain
                stage = -stage * gain + ap_out
                idx += 1
                if idx >= allpass_lengths[i]:
                    idx = 0
                allpass_indices[i] = idx

            out[n] = (1.0 - mix) * sample + mix * stage

        self._pre_idx = pre_idx
        return out

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
