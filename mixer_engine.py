"""Mixing utilities for DawlessGPT-Σ."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fx_processors import Delay, SchroederReverb, TruePeakLimiter


@dataclass
class TrackSettings:
    """Mutable mix controls for a single track."""

    name: str
    volume_db: float = 0.0
    pan: float = 0.0
    reverb_send: float = 0.0
    delay_send: float = 0.0


@dataclass
class SidechainRoute:
    """Describe a sidechain relation from one or many ``sources`` to ``dst``."""

    sources: List[str]
    dst: str
    depth_db: float
    attack_ms: float
    release_ms: float
    shape: str = "exp"

    def __post_init__(self) -> None:
        if not isinstance(self.sources, list):
            self.sources = list(self.sources)
        self.sources = [str(src) for src in self.sources]
        self.shape = str(self.shape).lower()


class MixerEngine:
    """Handle bus routing, gain staging and brickwall limiting.

    The mixer accumulates signals in float64 to reduce rounding noise, applies an
    equal-power pan-law, enforces low-frequency mono compatibility and ensures a
    single brickwall limiter is applied before the signal leaves the engine.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        audio_cfg = config.get("audio", {})
        export_cfg = config.get("export", {})
        fx_defaults = config.get("fx_defaults", {})

        self.sample_rate = int(audio_cfg.get("sample_rate", 44100))
        self.mono_bass_hz = float(audio_cfg.get("mono_bass_hz", 0.0) or 0.0)
        self.headroom_db = float(export_cfg.get("headroom_db", 1.0))
        self._limiter_ceiling_db = float(export_cfg.get("target_peak_dbtp", -0.3))

        self.tempo_bpm: float = float(audio_cfg.get("tempo_default", 120.0))

        self.cfg_fx_delay_time = fx_defaults.get("delay_time", "1/8.")
        self.cfg_fx_delay_feedback = float(fx_defaults.get("delay_feedback", 0.4))
        self.cfg_fx_delay_mix = float(fx_defaults.get("delay_mix", 0.25))

        self.cfg_fx_reverb_mix = float(fx_defaults.get("reverb_mix", 0.22))
        self.cfg_fx_reverb_decay = float(fx_defaults.get("reverb_decay", 0.6))
        self.cfg_fx_reverb_room_size = float(fx_defaults.get("reverb_room_size", 0.8))
        self.cfg_fx_reverb_pre_delay = fx_defaults.get("reverb_pre_delay", "1/64")

        self.master_pregain: float = 1.0

        self.reverb_bus = SchroederReverb(
            self.sample_rate,
            tempo_bpm=self.tempo_bpm,
            room_size=self.cfg_fx_reverb_room_size,
            decay=self.cfg_fx_reverb_decay,
            mix=self.cfg_fx_reverb_mix,
            pre_delay=self.cfg_fx_reverb_pre_delay,
        )
        self.delay_bus = Delay(
            self.sample_rate,
            tempo_bpm=self.tempo_bpm,
            feedback=self.cfg_fx_delay_feedback,
            mix=self.cfg_fx_delay_mix,
            time=self.cfg_fx_delay_time,
        )
        self._tp_limiter = TruePeakLimiter(self._limiter_ceiling_db)

        self.track_order: List[str] = []
        self.tracks: Dict[str, TrackSettings] = {}
        self._sidechain_routes: List[SidechainRoute] = []

    # ------------------------------------------------------------------ setup --
    def set_track_order(self, names: Sequence[str]) -> None:
        """Declare the ordering used during mixdown."""

        self.track_order = list(names)
        for name in self.track_order:
            self.tracks.setdefault(name, TrackSettings(name=name))

    def configure_track(self, name: str, **overrides: float) -> None:
        """Update mix settings for ``name`` (creating it when necessary)."""

        track = self.tracks.setdefault(name, TrackSettings(name=name))
        for key, value in overrides.items():
            if hasattr(track, key):
                setattr(track, key, float(value))

    def set_tempo(self, bpm: float) -> None:
        """Update tempo-aware processors (currently the delay bus)."""

        self.tempo_bpm = float(bpm)
        self.configure_delay(
            time=self.cfg_fx_delay_time,
            feedback=self.cfg_fx_delay_feedback,
            mix=self.cfg_fx_delay_mix,
            tempo_bpm=bpm,
        )
        self.configure_reverb(
            mix=self.cfg_fx_reverb_mix,
            decay=self.cfg_fx_reverb_decay,
            room_size=self.cfg_fx_reverb_room_size,
            pre_delay=self.cfg_fx_reverb_pre_delay,
            tempo_bpm=bpm,
        )

    def configure_delay(
        self,
        *,
        time: str,
        feedback: float,
        mix: float,
        tempo_bpm: float,
    ) -> None:
        """(Re)initialise the tempo-synchronised delay bus."""

        self.cfg_fx_delay_time = time
        self.cfg_fx_delay_feedback = float(feedback)
        self.cfg_fx_delay_mix = float(mix)
        self.tempo_bpm = float(tempo_bpm)
        self.delay_bus = Delay(
            self.sample_rate,
            feedback=self.cfg_fx_delay_feedback,
            mix=self.cfg_fx_delay_mix,
            time=self.cfg_fx_delay_time,
            tempo_bpm=self.tempo_bpm,
        )

    def configure_reverb(
        self,
        *,
        mix: float,
        decay: float,
        room_size: float,
        pre_delay: str,
        tempo_bpm: Optional[float] = None,
    ) -> None:
        """(Re)initialise the Schroeder reverb bus with tempo-synced pre-delay."""

        self.cfg_fx_reverb_mix = float(mix)
        self.cfg_fx_reverb_decay = float(decay)
        self.cfg_fx_reverb_room_size = float(room_size)
        self.cfg_fx_reverb_pre_delay = str(pre_delay)
        if tempo_bpm is not None:
            self.tempo_bpm = float(tempo_bpm)

        self.reverb_bus = SchroederReverb(
            self.sample_rate,
            tempo_bpm=self.tempo_bpm,
            room_size=self.cfg_fx_reverb_room_size,
            decay=self.cfg_fx_reverb_decay,
            mix=self.cfg_fx_reverb_mix,
            pre_delay=self.cfg_fx_reverb_pre_delay,
        )

    def clear_sidechains(self) -> None:
        self._sidechain_routes.clear()

    def create_sidechain(
        self,
        src_names: Sequence[str] | str,
        dst_name: str,
        *,
        depth_db: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 80.0,
        shape: str = "exp",
    ) -> None:
        """Register a sidechain route supporting multiple sources."""

        if isinstance(src_names, (str, bytes)):
            sources = [str(src_names)]
        else:
            sources = [str(src) for src in src_names]
        if not sources:
            raise ValueError("sidechain sources list must not be empty")

        shape_lc = str(shape).lower()
        if shape_lc not in {"lin", "exp", "log"}:
            raise ValueError(f"invalid sidechain shape '{shape}' (expected lin/exp/log)")

        route = SidechainRoute(
            sources=sources,
            dst=str(dst_name),
            depth_db=float(depth_db),
            attack_ms=float(attack_ms),
            release_ms=float(release_ms),
            shape=shape_lc,
        )
        self._sidechain_routes.append(route)

    def create_sc(
        self,
        src_name: str,
        dst_name: str,
        depth_db: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 80.0,
        shape: str = "exp",
    ) -> None:
        """Backward-compatible wrapper for legacy renderers."""

        self.create_sidechain(
            src_name,
            dst_name,
            depth_db=float(depth_db),
            attack_ms=float(attack_ms),
            release_ms=float(release_ms),
            shape=str(shape),
        )

    # --------------------------------------------------------------- rendering --
    @staticmethod
    def _first_order_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
        if cutoff_hz <= 0.0:
            return np.zeros_like(signal)
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        dt = 1.0 / float(sample_rate)
        alpha = dt / (rc + dt)
        out = np.zeros_like(signal)
        acc = 0.0
        for i, value in enumerate(signal):
            acc += alpha * (value - acc)
            out[i] = acc
        return out

    def _track_signal(self, name: str, buffers: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        buf = buffers.get(name)
        if buf is None:
            return None
        return np.asarray(buf)

    def _sidechain_envelope(
        self,
        signal: np.ndarray,
        *,
        attack_ms: float,
        release_ms: float,
    ) -> np.ndarray:
        rectified = np.abs(signal.astype(np.float64, copy=False))
        if not rectified.size:
            return rectified

        eps_time = 1e-9
        attack_s = max(float(attack_ms) / 1000.0, eps_time)
        release_s = max(float(release_ms) / 1000.0, eps_time)
        attack_coeff = np.exp(-1.0 / (attack_s * self.sample_rate))
        release_coeff = np.exp(-1.0 / (release_s * self.sample_rate))

        env = np.zeros_like(rectified)
        acc = 0.0
        for i, sample in enumerate(rectified):
            coeff = attack_coeff if sample > acc else release_coeff
            acc = (1.0 - coeff) * sample + coeff * acc
            env[i] = acc
        return env

    def _sidechain_gain_from_signal(
        self,
        signal: np.ndarray,
        *,
        length: int,
        depth_db: float,
        attack_ms: float,
        release_ms: float,
        shape: str,
    ) -> np.ndarray:
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = np.asarray(signal, dtype=np.float64)
        padded = np.zeros(length, dtype=np.float64)
        copy_len = min(length, signal.size)
        if copy_len:
            padded[:copy_len] = signal[:copy_len]

        env = self._sidechain_envelope(padded, attack_ms=attack_ms, release_ms=release_ms)
        eps = 1e-12
        peak = float(np.max(env))
        if peak <= eps:
            return np.ones(length, dtype=np.float64)
        env_norm = np.clip(env / (peak + eps), 0.0, 1.0)

        shape_lc = shape.lower()
        if shape_lc == "exp":
            shaped = env_norm ** 2.0
        elif shape_lc == "log":
            shaped = np.sqrt(env_norm)
        else:
            shaped = env_norm

        depth = float(depth_db)
        if depth <= 0.0:
            return np.ones(length, dtype=np.float64)
        ratio = 1.0 + max(1.0, depth / 1.5)
        target_gain = 1.0 / ratio
        gain = 1.0 - (1.0 - target_gain) * shaped
        return np.clip(gain, target_gain, 1.0)

    def render_mix(
        self,
        track_buffers: Dict[str, np.ndarray],
        *,
        sidechain_kick: Optional[np.ndarray] = None,
        sidechain_sources: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Mix ``track_buffers`` down to stereo float32.

        The returned buffer is always two channels in float32 format and has the
        length of the longest input track.
        """

        if not track_buffers:
            return np.zeros((0, 2), dtype=np.float32)

        order = self.track_order or list(track_buffers.keys())
        max_len = max(len(track_buffers[name]) for name in order if name in track_buffers)
        if max_len == 0:
            return np.zeros((0, 2), dtype=np.float32)

        master = np.zeros((max_len, 2), dtype=np.float64)
        reverb_input = np.zeros(max_len, dtype=np.float64)
        delay_input = np.zeros(max_len, dtype=np.float64)

        sc_by_dst: Dict[str, List[SidechainRoute]] = {}
        for route in self._sidechain_routes:
            sc_by_dst.setdefault(route.dst, []).append(route)
        extra_sources = sidechain_sources or {}

        for name in order:
            buf = self._track_signal(name, track_buffers)
            if buf is None:
                continue

            track = self.tracks.setdefault(name, TrackSettings(name=name))
            volume = 10.0 ** (track.volume_db / 20.0)
            pan = float(np.clip(track.pan, -1.0, 1.0))
            angle = (pan + 1.0) * 0.25 * np.pi
            left_gain = np.cos(angle)
            right_gain = np.sin(angle)

            track_len = buf.shape[0] if buf.ndim > 1 else buf.size
            routes = sc_by_dst.get(name, [])
            gain_curve: Optional[np.ndarray] = None
            if routes:
                total_gain = np.ones(track_len, dtype=np.float64)
                applied = False
                for route in routes:
                    route_gain = np.ones(track_len, dtype=np.float64)
                    for src_name in route.sources:
                        sc_signal = extra_sources.get(src_name)
                        if sc_signal is None and src_name in track_buffers:
                            sc_signal = track_buffers[src_name]
                        if (
                            sc_signal is None
                            and sidechain_kick is not None
                            and src_name.lower() == "kick"
                        ):
                            sc_signal = sidechain_kick
                        if sc_signal is None:
                            continue
                        gain = self._sidechain_gain_from_signal(
                            sc_signal,
                            length=track_len,
                            depth_db=route.depth_db,
                            attack_ms=route.attack_ms,
                            release_ms=route.release_ms,
                            shape=route.shape,
                        )
                        route_gain *= gain
                        applied = True
                    total_gain *= route_gain
                if applied:
                    gain_curve = total_gain

            if buf.ndim == 2 and buf.shape[1] >= 2:
                left_sig = np.asarray(buf[:, 0], dtype=np.float64)
                right_sig = np.asarray(buf[:, 1], dtype=np.float64)
                if gain_curve is not None:
                    left_sig = left_sig * gain_curve[: left_sig.size]
                    right_sig = right_sig * gain_curve[: right_sig.size]
                left_sig *= volume * left_gain
                right_sig *= volume * right_gain
                send_signal = 0.5 * (left_sig + right_sig)
            else:
                mono = np.asarray(buf, dtype=np.float64)
                if gain_curve is not None:
                    mono = mono * gain_curve[: mono.size]
                mono *= volume

                left_sig = mono * left_gain
                right_sig = mono * right_gain
                send_signal = mono

            left_len = min(max_len, left_sig.size)
            right_len = min(max_len, right_sig.size)
            master[:left_len, 0] += left_sig[:left_len]
            master[:right_len, 1] += right_sig[:right_len]

            if track.reverb_send:
                send_len = min(max_len, send_signal.size)
                reverb_input[:send_len] += send_signal[:send_len] * float(track.reverb_send)
            if track.delay_send:
                send_len = min(max_len, send_signal.size)
                delay_input[:send_len] += send_signal[:send_len] * float(track.delay_send)

        if np.any(reverb_input):
            rev = self.reverb_bus.process(reverb_input.astype(np.float32))
            rev = np.asarray(rev, dtype=np.float64)
            length = min(max_len, rev.size)
            master[:length, 0] += rev[:length] * 0.5
            master[:length, 1] += rev[:length] * 0.5

        if np.any(delay_input):
            delay = self.delay_bus.process(delay_input.astype(np.float32))
            delay = np.asarray(delay, dtype=np.float64)
            length = min(max_len, delay.size)
            master[:length, 0] += delay[:length] * 0.5
            master[:length, 1] += delay[:length] * 0.5

        if self.mono_bass_hz > 0.0:
            low_l = self._first_order_lowpass(master[:, 0], self.mono_bass_hz, self.sample_rate)
            low_r = self._first_order_lowpass(master[:, 1], self.mono_bass_hz, self.sample_rate)
            low_m = 0.5 * (low_l + low_r)
            master[:, 0] = low_m + (master[:, 0] - low_l)
            master[:, 1] = low_m + (master[:, 1] - low_r)

        master *= float(self.master_pregain)

        limited = self._tp_limiter.process(master.astype(np.float32))
        if limited.ndim == 1:
            limited = limited[:, None]
        return np.asarray(limited, dtype=np.float32)
