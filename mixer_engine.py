"""Mixing utilities for DawlessGPT-Σ."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dsp_core import brickwall_limit
from fx_processors import Compressor, Delay, SchroederReverb


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
    """Describe a sidechain relation from ``src`` to ``dst``."""

    src: str
    dst: str
    depth_db: float
    attack_ms: float
    release_ms: float
    shape: str = "exp"


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

        self.cfg_fx_delay_time = fx_defaults.get("delay_time", "1/8.")
        self.cfg_fx_delay_feedback = float(fx_defaults.get("delay_feedback", 0.4))
        self.cfg_fx_delay_mix = float(fx_defaults.get("delay_mix", 0.25))

        room_size = float(fx_defaults.get("reverb_room_size", 0.8))
        decay = float(fx_defaults.get("reverb_decay", 0.6))
        reverb_mix = float(fx_defaults.get("reverb_mix", 0.22))

        self.reverb_bus = SchroederReverb(
            self.sample_rate,
            room_size=room_size,
            decay=decay,
            mix=reverb_mix,
        )
        self.delay_bus = Delay(
            self.sample_rate,
            feedback=self.cfg_fx_delay_feedback,
            mix=self.cfg_fx_delay_mix,
            time=self.cfg_fx_delay_time,
            tempo_bpm=float(audio_cfg.get("tempo_default", 120.0)),
        )

        self.track_order: List[str] = []
        self.tracks: Dict[str, TrackSettings] = {}
        self._sidechain_routes: List[SidechainRoute] = []
        self.tempo_bpm: float = float(audio_cfg.get("tempo_default", 120.0))

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

    def clear_sidechains(self) -> None:
        self._sidechain_routes.clear()

    def create_sc(
        self,
        src_name: str,
        dst_name: str,
        depth_db: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 80.0,
        shape: str = "exp",
    ) -> None:
        """Register a sidechain compression route."""

        route = SidechainRoute(
            src=str(src_name),
            dst=str(dst_name),
            depth_db=float(depth_db),
            attack_ms=float(attack_ms),
            release_ms=float(release_ms),
            shape=str(shape),
        )
        self._sidechain_routes.append(route)

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

            if buf.ndim == 2 and buf.shape[1] >= 2:
                left_sig = np.asarray(buf[:, 0], dtype=np.float64) * volume
                right_sig = np.asarray(buf[:, 1], dtype=np.float64) * volume
                send_signal = 0.5 * (left_sig + right_sig)
            else:
                mono = np.asarray(buf, dtype=np.float64)

                for route in sc_by_dst.get(name, []):
                    sc_signal = None
                    if route.src in track_buffers:
                        sc_signal = track_buffers[route.src]
                    if route.src in extra_sources:
                        sc_signal = extra_sources[route.src]
                    if sc_signal is None and sidechain_kick is not None and route.src.lower() == "kick":
                        sc_signal = sidechain_kick
                    if sc_signal is None:
                        continue
                    comp = Compressor(
                        self.sample_rate,
                        threshold_db=-24.0,
                        ratio=1.0 + max(1.0, route.depth_db / 1.5),
                        attack_ms=route.attack_ms,
                        release_ms=route.release_ms,
                    )
                    sc_arr = np.asarray(sc_signal, dtype=np.float32)
                    if sc_arr.ndim > 1:
                        sc_arr = sc_arr[:, 0]
                    mono32 = mono.astype(np.float32, copy=False)
                    ducked = comp.process(mono32, sc_arr[: mono32.size])
                    gain = ducked.astype(np.float64) / (mono + 1e-12)
                    if route.shape == "exp":
                        gain = gain ** 2
                    elif route.shape == "log":
                        gain = np.sqrt(np.maximum(gain, 0.0))
                    mono = mono * gain

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

        headroom_gain = 10.0 ** (-self.headroom_db / 20.0)
        master *= headroom_gain

        stereo32 = master.astype(np.float32)
        left_limited = brickwall_limit(stereo32[:, 0], self._limiter_ceiling_db)
        right_limited = brickwall_limit(stereo32[:, 1], self._limiter_ceiling_db)
        return np.stack([left_limited, right_limited], axis=1)
