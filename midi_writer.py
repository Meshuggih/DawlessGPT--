# /mnt/data/midi_writer.py
"""Minimal MIDI writer tailored for DawlessGPT exports.

The previous iterations of this module exposed a handful of helper
functions but the project now requires a proper ``MIDIFile`` class that can
manage tempo, time signature, tracks and arbitrary MIDI events.

This implementation focuses on the features that DawlessGPT needs:

* format 1 MIDI file with a dedicated tempo/meta track;
* per-track note and control change events, including optional program
  changes;
* helpers for injecting tempo and time-signature meta events;
* deterministic ordering of events so overlapping notes (used for slides)
  remain intact.

The class intentionally keeps the surface small while remaining fully
functional for the test-suite and the renderer.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
import math
import struct

__all__ = [
    "MIDINote",
    "MIDIControlChange",
    "MIDITrack",
    "MIDIFile",
]

def _var_len(value: int) -> bytes:
    buffer = value & 0x7F
    out = bytearray()
    while (value >> 7):
        value >>= 7
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)

@dataclass
class MIDINote:
    time: float       # beats
    note: int
    velocity: int
    duration: float   # beats
    channel: int = 0

@dataclass
class MIDIControlChange:
    time: float        # beats
    cc_number: int     # 0..127
    value: int         # 0..127
    channel: int = 0

@dataclass
class MIDITrack:
    name: str = "Track"
    channel: int = 0
    notes: List[MIDINote] = field(default_factory=list)
    cc_events: List[MIDIControlChange] = field(default_factory=list)
    program: Optional[int] = None

class MIDIFile:
    """Simple MIDI writer with tempo and time-signature support."""

    def __init__(self, ticks_per_beat: int = 480) -> None:
        if ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive")
        self.ticks_per_beat = int(ticks_per_beat)
        self.tracks: List[MIDITrack] = []
        self._tempo_bpm = 120.0
        self._time_sig = (4, 4)

    # ------------------------------------------------------------------
    # Global metadata helpers
    def set_tempo(self, bpm: float) -> None:
        """Set the project tempo in beats per minute."""

        if bpm <= 0:
            raise ValueError("Tempo must be > 0 BPM")
        self._tempo_bpm = float(bpm)

    def set_time_signature(self, numerator: int, denominator: int) -> None:
        """Set the global time signature (e.g. ``4/4``)."""

        if numerator <= 0 or denominator <= 0:
            raise ValueError("Time signature must use positive integers")
        # MIDI expects denominators as power-of-two exponents. Clamp to sensible
        # values to keep malformed configs in check.
        power = int(round(math.log(denominator, 2))) if denominator > 0 else 2
        power = max(0, min(power, 7))
        self._time_sig = (int(numerator), int(2 ** power))

    # ------------------------------------------------------------------
    # Track / event management
    def add_track(self, name: str, channel: int) -> int:
        """Create a new MIDI track and return its index."""

        track = MIDITrack(name=name or "Track", channel=int(channel))
        self.tracks.append(track)
        return len(self.tracks) - 1

    def set_program(self, track_index: int, program: int) -> None:
        """Optionally attach a program change to a track."""

        self.tracks[track_index].program = int(program) & 0x7F

    def add_note(
        self,
        track_index: int,
        time_beats: float,
        note: int,
        velocity: int,
        duration_beats: float,
        channel: Optional[int] = None,
    ) -> None:
        tr = self.tracks[track_index]
        ch = tr.channel if channel is None else int(channel)
        note_event = MIDINote(
            time=float(time_beats),
            note=int(note),
            velocity=int(max(1, min(127, velocity))),
            duration=max(0.0, float(duration_beats)),
            channel=int(ch),
        )
        tr.notes.append(note_event)

    def add_cc(
        self,
        track_index: int,
        time_beats: float,
        cc_number: int,
        value: int,
        channel: Optional[int] = None,
    ) -> None:
        tr = self.tracks[track_index]
        ch = tr.channel if channel is None else int(channel)
        cc_event = MIDIControlChange(
            time=float(time_beats),
            cc_number=int(cc_number) & 0x7F,
            value=int(max(0, min(127, value))),
            channel=int(ch),
        )
        tr.cc_events.append(cc_event)

    # ------------------------------------------------------------------
    # Internal helpers
    def beats_to_ticks(self, beats: float) -> int:
        return int(round(float(beats) * self.ticks_per_beat))

    def _meta_tempo(self) -> bytes:
        mpqn = int(round(60000000.0 / self._tempo_bpm))
        mpqn = max(1, min(mpqn, 0xFFFFFF))
        return b"\xFF\x51\x03" + struct.pack(">I", mpqn)[1:]

    def _meta_timesig(self) -> bytes:
        nn, dd_val = self._time_sig
        dd = int(round(math.log(dd_val, 2))) if dd_val > 0 else 2
        dd = max(0, min(dd, 7))
        cc = 24  # MIDI clocks per metronome click
        bb = 8   # 32nd notes per MIDI quarter
        return b"\xFF\x58\x04" + bytes([nn & 0xFF, dd & 0xFF, cc & 0xFF, bb & 0xFF])

    @staticmethod
    def _meta_end() -> bytes:
        return b"\xFF\x2F\x00"

    @staticmethod
    def _meta_track_name(name: str) -> bytes:
        data = (name or "Track").encode("utf-8", "ignore")
        return b"\xFF\x03" + _var_len(len(data)) + data

    def _program_change(self, program: Optional[int], channel: int) -> Optional[bytes]:
        if program is None:
            return None
        return bytes([0xC0 | (channel & 0x0F), program & 0x7F])

    def _note_events(self, tr: MIDITrack) -> Iterable[Tuple[int, int, bytes]]:
        for n in tr.notes:
            on_tick = self.beats_to_ticks(n.time)
            off_tick = self.beats_to_ticks(n.time + n.duration)
            if off_tick <= on_tick:
                off_tick = on_tick + 1
            ch = n.channel & 0x0F
            on_msg = bytes([0x90 | ch, n.note & 0x7F, n.velocity & 0x7F])
            off_msg = bytes([0x80 | ch, n.note & 0x7F, 0x00])
            yield (on_tick, 30, on_msg)
            yield (off_tick, 20, off_msg)

    def _cc_events(self, tr: MIDITrack) -> Iterable[Tuple[int, int, bytes]]:
        for c in tr.cc_events:
            tick = self.beats_to_ticks(c.time)
            ch = c.channel & 0x0F
            msg = bytes([0xB0 | ch, c.cc_number & 0x7F, c.value & 0x7F])
            yield (tick, 10, msg)

    def _build_track_chunk(self, tr: MIDITrack) -> bytes:
        events: List[Tuple[int, int, bytes]] = []
        events.append((0, 0, self._meta_track_name(tr.name)))
        prog_msg = self._program_change(tr.program, tr.channel)
        if prog_msg is not None:
            events.append((0, 5, prog_msg))
        events.extend(self._cc_events(tr))
        events.extend(self._note_events(tr))
        events.sort(key=lambda item: (item[0], item[1]))

        out = bytearray()
        last_tick = 0
        for tick, _prio, msg in events:
            delta = max(0, tick - last_tick)
            out += _var_len(delta) + msg
            last_tick = tick
        out += _var_len(0) + self._meta_end()
        return b"MTrk" + struct.pack(">I", len(out)) + bytes(out)

    def _tempo_track(self) -> bytes:
        events = bytearray()
        events += _var_len(0) + self._meta_timesig()
        events += _var_len(0) + self._meta_tempo()
        events += _var_len(0) + self._meta_end()
        return b"MTrk" + struct.pack(">I", len(events)) + bytes(events)

    def build_file_bytes(self) -> bytes:
        header = b"MThd" + struct.pack(">IHHH", 6, 1, len(self.tracks) + 1, self.ticks_per_beat)
        chunks = [self._tempo_track()] + [self._build_track_chunk(tr) for tr in self.tracks]
        return header + b"".join(chunks)

    def save(self, filename: str) -> None:
        data = self.build_file_bytes()
        with open(filename, "wb") as fh:
            fh.write(data)