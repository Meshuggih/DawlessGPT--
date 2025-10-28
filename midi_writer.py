# /mnt/data/midi_writer.py
from dataclasses import dataclass, field
from typing import List, Optional
import struct, math

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
    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = int(ticks_per_beat)
        self.tracks: List[MIDITrack] = []
        self._tempo_bpm = 120.0
        self._time_sig = (4,4)

    def add_track(self, name: str, channel: int) -> int:
        self.tracks.append(MIDITrack(name=name, channel=channel))
        return len(self.tracks) - 1

    def add_note(self, track_index: int, time_beats: float, note: int, velocity: int,
                 duration_beats: float, channel: Optional[int]=None):
        tr = self.tracks[track_index]
        ch = tr.channel if channel is None else channel
        tr.notes.append(MIDINote(time=time_beats, note=note, velocity=velocity, duration=duration_beats, channel=ch))

    def add_cc(self, track_index: int, time_beats: float, cc_number: int, value: int, channel: Optional[int]=None):
        tr = self.tracks[track_index]
        ch = tr.channel if channel is None else channel
        tr.cc_events.append(MIDIControlChange(time=time_beats, cc_number=cc_number, value=value, channel=ch))

    def set_tempo(self, bpm: float) -> None:
        self._tempo_bpm = float(bpm)

    def set_time_signature(self, numerator: int, denominator: int) -> None:
        self._time_sig = (int(numerator), int(denominator))

    def _meta_tempo(self) -> bytes:
        mpqn = int(60000000 / self._tempo_bpm)
        return b'\x00\xFF\x51\x03' + struct.pack('>I', mpqn)[1:]

    def _meta_timesig(self) -> bytes:
        nn, ddv = self._time_sig
        dd = int(round(math.log2(ddv))) if ddv > 0 else 2
        cc = 24  # MIDI clocks per metronome click
        bb = 8   # 32nd notes per MIDI quarter
        return b'\x00\xFF\x58\x04' + bytes([nn & 0xFF, dd & 0xFF, cc & 0xFF, bb & 0xFF])

    def _meta_end(self) -> bytes:
        return b'\x00\xFF\x2F\x00'

    def _build_note_events(self, tr: MIDITrack) -> List[tuple]:
        events = []
        for n in tr.notes:
            on_tick  = int(round(n.time * self.ticks_per_beat))
            off_tick = int(round((n.time + n.duration) * self.ticks_per_beat))
            ch = n.channel & 0x0F
            events.append((on_tick,  bytes([0x90 | ch, n.note & 0x7F, n.velocity & 0x7F])))
            events.append((off_tick, bytes([0x80 | ch, n.note & 0x7F, 0x00])))
        return events

    def _build_cc_events(self, tr: MIDITrack) -> List[tuple]:
        events = []
        for c in tr.cc_events:
            tick = int(round(c.time * self.ticks_per_beat))
            ch = c.channel & 0x0F
            events.append((tick, bytes([0xB0 | ch, c.cc_number & 0x7F, c.value & 0x7F])))
        return events

    def _build_track_chunk(self, tr: MIDITrack) -> bytes:
        events = self._build_note_events(tr) + self._build_cc_events(tr)
        events.sort(key=lambda e: e[0])
        out = bytearray()
        last_tick = 0
        for tick, msg in events:
            delta = tick - last_tick
            out += _var_len(max(0, delta)) + msg
            last_tick = tick
        out += self._meta_end()
        return b'MTrk' + struct.pack('>I', len(out)) + bytes(out)

    def save(self, filename: str) -> None:
        header = b'MThd' + struct.pack('>IHHH', 6, 1, max(1,len(self.tracks))+1, self.ticks_per_beat)
        tempo_stream = bytearray()
        tempo_stream += self._meta_timesig()
        tempo_stream += self._meta_tempo()
        tempo_stream += self._meta_end()
        tempo_chunk = b'MTrk' + struct.pack('>I', len(tempo_stream)) + bytes(tempo_stream)
        chunks = [tempo_chunk] + [self._build_track_chunk(tr) for tr in self.tracks]
        with open(filename, 'wb') as f:
            f.write(header)
            for ch in chunks:
                f.write(ch)