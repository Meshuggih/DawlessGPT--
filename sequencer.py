# /mnt/data/sequencer.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random

@dataclass
class Step:
    beat: float
    note: Optional[int] = None
    vel: int = 100
    length_beats: float = 1.0
    locks: Dict[str, float] = field(default_factory=dict)   # 0..1
    slide: bool = False

@dataclass
class Pattern:
    steps: List[Step] = field(default_factory=list)
    channel: int = 0
    device: Optional[str] = None

class Sequencer:
    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = int(ticks_per_beat)
        self.patterns: Dict[str, Pattern] = {}
        self.mod_matrix = None
        self.humanize = dict(time_ms=0.0, vel_var=0)

    def set_pattern(self, track_name: str, pattern: Pattern):
        self.patterns[track_name] = pattern

    def to_midi(self, midi, cc_resolver, bpm: float, collect_cc: Optional[list] = None):
        dt_beats = (self.humanize.get("time_ms", 0.0) / 1000.0) * (bpm / 60.0)
        vel_var = int(self.humanize.get("vel_var", 0) or 0)

        def _register_cc(
            track: str,
            device: Optional[str],
            param: str,
            cc_number: int,
            origin: str,
        ) -> None:
            if collect_cc is None:
                return
            device_name = device or "unknown"
            for entry in collect_cc:
                if (
                    entry.get("device") == device_name
                    and entry.get("param") == param
                    and int(entry.get("cc", -1)) == int(cc_number)
                    and entry.get("origin") == origin
                ):
                    entry["count"] = int(entry.get("count", 0)) + 1
                    tracks = entry.setdefault("tracks", {})
                    tracks[track] = int(tracks.get(track, 0)) + 1
                    return
            collect_cc.append(
                {
                    "device": device_name,
                    "param": param,
                    "cc": int(cc_number),
                    "origin": origin,
                    "count": 1,
                    "tracks": {track: 1},
                }
            )

        for track_name, pat in self.patterns.items():
            ti = midi.add_track(track_name, channel=pat.channel)
            steps = sorted(pat.steps, key=lambda s: s.beat)
            note_indices = [i for i, st in enumerate(steps) if st.note is not None]
            for idx in note_indices:
                st = steps[idx]
                assert st.note is not None
                base_dur = max(0.05, float(st.length_beats))
                dv = random.randint(-vel_var, vel_var) if vel_var > 0 else 0
                start = float(st.beat) + dt_beats
                velocity = int(max(1, min(127, st.vel + dv)))

                # Determine overlap for slides.
                duration = base_dur
                if st.slide:
                    slide_overlap = 0.05
                    next_st = None
                    for j in note_indices:
                        if j > idx:
                            next_st = steps[j]
                            break
                    if next_st is not None:
                        next_start = float(next_st.beat) + dt_beats
                        duration = max(duration, (next_start - start) + slide_overlap)
                midi.add_note(
                    ti,
                    time_beats=start,
                    note=int(st.note),
                    velocity=velocity,
                    duration_beats=duration,
                    channel=pat.channel,
                )

                if st.slide and pat.device:
                    cc, origin = cc_resolver.resolve("glide", device=pat.device)
                    midi.add_cc(
                        ti,
                        time_beats=float(st.beat),
                        cc_number=cc,
                        value=cc_resolver.scale_0_1_to_0_127(1.0),
                        channel=pat.channel,
                    )
                    _register_cc(track_name, pat.device, "glide", cc, origin)

            for st in steps:
                if not st.locks:
                    continue
                for param, value in st.locks.items():
                    cc, origin = cc_resolver.resolve(param, device=pat.device)
                    midi.add_cc(
                        ti,
                        time_beats=float(st.beat),
                        cc_number=cc,
                        value=cc_resolver.scale_0_1_to_0_127(value),
                        channel=pat.channel,
                    )
                    _register_cc(track_name, pat.device, param, cc, origin)

    def tick(self, ctx):
        if self.mod_matrix is not None:
            self.mod_matrix.apply(ctx)