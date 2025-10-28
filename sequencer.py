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

    def to_midi(self, midi, cc_resolver, bpm: float, collect_cc: Optional[list]=None):
        dt_beats = (self.humanize.get("time_ms",0.0)/1000.0) * (bpm/60.0)
        vel_var  = int(self.humanize.get("vel_var",0) or 0)
        for track_name, pat in self.patterns.items():
            ti = midi.add_track(track_name, channel=pat.channel)
            for st in pat.steps:
                if st.note is not None:
                    dur = max(0.05, float(st.length_beats))
                    dv = random.randint(-vel_var, vel_var) if vel_var>0 else 0
                    midi.add_note(ti, time_beats=float(st.beat)+dt_beats, note=int(st.note),
                                  velocity=int(max(1,min(127,st.vel + dv))),
                                  duration_beats=dur, channel=pat.channel)
                    if st.slide and pat.device:
                        cc, origin = cc_resolver.resolve("glide", device=pat.device)
                        if collect_cc is not None:
                            collect_cc.append({"track": track_name, "device": pat.device or "unknown",
                                               "param": "glide", "cc": int(cc), "origin": origin})
                        midi.add_cc(ti, time_beats=st.beat, cc_number=cc,
                                    value=cc_resolver.scale_0_1_to_0_127(1.0), channel=pat.channel)
            for st in pat.steps:
                if not st.locks: continue
                for param, v in st.locks.items():
                    cc, origin = cc_resolver.resolve(param, device=pat.device)
                    if collect_cc is not None:
                        collect_cc.append({"track": track_name, "device": pat.device or "unknown",
                                           "param": param, "cc": int(cc), "origin": origin})
                    midi.add_cc(ti, time_beats=st.beat, cc_number=cc,
                                value=cc_resolver.scale_0_1_to_0_127(v), channel=pat.channel)

    def tick(self, ctx):
        if self.mod_matrix is not None:
            self.mod_matrix.apply(ctx)