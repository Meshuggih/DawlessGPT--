# /mnt/data/sequencer.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable
import random
import json
import os
from math import floor

# Cache global pour les templates de styles afin d'éviter les relectures.
_STYLE_TEMPLATES_CACHE: Optional[Dict[str, Dict[str, object]]] = None


def _load_style_templates() -> Dict[str, Dict[str, object]]:
    """Charge les templates de styles à partir des assets connus.

    Si `set_style_templates` a été invoqué, on réutilise la copie en mémoire.
    Dans le cas contraire, on lit depuis le répertoire courant en tolérant les
    deux orthographes du fichier historique (avec ou sans «S» final).
    """

    global _STYLE_TEMPLATES_CACHE
    if _STYLE_TEMPLATES_CACHE is not None:
        return _STYLE_TEMPLATES_CACHE

    candidates = (
        "STYLE_TEMPLATES.json",
        "STYLE_TEMPLATE.json",
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data: Dict[str, Dict[str, object]] = {}
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
        cleaned = "\n".join(lines)
        try:
            parsed = json.loads(cleaned or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed  # dernière occurrence valide gagne
            break

    _STYLE_TEMPLATES_CACHE = data
    return data


def set_style_templates(templates: Optional[Dict[str, Dict[str, object]]]) -> None:
    """Injecte un cache pré-chargé de templates.

    On effectue une copie défensive pour garantir que l'appelant peut garder
    le contrôle de sa structure sans risque de mutation involontaire.
    """

    global _STYLE_TEMPLATES_CACHE
    if templates is None:
        _STYLE_TEMPLATES_CACHE = None
    else:
        _STYLE_TEMPLATES_CACHE = json.loads(json.dumps(templates))


def _bjorklund(steps: int, pulses: int) -> List[int]:
    """Renvoie une séquence binaire de type Euclidienne.

    Implémentation compacte utilisant la formule des différences de plancher
    pour obtenir une distribution quasi régulière sans dépendre d'une boucle
    récursive plus coûteuse. Les valeurs retournées sont 1 pour un coup et 0
    pour un silence.
    """

    if steps <= 0:
        return []
    pulses = max(0, min(pulses, steps))
    if pulses == 0:
        return [0] * steps
    if pulses == steps:
        return [1] * steps
    return [
        floor(((i + 1) * pulses) / steps) - floor((i * pulses) / steps)
        for i in range(steps)
    ]


def _roll_steps(base: Iterable["Step"], rotation: int) -> List["Step"]:
    steps = list(base)
    if not steps:
        return steps
    rotation = rotation % len(steps)
    if rotation == 0:
        return steps
    return steps[rotation:] + steps[:rotation]


def generate_pattern(style: str, steps: int, seed: int) -> Dict[str, List["Step"]]:
    """Génère des motifs pour un style donné.

    Les pistes percussives (kick, snare, hats, perc, break) utilisent une
    répartition Euclidienne tandis que les pistes mélodiques (pad, keys, stabs,
    bass) produisent un arpège cyclique. Le résultat est déterministe pour un
    triplet `(style, steps, seed)` donné.
    """

    templates = _load_style_templates()
    tpl = templates.get(style, {}) if isinstance(templates, dict) else {}
    track_list = []
    if isinstance(tpl, dict):
        tracks_val = tpl.get("tracks")  # type: ignore[assignment]
        if isinstance(tracks_val, list):
            track_list = [str(t) for t in tracks_val]
    if not track_list:
        track_list = ["kick", "pad"]
    else:
        for essential in ("kick", "pad"):
            if essential not in track_list:
                track_list.append(essential)

    rng = random.Random((hash(style) & 0xFFFFFFFF) ^ int(seed))
    patterns: Dict[str, List[Step]] = {}

    beat_resolution = max(1, int(steps))
    loop_length_beats = float(beat_resolution)
    beat_spacing = loop_length_beats / max(1, beat_resolution)

    def _make_euclidean(track_name: str, note: int, pulses: int) -> List[Step]:
        rotation = rng.randint(0, max(0, beat_resolution - 1))
        pattern_bits = _bjorklund(beat_resolution, pulses)
        velocity_base = 112 if track_name == "kick" else 96
        hits: List[Step] = []
        for idx, bit in enumerate(pattern_bits):
            if bit:
                beat = float(idx) * beat_spacing
                vel = int(max(1, min(127, velocity_base + rng.randint(-6, 6))))
                hits.append(
                    Step(
                        beat=beat,
                        note=note,
                        vel=vel,
                        length_beats=beat_spacing * 0.95,
                    )
                )
        return _roll_steps(hits, rotation)

    def _make_arp(root: int) -> List[Step]:
        chord_pool = [
            (0, 3, 7, 10),  # m7
            (0, 4, 7, 11),  # maj7
            (0, 5, 7, 10),  # sus4 add7
        ]
        progression = rng.choices(chord_pool, k=max(1, beat_resolution // 4))
        steps_out: List[Step] = []
        vel_base = 78
        for block, chord in enumerate(progression):
            beat = float(block * 4)
            if beat >= loop_length_beats:
                break
            offset = rng.choice(chord)
            vel = int(max(1, min(120, vel_base + rng.randint(-8, 8))))
            lock_amount = 0.35 + 0.25 * (block / max(1, len(progression) - 1))
            steps_out.append(
                Step(
                    beat=beat,
                    note=root + offset,
                    vel=vel,
                    length_beats=4.0,
                    locks={"cutoff": min(0.95, lock_amount)},
                )
            )
        return steps_out

    for track_name in track_list:
        lower = track_name.lower()
        if lower in {"kick", "snare", "break"}:
            pulses = max(1, beat_resolution // (4 if lower == "kick" else 8))
            note = 36 if lower == "kick" else 38
            patterns[track_name] = _make_euclidean(track_name, note, pulses)
        elif lower in {"hats", "perc"}:
            pulses = max(1, beat_resolution // 2)
            note = 42 if lower == "hats" else 39
            patterns[track_name] = _make_euclidean(track_name, note, pulses)
        elif lower in {"pad", "keys", "stabs", "bass"}:
            root = 48 if lower == "pad" else 36
            patterns[track_name] = _make_arp(root)

    return patterns

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
        jitter_range_beats = abs(self.humanize.get("time_ms", 0.0)) / 1000.0
        jitter_range_beats *= bpm / 60.0
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
                jitter = (
                    random.uniform(-jitter_range_beats, jitter_range_beats)
                    if jitter_range_beats > 0.0
                    else 0.0
                )
                start = float(st.beat) + jitter
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
                        next_start = float(next_st.beat)
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