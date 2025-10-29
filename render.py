# /mnt/data/render.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import copy
import ast
import json
import os
import warnings
import struct
import wave
import numpy as np
import logging
import re
from datetime import datetime, timezone

from mixer_engine import MixerEngine
from fx_processors import Delay
from dsp_core import brickwall_limit
from arrangement_engine import ArrangementEngine
from modulation_matrix import ModulationMatrix, DropBusSource
from midi_writer import MIDIFile
from midi_cc_db import CCResolver
from sequencer import (
    Sequencer,
    Pattern,
    Step,
    generate_pattern,
    set_style_templates,
)

CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
ASSET_CACHE: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

_DEFAULT_AUDIO = {
    "sample_rate": 44100,
    "bit_depth": 24,
    "buffer": 512,
    "time_signature": "4/4",
    "tempo_default": 120,
    "key_default": "Cm",
    "tuning_ref_hz": 440.0,
    "mono_bass_hz": 150,
}

_DEFAULT_ASSET_PATHS = {
    "style_templates": "./STYLE_TEMPLATES.json",
    "drop_protocols": "./DROP_PROTOCOLS.yaml",
    "synth_presets": "./SYNTH_PRESETS.json",
    "drum_kits": "./DRUM_KITS.json",
    "studio_theory": "./STUDIO_THEORY.json",
    "studio_config": "./STUDIO_CONFIG.json",
    "cc_overrides": "./cc_mappings.json",
}

_DEFAULT_PATHS = {
    "base": ".",
    "output": "./output",
    "midi_export": "./midi_export",
    "assets": _DEFAULT_ASSET_PATHS,
}

_ASSET_ALIASES = {
    "style_templates": ("STYLE_TEMPLATES.json", "STYLE_TEMPLATE.json"),
}

_ASSET_VALIDATORS: Dict[str, Tuple[type, str]] = {
    "style_templates": (dict, "Le fichier de templates de styles doit contenir un objet JSON."),
    "drop_protocols": (dict, "Les protocoles de drop doivent être une table YAML."),
    "synth_presets": (dict, "Les presets synthé doivent être un objet JSON."),
    "drum_kits": (dict, "Les kits de batterie doivent être un objet JSON."),
    "studio_theory": (dict, "La théorie studio doit être un objet JSON."),
    "studio_config": (dict, "La configuration studio doit être un objet JSON."),
    "cc_mappings": (dict, "Les mappings CC doivent être un objet JSON."),
}


def _parse_scalar(value: str) -> Any:
    token = value.strip()
    if not token or token.lower() in {"null", "~"}:
        return None
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_inline_dict(text: str) -> Dict[str, Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        current.append(ch)
    if current:
        parts.append("".join(current).strip())

    out: Dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        key, _, raw_val = part.partition(":")
        key = key.strip().strip('"\'')
        out[key] = _parse_value(raw_val.strip())
    return out


def _parse_value(token: str) -> Any:
    if not token:
        return None
    stripped = token.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return _parse_inline_dict(stripped)
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return stripped
    return _parse_scalar(stripped)


def _load_yaml_like(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except Exception:
        pass

    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        trimmed = raw.split("#", 1)[0].rstrip()
        if not trimmed.strip():
            continue
        indent = len(trimmed) - len(trimmed.lstrip(" "))
        content = trimmed.strip()
        lines.append((indent, content))

    if not lines:
        return {}

    root: Any = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(lines):
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("unexpected list item outside list context")
            value_part = content[2:].strip()
            if not value_part:
                container: Any
                if next_line and next_line[0] > indent and next_line[1].startswith("- "):
                    container = []
                else:
                    container = {}
                parent.append(container)
                stack.append((indent + 2, container))
            else:
                parent.append(_parse_value(value_part))
            continue

        key, _, raw_val = content.partition(":")
        if not _:
            raise ValueError(f"ligne YAML invalide: {content}")
        key = key.strip().strip('"\'')
        value_part = raw_val.strip()
        if not value_part:
            container = [] if (next_line and next_line[0] > indent and next_line[1].startswith("- ")) else {}
            if isinstance(parent, list):
                parent.append({key: container})
                stack.append((indent + 2, container))
            else:
                parent[key] = container
                stack.append((indent + 2, container))
        else:
            value = _parse_value(value_part)
            if isinstance(parent, list):
                parent.append({key: value})
            else:
                parent[key] = value

    return root


def _resolve_path(base_dir: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = os.path.expanduser(str(value))
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(base_dir, path))
    return path


def load_asset(name: str, path: str, *, use_cache: bool = True) -> Any:
    """Load an asset file (JSON or YAML) with light validation and caching."""

    if not path:
        raise ValueError(f"Chemin vide pour l'asset '{name}'.")

    abs_path = os.path.abspath(path)
    if use_cache and abs_path in ASSET_CACHE:
        return copy.deepcopy(ASSET_CACHE[abs_path])

    candidate_paths = [abs_path]
    for alt in _ASSET_ALIASES.get(name, ()):  # try known aliases
        if not os.path.isabs(alt):
            candidate_paths.append(os.path.join(os.path.dirname(abs_path), alt))
        else:
            candidate_paths.append(alt)

    actual_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            actual_path = candidate
            break
    if actual_path is None:
        raise FileNotFoundError(f"Asset '{name}' introuvable (recherché: {candidate_paths}).")

    _, ext = os.path.splitext(actual_path)
    ext = ext.lower()
    with open(actual_path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            text = f.read()
            data = _load_yaml_like(text) or {}
        elif ext == ".json":
            text = f.read()
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
            cleaned = "\n".join(lines)
            data = json.loads(cleaned or "{}")
        else:
            raise ValueError(f"Extension de fichier non supportée pour l'asset '{name}': {ext}")

    expected_type, err_msg = _ASSET_VALIDATORS.get(name, (dict, ""))
    if expected_type and not isinstance(data, expected_type):
        warnings.warn(err_msg or f"Asset '{name}' invalide: type attendu {expected_type}.")
        data = expected_type()  # type: ignore[call-arg]

    snapshot = copy.deepcopy(data)
    ASSET_CACHE[abs_path] = copy.deepcopy(snapshot)
    ASSET_CACHE[actual_path] = copy.deepcopy(snapshot)
    return snapshot


def load_config_safe(path: str = "config.yaml", *, use_cache: bool = True) -> Dict[str, Any]:
    """Load YAML config safely, injecting defaults and caching the result."""

    abs_path = os.path.abspath(path)
    if use_cache and abs_path in CONFIG_CACHE:
        return copy.deepcopy(CONFIG_CACHE[abs_path])

    base_dir = os.path.dirname(abs_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            cfg_raw = _load_yaml_like(f.read()) or {}
    except FileNotFoundError:
        warnings.warn(f"Fichier de configuration '{path}' introuvable. Utilisation des valeurs par défaut.")
        cfg_raw = {}

    if not isinstance(cfg_raw, dict):
        warnings.warn("La configuration doit être une structure de type dictionnaire. Valeurs par défaut appliquées.")
        cfg_raw = {}

    cfg: Dict[str, Any] = copy.deepcopy(cfg_raw)

    def _ensure_section(key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        section = cfg.get(key)
        if not isinstance(section, dict):
            warnings.warn(f"Section '{key}' manquante ou invalide — utilisation des valeurs par défaut.")
            section = copy.deepcopy(defaults)
        else:
            for sub_key, default_value in defaults.items():
                if isinstance(default_value, dict):
                    sub_section = section.get(sub_key)
                    if not isinstance(sub_section, dict):
                        warnings.warn(f"Clé '{key}.{sub_key}' manquante ou invalide — valeurs par défaut injectées.")
                        section[sub_key] = copy.deepcopy(default_value)
                    else:
                        for leaf_key, leaf_default in default_value.items():
                            if leaf_key not in sub_section:
                                sub_section[leaf_key] = copy.deepcopy(leaf_default)
                else:
                    section.setdefault(sub_key, copy.deepcopy(default_value))
        cfg[key] = section
        return section

    _ensure_section("audio", _DEFAULT_AUDIO)
    paths = _ensure_section("paths", _DEFAULT_PATHS)
    styles = cfg.get("styles")
    if not isinstance(styles, dict):
        warnings.warn("Section 'styles' manquante ou invalide — dictionnaire vide utilisé.")
        styles = {}
    cfg["styles"] = styles

    paths["base"] = _resolve_path(base_dir, paths.get("base", base_dir)) or base_dir
    for key in ("output", "midi_export"):
        paths[key] = _resolve_path(paths["base"], paths.get(key, _DEFAULT_PATHS[key]))

    assets_cfg = paths.get("assets", {})
    if not isinstance(assets_cfg, dict):
        warnings.warn("Section 'paths.assets' invalide — valeurs par défaut utilisées.")
        assets_cfg = copy.deepcopy(_DEFAULT_ASSET_PATHS)
    for asset_key, default_path in _DEFAULT_ASSET_PATHS.items():
        raw_value = assets_cfg.get(asset_key, default_path)
        if asset_key not in assets_cfg or raw_value is None:
            warnings.warn(f"Chemin pour l'asset '{asset_key}' manquant — valeur par défaut utilisée.")
            raw_value = default_path
        resolved = _resolve_path(paths["base"], raw_value)
        assets_cfg[asset_key] = resolved if resolved is not None else _resolve_path(paths["base"], default_path)
    paths["assets"] = assets_cfg

    cfg["_base_dir"] = paths["base"]

    loaded_assets: Dict[str, Any] = {}
    for asset_key in (
        "style_templates",
        "drop_protocols",
        "synth_presets",
        "drum_kits",
        "studio_theory",
        "studio_config",
        "cc_overrides",
    ):
        asset_path = assets_cfg.get(asset_key)
        if not asset_path:
            continue
        try:
            loaded_assets[asset_key] = load_asset(asset_key, asset_path)
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
        except Exception as exc:
            warnings.warn(f"Échec du chargement de l'asset '{asset_key}': {exc}")

    cfg["_assets"] = loaded_assets

    CONFIG_CACHE[abs_path] = copy.deepcopy(cfg)
    return copy.deepcopy(cfg)

@dataclass
class SessionPlan:
    style: str
    bpm: float
    seed: int
    duration_s: float
    sections: List[str]

def _fmt_name(template: str, **kw) -> str:
    return template.format(**kw)

def _init_rng(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32-1))

def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _sanitize_component(*parts: Any) -> str:
    raw = "_".join(str(p) for p in parts if p is not None)
    raw = raw.strip()
    if not raw:
        return "session"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    cleaned = cleaned.strip("-_")
    return cleaned or "session"


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    if data.ndim == 1:
        data_2d = data[:, None]
    else:
        data_2d = data
    n_samples = data_2d.shape[0]
    if n_samples == 0:
        return data
    ratio = float(sr_out) / float(sr_in)
    new_len = max(1, int(round(n_samples * ratio)))
    xp = np.linspace(0.0, float(n_samples - 1), num=n_samples, dtype=np.float64)
    x_new = np.linspace(0.0, float(n_samples - 1), num=new_len, dtype=np.float64)
    resampled = np.stack([
        np.interp(x_new, xp, data_2d[:, ch]).astype(np.float32)
        for ch in range(data_2d.shape[1])
    ], axis=1)
    if data.ndim == 1:
        return resampled[:, 0]
    return resampled

def _apply_style_to_mixer(cfg: Dict[str, Any], style: str, mixer: MixerEngine, bpm: float):
    fxd = cfg.get("fx_defaults", {})
    mixer.configure_delay(
        time=fxd.get("delay_time", "1/8."),
        feedback=float(fxd.get("delay_feedback", 0.4)),
        mix=float(fxd.get("delay_mix", 0.25)),
        tempo_bpm=bpm,
    )
    mixer.configure_reverb(
        mix=float(fxd.get("reverb_mix", 0.22)),
        decay=float(fxd.get("reverb_decay", 0.6)),
        room_size=float(fxd.get("reverb_room_size", 0.8)),
        pre_delay=fxd.get("reverb_pre_delay", "1/64"),
        tempo_bpm=bpm,
    )
    mixer.clear_sidechains()
    style_cfg = cfg["styles"][style]
    for r in style_cfg.get("sidechain_routes", []):
        mixer.create_sidechain(
            r["src"],
            r["dst"],
            depth_db=r.get("depth_db", 3.0),
            attack_ms=r.get("attack_ms", 5),
            release_ms=r.get("release_ms", 80),
            shape=r.get("shape", "exp"),
        )

def _wire_modmatrix_drop(
    cfg: Dict[str, Any],
    arranger: ArrangementEngine,
) -> Tuple[ModulationMatrix, List[Dict[str, Any]]]:
    matrix_cfg = cfg.get("modulation_matrix", {})
    mm = ModulationMatrix(sum_mode=matrix_cfg.get("sum_mode", "weighted_clamp"))
    drop_bus = matrix_cfg.get("drop_bus_name", "DROP")
    drop_routes: List[Dict[str, Any]] = [
        {
            "protocol": "filter_sweep_master",
            "destination": "pad.reverb_send",
            "amount": 1.0,
            "range": (0.0, 0.85),
            "curve": "exp",
        },
        {
            "protocol": "reverb_wash_bass_filter",
            "destination": "bass.reverb_send",
            "amount": 1.0,
            "range": (0.0, 0.7),
            "curve": "lin",
        },
    ]
    for route in drop_routes:
        src = DropBusSource(
            route["protocol"],
            bus=drop_bus,
            protocols_path=arranger.protocols_path,
            protocols=arranger.protocols,
        )
        rng = route.get("range", (0.0, 1.0))
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            rng = (0.0, 1.0)
        mm.add(
            src,
            route["destination"],
            amt=float(route.get("amount", 1.0)),
            rng=(float(rng[0]), float(rng[1])),
            curve=str(route.get("curve", "lin")),
        )
    return mm, drop_routes

def _render_tracks(plan: SessionPlan, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    sr = int(cfg["audio"]["sample_rate"])
    N = int(plan.duration_s * sr)
    t = np.arange(N) / sr
    beat_duration = 60.0 / max(1e-6, float(plan.bpm))

    pattern_steps = max(16, int(cfg.get("patterns", {}).get("steps", 16)))
    generated = generate_pattern(plan.style, pattern_steps, plan.seed)

    loop_beats = float(pattern_steps)
    for step_list in generated.values():
        for st in step_list:
            loop_beats = max(loop_beats, float(st.beat) + float(st.length_beats))
    loops = max(1, int(np.ceil((plan.duration_s / beat_duration) / loop_beats)))

    def _schedule_hits(steps: List[Step], env: np.ndarray) -> np.ndarray:
        buffer = np.zeros(N, dtype=np.float32)
        hit_len = env.shape[0]
        for loop in range(loops):
            loop_offset_beats = loop * loop_beats
            for st in steps:
                if st.note is None:
                    continue
                start_time = (loop_offset_beats + float(st.beat)) * beat_duration
                start_idx = int(round(start_time * sr))
                if start_idx >= N:
                    continue
                amp = float(st.vel) / 127.0
                end_idx = min(N, start_idx + hit_len)
                seg = end_idx - start_idx
                if seg <= 0:
                    continue
                buffer[start_idx:end_idx] += amp * env[:seg]
        return buffer

    # Kick: décroissance rapide inspirée d'un circuit analogique.
    kick_len = int(0.8 * sr)
    kick_t = np.arange(kick_len) / sr
    kick_env = np.sin(2 * np.pi * 52.0 * kick_t) * np.exp(-kick_t * 9.5)
    kick_steps = generated.get("kick", [])
    kick = _schedule_hits(kick_steps, kick_env.astype(np.float32))

    # Bass: reste un sinus simple pour servir de référence grave stable.
    bass = (0.3 * np.sin(2 * np.pi * 55 * t)).astype(np.float32)

    # Pad : arpège cyclique avec enveloppe douce et suivi de hauteur.
    pad = np.zeros(N, dtype=np.float32)
    pad_steps = generated.get("pad", [])
    attack = int(0.25 * sr)
    release = int(0.6 * sr)
    for loop in range(loops):
        loop_offset_beats = loop * loop_beats
        for st in pad_steps:
            if st.note is None:
                continue
            start_time = (loop_offset_beats + float(st.beat)) * beat_duration
            length_s = max(beat_duration * float(st.length_beats), 0.5)
            start_idx = int(round(start_time * sr))
            if start_idx >= N:
                continue
            stop_idx = min(N, int(round((start_time + length_s) * sr)) + release)
            if stop_idx <= start_idx:
                continue
            local_len = stop_idx - start_idx
            local_t = np.arange(local_len) / sr
            freq = 440.0 * (2.0 ** ((float(st.note) - 69.0) / 12.0))
            osc = np.sin(2 * np.pi * freq * local_t)
            env = np.ones(local_len)
            a = min(attack, local_len)
            if a > 0:
                env[:a] *= np.linspace(0.0, 1.0, a, endpoint=False)
            r = min(release, local_len)
            if r > 0:
                env[-r:] *= np.linspace(1.0, 0.0, r, endpoint=True)
            vel = float(st.vel) / 127.0
            pad[start_idx:stop_idx] += (vel * 0.3) * (osc * env)

    return {"kick": kick.astype(np.float32), "bass": bass, "pad": pad.astype(np.float32)}

def _float_to_int24_pcm(x: np.ndarray, dither: bool=False) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    if x.ndim == 1:
        x = x[:, None]
    if dither:
        lsb = 1.0 / 8388607.0
        rng = np.random.default_rng()
        d = (rng.random(x.shape) - rng.random(x.shape)) * lsb
        x = np.clip(x + d, -1.0, 1.0)
    s = (x * 8388607.0).astype(np.int32)
    out = bytearray()
    for frame in s:
        for ch in frame:
            out += bytes((ch & 0xFF, (ch >> 8) & 0xFF, (ch >> 16) & 0xFF))
    return bytes(out)

def _write_wav_24(path: str, data: np.ndarray, sr: int, normalize: bool=False, target_peak_db: float=-0.3, dither: bool=True):
    nchan = data.shape[1] if data.ndim == 2 else 1
    if normalize:
        peak = float(np.max(np.abs(data)) + 1e-12)
        target = 10.0**(target_peak_db/20.0)
        if peak > 0: data = data * min(1.0, target/peak)
    with wave.open(path, 'wb') as w:
        w.setnchannels(nchan)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(_float_to_int24_pcm(data, dither=dither))

def _export_audio_and_midi(
    stereo: np.ndarray,
    trackbufs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    plan: SessionPlan,
    session_dir: str,
    midi_dir: str,
    filename_ctx: Dict[str, Any],
    *,
    target_sample_rate: int = 48000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    sr_cfg = int(cfg["audio"]["sample_rate"])
    stereo_to_write = _resample(stereo, sr_cfg, target_sample_rate)
    master_name = _fmt_name(cfg["export"]["filenames"]["master"], **filename_ctx)
    master_path = os.path.join(session_dir, master_name)
    _write_wav_24(
        master_path,
        stereo_to_write.astype(np.float32),
        target_sample_rate,
        normalize=cfg["export"].get("normalize", True),
        target_peak_db=float(cfg["export"].get("target_peak_dbtp", -0.3)),
        dither=cfg["export"].get("dither", True),
    )

    stems: Dict[str, str] = {}
    if cfg["export"].get("render_stems", False):
        for name, buf in trackbufs.items():
            stem_ctx = dict(filename_ctx)
            stem_ctx["track"] = _sanitize_component(name)
            nm = _fmt_name(cfg["export"]["filenames"]["stem"], **stem_ctx)
            p = os.path.join(session_dir, nm)
            stem_buf = _resample(buf, sr_cfg, target_sample_rate)
            _write_wav_24(
                p,
                np.stack([stem_buf, stem_buf], axis=1).astype(np.float32),
                target_sample_rate,
                normalize=cfg["export"].get("normalize", True),
                target_peak_db=float(cfg["export"].get("target_peak_dbtp", -0.3)),
                dither=cfg["export"].get("dither", True),
            )
            stems[name] = p

    midifile = MIDIFile()
    midifile.set_tempo(float(plan.bpm))
    n, d = map(
        int,
        (cfg["audio"].get("time_signature", "4/4") or "4/4").split("/"),
    )
    midifile.set_time_signature(n, d)

    assets = cfg.get("_assets", {})
    resolver = CCResolver(
        studio_config_path=cfg["paths"]["assets"].get("studio_config"),
        cc_overrides_path=cfg["paths"]["assets"].get("cc_overrides"),
        fallback_profile=cfg.get("midi_cc", {}).get("fallback_profile", {}),
        studio_config=assets.get("studio_config"),
        cc_overrides=assets.get("cc_overrides"),
    )
    seq = Sequencer(ticks_per_beat=480)
    hp_name = cfg["styles"][plan.style].get("humanize_profile", None)
    if hp_name:
        hp = cfg["humanize"]["profiles"].get(hp_name, {})
        seq.humanize = dict(time_ms=hp.get("time_ms", 0.0), vel_var=hp.get("vel_variation", 0))

    devmap = cfg.get("midi_cc", {}).get("device_map", {})
    pattern_steps = max(16, int(cfg.get("patterns", {}).get("steps", 16)))
    generated_patterns = generate_pattern(plan.style, pattern_steps, plan.seed)

    kick_steps = generated_patterns.get("kick", [])
    if kick_steps:
        pat_kick = Pattern(channel=9, device=devmap.get("kick", "drum_machine"))
        pat_kick.steps.extend(kick_steps)
        seq.set_pattern("kick", pat_kick)

    pat_bass = Pattern(channel=0, device=devmap.get("bass", "moog_minitaur"))
    pat_bass.steps += [
        Step(beat=0.0, note=36, vel=110, length_beats=2.0),
        Step(beat=4.0, locks={"cutoff": 0.62, "resonance": 0.55}),
        Step(beat=8.0, locks={"reverb_send": 0.18}),
        Step(beat=12.0, note=38, vel=108, length_beats=1.5, slide=True),
    ]
    seq.set_pattern("bass", pat_bass)
    pat_stabs = Pattern(channel=1, device=devmap.get("stabs", "subsequent_37"))
    pat_stabs.steps += [
        Step(beat=0.0, note=55, vel=96, length_beats=1.0),
        Step(beat=8.0, locks={"cutoff": 0.58}),
        Step(beat=24.0, locks={"delay_send": 0.50}),
    ]
    seq.set_pattern("stabs", pat_stabs)
    pat_pad = Pattern(channel=2, device=devmap.get("pad", "roland_juno106"))
    pad_steps = generated_patterns.get("pad")
    if pad_steps:
        pat_pad.steps.extend(pad_steps)
    else:
        pat_pad.steps += [
            Step(beat=0.0, note=48, vel=80, length_beats=4.0),
            Step(beat=16.0, locks={"cutoff": 0.42}),
            Step(beat=48.0, locks={"reverb_send": 0.65}),
        ]
    seq.set_pattern("pad", pat_pad)
    pat_keys = Pattern(channel=3, device=devmap.get("keys", "generic_synth"))
    pat_keys.steps += [
        Step(beat=4.0, note=60, vel=92, length_beats=0.5),
        Step(beat=12.0, locks={"delay_send": 0.60}),
        Step(beat=28.0, locks={"chorus_send": 0.55}),
    ]
    seq.set_pattern("keys", pat_keys)

    _cc_used: List[Dict[str, Any]] = []
    seq.to_midi(midifile, resolver, bpm=plan.bpm, collect_cc=_cc_used)

    midi_name = _fmt_name(cfg["export"]["filenames"]["midi"], **filename_ctx)
    midi_path = os.path.join(midi_dir, midi_name)
    midifile.save(midi_path)

    cc_origin_stats: Dict[str, int] = {}
    for entry in _cc_used:
        origin = entry.get("origin", "HYPOTHÈSE")
        count = int(entry.get("count", 1))
        cc_origin_stats[origin] = cc_origin_stats.get(origin, 0) + count

    return {"master": master_path, "stems": stems, "midi": midi_path}, _cc_used, cc_origin_stats

def _evaluate_analysis(metrics: Dict[str, float], cfg: Dict[str, Any], style: str) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    tolerances = analysis_cfg.get("tolerances", {})
    evaluation: Dict[str, Any] = {}

    lufs_target = tolerances.get("lufs_i", {}).get(style)
    if lufs_target and metrics.get("lufs_i") is not None:
        lo, hi = float(lufs_target[0]), float(lufs_target[1])
        val = float(metrics["lufs_i"])
        ok = lo <= val <= hi
        evaluation["lufs_i"] = {"ok": ok, "range": [lo, hi], "value": val}
        if not ok:
            logger.warning("[Σ] Loudness hors tolérance pour %s: %.2f LUFS (cible %.2f..%.2f).", style, val, lo, hi)

    dbtp_max = tolerances.get("dbtp_max")
    if dbtp_max is not None and metrics.get("dbtp") is not None:
        val = float(metrics["dbtp"])
        limit = float(dbtp_max)
        ok = val <= limit + 0.2
        evaluation["dbtp"] = {"ok": ok, "max": limit, "value": val}
        if not ok:
            logger.warning("[Σ] Pic réel au-delà du plafond: %.2f dBTP (max %.2f).", val, limit)

    corr_min = tolerances.get("correlation_min")
    if corr_min is not None and metrics.get("correlation") is not None:
        val = float(metrics["correlation"])
        min_val = float(corr_min)
        ok = val >= min_val
        evaluation["correlation"] = {"ok": ok, "min": min_val, "value": val}
        if not ok:
            logger.warning("[Σ] Corrélation stéréo basse: %.2f (min %.2f).", val, min_val)

    mono_floor = tolerances.get("mono_bass_min", 0.85)
    if metrics.get("mono_bass_check") is not None:
        val = float(metrics["mono_bass_check"])
        ok = val >= float(mono_floor)
        evaluation["mono_bass_check"] = {"ok": ok, "min": float(mono_floor), "value": val}
        if not ok:
            logger.warning("[Σ] Compatibilité mono basse douteuse: %.2f (min %.2f).", val, mono_floor)

    clicks_limit = tolerances.get("clicks_max", 0)
    if metrics.get("clicks") is not None:
        val = int(metrics["clicks"])
        ok = val <= int(clicks_limit)
        evaluation["clicks"] = {"ok": ok, "max": int(clicks_limit), "value": val}
        if not ok:
            logger.warning("[Σ] Détection de %d clic(s) (max autorisé %d).", val, clicks_limit)

    return evaluation


def _analyze(stereo: np.ndarray, cfg: Dict[str, Any], style: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    try:
        from analysis_tools import (
            detect_clicks,
            measure_correlation,
            measure_lufs,
            measure_true_peak,
            mono_bass_check,
        )

        sr = int(cfg["audio"]["sample_rate"])
        lufs = float(measure_lufs(stereo, sr))
        dbtp = float(measure_true_peak(stereo, sr))
        corr = float(measure_correlation(stereo))
        mbc = float(mono_bass_check(stereo, sr, float(cfg["audio"].get("mono_bass_hz", 150.0))))
        clk = int(detect_clicks(stereo))
        metrics = {
            "lufs_i": lufs,
            "dbtp": dbtp,
            "correlation": corr,
            "mono_bass_check": mbc,
            "clicks": clk,
        }
        evaluation = _evaluate_analysis(metrics, cfg, style)
        return metrics, evaluation
    except Exception as exc:
        logger.warning("[Σ] Analyse audio échouée: %s", exc)
        return {
            "lufs_i": None,
            "dbtp": None,
            "correlation": None,
            "mono_bass_check": None,
            "clicks": None,
        }, {}

def run_session(plan: Dict[str, Any] | SessionPlan, config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config_safe(config_path)
    set_style_templates(cfg.get("_assets", {}).get("style_templates"))
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(level=level)

    paths_cfg = cfg["paths"]
    output_dir = paths_cfg["output"]
    midi_dir = paths_cfg["midi_export"]
    _ensure_dirs(output_dir, midi_dir)

    if isinstance(plan, SessionPlan):
        session_plan = plan
        style = session_plan.style
        if style not in cfg["styles"]:
            raise KeyError(f"Style inconnu '{style}' dans la configuration.")
    elif isinstance(plan, dict):
        if "style" not in plan:
            raise KeyError("Le plan de session doit préciser un 'style'.")
        style = str(plan["style"])
        if style not in cfg["styles"]:
            raise KeyError(f"Style inconnu '{style}' dans la configuration.")
        bpm = float(plan.get("bpm", cfg["styles"][style].get("bpm_default", cfg["audio"].get("tempo_default", 120.0))))
        duration = float(plan.get("duration_s", plan.get("duration", 84.0)))
        seed = int(plan.get("seed", cfg.get("project", {}).get("seed", 0)))
        sections = plan.get("sections") or list(cfg["styles"][style].get("structure", []))
        session_plan = SessionPlan(style=style, bpm=bpm, seed=seed, duration_s=duration, sections=sections)
    else:
        raise TypeError("Le plan de session doit être un dictionnaire ou un SessionPlan.")

    seed = int(session_plan.seed)
    _init_rng(seed)

    sections = session_plan.sections or list(cfg["styles"][session_plan.style].get("structure", []))
    if not sections:
        sections = ["intro", "outro"]
    session_plan = SessionPlan(
        style=session_plan.style,
        bpm=float(session_plan.bpm),
        seed=seed,
        duration_s=float(session_plan.duration_s),
        sections=sections,
    )

    session_slug = _sanitize_component(
        cfg.get("project", {}).get("name", "project"),
        session_plan.style,
        f"seed{session_plan.seed}",
    )
    session_dir = os.path.join(output_dir, session_slug)
    _ensure_dirs(session_dir)

    filename_ctx = {
        "project": _sanitize_component(cfg.get("project", {}).get("name", "project")),
        "style": _sanitize_component(session_plan.style),
        "bpm": int(round(session_plan.bpm)),
        "seed": session_plan.seed,
    }

    drop_protocols = cfg.get("_assets", {}).get("drop_protocols", {})
    arranger = ArrangementEngine(
        protocols_path=cfg["paths"]["assets"]["drop_protocols"],
        protocols=drop_protocols,
    )
    modmat, drop_routes = _wire_modmatrix_drop(cfg, arranger)

    tracks = _render_tracks(session_plan, cfg)
    mx = MixerEngine(cfg)
    mx.master_pregain = 10.0 ** (
        -float(cfg["export"].get("headroom_db", 1.0)) / 20.0
    )
    names = list(tracks.keys())
    mx.set_track_order(names)
    for name in names:
        settings = {"volume_db": 0.0, "pan": 0.0, "reverb_send": 0.0, "delay_send": 0.0}
        if name == "pad":
            settings["reverb_send"] = 0.35
        if name == "stabs":
            settings["delay_send"] = 0.25
        if name == "bass":
            settings["reverb_send"] = 0.10
        mx.configure_track(name, **settings)

    _apply_style_to_mixer(cfg, session_plan.style, mx, session_plan.bpm)

    drop_destinations = arranger.prepare_destination_context(mx, drop_routes)
    if drop_destinations:
        arranger.apply_modulations(modmat, drop_destinations)
        arranger.apply_final_values_to_mixer(mx, drop_destinations)
    drop_summary = arranger.summarise_modulations(drop_destinations) if drop_destinations else []
    sc_routes_summary = [
        {
            "sources": list(route.sources),
            "dst": route.dst,
            "depth_db": float(route.depth_db),
            "attack_ms": float(route.attack_ms),
            "release_ms": float(route.release_ms),
            "shape": route.shape,
        }
        for route in getattr(mx, "_sidechain_routes", [])
    ]

    fxov = cfg["styles"][session_plan.style].get("fx_overrides", {})
    if fxov:
        mix_val = fxov.get("reverb_mix")
        decay_factor = fxov.get("reverb_decay")
        room_factor = fxov.get("reverb_room_size")
        pre_delay_override = fxov.get("reverb_pre_delay")

        mix_value = (
            float(np.clip(mix_val, 0.0, 1.0))
            if mix_val is not None
            else mx.cfg_fx_reverb_mix
        )
        decay_value = mx.cfg_fx_reverb_decay
        if decay_factor is not None:
            decay_value = float(
                np.clip(mx.cfg_fx_reverb_decay * float(decay_factor), 0.05, 12.0)
            )
        room_value = mx.cfg_fx_reverb_room_size
        if room_factor is not None:
            room_value = float(
                np.clip(mx.cfg_fx_reverb_room_size * float(room_factor), 0.1, 3.0)
            )
        pre_delay_value = (
            pre_delay_override if pre_delay_override is not None else mx.cfg_fx_reverb_pre_delay
        )
        mx.configure_reverb(
            mix=mix_value,
            decay=decay_value,
            room_size=room_value,
            pre_delay=pre_delay_value,
            tempo_bpm=session_plan.bpm,
        )

        if "delay_time" in fxov or "delay_mix" in fxov or "delay_feedback" in fxov:
            delay_time = fxov.get("delay_time", mx.cfg_fx_delay_time)
            delay_mix = fxov.get("delay_mix", mx.cfg_fx_delay_mix)
            delay_feedback = fxov.get("delay_feedback", mx.cfg_fx_delay_feedback)
            mx.configure_delay(
                time=delay_time,
                feedback=float(delay_feedback),
                mix=float(delay_mix),
                tempo_bpm=session_plan.bpm,
            )

    mx.delay_bus = Delay(
        mx.sample_rate,
        tempo_bpm=session_plan.bpm,
        feedback=mx.cfg_fx_delay_feedback,
        mix=mx.cfg_fx_delay_mix,
        time=mx.cfg_fx_delay_time,
    )

    sc_sources: Dict[str, np.ndarray] = {}
    for r in cfg["styles"][session_plan.style].get("sidechain_routes", []):
        src = r["src"]
        if src in tracks:
            sc_sources[src] = tracks[src]

    stereo = mx.render_mix(
        tracks,
        sidechain_kick=tracks.get("kick", None),
        sidechain_sources=sc_sources,
    )

    target_lufs = cfg["styles"].get(session_plan.style, {}).get("target_lufs")
    if target_lufs is not None:
        try:
            from analysis_tools import measure_lufs

            sample_rate = int(cfg["audio"].get("sample_rate", 48000))
            measured_lufs = float(measure_lufs(stereo, sample_rate))
        except Exception:
            measured_lufs = float("nan")
        if not np.isnan(measured_lufs):
            diff_db = float(target_lufs) - measured_lufs
            diff_db = float(np.clip(diff_db, -12.0, 12.0))
            if abs(diff_db) > 1e-3:
                gain = float(10.0 ** (diff_db / 20.0))
                stereo = stereo * gain
                for name, buf in list(tracks.items()):
                    tracks[name] = buf * gain
                ceiling_db = float(cfg["export"].get("target_peak_dbtp", -0.3))
                stereo = np.stack(
                    [
                        brickwall_limit(stereo[:, 0], ceiling_db),
                        brickwall_limit(stereo[:, 1], ceiling_db),
                    ],
                    axis=1,
                )

    outs, cc_used, cc_origin_stats = _export_audio_and_midi(
        stereo,
        tracks,
        cfg,
        session_plan,
        session_dir,
        midi_dir,
        filename_ctx,
    )

    if cfg["analysis"].get("run_after_export", False):
        metrics, analysis_eval = _analyze(stereo, cfg, session_plan.style)
    else:
        metrics, analysis_eval = {}, {}

    cc_origin_stats.setdefault("real", 0)
    cc_origin_stats.setdefault("HYPOTHÈSE", 0)
    hyp_count = cc_origin_stats.get("HYPOTHÈSE", 0)
    if hyp_count:
        logger.warning(
            "[Σ] CC mappings fallback used %d time(s); consider updating hardware profiles.",
            hyp_count,
        )

    report_name = _fmt_name(cfg["export"]["filenames"]["report"], **filename_ctx)
    report_path = os.path.join(session_dir, report_name)

    paths_info: Dict[str, Any] = dict(outs)
    paths_info["session_dir"] = session_dir
    paths_info["report"] = report_path

    project_cfg = cfg.get("project", {})
    project_version = str(project_cfg.get("version", "1.0"))
    raw_date = project_cfg.get("date_utc", "auto")
    if isinstance(raw_date, str) and raw_date.lower() == "auto":
        date_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    else:
        date_utc = raw_date

    report = {
        "project": project_cfg.get("name", "dawless"),
        "version": project_version,
        "date_utc": date_utc,
        "seed": session_plan.seed,
        "style": session_plan.style,
        "bpm": session_plan.bpm,
        "duration_s": session_plan.duration_s,
        "sections": session_plan.sections,
        "targets": {
            "lufs": float(cfg["styles"][session_plan.style].get("target_lufs", -12.0)),
            "dBTP": float(cfg["export"].get("target_peak_dbtp", -0.3)),
            "mono_bass_hz": int(cfg["audio"].get("mono_bass_hz", 150)),
        },
        "metrics": metrics,
        "analysis_evaluation": analysis_eval,
        "sc_routes": sc_routes_summary,
        "cc_used": cc_used,
        "cc_origin_stats": cc_origin_stats,
        "device_map": cfg.get("midi_cc", {}).get("device_map", {}),
        "drops": drop_summary,
        "paths": paths_info,
        "schema_version": cfg.get("logging", {}).get("report_schema_version", "1.0"),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
