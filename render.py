# /mnt/data/render.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import copy
import json
import os
import time
import warnings
import yaml
import struct
import wave
import numpy as np
import logging
import re

from mixer_engine import MixerEngine
from arrangement_engine import ArrangementEngine
from modulation_matrix import ModulationMatrix, DropBusSource
from midi_writer import MIDIFile
from midi_cc_db import CCResolver
from sequencer import Sequencer, Pattern, Step

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
            data = yaml.safe_load(f) or {}
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
            cfg_raw = yaml.safe_load(f) or {}
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
    N  = int(plan.duration_s * sr)
    t  = np.arange(N)/sr
    kick = (np.sin(2*np.pi*50*t) * (np.exp(-t*8))).astype(np.float32)
    bass = (0.3*np.sin(2*np.pi*55*t)).astype(np.float32)
    pad  = (0.05*np.sin(2*np.pi*0.2*t)).astype(np.float32)
    return {"kick": kick, "bass": bass, "pad": pad}  # placeholders

def _float_to_int24_pcm(x: np.ndarray, dither: bool=False) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    if x.ndim == 1: x = x[:, None]
    if dither:
        lsb = 1.0/8388607.0
        rng = np.random.RandomState(12345)
        d = (rng.rand(*x.shape) - rng.rand(*x.shape)) * lsb
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
    midifile.set_tempo(plan.bpm)
    ts = (cfg["audio"].get("time_signature", "4/4") or "4/4").split("/")
    midifile.set_time_signature(int(ts[0]), int(ts[1]))

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
    if dbtp_max is not None and metrics.get("dBTP") is not None:
        val = float(metrics["dBTP"])
        limit = float(dbtp_max)
        ok = val <= limit + 0.2
        evaluation["dBTP"] = {"ok": ok, "max": limit, "value": val}
        if not ok:
            logger.warning("[Σ] Pic réel au-delà du plafond: %.2f dBTP (max %.2f).", val, limit)

    corr_min = tolerances.get("correlation_min")
    if corr_min is not None and metrics.get("corr") is not None:
        val = float(metrics["corr"])
        min_val = float(corr_min)
        ok = val >= min_val
        evaluation["corr"] = {"ok": ok, "min": min_val, "value": val}
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
            "dBTP": dbtp,
            "corr": corr,
            "mono_bass_check": mbc,
            "clicks": clk,
        }
        evaluation = _evaluate_analysis(metrics, cfg, style)
        return metrics, evaluation
    except Exception as exc:
        logger.warning("[Σ] Analyse audio échouée: %s", exc)
        return {"lufs_i": None, "dBTP": None, "corr": None}, {}

def run_session(plan: Dict[str, Any] | SessionPlan, config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config_safe(config_path)
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

    fxov = cfg["styles"][session_plan.style].get("fx_overrides", {})
    reverb_overrides: Dict[str, Any] = {}
    if "reverb_mix" in fxov:
        reverb_overrides["mix"] = float(fxov["reverb_mix"])
    if "reverb_decay" in fxov:
        reverb_overrides["decay"] = float(fxov["reverb_decay"])
    if "reverb_room_size" in fxov:
        reverb_overrides["room_size"] = float(fxov["reverb_room_size"])
    if "reverb_pre_delay" in fxov:
        reverb_overrides["pre_delay"] = fxov["reverb_pre_delay"]
    if reverb_overrides:
        mx.configure_reverb(
            mix=reverb_overrides.get("mix", mx.cfg_fx_reverb_mix),
            decay=reverb_overrides.get("decay", mx.cfg_fx_reverb_decay),
            room_size=reverb_overrides.get("room_size", mx.cfg_fx_reverb_room_size),
            pre_delay=reverb_overrides.get("pre_delay", mx.cfg_fx_reverb_pre_delay),
            tempo_bpm=session_plan.bpm,
        )
    if "delay_time" in fxov:
        mx.configure_delay(
            time=fxov["delay_time"],
            feedback=mx.cfg_fx_delay_feedback,
            mix=mx.cfg_fx_delay_mix,
            tempo_bpm=session_plan.bpm,
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

    report = {
        "project": cfg.get("project", {}).get("name", "dawless"),
        "version": cfg.get("project", {}).get("version", "unknown"),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "sc_routes": [
            f"{r['src']}->{r['dst']}" for r in cfg["styles"][session_plan.style].get("sidechain_routes", [])
        ],
        "cc_used": cc_used,
        "cc_origin_stats": cc_origin_stats,
        "device_map": cfg.get("midi_cc", {}).get("device_map", {}),
        "drops": drop_summary,
        "paths": paths_info,
        "schema_version": "1.0",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
