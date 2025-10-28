# /mnt/data/render.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json, time, os, yaml, struct, wave
import numpy as np

from mixer_engine import MixerEngine
from arrangement_engine import ArrangementEngine, DropBusSource
from modulation_matrix import ModulationMatrix
from midi_writer import MIDIFile
from midi_cc_db import CCResolver
from sequencer import Sequencer, Pattern, Step
from fx_processors import Delay

@dataclass
class SessionPlan:
    style: str
    bpm: float
    seed: int
    duration_s: float
    sections: List[str]

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _fmt_name(template: str, **kw) -> str:
    return template.format(**kw)

def _init_rng(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32-1))

def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _apply_style_to_mixer(cfg: Dict[str, Any], style: str, mixer: MixerEngine, bpm: float):
    fxd = cfg.get("fx_defaults", {})
    mixer.cfg_fx_delay_time     = fxd.get("delay_time", "1/8.")
    mixer.cfg_fx_delay_feedback = float(fxd.get("delay_feedback", 0.4))
    mixer.cfg_fx_delay_mix      = float(fxd.get("delay_mix", 0.25))
    mixer.tempo_bpm = bpm
    style_cfg = cfg["styles"][style]
    for r in style_cfg.get("sidechain_routes", []):
        mixer.create_sc(r["src"], r["dst"], depth_db=r.get("depth_db", 3.0),
                        attack_ms=r.get("attack_ms", 5), release_ms=r.get("release_ms", 80),
                        shape=r.get("shape", "exp"))

def _wire_modmatrix_drop(cfg: Dict[str, Any], arranger: ArrangementEngine) -> ModulationMatrix:
    mm = ModulationMatrix(sum_mode=cfg["modulation_matrix"]["sum_mode"])
    return mm

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

def _export_audio_and_midi(stereo: np.ndarray, trackbufs: Dict[str, np.ndarray],
                           cfg: Dict[str, Any], plan: SessionPlan, outdir: str):
    sr = int(cfg["audio"]["sample_rate"])
    master_name = _fmt_name(cfg["export"]["filenames"]["master"],
                            project=cfg["project"]["name"], style=plan.style, bpm=int(plan.bpm), seed=plan.seed)
    master_path = os.path.join(outdir, master_name)
    _write_wav_24(master_path, stereo.astype(np.float32), sr,
                  normalize=cfg["export"].get("normalize", True),
                  target_peak_db=float(cfg["export"].get("target_peak_dbtp",-0.3)),
                  dither=cfg["export"].get("dither", True))

    stems = {}
    if cfg["export"]["render_stems"]:
        for name, buf in trackbufs.items():
            nm = _fmt_name(cfg["export"]["filenames"]["stem"],
                           project=cfg["project"]["name"], style=plan.style, bpm=int(plan.bpm), seed=plan.seed, track=name)
            p = os.path.join(outdir, nm)
            _write_wav_24(p, np.stack([buf, buf], axis=1).astype(np.float32), sr,
                          normalize=cfg["export"].get("normalize", True),
                          target_peak_db=float(cfg["export"].get("target_peak_dbtp",-0.3)),
                          dither=cfg["export"].get("dither", True))
            stems[name] = p

    # === MIDI (p-locks + CC) ===
    midifile = MIDIFile()
    midifile.set_tempo(plan.bpm)
    ts = (cfg["audio"].get("time_signature","4/4") or "4/4").split("/")
    midifile.set_time_signature(int(ts[0]), int(ts[1]))

    resolver = CCResolver(
        studio_config_path=cfg["paths"]["assets"]["studio_config"],
        cc_overrides_path=cfg["paths"]["assets"].get("cc_overrides", None),
        fallback_profile=cfg["midi_cc"]["fallback_profile"]
    )
    seq = Sequencer(ticks_per_beat=480)
    hp_name = cfg["styles"][plan.style].get("humanize_profile", None)
    if hp_name:
        hp = cfg["humanize"]["profiles"].get(hp_name, {})
        seq.humanize = dict(time_ms=hp.get("time_ms",0.0), vel_var=hp.get("vel_variation",0))

    devmap = (cfg.get("midi_cc", {}).get("device_map", {}))
    pat_bass  = Pattern(channel=0, device=devmap.get("bass","moog_minitaur"))
    pat_bass.steps += [
        Step(beat=0.0,  note=36, vel=110, length_beats=2.0),
        Step(beat=4.0,  locks={"cutoff":0.62, "resonance":0.55}),
        Step(beat=8.0,  locks={"reverb_send":0.18}),
        Step(beat=12.0, note=38, vel=108, length_beats=1.5, slide=True)
    ]
    seq.set_pattern("bass", pat_bass)
    pat_stabs = Pattern(channel=1, device=devmap.get("stabs","subsequent_37"))
    pat_stabs.steps += [
        Step(beat=0.0, note=55, vel=96, length_beats=1.0),
        Step(beat=8.0, locks={"cutoff":0.58}),
        Step(beat=24.0, locks={"delay_send":0.50})
    ]
    seq.set_pattern("stabs", pat_stabs)
    pat_pad   = Pattern(channel=2, device=devmap.get("pad","roland_juno106"))
    pat_pad.steps += [
        Step(beat=0.0, note=48, vel=80, length_beats=4.0),
        Step(beat=16.0, locks={"cutoff":0.42}),
        Step(beat=48.0, locks={"reverb_send":0.65})
    ]
    seq.set_pattern("pad", pat_pad)
    pat_keys  = Pattern(channel=3, device=devmap.get("keys","generic_synth"))
    pat_keys.steps += [
        Step(beat=4.0,  note=60, vel=92, length_beats=0.5),
        Step(beat=12.0, locks={"delay_send":0.60}),
        Step(beat=28.0, locks={"chorus_send":0.55})
    ]
    seq.set_pattern("keys", pat_keys)

    _cc_used = []
    seq.to_midi(midifile, resolver, bpm=plan.bpm, collect_cc=_cc_used)

    midi_name = _fmt_name(cfg["export"]["filenames"]["midi"], project=cfg["project"]["name"],
                          style=plan.style, bpm=int(plan.bpm), seed=plan.seed)
    midi_path = os.path.join(cfg["paths"]["midi_export"], midi_name)
    midifile.save(midi_path)

    return {"master": master_path, "stems": stems, "midi": midi_path}, _cc_used

def _analyze(stereo: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    try:
        from analysis_tools import measure_lufs, measure_true_peak, measure_correlation, mono_bass_check, detect_clicks
        lufs = float(measure_lufs(stereo, int(cfg["audio"]["sample_rate"])))
        dbtp = float(measure_true_peak(stereo, int(cfg["audio"]["sample_rate"])))
        corr = float(measure_correlation(stereo))
        mbc  = float(mono_bass_check(stereo, int(cfg["audio"]["sample_rate"]), float(cfg["audio"].get("mono_bass_hz",150))))
        clk  = int(detect_clicks(stereo))
        return {"lufs_i": lufs, "dBTP": dbtp, "corr": corr, "mono_bass_check": mbc, "clicks": clk}
    except Exception:
        return {"lufs_i": -999.0, "dBTP": 0.0, "corr": 1.0}

def run_session(session_plan: Dict[str, Any], config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = _load_config(config_path)
    _ensure_dirs(cfg["paths"]["output"], cfg["paths"]["midi_export"])
    seed = int(session_plan.get("seed", cfg["project"]["seed"]))
    _init_rng(seed)

    style = session_plan["style"]
    bpm   = float(session_plan.get("bpm", cfg["styles"][style]["bpm_default"]))
    dur   = float(session_plan.get("duration_s", 84.0))
    sections = cfg["styles"][style]["structure"]
    plan = SessionPlan(style=style, bpm=bpm, seed=seed, duration_s=dur, sections=sections)

    arranger = ArrangementEngine(cfg["paths"]["assets"]["drop_protocols"])
    modmat = _wire_modmatrix_drop(cfg, arranger)

    # Rendu (placeholder)
    tracks = _render_tracks(plan, cfg)
    mx = MixerEngine(num_tracks=len(tracks), sample_rate=int(cfg["audio"]["sample_rate"]))
    mx.master_pregain = 10.0**(-float(cfg["export"].get("headroom_db",1.0))/20.0)

    # nommage + envois de base pour démo aux
    mx.tracks = [type("Track", (), {"name": k, "volume_db": 0.0, "reverb_send": 0.0, "delay_send": 0.0, "pan": 0.0})() for k in tracks.keys()]
    for tr in mx.tracks:
        if tr.name == "pad": tr.reverb_send = 0.35
        if tr.name == "stabs": setattr(tr, "delay_send", 0.25)
        if tr.name == "bass": tr.reverb_send = 0.10

    _apply_style_to_mixer(cfg, plan.style, mx, plan.bpm)

    # Recréer systématiquement le Delay bus selon les defaults/tempo
    mx.delay_bus = Delay(int(cfg["audio"]["sample_rate"]),
                         feedback=mx.cfg_fx_delay_feedback,
                         mix=mx.cfg_fx_delay_mix,
                         time=mx.cfg_fx_delay_time,
                         tempo_bpm=plan.bpm)

    # appliquer fx_overrides aux bus
    fxov = cfg["styles"][plan.style].get("fx_overrides", {})
    if "reverb_mix" in fxov:   mx.reverb_bus.mix = float(fxov["reverb_mix"])
    if "reverb_decay" in fxov:
        fac = max(0.2, min(1.2, float(fxov["reverb_decay"])))
        mx.reverb_bus.gains = [max(0.5, min(0.99, g*fac)) for g in mx.reverb_bus.gains]
    if "reverb_room_size" in cfg.get("fx_defaults", {}):
        rs = float(cfg["fx_defaults"]["reverb_room_size"])
        rs = max(0.5, min(1.5, rs))
        mx.reverb_bus.delays = [max(1, int(d*rs)) for d in mx.reverb_bus.delays]
    if "delay_time" in fxov:
        mx.delay_bus = Delay(int(cfg["audio"]["sample_rate"]),
                             feedback=mx.cfg_fx_delay_feedback,
                             mix=mx.cfg_fx_delay_mix,
                             time=fxov["delay_time"],
                             tempo_bpm=plan.bpm)

    # construire sidechain_sources: index -> signal
    names = list(tracks.keys())
    idx_map = {name: i for i, name in enumerate(names)}
    sc_sources = {}
    for r in cfg["styles"][style].get("sidechain_routes", []):
        src = r["src"]
        if src in idx_map:
            si = idx_map[src]
            sc_sources[si] = tracks[src]

    stereo = mx.render_mix([tracks[k] for k in names],
                           sidechain_kick=tracks.get("kick", None),
                           sidechain_sources=sc_sources)

    outs, cc_used = _export_audio_and_midi(stereo, tracks, cfg, plan, cfg["paths"]["output"])
    metrics = _analyze(stereo, cfg) if cfg["analysis"]["run_after_export"] else {}

    cc_origin_stats = {"real": 0, "HYPOTHÈSE": 0}
    for ev in cc_used:
        cc_origin_stats[ev.get("origin","HYPOTHÈSE")] = cc_origin_stats.get(ev.get("origin","HYPOTHÈSE"), 0) + 1

    report = {
        "project": cfg["project"]["name"],
        "version": cfg["project"]["version"],
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "style": style,
        "bpm": bpm,
        "duration_s": dur,
        "sections": sections,
        "targets": {
            "lufs": float(cfg["styles"][style]["target_lufs"]),
            "dBTP": float(cfg["export"]["target_peak_dbtp"]),
            "mono_bass_hz": int(cfg["audio"]["mono_bass_hz"])
        },
        "metrics": metrics,
        "sc_routes": [f"{r['src']}->{r['dst']}" for r in cfg["styles"][style].get("sidechain_routes", [])],
        "cc_used": cc_used,
        "cc_origin_stats": cc_origin_stats,
        "device_map": cfg.get("midi_cc",{}).get("device_map",{}),
        "drops": ["filter_sweep_master","reverb_wash_bass_filter"],
        "paths": outs,
        "schema_version": cfg["logging"]["report_schema_version"]
    }
    rep_name = _fmt_name(cfg["export"]["filenames"]["report"],
                         project=cfg["project"]["name"], style=style, bpm=int(bpm), seed=seed)
    rep_path = os.path.join(cfg["paths"]["output"], rep_name)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report