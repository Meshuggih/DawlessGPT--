# /mnt/data/cli.py
import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from render import load_config_safe, run_session

MAX_PRODUCT_FILES = 20
PRODUCT_EXTENSIONS: Sequence[str] = (
    ".wav",
    ".wave",
    ".aiff",
    ".aif",
    ".flac",
    ".mp3",
    ".mid",
    ".midi",
    ".json",
)
IGNORED_DIRECTORIES: Sequence[str] = ("output", "midi_export")


def check_file_budget(cfg: dict | None = None, base_dir: Path | None = None, limit: int = MAX_PRODUCT_FILES) -> int:
    """Ensure the number of produced files stays within the strict budget."""
    base = base_dir or Path(__file__).resolve().parent
    count = 0
    for root, dirs, files in os.walk(base):
        # Remove ignored directories from traversal (strict budget ignores runtime outputs)
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
        for filename in files:
            if Path(filename).suffix.lower() in PRODUCT_EXTENSIONS:
                count += 1
    if count > limit:
        raise RuntimeError(
            f"Budget de fichiers produit dépassé: {count} > {limit}. Nettoyez les exports avant de continuer."
        )
    return count


def run_selftest(cfg: dict, base_dir: Path) -> None:
    """Run an extensive self-test battery and raise if any check fails."""

    time_limit_s = 15.0
    memory_limit_bytes = 512 * 1024 * 1024

    tracemalloc.start()
    start_time = time.perf_counter()

    def run_check(name: str, func: Callable[[], str]) -> None:
        try:
            message = func()
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[Σ][FAIL] {name} — {exc}")
            raise
        else:
            print(f"[Σ][PASS] {name} — {message}")

    def _check_file_budget() -> str:
        count = check_file_budget(cfg, base_dir=base_dir)
        return f"{count} fichier(s) produit(s) ≤ limite {MAX_PRODUCT_FILES}"

    def _check_dsp_core() -> str:
        import numpy as np

        from dsp_core import brickwall_limit, db_to_lin, downsample_linear, oversample_linear
        from mixer_engine import MixerEngine

        rng = np.random.default_rng(1337)
        signal = np.array(rng.standard_normal(4096), dtype=np.float32) * 0.25
        factor = 4
        oversampled = oversample_linear(signal, factor)
        recovered = downsample_linear(oversampled, factor)
        rms_err = float(np.sqrt(np.mean((signal - recovered) ** 2)))
        rms_tolerance = 5e-7
        if rms_err > rms_tolerance:
            raise RuntimeError(
                f"oversample/downsample rms={rms_err:.3e} tol={rms_tolerance:.1e}"
            )

        ceiling_db = -1.0
        ceiling = np.float32(db_to_lin(ceiling_db))
        limiter_input = np.array([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=np.float32)
        limited = brickwall_limit(limiter_input, ceiling_db)
        excess = float(max(0.0, np.max(np.abs(limited)) - ceiling))
        ceiling_tolerance = 1e-7
        if excess > ceiling_tolerance:
            raise RuntimeError(
                f"brickwall_limit excès={excess:.3e} tol={ceiling_tolerance:.1e}"
            )

        mixer_cfg = {
            "audio": {"sample_rate": 48000, "mono_bass_hz": 160, "tempo_default": 120},
            "export": {"headroom_db": 6.0, "target_peak_dbtp": 0.0},
            "fx_defaults": {
                "delay_time": "1/8.",
                "delay_feedback": 0.0,
                "delay_mix": 0.0,
                "reverb_mix": 0.0,
                "reverb_decay": 0.6,
                "reverb_room_size": 0.8,
            },
        }
        sr = int(mixer_cfg["audio"]["sample_rate"])
        t = np.arange(sr) / sr

        headroom_tone = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        headroom_mixer = MixerEngine(mixer_cfg)
        headroom_mixer.set_track_order(["tone"])
        headroom_mixer.configure_track("tone", pan=-1.0)
        headroom_mix = headroom_mixer.render_mix({"tone": headroom_tone})
        peak = float(np.max(np.abs(headroom_mix[:, 0])))
        processed_tone = headroom_tone.astype(np.float64)
        mono_bass = float(mixer_cfg["audio"].get("mono_bass_hz", 0.0) or 0.0)
        if mono_bass > 0.0:
            low_l = headroom_mixer._first_order_lowpass(processed_tone, mono_bass, sr)
            low_r = np.zeros_like(low_l)
            low_m = 0.5 * (low_l + low_r)
            processed_tone = low_m + (processed_tone - low_l)
        expected_peak = float(
            np.max(np.abs(processed_tone)) * 10.0 ** (-mixer_cfg["export"]["headroom_db"] / 20.0)
        )
        headroom_tol = 5e-4
        if abs(peak - expected_peak) > headroom_tol:
            raise RuntimeError(
                f"headroom pic={peak:.4f} attendu={expected_peak:.4f} tol={headroom_tol:.1e}"
            )

        mono_bass_tone = (0.5 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        mono_mixer = MixerEngine(mixer_cfg)
        mono_mixer.set_track_order(["tone"])
        mono_mixer.configure_track("tone", pan=1.0)
        mono_mix = mono_mixer.render_mix({"tone": mono_bass_tone})
        low_l = headroom_mixer._first_order_lowpass(mono_mix[:, 0], mono_bass, sr)
        low_r = headroom_mixer._first_order_lowpass(mono_mix[:, 1], mono_bass, sr)
        energy_l = float(np.sum(low_l ** 2))
        energy_r = float(np.sum(low_r ** 2))
        if energy_l <= 0.0 or energy_r <= 0.0:
            raise RuntimeError("mono-bass énergie basse fréquence nulle")
        balance = min(energy_l, energy_r) / max(energy_l, energy_r)
        if balance < 0.6:
            raise RuntimeError(f"mono-bass balance={balance:.3f} min=0.600")

        return (
            f"oversample rms={rms_err:.3e}; limiter excès={excess:.3e}; "
            f"headroom pic={peak:.4f}; mono balance={balance:.3f}"
        )

    def _check_delay_alignment() -> str:
        import numpy as np

        from fx_processors import Delay

        delay_divisions: list[str] = []
        for base in Delay.SUPPORTED_BASE_DIVISIONS:
            delay_divisions.append(base)
            if not base.endswith("/1"):
                delay_divisions.append(f"{base}.")
            delay_divisions.append(f"{base}t")

        tempo_test = 120.0
        sr_test = 48000
        tol_samples = 2
        for division in delay_divisions:
            delay = Delay(sr_test, tempo_bpm=tempo_test, time=division, feedback=0.0, mix=1.0)
            length = delay.delay_samples + 64
            impulse = np.zeros(length, dtype=np.float32)
            impulse[0] = 1.0
            echoed = delay.process(impulse)
            delayed_region = np.abs(echoed[1:])
            if not np.any(delayed_region > 0.5):
                raise RuntimeError(f"division {division} sans écho détecté")
            first_idx = int(np.argmax(delayed_region > 0.5)) + 1
            delta = abs(first_idx - delay.delay_samples)
            if delta > tol_samples:
                raise RuntimeError(
                    f"division {division} alignement {first_idx} vs {delay.delay_samples} tol ±{tol_samples}"
                )
        return f"divisions delay alignées à ±{tol_samples} échantillons"

    def _check_sidechain() -> str:
        import numpy as np

        from mixer_engine import MixerEngine

        mixer_cfg = {
            "audio": {"sample_rate": 48000, "mono_bass_hz": 160, "tempo_default": 120},
            "export": {"headroom_db": 6.0, "target_peak_dbtp": 0.0},
            "fx_defaults": {},
        }

        sr = int(mixer_cfg["audio"]["sample_rate"])
        sc_len = sr
        pad_signal = np.ones(sc_len, dtype=np.float32)
        control_signal = np.linspace(0.0, 1.0, sc_len, dtype=np.float32)
        baseline_engine = MixerEngine(mixer_cfg)
        baseline_engine.set_track_order(["pad"])
        baseline_mix = baseline_engine.render_mix({"pad": pad_signal})
        baseline_mean = float(np.mean(np.abs(baseline_mix[:, 0])) + 1e-12)

        depth_db = 6.0
        attack_ms = 5.0
        release_ms = 80.0
        tol_ratio = 0.10
        ratios: list[str] = []
        for shape in ("lin", "exp", "log"):
            sc_engine = MixerEngine(mixer_cfg)
            sc_engine.set_track_order(["pad"])
            sc_engine.create_sidechain(
                ["control"],
                "pad",
                depth_db=depth_db,
                attack_ms=attack_ms,
                release_ms=release_ms,
                shape=shape,
            )
            mixed = sc_engine.render_mix({"pad": pad_signal, "control": control_signal})
            mixed_mean = float(np.mean(np.abs(mixed[:, 0])) + 1e-12)
            ratio = mixed_mean / baseline_mean

            expected_gain = sc_engine._sidechain_gain_from_signal(
                control_signal,
                length=sc_len,
                depth_db=depth_db,
                attack_ms=attack_ms,
                release_ms=release_ms,
                shape=shape,
            )
            expected_ratio = float(np.mean(expected_gain))
            if expected_ratio <= 0.0:
                raise RuntimeError("gain moyen attendu nul")
            if abs(ratio - expected_ratio) > tol_ratio * expected_ratio:
                raise RuntimeError(
                    f"shape={shape} ratio={ratio:.3f} attendu={expected_ratio:.3f} tol ±{tol_ratio*100:.1f}%"
                )
            ratios.append(f"{shape}={ratio:.3f}")
        return "ratios moyens " + ", ".join(ratios)

    def _check_analysis() -> str:
        import numpy as np

        from analysis_tools import (
            detect_clicks,
            measure_correlation,
            measure_lufs,
            measure_true_peak,
            mono_bass_check,
        )

        analysis_cfg = cfg.get("analysis", {})
        style_lufs = analysis_cfg.get("tolerances", {}).get("lufs_i", {})
        sr_analysis = int(cfg["audio"].get("sample_rate", 48000))
        analysis_results: list[str] = []
        if style_lufs:
            noise_len = int(sr_analysis * 6.0)
            t = np.linspace(0.0, noise_len / sr_analysis, noise_len, endpoint=False, dtype=np.float32)
            base_wave = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
            stereo_template = np.stack([base_wave, base_wave], axis=1)
            for style_name, bounds in style_lufs.items():
                style_cfg = cfg.get("styles", {}).get(style_name)
                if not style_cfg or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    continue
                target_lufs = float(style_cfg.get("target_lufs", np.mean(bounds)))
                measured_base = measure_lufs(stereo_template, sr_analysis)
                scale = float(10.0 ** ((target_lufs - measured_base) / 20.0))
                test_signal = np.clip(stereo_template * scale, -1.0, 1.0)
                lufs = float(measure_lufs(test_signal, sr_analysis))
                lo, hi = float(bounds[0]), float(bounds[1])
                if not (lo <= lufs <= hi):
                    raise RuntimeError(
                        f"style {style_name} LUFS {lufs:.2f} tol {lo:.2f}..{hi:.2f}"
                    )
                dbtp_limit = float(
                    analysis_cfg.get("tolerances", {}).get(
                        "dbtp_max", cfg.get("export", {}).get("target_peak_dbtp", 0.0)
                    )
                )
                dbtp = float(measure_true_peak(test_signal, sr_analysis))
                if dbtp > dbtp_limit + 0.2:
                    raise RuntimeError(
                        f"style {style_name} dBTP {dbtp:.2f} > {dbtp_limit:.2f}"
                    )
                corr = float(measure_correlation(test_signal))
                corr_min = analysis_cfg.get("tolerances", {}).get("correlation_min")
                if corr_min is not None and corr < float(corr_min) - 1e-3:
                    raise RuntimeError(
                        f"style {style_name} corr {corr:.3f} < {float(corr_min):.3f}"
                    )
                bass = float(
                    mono_bass_check(
                        test_signal,
                        sr_analysis,
                        float(cfg["audio"].get("mono_bass_hz", 150.0)),
                    )
                )
                if bass < 0.9:
                    raise RuntimeError(
                        f"style {style_name} mono bass {bass:.3f} < 0.900"
                    )
                clicks = int(detect_clicks(test_signal))
                if clicks != 0:
                    raise RuntimeError(
                        f"style {style_name} {clicks} clic(s)"
                    )
                analysis_results.append(
                    f"{style_name}: LUFS={lufs:.2f}, dBTP={dbtp:.2f}, corr={corr:.3f}, mono={bass:.3f}"
                )
        if analysis_results:
            return "; ".join(analysis_results)
        return "aucune tolérance définie"

    def _check_midi_metadata() -> str:
        from midi_writer import MIDIFile

        def _assert_meta_presence(numerator: int, denominator: int) -> None:
            midi = MIDIFile()
            midi.set_tempo(123.0)
            midi.set_time_signature(numerator, denominator)
            ti = midi.add_track("test", channel=0)
            midi.add_note(ti, time_beats=0.0, note=60, velocity=100, duration_beats=1.0)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                path = tmp.name
            try:
                midi.save(path)
                with open(path, "rb") as fh:
                    data = fh.read()
            finally:
                os.remove(path)
            if b"\xFF\x51\x03" not in data:
                raise RuntimeError("meta tempo absente")
            import math

            denom_power = int(round(math.log(denominator, 2))) if denominator > 0 else 2
            ts_bytes = bytes((0xFF, 0x58, 0x04, numerator & 0xFF, denom_power & 0xFF))
            if ts_bytes not in data:
                raise RuntimeError(f"signature {numerator}/{denominator} introuvable")

        _assert_meta_presence(4, 4)
        _assert_meta_presence(3, 4)
        return "tempo + signatures 4/4 et 3/4 présents"

    def _check_cc_stats() -> str:
        from midi_cc_db import CCResolver
        from midi_writer import MIDIFile
        from sequencer import Pattern, Sequencer, Step

        resolver = CCResolver(
            studio_config_path=None,
            fallback_profile={"resonance": 71},
            studio_config={"midi_cc_hardware": {"synth_test": {"cutoff": 74}}},
        )
        seq = Sequencer(ticks_per_beat=240)
        pat = Pattern(channel=0, device="synth_test")
        pat.steps = [
            Step(beat=0.0, note=60, vel=100, length_beats=1.0, locks={"cutoff": 0.5}),
            Step(beat=1.0, note=62, vel=100, length_beats=1.0, locks={"resonance": 0.4}),
        ]
        seq.set_pattern("test", pat)
        midi = MIDIFile(ticks_per_beat=240)
        cc_summary: list[dict] = []
        seq.to_midi(midi, resolver, bpm=120.0, collect_cc=cc_summary)
        total_cc_events = sum(len(tr.cc_events) for tr in midi.tracks)
        counted_events = sum(int(entry.get("count", 1)) for entry in cc_summary)
        if total_cc_events != counted_events:
            raise RuntimeError("incohérence événements vs statistiques")
        origin_stats: dict[str, int] = {}
        for entry in cc_summary:
            origin = entry.get("origin", "HYPOTHÈSE")
            origin_stats[origin] = origin_stats.get(origin, 0) + int(entry.get("count", 1))
        if origin_stats.get("real", 0) < 1 or origin_stats.get("HYPOTHÈSE", 0) < 1:
            raise RuntimeError("origines real/HYPOTHÈSE absentes")
        if not any("test" in entry.get("tracks", {}) for entry in cc_summary):
            raise RuntimeError("piste 'test' manquante dans le résumé CC")
        return f"événements CC={total_cc_events}, origines={origin_stats}"

    def _check_drop_bus() -> str:
        from arrangement_engine import ArrangementEngine
        from mixer_engine import MixerEngine
        from modulation_matrix import DropBusSource, ModulationMatrix

        drop_assets = cfg.get("paths", {}).get("assets", {})
        drop_path = drop_assets.get("drop_protocols", "DROP_PROTOCOLS.yaml")
        drop_cache = cfg.get("_assets", {}).get("drop_protocols", {})
        arranger_test = ArrangementEngine(protocols_path=str(drop_path), protocols=drop_cache)
        drop_bus_name = cfg.get("modulation_matrix", {}).get("drop_bus_name", "DROP")

        drop_matrix = ModulationMatrix(sum_mode="weighted_clamp")
        drop_source = DropBusSource(
            "filter_sweep_master",
            bus=drop_bus_name,
            protocols_path=str(drop_path),
            protocols=arranger_test.protocols,
        )
        drop_matrix.add(drop_source, "pad.reverb_send", amt=1.0, rng=(0.0, 0.85), curve="exp")

        drop_mixer = MixerEngine(cfg)
        drop_mixer.set_track_order(["pad"])
        drop_mixer.configure_track("pad", reverb_send=0.35)

        drop_routes = [
            {"protocol": "filter_sweep_master", "destination": "pad.reverb_send", "range": (0.0, 0.85)}
        ]
        destinations = arranger_test.prepare_destination_context(drop_mixer, drop_routes)
        beats = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
        arranger_test.apply_modulations(drop_matrix, destinations, beats=beats)
        summary = arranger_test.summarise_modulations(destinations)
        pad_summary = next((entry for entry in summary if entry["destination"] == "pad.reverb_send"), None)
        if not pad_summary:
            raise RuntimeError("destination pad.reverb_send absente")
        avg_abs_delta = float(pad_summary.get("average_abs_delta", 0.0))
        if avg_abs_delta <= 0.05:
            raise RuntimeError(f"Δ moyen absolu {avg_abs_delta:.3f} ≤ 0.050")
        return f"Δ moyen absolu {avg_abs_delta:.3f} sur pad.reverb_send"

    def _check_offline_imports() -> str:
        modules = (
            "render",
            "sequencer",
            "mixer_engine",
            "dsp_core",
            "analysis_tools",
            "fx_processors",
            "arrangement_engine",
            "modulation_matrix",
            "midi_writer",
            "midi_cc_db",
        )
        for module in modules:
            importlib.import_module(module)
        return f"{len(modules)} module(s) importés"

    def _check_smoke_and_determinism() -> str:
        import wave

        def _hash_file(path: Path) -> str:
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    if not chunk:
                        break
                    digest.update(chunk)
            return digest.hexdigest()

        def _hash_report(path: Path) -> str:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["date_utc"] = "NORMALISÉE"
            serialised = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(serialised).hexdigest()

        def _run_once(tmp_path: Path) -> tuple[dict, dict[str, Path], dict[str, str]]:
            quick_style = next(iter(cfg["styles"]))
            quick_plan = {
                "style": quick_style,
                "bpm": float(
                    cfg["styles"][quick_style].get(
                        "bpm_default", cfg["audio"].get("tempo_default", 120.0)
                    )
                ),
                "seed": 2024,
                "duration_s": 1.5,
            }

            quick_cfg = copy.deepcopy(cfg)
            for transient in ("_assets", "_base_dir"):
                quick_cfg.pop(transient, None)
            quick_cfg.setdefault("analysis", {})["run_after_export"] = False
            quick_cfg.setdefault("export", {})
            if "filenames" not in quick_cfg["export"] and "export" in cfg:
                quick_cfg["export"]["filenames"] = copy.deepcopy(cfg["export"].get("filenames", {}))
            quick_cfg["export"]["render_stems"] = False
            quick_cfg.setdefault("paths", {})
            output_dir = tmp_path / "quick_output"
            midi_dir = tmp_path / "quick_midi"
            quick_cfg["paths"]["base"] = str(tmp_path)
            quick_cfg["paths"]["output"] = str(output_dir)
            quick_cfg["paths"]["midi_export"] = str(midi_dir)
            assets_paths = cfg.get("paths", {}).get("assets", {})
            quick_cfg["paths"]["assets"] = {k: str(v) for k, v in assets_paths.items()}
            project_cfg = quick_cfg.setdefault("project", {})
            project_cfg.setdefault("name", cfg.get("project", {}).get("name", "dawless"))
            project_cfg.setdefault("version", cfg.get("project", {}).get("version", "1.0"))

            quick_cfg_path = tmp_path / "quick_config.json"
            with quick_cfg_path.open("w", encoding="utf-8") as fh:
                json.dump(quick_cfg, fh, indent=2)

            report = run_session(quick_plan, config_path=str(quick_cfg_path))

            for key in ("paths", "metrics", "cc_used", "drops", "schema_version"):
                if key not in report:
                    raise RuntimeError(f"rapport Σ incomplet — clé '{key}' absente")

            paths = {k: Path(v) for k, v in report["paths"].items() if isinstance(v, str)}
            session_dir = paths.get("session_dir")
            if not session_dir or not session_dir.exists():
                raise RuntimeError("rapport Σ invalide — répertoire de session manquant")

            master_path = paths.get("master")
            if not master_path or not master_path.exists():
                raise RuntimeError("rapport Σ invalide — export master introuvable")

            with wave.open(str(master_path), "rb") as wh:
                if wh.getframerate() != 48000:
                    raise RuntimeError("master fréquence ≠ 48 kHz")
                if wh.getsampwidth() != 3:
                    raise RuntimeError("master profondeur ≠ 24-bit")

            midi_path = paths.get("midi")
            if not midi_path or not midi_path.exists():
                raise RuntimeError("rapport Σ invalide — export MIDI introuvable")

            report_path = paths.get("report")
            if not report_path or not report_path.exists():
                raise RuntimeError("rapport Σ invalide — fichier JSON absent")

            with report_path.open("r", encoding="utf-8") as fh:
                persisted = json.load(fh)
            if persisted.get("schema_version") != "1.0":
                raise RuntimeError("rapport Σ invalide — schema_version ≠ 1.0")
            if persisted.get("paths") != report["paths"]:
                raise RuntimeError("rapport Σ invalide — divergence mémoire/disque")

            hashes = {
                "master": _hash_file(master_path),
                "midi": _hash_file(midi_path),
                "report": _hash_report(report_path),
            }

            return report, paths, hashes

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_a, paths_a, hashes_a = _run_once(tmp_path)
            shutil.rmtree(paths_a["session_dir"], ignore_errors=True)
            midi_parent = Path(report_a["paths"]["midi"]).parent
            shutil.rmtree(midi_parent, ignore_errors=True)
            _report_b, _paths_b, hashes_b = _run_once(tmp_path)

        if hashes_a != hashes_b:
            raise RuntimeError("hashes divergents entre deux exécutions identiques")

        return (
            "exports master/MIDI/report valides et reproductibles — "
            f"hash master {hashes_a['master'][:8]}…"
        )

    run_check("Budget fichiers", _check_file_budget)
    run_check("DSP de base", _check_dsp_core)
    run_check("Alignement delay", _check_delay_alignment)
    run_check("Sidechain", _check_sidechain)
    run_check("Analyses audio", _check_analysis)
    run_check("MIDI metadata", _check_midi_metadata)
    run_check("Statistiques CC", _check_cc_stats)
    run_check("Drop bus", _check_drop_bus)
    run_check("Scan offline imports", _check_offline_imports)
    run_check("Session smoke + déterminisme", _check_smoke_and_determinism)

    duration = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def _check_resources() -> str:
        peak_mb = peak / (1024 * 1024)
        if duration > time_limit_s:
            raise RuntimeError(f"temps {duration:.2f}s > {time_limit_s:.2f}s")
        if peak > memory_limit_bytes:
            raise RuntimeError(
                f"mémoire pic {peak_mb:.1f} MiB > {memory_limit_bytes / (1024 * 1024):.0f} MiB"
            )
        return f"temps {duration:.2f}s, pic mémoire {peak_mb:.1f} MiB"

    run_check("Budget ressources", _check_resources)
    print(f"[Σ] self-test terminé en {duration:.2f}s, pic mémoire {peak / (1024 * 1024):.1f} MiB.")


def main() -> None:
    ap = argparse.ArgumentParser(description="DawlessGPT-Σ — rendu et exports automatiques")
    ap.add_argument("--style", help="Style de rendu à sélectionner (obligatoire sauf en mode --selftest)")
    ap.add_argument("--bpm", type=float, help="Tempo spécifique (sinon valeur par défaut du style)")
    ap.add_argument("--seed", type=int, default=1337, help="Graine pseudo-aléatoire pour le rendu")
    ap.add_argument("--dur", type=float, default=84.0, help="Durée cible du morceau en secondes")
    ap.add_argument("--config", default="config.yaml", help="Chemin vers le fichier de configuration YAML")
    ap.add_argument("--selftest", action="store_true", help="Exécuter une vérification rapide sans rendu complet")
    ap.add_argument("--dry-run", action="store_true", help="Afficher le plan de session sans lancer de rendu")
    ap.add_argument("--no-audio", action="store_true", help="Désactiver l'export audio (expérimental)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config_safe(str(cfg_path))

    base_dir = Path(cfg.get("_base_dir", cfg_path.resolve().parent))

    if args.selftest:
        run_selftest(cfg, base_dir=base_dir)
        return

    if not args.style:
        ap.error("--style est requis pour lancer un rendu (utilisez --selftest pour la vérification seule)")

    check_file_budget(cfg, base_dir=base_dir)

    if cfg.get("logging", {}).get("level", "INFO") == "INFO":
        print(
            f"[Σ] launching style={args.style} bpm={args.bpm or 'default'} seed={args.seed} dur={args.dur}s"
        )

    plan = {"style": args.style, "seed": args.seed, "duration_s": args.dur}
    if args.bpm is not None:
        plan["bpm"] = args.bpm
    if args.no_audio:
        plan["no_audio"] = True

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    rep = run_session(plan, args.config)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
