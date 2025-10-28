# /mnt/data/cli.py
import argparse
import json
import os
from pathlib import Path
from typing import Sequence
from render import run_session, load_config_safe

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
        check_file_budget(cfg, base_dir=base_dir)

        import numpy as np

        from dsp_core import (
            brickwall_limit,
            db_to_lin,
            downsample_linear,
            oversample_linear,
        )
        from mixer_engine import MixerEngine
        from fx_processors import Delay

        rng = np.random.default_rng(1337)
        signal = np.array(rng.standard_normal(4096), dtype=np.float32) * 0.25
        factor = 4
        oversampled = oversample_linear(signal, factor)
        recovered = downsample_linear(oversampled, factor)
        rms_err = float(np.sqrt(np.mean((signal - recovered) ** 2)))
        rms_tolerance = 5e-7
        if rms_err > rms_tolerance:
            raise RuntimeError(
                f"Échec du test oversample/downsample: rms={rms_err:.3e} tol={rms_tolerance:.1e}"
            )
        print(
            f"[Σ] self-test DSP oversample/downsample — rms={rms_err:.3e} (tol {rms_tolerance:.1e})."
        )

        ceiling_db = -1.0
        ceiling = np.float32(db_to_lin(ceiling_db))
        limiter_input = np.array([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=np.float32)
        limited = brickwall_limit(limiter_input, ceiling_db)
        excess = float(max(0.0, np.max(np.abs(limited)) - ceiling))
        ceiling_tolerance = 1e-7
        if excess > ceiling_tolerance:
            raise RuntimeError(
                f"Échec du test brickwall_limit: dépassement de {excess:.3e} (tol {ceiling_tolerance:.1e})"
            )
        print(
            f"[Σ] self-test DSP brickwall_limit — excès max={excess:.3e} (tol {ceiling_tolerance:.1e})."
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
                f"Échec du test de headroom: pic={peak:.4f} attendu={expected_peak:.4f} (tol {headroom_tol:.1e})"
            )
        print(
            f"[Σ] self-test mixer headroom — pic={peak:.4f} (attendu {expected_peak:.4f} ±{headroom_tol:.1e})."
        )

        def _lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
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

        mono_bass_tone = (0.5 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        mono_mixer = MixerEngine(mixer_cfg)
        mono_mixer.set_track_order(["tone"])
        mono_mixer.configure_track("tone", pan=1.0)
        mono_mix = mono_mixer.render_mix({"tone": mono_bass_tone})
        low_l = _lowpass(mono_mix[:, 0], float(mixer_cfg["audio"]["mono_bass_hz"]), sr)
        low_r = _lowpass(mono_mix[:, 1], float(mixer_cfg["audio"]["mono_bass_hz"]), sr)
        energy_l = float(np.sum(low_l ** 2))
        energy_r = float(np.sum(low_r ** 2))
        if energy_l <= 0.0 or energy_r <= 0.0:
            raise RuntimeError("Échec du test mono-bass: énergie basse fréquence nulle sur une des voies.")
        balance = min(energy_l, energy_r) / max(energy_l, energy_r)
        if balance < 0.6:
            raise RuntimeError(
                f"Échec du test mono-bass: déséquilibre d'énergie basse fréquence {balance:.3f} (< 0.600 attendu)"
            )
        print(
            f"[Σ] self-test mixer mono-bass — équilibre basse fréquence {balance:.3f} (min 0.600)."
        )

        delay_divisions = []
        for base in Delay.SUPPORTED_BASE_DIVISIONS:
            delay_divisions.append(base)
            if not base.endswith("/1"):
                delay_divisions.append(f"{base}.")
            delay_divisions.append(f"{base}t")

        tempo_test = 120.0
        sr_test = 48000
        tol_samples = 2
        for division in delay_divisions:
            delay = Delay(
                sr_test,
                tempo_bpm=tempo_test,
                time=division,
                feedback=0.0,
                mix=1.0,
            )
            length = delay.delay_samples + 64
            impulse = np.zeros(length, dtype=np.float32)
            impulse[0] = 1.0
            echoed = delay.process(impulse)
            delayed_region = np.abs(echoed[1:])
            if not np.any(delayed_region > 0.5):
                raise RuntimeError(
                    f"Échec du test délai tempo-sync: aucun écho détecté pour division {division}."
                )
            first_idx = int(np.argmax(delayed_region > 0.5)) + 1
            delta = abs(first_idx - delay.delay_samples)
            if delta > tol_samples:
                raise RuntimeError(
                    f"Échec du test délai tempo-sync: division {division} alignement {first_idx} vs {delay.delay_samples} (tol ±{tol_samples})."
                )
        print(
            f"[Σ] self-test delay tempo-sync — toutes les divisions alignées à ±{tol_samples} échantillons."
        )

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
                raise RuntimeError("Échec du test sidechain: gain moyen attendu nul.")
            if abs(ratio - expected_ratio) > tol_ratio * expected_ratio:
                raise RuntimeError(
                    "Échec du test sidechain: "
                    f"shape={shape} ratio={ratio:.3f} attendu={expected_ratio:.3f} (tol ±{tol_ratio*100:.1f} %)."
                )
        print(
            "[Σ] self-test sidechain — réduction moyenne conforme pour shapes lin/exp/log (±10%)."
        )

        print("[Σ] self-test ok — budget fichiers respecté et configuration chargée.")
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
