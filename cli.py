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
