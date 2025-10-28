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
