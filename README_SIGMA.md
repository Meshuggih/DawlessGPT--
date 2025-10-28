# /mnt/data/README_SIGMA.md
# DawlessGPT-Σ v1.0 — README

## Exécution rapide (offline)
```bash
python /mnt/data/cli.py --style techno_peak --bpm 130 --seed 4242 --dur 84
python /mnt/data/cli.py --style ambient     --bpm 90  --seed 4243 --dur 90
python /mnt/data/cli.py --style dnb         --bpm 168 --seed 4247 --dur 72
```

## Budget fichiers (strict)
**La CLI vérifie automatiquement que le budget de 20 fichiers de production est respecté (répertoires `output/` et `midi_export/` exclus) avant chaque rendu ou `--selftest`.**
