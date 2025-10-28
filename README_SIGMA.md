# /mnt/data/README_SIGMA.md
# DawlessGPT-Σ v1.0 — README

## Exécution rapide (offline)
```bash
python /mnt/data/cli.py --style techno_peak --bpm 130 --seed 4242 --dur 84
python /mnt/data/cli.py --style ambient     --bpm 90  --seed 4243 --dur 90
python /mnt/data/cli.py --style dnb         --bpm 168 --seed 4247 --dur 72
```

## Export audio / MIDI / rapport Σ
* Les exports WAV sont générés en 24-bit / 48 kHz et regroupés dans `output/<projet>_<style>_seed<seed>/`.
* Le fichier MIDI correspondant est écrit dans `midi_export/` en reprenant le même tag (`project_style_bpm_seed.mid`).
* Le rapport JSON Σ (`schema_version=1.0`) est sauvegardé aux côtés du master dans le sous-dossier de session et contient les chemins réels des exports, les métriques d'analyse et le résumé des drops/CC.
* Les noms de fichiers sont assainis à partir du projet, du style et de la seed afin d'éviter les caractères problématiques.

## Budget fichiers (strict)
**La CLI vérifie automatiquement que le budget de 20 fichiers de production est respecté (répertoires `output/` et `midi_export/` exclus) avant chaque rendu ou `--selftest`.**

## Auto-test (`--selftest`)
La commande `python /mnt/data/cli.py --selftest` exécute une batterie de contrôles rapides incluant désormais un rendu miniature garantissant la présence des fichiers master/MIDI/rapport, la fréquence d'échantillonnage 48 kHz, la résolution 24-bit et l'intégrité du rapport Σ.
