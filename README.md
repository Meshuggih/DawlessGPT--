# DawlessGPT-Σ

DawlessGPT-Σ est une toolchain de rendu autonome orientée techno/ambient/dnb. Le projet fonctionne **offline** avec Python (stdlib + numpy) et respecte un budget de **≤ 20 fichiers produit** (hors annexe `PROMPT_SYSTEME_SIGMA.md`).

## 🚀 Démarrage rapide (offline)
```bash
python /mnt/data/cli.py --style techno_peak --bpm 130 --seed 4242 --dur 84
python /mnt/data/cli.py --style ambient     --bpm  90 --seed 4243 --dur 90
python /mnt/data/cli.py --style dnb         --bpm 168 --seed 4247 --dur 72
```

Les exports (WAV 24-bit / 48 kHz, MIDI et rapport Σ JSON) sont regroupés dans `output/<projet>_<style>_seed<seed>/` et `midi_export/`.

## QA intégré (offline)
```bash
python /mnt/data/cli.py --selftest --style techno_peak --bpm 130 --seed 4242 --dur 8
```

- Delay tempo-sync : subdivisions 1/8, 1/8., 1/8T, 1/16, 3/16 (1er écho ±2 échantillons).
- Smoke render : techno_peak / ambient / dnb → présence des exports WAV / MID / REPORT.
- Analyse : métriques `lufs_i`, `dbtp`, `correlation`, `mono_bass_check`, `clicks` respectant `config.yaml`.

## 🧪 Acceptance criteria (mesurables)
- **Budget** : ≤ 20 fichiers produit + 1 annexe ; `README_SIGMA.md` supprimé ; unique `STYLE_TEMPLATES.json`.
- **Delay tempo-sync** : test intégré passe (±2 samples pour 1/8, 1/8., 1/8T, 1/16, 3/16).
- **Sidechain** : paramètre `shape` (lin/exp/log) modifie la courbe de ducking.
- **True-Peak** : `headroom_db` pré-limiteur appliqué, dBTP ≤ −0,30 dB (selon métrique approximée).
- **MIDI** : méta 0x51 (tempo) + méta 0x58 (signature temporelle) présents.
- **Métriques** : clés exactes `lufs_i`, `dbtp`, `correlation`, `mono_bass_check`, `clicks`.
- **Report** : inclut `version` et `date_utc` (auto ISO-8601 si “auto”).
- **CLI** : `--selftest` disponible, retourne succès si tout passe.
- **README** : usage `--selftest` documenté ; rappel “stdlib + numpy uniquement ; offline ; pas d’ajout de fichiers”.

## 🔒 Contraintes non-négociables
- Aucune dépendance additionnelle (stdlib + numpy).
- Aucun nouveau fichier produit (budget strict ≤ 20 + 1 annexe).
- Compat ascendante Σ v1.0 : pas de renommage d’APIs publiques.

## 🧭 Commandes de validation
- `python /mnt/data/cli.py --selftest --style techno_peak --bpm 130 --seed 4242 --dur 8`
- `python /mnt/data/cli.py --style ambient --bpm 90 --seed 4243 --dur 10`
- Inspecter le rapport Σ (`paths.report`) : métriques harmonisées, `cc_origin_stats`, `version`, `date_utc`.
