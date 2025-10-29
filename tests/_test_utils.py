import json
import os
import sys
from typing import Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from render import load_config_safe


def prepare_config(tmp_dir: str, *, stems: bool = False) -> Tuple[str, dict]:
    cfg = load_config_safe("config.yaml", use_cache=False)
    output_dir = os.path.join(tmp_dir, "output")
    midi_dir = os.path.join(tmp_dir, "midi")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)

    cfg["paths"]["base"] = os.getcwd()
    cfg["paths"]["output"] = output_dir
    cfg["paths"]["midi_export"] = midi_dir
    cfg["export"]["render_stems"] = bool(stems)
    cfg["export"]["normalize"] = True
    cfg["export"]["dither"] = True
    cfg["analysis"]["run_after_export"] = True

    cfg.pop("_assets", None)
    cfg.pop("_base_dir", None)

    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return config_path, cfg
