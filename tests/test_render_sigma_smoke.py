import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from render import run_session

from tests._test_utils import prepare_config


def _make_plan(style: str, cfg: dict) -> dict:
    style_cfg = cfg["styles"][style]
    bpm = float(style_cfg.get("bpm_default", cfg["audio"].get("tempo_default", 120.0)))
    return {
        "style": style,
        "bpm": bpm,
        "duration_s": 6.0,
        "seed": int(cfg.get("project", {}).get("seed", 1337)),
    }


def test_render_sigma_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        config_path, cfg = prepare_config(tmp, stems=False)
        styles = ["techno_peak", "ambient", "dnb"]
        reports = []
        for style in styles:
            plan = _make_plan(style, cfg)
            report = run_session(plan, config_path=config_path)
            reports.append(report)
            master = report["paths"]["master"]
            midi = report["paths"]["midi"]
            json_report = report["paths"]["report"]
            assert os.path.exists(master)
            assert os.path.exists(midi)
            assert os.path.exists(json_report)
            print(f"[{style}] metrics:\n" + json.dumps(report["metrics"], indent=2, sort_keys=True))

        # Reload reports to ensure they are valid JSON.
        for rep in reports:
            with open(rep["paths"]["report"], "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            assert parsed["project"] == cfg.get("project", {}).get("name", "dawless")
            assert parsed["paths"]["master"]
