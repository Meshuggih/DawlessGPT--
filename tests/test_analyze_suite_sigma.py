import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from render import run_session

from tests._test_utils import prepare_config


def _plan(style: str, cfg: dict) -> dict:
    style_cfg = cfg["styles"][style]
    bpm = float(style_cfg.get("bpm_default", cfg["audio"].get("tempo_default", 120.0)))
    return {
        "style": style,
        "bpm": bpm,
        "duration_s": 6.0,
        "seed": int(cfg.get("project", {}).get("seed", 1337)) + hash(style) % 1000,
    }


def test_analyze_suite_sigma():
    with tempfile.TemporaryDirectory() as tmp:
        config_path, cfg = prepare_config(tmp, stems=False)
        tol = cfg["analysis"]["tolerances"]
        lufs_ranges = tol.get("lufs_i", {})
        corr_min = float(tol.get("correlation_min", 0.0))
        dbtp_max = float(tol.get("dbtp_max", 0.0))
        mono_min = float(tol.get("mono_bass_min", 0.85))

        for style in ["techno_peak", "ambient", "dnb"]:
            plan = _plan(style, cfg)
            report = run_session(plan, config_path=config_path)
            metrics = report["metrics"]
            l_lo, l_hi = lufs_ranges[style]
            assert l_lo <= metrics["lufs_i"] <= l_hi, f"LUFS {metrics['lufs_i']} not in [{l_lo}, {l_hi}]"
            assert metrics["dbtp"] <= dbtp_max + 0.2
            assert metrics["correlation"] >= corr_min - 1e-6
            assert metrics["mono_bass_check"] >= mono_min - 0.05
            assert isinstance(metrics["clicks"], int)
            with open(report["paths"]["report"], "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            assert parsed["metrics"]["dbtp"] == metrics["dbtp"]
            assert parsed["sc_routes"], "Sidechain routes missing in report"
