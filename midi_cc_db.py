# /mnt/data/midi_cc_db.py
"""Helpers for resolving MIDI CC mappings.

The resolver merges hardware mappings defined in ``STUDIO_CONFIG`` with
optional overrides (``cc_mappings.json``). When a mapping cannot be
confirmed by real hardware data the resolver falls back to a generic
profile and marks the origin as ``"HYPOTHÈSE"`` so the renderer can log a
warning and keep track of the assumptions made.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


class CCResolver:
    def __init__(
        self,
        studio_config_path: Optional[str],
        cc_overrides_path: Optional[str] = None,
        fallback_profile: Optional[Dict[str, int]] = None,
        studio_config: Optional[Dict[str, Any]] = None,
        cc_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._device_maps: Dict[str, Dict[str, int]] = {}
        self._global_overrides: Dict[str, int] = {}
        self._aliases: Dict[str, str] = {}
        self.fallback: Dict[str, int] = {k.lower(): int(v) for k, v in (fallback_profile or {}).items()}

        cfg_data: Dict[str, Any]
        if studio_config is not None:
            cfg_data = dict(studio_config)
        elif studio_config_path and os.path.exists(studio_config_path):
            with open(studio_config_path, "r", encoding="utf-8") as fh:
                cfg_data = json.load(fh)
        else:
            cfg_data = {}

        hw = cfg_data.get("midi_cc_hardware", {}) if isinstance(cfg_data, dict) else {}
        for dev, mapping in hw.items():
            self._device_maps[dev.lower()] = {
                str(param).lower(): int(value) for param, value in mapping.items()
            }

        overrides_data: Dict[str, Any]
        if cc_overrides is not None:
            overrides_data = dict(cc_overrides)
        elif cc_overrides_path and os.path.exists(cc_overrides_path):
            with open(cc_overrides_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
            overrides_data = json.loads("\n".join(lines) or "{}")
        else:
            overrides_data = {}

        if isinstance(overrides_data, dict):
            aliases = overrides_data.get("aliases", {})
            if isinstance(aliases, dict):
                for alias, target in aliases.items():
                    self._aliases[str(alias).lower()] = str(target).lower()

            defaults = overrides_data.get("defaults") or overrides_data.get("global")
            if isinstance(defaults, dict):
                self._global_overrides.update({str(k).lower(): int(v) for k, v in defaults.items()})

            devices = overrides_data.get("devices") or overrides_data
            if isinstance(devices, dict):
                for dev, mapping in devices.items():
                    if not isinstance(mapping, dict):
                        continue
                    # skip special keys
                    if dev in {"aliases", "defaults", "global"}:
                        continue
                    normalized_dev = str(dev).lower()
                    device_map = self._device_maps.setdefault(normalized_dev, {})
                    for param, value in mapping.items():
                        if isinstance(value, (int, float)):
                            device_map[str(param).lower()] = int(value)

    # ------------------------------------------------------------------
    def _canonical_param(self, param: str) -> str:
        key = (param or "").lower()
        return self._aliases.get(key, key)

    def resolve(self, param: str, device: Optional[str] = None) -> Tuple[int, str]:
        """Resolve a parameter to a CC number.

        Returns a tuple ``(cc_number, origin)`` where ``origin`` is either
        ``"real"`` (confirmed hardware mapping) or ``"HYPOTHÈSE"`` when the
        resolver had to rely on a generic fallback.
        """

        canonical = self._canonical_param(param)
        if device:
            dev_key = device.lower()
            dev_map = self._device_maps.get(dev_key)
            if dev_map and canonical in dev_map:
                return int(dev_map[canonical]), "real"

        if canonical in self._global_overrides:
            return int(self._global_overrides[canonical]), "real"

        if canonical in self.fallback:
            return int(self.fallback[canonical]), "HYPOTHÈSE"

        default_map = {
            "cutoff": 74,
            "resonance": 71,
            "volume": 7,
            "pan": 10,
            "expression": 11,
            "reverb_send": 91,
            "chorus_send": 93,
            "delay_send": 94,
            "glide": 5,
        }
        cc = default_map.get(canonical, 74)
        return int(cc), "HYPOTHÈSE"

    # ------------------------------------------------------------------
    @staticmethod
    def scale_0_1_to_0_127(v: float) -> int:
        v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
        return int(round(v * 127.0))
