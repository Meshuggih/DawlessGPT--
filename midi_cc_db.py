# /mnt/data/midi_cc_db.py
import json
import os
from typing import Any, Dict, Optional, Tuple

class CCResolver:
    def __init__(self, studio_config_path: Optional[str], cc_overrides_path: Optional[str]=None,
                 fallback_profile: Optional[Dict[str,int]]=None,
                 studio_config: Optional[Dict[str, Any]] = None,
                 cc_overrides: Optional[Dict[str, Any]] = None):
        self.db: Dict[str, Dict[str,int]] = {}
        self.fallback: Dict[str,int] = fallback_profile or {}
        cfg_data: Dict[str, Any] = {}
        if studio_config is not None:
            cfg_data = dict(studio_config)
        elif studio_config_path and os.path.exists(studio_config_path):
            with open(studio_config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

        hw = cfg_data.get("midi_cc_hardware", {}) if isinstance(cfg_data, dict) else {}
        for dev, mapping in hw.items():
            self.db[dev.lower()] = {k.lower(): int(v) for k, v in mapping.items()}

        overrides_data: Dict[str, Any] = {}
        if cc_overrides is not None:
            overrides_data = dict(cc_overrides)
        elif cc_overrides_path and os.path.exists(cc_overrides_path):
            with open(cc_overrides_path, "r", encoding="utf-8") as f:
                overrides_data = json.load(f)

        for dev, mapping in overrides_data.items():
            d = self.db.setdefault(dev.lower(), {})
            d.update({k.lower(): int(v) for k, v in mapping.items()})

    def resolve(self, param: str, device: Optional[str]=None) -> Tuple[int,str]:
        p = (param or "").lower()
        if device:
            dev = device.lower()
            cc = self.db.get(dev, {}).get(p, None)
            if cc is not None:
                return int(cc), "real"
        if p in self.fallback:
            return int(self.fallback[p]), "HYPOTHÈSE"
        default_map = {"cutoff":74, "resonance":71, "volume":7, "pan":10, "reverb_send":91, "delay_send":94, "glide":5}
        cc = default_map.get(p, 74)
        return int(cc), "HYPOTHÈSE"

    @staticmethod
    def scale_0_1_to_0_127(v: float) -> int:
        v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
        return int(round(v * 127.0))
