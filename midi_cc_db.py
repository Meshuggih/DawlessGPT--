# /mnt/data/midi_cc_db.py
import json, os
from typing import Dict, Optional, Tuple

class CCResolver:
    def __init__(self, studio_config_path: str, cc_overrides_path: Optional[str]=None,
                 fallback_profile: Optional[Dict[str,int]]=None):
        self.db: Dict[str, Dict[str,int]] = {}
        self.fallback: Dict[str,int] = fallback_profile or {}
        if studio_config_path and os.path.exists(studio_config_path):
            with open(studio_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            hw = cfg.get("midi_cc_hardware", {})
            for dev, mapping in hw.items():
                self.db[dev.lower()] = {k.lower(): int(v) for k,v in mapping.items()}
        if cc_overrides_path and os.path.exists(cc_overrides_path):
            with open(cc_overrides_path, "r", encoding="utf-8") as f:
                ov = json.load(f)
            for dev, mapping in ov.items():
                d = self.db.setdefault(dev.lower(), {})
                d.update({k.lower(): int(v) for k,v in mapping.items()})

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