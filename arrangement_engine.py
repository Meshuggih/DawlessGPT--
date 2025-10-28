# /mnt/data/arrangement_engine.py
from typing import Any, Dict, Optional

import yaml

class ArrangementEngine:
    def __init__(self, protocols_path: str = 'DROP_PROTOCOLS.yaml', protocols: Optional[Dict[str, Any]] = None):
        self.protocols_path = protocols_path
        if protocols is not None:
            self.protocols = dict(protocols)
        else:
            with open(self.protocols_path, 'r', encoding='utf-8') as f:
                self.protocols = yaml.safe_load(f) or {}

    def get_protocol(self, name: str) -> dict:
        return self.protocols.get(name, {})

    def get_value(self, name: str, beat: float) -> float:
        p = self.get_protocol(name)
        if not p: return 0.0
        pts = p.get("curve", [])
        if not pts: return 0.0
        for (b0,v0),(b1,v1) in zip(pts, pts[1:]):
            if b0 <= beat <= b1:
                t = 0.0 if b1==b0 else (beat-b0)/(b1-b0)
                return v0 + t*(v1-v0)
        return float(pts[-1][1])

class DropBusSource:
    def __init__(self, arranger: ArrangementEngine, name: str):
        self.arranger, self.name = arranger, name
    def get(self, ctx) -> float:
        beat = getattr(ctx, "beat", 0.0)
        return float(self.arranger.get_value(self.name, beat))
