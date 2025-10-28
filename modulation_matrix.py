# /mnt/data/modulation_matrix.py
from dataclasses import dataclass
from typing import Any, List, Tuple

@dataclass
class Route:
    src: Any
    dst: Any
    amt: float = 1.0
    rng: Tuple[float,float] = (0.0, 1.0)
    curve: str = "lin"

def _shape(name: str, x: float) -> float:
    if name == "lin": return x
    if name == "exp": return x*x
    if name == "log": return x**0.5
    return x

def _remap(x: float, lo: float, hi: float) -> float:
    x = min(max(x,0.0),1.0)
    return lo + (hi-lo)*x

class MacroSource:
    def __init__(self, name: str): self.name, self.v = name, 0.0
    def set(self, v: float): self.v = float(v)
    def get(self, ctx=None) -> float: return self.v

class CCSource:
    def __init__(self, cc_number: int): self.cc = cc_number; self.v = 0.0
    def feed(self, value_0_1: float): self.v = float(value_0_1)
    def get(self, ctx=None) -> float: return self.v

class ModulationMatrix:
    def __init__(self, sum_mode: str = "weighted_clamp"):
        self.routes: List[Route] = []
        self.sum_mode = sum_mode
        self.macros = {k: MacroSource(k) for k in ("MacroA","MacroB","MacroC","MacroD")}
        self.cc_sources = {}

    def get_macro(self, name: str) -> MacroSource: return self.macros[name]
    def get_cc(self, cc_number: int) -> CCSource:
        if cc_number not in self.cc_sources:
            self.cc_sources[cc_number] = CCSource(cc_number)
        return self.cc_sources[cc_number]

    def add(self, src: Any, dst: Any, amt: float=1.0, rng=(0.0,1.0), curve="lin"):
        self.routes.append(Route(src=src, dst=dst, amt=amt, rng=rng, curve=curve))

    def apply(self, ctx) -> None:
        for r in self.routes:
            s = r.src.get(ctx)
            val = _remap(_shape(r.curve, s) * r.amt, r.rng[0], r.rng[1])
            if hasattr(r.dst, "set") and callable(getattr(r.dst, "set")):
                r.dst.set(val)
            else:
                try: r.dst(val, ctx)
                except TypeError: pass