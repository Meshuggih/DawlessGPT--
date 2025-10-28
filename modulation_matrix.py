# /mnt/data/modulation_matrix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class Route:
    src: Any
    dst: Any
    amt: float = 1.0
    rng: Tuple[float, float] = (0.0, 1.0)
    curve: str = "lin"


def _shape(name: str, x: float) -> float:
    if name == "lin":
        return x
    if name == "exp":
        return x * x
    if name == "log":
        return x ** 0.5
    return x


def _remap(x: float, lo: float, hi: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return lo + (hi - lo) * x


class DropBusSource:
    """Evaluate drop automation curves defined in ``DROP_PROTOCOLS.yaml``."""

    _CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        protocol_name: str,
        *,
        bus: str = "DROP",
        protocols_path: str = "DROP_PROTOCOLS.yaml",
        protocols: Dict[str, Any] | None = None,
    ) -> None:
        self.protocol_name = str(protocol_name)
        self.bus = str(bus)
        self.name = f"{self.bus}.{self.protocol_name}"
        self.protocols_path = protocols_path
        if protocols is not None:
            self._protocols = dict(protocols)
        else:
            self._protocols = self._load_protocols(protocols_path)

    # ------------------------------------------------------------------ util --
    @classmethod
    def _load_protocols(cls, path: str) -> Dict[str, Any]:
        cached = cls._CACHE.get(path)
        if cached is not None:
            return cached
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        cls._CACHE[path] = data
        return data

    def _get_curve(self) -> List[Tuple[float, float]]:
        proto = self._protocols.get(self.protocol_name, {})
        curve = proto.get("curve", []) if isinstance(proto, dict) else []
        pts: List[Tuple[float, float]] = []
        for entry in curve:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            try:
                beat = float(entry[0])
                value = float(entry[1])
            except (TypeError, ValueError):
                continue
            pts.append((beat, value))
        pts.sort(key=lambda item: item[0])
        return pts

    def _sample_curve(self, beat: float) -> float:
        pts = self._get_curve()
        if not pts:
            return 0.0
        if beat <= pts[0][0]:
            return float(pts[0][1])
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b0 <= beat <= b1:
                if b1 == b0:
                    return float(v1)
                t = (beat - b0) / (b1 - b0)
                return float(v0 + t * (v1 - v0))
        return float(pts[-1][1])

    def get(self, ctx: Any | None = None) -> float:
        beat = 0.0
        if ctx is not None:
            beat = float(getattr(ctx, "beat", getattr(ctx, "time", 0.0)))
            if isinstance(ctx, dict):  # allow dict-based contexts
                beat = float(ctx.get("beat", beat))
        return self._sample_curve(beat)


class MacroSource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.v = 0.0

    def set(self, v: float) -> None:
        self.v = float(v)

    def get(self, ctx: Any | None = None) -> float:
        return self.v


class CCSource:
    def __init__(self, cc_number: int) -> None:
        self.cc = cc_number
        self.v = 0.0

    def feed(self, value_0_1: float) -> None:
        self.v = float(value_0_1)

    def get(self, ctx: Any | None = None) -> float:
        return self.v


class ModulationMatrix:
    def __init__(self, sum_mode: str = "weighted_clamp") -> None:
        self.routes: List[Route] = []
        self.sum_mode = sum_mode
        self.macros = {k: MacroSource(k) for k in ("MacroA", "MacroB", "MacroC", "MacroD")}
        self.cc_sources: Dict[int, CCSource] = {}

    def get_macro(self, name: str) -> MacroSource:
        return self.macros[name]

    def get_cc(self, cc_number: int) -> CCSource:
        if cc_number not in self.cc_sources:
            self.cc_sources[cc_number] = CCSource(cc_number)
        return self.cc_sources[cc_number]

    def add(self, src: Any, dst: Any, amt: float = 1.0, rng: Tuple[float, float] = (0.0, 1.0), curve: str = "lin") -> None:
        self.routes.append(Route(src=src, dst=dst, amt=amt, rng=rng, curve=curve))

    def _resolve_destination_name(self, dst: Any) -> str | None:
        if isinstance(dst, str):
            return dst
        name = getattr(dst, "name", None)
        if isinstance(name, str):
            return name
        return None

    def apply(self, ctx: Any) -> Dict[str, Any]:
        destinations = {}
        beat = float(getattr(ctx, "beat", getattr(ctx, "time", 0.0)))
        if isinstance(ctx, dict):
            destinations = ctx.get("destinations", {})
            beat = float(ctx.get("beat", beat))
        else:
            destinations = getattr(ctx, "destinations", {})

        if isinstance(destinations, dict):
            for slot in destinations.values():
                reset = getattr(slot, "reset_frame", None)
                if callable(reset):
                    reset()

        for r in self.routes:
            src_val = float(r.src.get(ctx))
            shaped = _shape(r.curve, src_val)
            scaled = shaped * float(r.amt)
            val = _remap(scaled, r.rng[0], r.rng[1])
            weight = abs(float(r.amt)) if r.amt is not None else 1.0
            dst_name = self._resolve_destination_name(r.dst)
            if dst_name and isinstance(destinations, dict) and dst_name in destinations:
                slot = destinations[dst_name]
                accumulate = getattr(slot, "accumulate", None)
                if callable(accumulate):
                    accumulate(
                        source=getattr(r.src, "name", repr(r.src)),
                        value=float(val),
                        weight=float(weight),
                        raw=float(src_val),
                        curve=r.curve,
                    )
                    continue
            if hasattr(r.dst, "set") and callable(getattr(r.dst, "set")):
                r.dst.set(val)
            else:
                try:
                    r.dst(val, ctx)
                except TypeError:
                    pass

        results: Dict[str, Any] = {}
        if isinstance(destinations, dict):
            for name, slot in destinations.items():
                commit = getattr(slot, "commit", None)
                if callable(commit):
                    results[name] = commit(beat, self.sum_mode)
        return results
