# /mnt/data/modulation_matrix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import ast
import json


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
                data = _load_yaml_like(fh.read()) or {}
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
def _parse_scalar(value: str) -> Any:
    token = value.strip()
    if not token or token.lower() in {"null", "~"}:
        return None
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_inline_dict(text: str) -> Dict[str, Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        current.append(ch)
    if current:
        parts.append("".join(current).strip())

    out: Dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        key, _, raw_val = part.partition(":")
        key = key.strip().strip('"\'')
        out[key] = _parse_value(raw_val.strip())
    return out


def _parse_value(token: str) -> Any:
    if not token:
        return None
    stripped = token.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return _parse_inline_dict(stripped)
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return stripped
    return _parse_scalar(stripped)


def _load_yaml_like(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except Exception:
        pass

    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        trimmed = raw.split("#", 1)[0].rstrip()
        if not trimmed.strip():
            continue
        indent = len(trimmed) - len(trimmed.lstrip(" "))
        content = trimmed.strip()
        lines.append((indent, content))

    if not lines:
        return {}

    root: Any = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(lines):
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("unexpected list item outside list context")
            value_part = content[2:].strip()
            if not value_part:
                container: Any
                if next_line and next_line[0] > indent and next_line[1].startswith("- "):
                    container = []
                else:
                    container = {}
                parent.append(container)
                stack.append((indent + 2, container))
            else:
                parent.append(_parse_value(value_part))
            continue

        key, _, raw_val = content.partition(":")
        if not _:
            raise ValueError(f"ligne YAML invalide: {content}")
        key = key.strip().strip('"\'')
        value_part = raw_val.strip()
        if not value_part:
            container = [] if (next_line and next_line[0] > indent and next_line[1].startswith("- ")) else {}
            if isinstance(parent, list):
                parent.append({key: container})
                stack.append((indent + 2, container))
            else:
                parent[key] = container
                stack.append((indent + 2, container))
        else:
            value = _parse_value(value_part)
            if isinstance(parent, list):
                parent.append({key: value})
            else:
                parent[key] = value

    return root

