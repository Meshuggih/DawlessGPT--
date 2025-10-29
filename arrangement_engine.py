# /mnt/data/arrangement_engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ast
import json


@dataclass
class DestinationSlot:
    name: str
    track: str
    parameter: str
    base_value: float
    min_value: float = 0.0
    max_value: float = 1.0
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_value = float(self.base_value)
        lo = float(min(self.min_value, self.max_value))
        hi = float(max(self.min_value, self.max_value))
        self.min_value = lo
        self.max_value = hi
        self.value = self.base_value
        self._frame_contribs: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------- frame --
    def reset_frame(self) -> None:
        self.value = self.base_value
        self._frame_contribs = []

    def accumulate(self, *, source: str, value: float, weight: float, raw: float, curve: str) -> None:
        self._frame_contribs.append(
            {
                "source": source,
                "value": float(value),
                "weight": float(weight),
                "raw": float(raw),
                "curve": curve,
            }
        )

    def _combine_frame(self, sum_mode: str) -> float:
        if not self._frame_contribs:
            return self.base_value
        if sum_mode == "weighted_clamp":
            total_weight = 0.0
            weighted_sum = 0.0
            for contrib in self._frame_contribs:
                weight = float(contrib.get("weight", 0.0))
                if weight <= 0.0:
                    weight = 1.0
                weighted_sum += float(contrib["value"]) * weight
                total_weight += weight
            target = weighted_sum / total_weight if total_weight > 0.0 else self._frame_contribs[-1]["value"]
        elif sum_mode == "sum_clamp":
            target = self.base_value
            for contrib in self._frame_contribs:
                target += float(contrib["value"]) - self.base_value
        else:
            target = float(self._frame_contribs[-1]["value"])
        return float(max(self.min_value, min(self.max_value, target)))

    def commit(self, beat: float, sum_mode: str) -> Dict[str, Any]:
        final_value = self._combine_frame(sum_mode)
        self.value = final_value
        delta = final_value - self.base_value
        entry = {
            "beat": float(beat),
            "value": final_value,
            "delta": delta,
            "abs_delta": abs(delta),
            "contributions": [dict(contrib) for contrib in self._frame_contribs],
        }
        self.timeline.append(entry)
        return entry

    # ---------------------------------------------------------------- summary --
    def summary(self) -> Dict[str, Any] | None:
        if not self.timeline:
            return None
        count = len(self.timeline)
        avg_value = sum(entry["value"] for entry in self.timeline) / count
        avg_delta = sum(entry["delta"] for entry in self.timeline) / count
        avg_abs_delta = sum(entry["abs_delta"] for entry in self.timeline) / count
        min_value = min(entry["value"] for entry in self.timeline)
        max_value = max(entry["value"] for entry in self.timeline)
        final_value = self.timeline[-1]["value"]

        source_stats: Dict[str, Dict[str, float]] = {}
        for entry in self.timeline:
            for contrib in entry["contributions"]:
                src = str(contrib.get("source"))
                stats = source_stats.setdefault(src, {"count": 0.0, "value_sum": 0.0, "weight_sum": 0.0})
                stats["count"] += 1.0
                stats["value_sum"] += float(contrib.get("value", 0.0))
                stats["weight_sum"] += float(contrib.get("weight", 0.0))
        sources = []
        for src, stats in sorted(source_stats.items()):
            count_src = max(1.0, stats["count"])
            sources.append(
                {
                    "name": src,
                    "average_value": stats["value_sum"] / count_src,
                    "average_weight": stats["weight_sum"] / count_src,
                    "count": int(stats["count"]),
                }
            )
        return {
            "destination": self.name,
            "track": self.track,
            "parameter": self.parameter,
            "base_value": self.base_value,
            "final_value": final_value,
            "average_value": avg_value,
            "average_delta": avg_delta,
            "average_abs_delta": avg_abs_delta,
            "min_value": min_value,
            "max_value": max_value,
            "frames": self.timeline,
            "sources": sources,
        }


class ArrangementEngine:
    def __init__(
        self,
        protocols_path: str = "DROP_PROTOCOLS.yaml",
        protocols: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.protocols_path = protocols_path
        if protocols is not None:
            self.protocols = dict(protocols)
        else:
            try:
                with open(self.protocols_path, "r", encoding="utf-8") as fh:
                    self.protocols = _load_yaml_like(fh.read()) or {}
            except FileNotFoundError:
                self.protocols = {}
        if not isinstance(self.protocols, dict):
            self.protocols = {}

    # ----------------------------------------------------------------- access --
    def get_protocol(self, name: str) -> Dict[str, Any]:
        proto = self.protocols.get(name, {})
        return proto if isinstance(proto, dict) else {}

    def _collect_beats(self) -> List[float]:
        beats = {0.0}
        for proto in self.protocols.values():
            if not isinstance(proto, dict):
                continue
            curve = proto.get("curve", [])
            if not isinstance(curve, list):
                continue
            for entry in curve:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    continue
                try:
                    beat = float(entry[0])
                except (TypeError, ValueError):
                    continue
                beats.add(beat)
        return sorted(beats)

    # ----------------------------------------------------------- destinations --
    def prepare_destination_context(
        self,
        mixer: Any,
        routes: Iterable[Dict[str, Any]],
    ) -> Dict[str, DestinationSlot]:
        destinations: Dict[str, DestinationSlot] = {}
        tracks = getattr(mixer, "tracks", {})
        if not isinstance(tracks, dict):
            return destinations

        for spec in routes:
            dest_name = spec.get("destination") if isinstance(spec, dict) else None
            if not isinstance(dest_name, str) or "." not in dest_name:
                continue
            track_name, parameter = dest_name.split(".", 1)
            track = tracks.get(track_name)
            if track is None or not hasattr(track, parameter):
                continue
            base_value = getattr(track, parameter)
            if base_value is None:
                continue
            rng = spec.get("range") if isinstance(spec, dict) else None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                try:
                    lo = float(rng[0])
                    hi = float(rng[1])
                except (TypeError, ValueError):
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = 0.0, 1.0
            slot = destinations.get(dest_name)
            if slot is None:
                slot = DestinationSlot(
                    name=dest_name,
                    track=track_name,
                    parameter=parameter,
                    base_value=float(base_value),
                    min_value=lo,
                    max_value=hi,
                )
                destinations[dest_name] = slot
            else:
                slot.min_value = min(slot.min_value, lo)
                slot.max_value = max(slot.max_value, hi)
        return destinations

    # --------------------------------------------------------------- apply mm --
    def apply_modulations(
        self,
        matrix: Any,
        destinations: Dict[str, DestinationSlot],
        beats: Iterable[float] | None = None,
    ) -> List[Dict[str, Any]]:
        if not destinations:
            return []
        beat_sequence = list(beats) if beats is not None else self._collect_beats()
        if not beat_sequence:
            beat_sequence = [0.0]
        ctx = SimpleNamespace(arranger=self, destinations=destinations, beat=0.0)
        snapshots: List[Dict[str, Any]] = []
        for beat in beat_sequence:
            ctx.beat = float(beat)
            matrix.apply(ctx)
            snapshot = {
                name: slot.timeline[-1]
                for name, slot in destinations.items()
                if slot.timeline
            }
            snapshots.append({"beat": float(beat), "destinations": snapshot})
        return snapshots

    def summarise_modulations(self, destinations: Dict[str, DestinationSlot]) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for slot in destinations.values():
            summary = slot.summary()
            if summary:
                summaries.append(summary)
        return summaries

    def apply_final_values_to_mixer(self, mixer: Any, destinations: Dict[str, DestinationSlot]) -> None:
        tracks = getattr(mixer, "tracks", {})
        if not isinstance(tracks, dict):
            return
        for slot in destinations.values():
            if not slot.timeline:
                continue
            track = tracks.get(slot.track)
            if track is None or not hasattr(track, slot.parameter):
                continue
            setattr(track, slot.parameter, float(slot.timeline[-1]["value"]))


__all__ = ["ArrangementEngine", "DestinationSlot"]
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

