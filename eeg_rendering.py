"""Display-independent EEG render preparation.

The Qt/OpenGL window consumes these compact float buffers, while benchmarks and
tests can exercise the complete graph path without importing GUI dependencies.
No selection or sampling happens here: every valid input node and edge is packed.
"""
from __future__ import annotations

from array import array
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple


Color = Tuple[float, float, float]
ColorResolver = Callable[[Mapping[str, Any]], Color]


RENDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "Quality": {
        "description": "soft glow, smooth synapses, world-sized neurons",
        "antialias": True,
        "edge_width": 1.35,
        "glow": True,
        "px_mode": False,
    },
    "Balanced": {
        "description": "smooth synapses with a single neuron pass",
        "antialias": True,
        "edge_width": 1.1,
        "glow": False,
        "px_mode": False,
    },
    "Throughput": {
        "description": "fixed-pixel neurons and lean, unsmoothed synapses",
        "antialias": False,
        "edge_width": 1.0,
        "glow": False,
        "px_mode": True,
    },
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pack_nodes(
    neurons: Iterable[Mapping[str, Any]],
    color_resolver: ColorResolver,
    *,
    profile: str = "Balanced",
    emotion_pulse: float = 1.0,
) -> Dict[str, Any]:
    """Pack every neuron into contiguous float32-compatible buffers."""
    settings = RENDER_PROFILES.get(profile, RENDER_PROFILES["Balanced"])
    positions = array("f")
    sizes = array("f")
    colors = array("f")
    count = 0

    for neuron in neurons:
        pos = neuron.get("pos")
        if pos is None or len(pos) < 3:
            continue
        activation = clamp(_number(neuron.get("activation")))
        base_r, base_g, base_b = color_resolver(neuron)
        tint = emotion_pulse if neuron.get("network_type") == "emotion" else 1.0
        alpha = clamp(0.35 + 0.55 * activation)
        positions.extend((_number(pos[0]), _number(pos[1]), _number(pos[2])))
        colors.extend(
            (
                clamp(base_r * (0.65 + 0.45 * activation) * tint),
                clamp(base_g * (0.65 + 0.45 * activation) * tint),
                clamp(base_b * (0.65 + 0.45 * activation) * tint),
                alpha,
            )
        )
        sizes.append(2.5 + 4.5 * activation if settings["px_mode"] else 0.5 + 1.8 * activation)
        count += 1

    return {"positions": positions, "sizes": sizes, "colors": colors, "count": count}


def pack_edges(
    edges: Iterable[Mapping[str, Any]],
    positions_by_id: Mapping[Any, Sequence[float]],
    color_resolver: ColorResolver,
) -> Dict[str, Any]:
    """Pack every edge whose two endpoints are present into GL line vertices."""
    positions = array("f")
    colors = array("f")
    count = 0

    for edge in edges:
        source = positions_by_id.get(edge.get("source"))
        target = positions_by_id.get(edge.get("target"))
        if source is None or target is None or len(source) < 3 or len(target) < 3:
            continue
        base_r, base_g, base_b = color_resolver(edge)
        alpha = 0.22 + 0.4 * clamp(_number(edge.get("weight")))
        color = (base_r, base_g, base_b, alpha)
        positions.extend(
            (
                _number(source[0]), _number(source[1]), _number(source[2]),
                _number(target[0]), _number(target[1]), _number(target[2]),
            )
        )
        colors.extend(color)
        colors.extend(color)
        count += 1

    return {"positions": positions, "colors": colors, "count": count}


def dangling_endpoint_ids(
    neurons: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
) -> Tuple[Any, ...]:
    """Return stable, unique edge endpoints that have no corresponding node."""
    node_ids = {node.get("id") for node in neurons if node.get("id") is not None}
    missing = set()
    for edge in edges:
        for field in ("source", "target"):
            endpoint = edge.get(field)
            if endpoint is not None and endpoint not in node_ids:
                missing.add(endpoint)
    return tuple(sorted(missing, key=str))


def buffer_bytes(*packed: Mapping[str, Any]) -> int:
    """Return the CPU buffer footprint for packed render data."""
    total = 0
    for group in packed:
        for value in group.values():
            if isinstance(value, array):
                total += len(value) * value.itemsize
    return total


def estimated_gl_vertices(node_count: int, edge_count: int, profile: str) -> int:
    """Comparable upload/draw workload; Quality has an extra glow node pass."""
    settings = RENDER_PROFILES.get(profile, RENDER_PROFILES["Balanced"])
    node_passes = 2 if settings["glow"] else 1
    return node_count * node_passes + edge_count * 2
