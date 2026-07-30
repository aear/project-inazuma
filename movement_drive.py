"""Shared movement-drive calculation for Ina's autonomous motor pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def calculate_movement_urge(
    emotions: Mapping[str, Any] | None,
    *,
    boredom: Any = 0.0,
    energy: Any = 0.5,
    sleep_pressure: Any = 0.0,
) -> Dict[str, Any]:
    """Return the canonical movement urge and its inspectable drivers."""
    snapshot: Mapping[str, Any] = emotions if isinstance(emotions, Mapping) else {}
    nested = snapshot.get("values")
    if isinstance(nested, Mapping) and nested:
        snapshot = nested

    curiosity = _unit(snapshot.get("curiosity"))
    novelty = _unit(snapshot.get("novelty"))
    intensity = _unit(snapshot.get("intensity"))
    attention = _unit(snapshot.get("attention"))
    boredom_level = _unit(boredom if boredom is not None else snapshot.get("boredom"))
    stress = _unit(snapshot.get("stress"))
    threat = _unit(snapshot.get("threat"))
    sleep = _unit(sleep_pressure)
    energy_level = _unit(energy, default=0.5)
    fatigue = 1.0 - energy_level

    drive = (
        (0.35 * curiosity)
        + (0.2 * novelty)
        + (0.2 * intensity)
        + (0.15 * attention)
        + (0.1 * boredom_level)
    )
    inhibition = min(
        0.75,
        (0.35 * stress) + (0.3 * threat) + (0.25 * sleep) + (0.2 * fatigue),
    )
    urge = _unit(drive * (1.0 - inhibition))
    return {
        "level": round(urge, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "movement_drive",
        "drivers": {
            "curiosity": round(curiosity, 3),
            "novelty": round(novelty, 3),
            "intensity": round(intensity, 3),
            "attention": round(attention, 3),
            "boredom": round(boredom_level, 3),
            "stress": round(stress, 3),
            "threat": round(threat, 3),
            "sleep_pressure": round(sleep, 3),
            "fatigue": round(fatigue, 3),
            "base_drive": round(drive, 3),
            "inhibition": round(inhibition, 3),
        },
    }
