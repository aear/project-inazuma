"""Dependency-free policy helpers for choosing a self-read file lane."""

from typing import Any, Dict, Mapping


SELF_READ_FOCUS_ENV = "SELF_READ_FOCUS"
VALID_SELF_READ_FOCUS = {"new", "seen", "balanced"}
SELF_READ_FOCUS_MARGIN = 0.12


def _active(values: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(values.get(name, default))))
    except (TypeError, ValueError):
        return default


def self_read_focus_from_emotions(emotions: Any) -> Dict[str, Any]:
    """
    Turn positive emotion activations into a non-binding new/seen hint.

    Emotion sliders are signed, so negative values contribute no pull.
    A small margin keeps neutral or conflicted states balanced.
    """
    values = emotions.get("values") if isinstance(emotions, dict) else {}
    if not isinstance(values, dict):
        values = emotions if isinstance(emotions, dict) else {}

    curiosity = _active(values, "curiosity")
    novelty = _active(values, "novelty")
    attention = _active(values, "attention")
    familiarity = _active(values, "familiarity")
    fuzziness = _active(values, "fuzziness", _active(values, "fuzz_level"))
    clarity = _active(values, "clarity", 0.5)

    new_score = (0.5 * curiosity) + (0.35 * novelty) + (0.15 * attention)
    seen_score = (
        (0.55 * familiarity)
        + (0.25 * fuzziness)
        + (0.2 * (1.0 - clarity))
    )
    if new_score > seen_score + SELF_READ_FOCUS_MARGIN:
        focus = "new"
    elif seen_score > new_score + SELF_READ_FOCUS_MARGIN:
        focus = "seen"
    else:
        focus = "balanced"

    return {
        "focus": focus,
        "new_score": round(new_score, 4),
        "seen_score": round(seen_score, 4),
        "drivers": {
            "curiosity": round(curiosity, 4),
            "novelty": round(novelty, 4),
            "attention": round(attention, 4),
            "familiarity": round(familiarity, 4),
            "fuzziness": round(fuzziness, 4),
            "clarity": round(clarity, 4),
        },
    }
