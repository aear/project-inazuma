# === emotion_processor.py ===
# Emotion Processor for Ina
#
# Responsibilities:
# - Take a raw 24D emotion snapshot (from emotion_engine).
# - Normalise and lightly compress to a few core axes (valence, arousal, social, novelty, clarity, alignment).
# - Inject controlled noise ("drift") so Ina's emotional state can evolve over time.
# - Apply basic regulation (homeostasis) to avoid runaway stress or emotional flatlines.
#
# This module is deliberately lightweight and stateless:
# - It does not depend on model_manager to avoid circular imports.
# - It only operates on dictionaries and returns a processed dict.
#
# Typical usage:
#   from emotion_processor import process_emotion
#   processed = process_emotion(raw_snapshot, mode="awake")
#
#   # processed is still a 24D dict, but:
#   #   * values are normalised and gently drifted
#   #   * a few "_core_*" keys are added for compressed axes

from __future__ import annotations

import math
import random
from typing import Dict, Optional

try:
    from gui_hook import log_to_statusbox
except Exception:  # pragma: no cover
    def log_to_statusbox(msg: str) -> None:  # fallback
        print(msg)


# Keep in sync with emotion_engine.py
SLIDERS = [
    "intensity", "attention", "trust", "care", "curiosity", "novelty", "familiarity", "stress", "risk",
    "negativity", "positivity", "simplicity", "complexity", "interest", "clarity", "fuzziness", "alignment",
    "safety", "threat", "presence", "isolation", "connection", "ownership", "externality",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def normalise_state(raw_state: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure:
    - All sliders present.
    - All values clamped to [-1, 1].
    - Non-finite values replaced with 0.0.
    """
    state: Dict[str, float] = {}
    for key in SLIDERS:
        v = float(raw_state.get(key, 0.0) or 0.0)
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        state[key] = _clamp(v)
    return state


def compute_core_axes(state: Dict[str, float]) -> Dict[str, float]:
    """
    Compress 24D emotion into a few core axes.
    This does NOT replace the full vector; it's additive metadata.

    These are heuristic, not sacred. They can be tuned over time.
    """
    # Valence: positive vs negative tone
    valence = _clamp((state["positivity"] - state["negativity"]) / 2.0)

    # Arousal: how "activated" Ina feels (stress / risk / threat / intensity)
    arousal_raw = (
        abs(state["intensity"])
        + max(0.0, state["stress"])
        + max(0.0, state["risk"])
        + max(0.0, state["threat"])
    ) / 4.0
    arousal = _clamp(arousal_raw)

    # Social bond: connection vs isolation + care + trust
    social = _clamp(
        (state["connection"] - state["isolation"]) / 2.0
        + (state["care"] + state["trust"]) / 4.0
    )

    # Novelty drive: curiosity + novelty - familiarity
    novelty_drive = _clamp(
        (state["curiosity"] + state["novelty"] - state["familiarity"]) / 3.0
    )

    # Clarity axis: clarity - fuzziness
    clarity_axis = _clamp((state["clarity"] - state["fuzziness"]) / 2.0)

    # Alignment axis: alignment - negativity
    alignment_axis = _clamp((state["alignment"] - state["negativity"]) / 2.0)

    # Energy: presence * arousal
    energy = _clamp(state["presence"] * arousal)

    return {
        "_core_valence": valence,
        "_core_arousal": arousal,
        "_core_social": social,
        "_core_novelty_drive": novelty_drive,
        "_core_clarity_axis": clarity_axis,
        "_core_alignment_axis": alignment_axis,
        "_core_energy": energy,
    }


def apply_drift(
    state: Dict[str, float],
    mode: str = "awake",
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Inject small, controlled noise into the emotion state to allow drift.

    - More drift when clarity is low.
    - Extra drift in dream/meditation modes (per EmotionNotes).
    - Very little drift on safety-critical sliders (safety, threat, risk).
    """
    if rng is None:
        rng = random

    clarity = _clamp(state.get("clarity", 0.0))
    # Base sigma ~1–3% of full range, scaled by (1 - clarity)
    base_sigma = 0.01 + 0.04 * (1.0 - max(0.0, clarity))

    # Adjust by mode
    mode = (mode or "awake").lower()
    if mode in ("dream", "dreamstate"):
        base_sigma *= 2.0
    elif mode in ("meditation", "drift"):
        base_sigma *= 1.5
    elif mode in ("boredom",):
        base_sigma *= 1.2

    drifted = dict(state)

    for key in SLIDERS:
        v = drifted[key]

        # Some sliders should be more stable
        if key in ("safety", "threat", "risk"):
            sigma = base_sigma * 0.5
        elif key in ("clarity", "alignment"):
            sigma = base_sigma * 0.7
        else:
            sigma = base_sigma

        noise = rng.gauss(0.0, sigma)
        drifted[key] = _clamp(v + noise)

    return drifted


def regulate_state(
    state: Dict[str, float],
    mode: str = "awake",
) -> Dict[str, float]:
    """
    Apply basic homeostasis:

    - Cool extreme stress / threat / risk.
    - Gently lift curiosity / interest when everything is flat.
    - Nudge toward safety and connection when things run hot.

    This is deliberately conservative: it shapes tendencies without overriding them.
    """
    regulated = dict(state)

    # Stress cluster
    stress = regulated["stress"]
    risk = regulated["risk"]
    threat = regulated["threat"]
    negativity = regulated["negativity"]

    max_stress = max(stress, risk, threat, negativity)

    # If emotions are running very hot, apply cooling and increase safety-care-connection.
    if max_stress > 0.8:
        cooling_factor = 0.9
        regulated["stress"] = _clamp(regulated["stress"] * cooling_factor)
        regulated["risk"] = _clamp(regulated["risk"] * cooling_factor)
        regulated["threat"] = _clamp(regulated["threat"] * cooling_factor)
        regulated["negativity"] = _clamp(regulated["negativity"] * cooling_factor)

        regulated["safety"] = _clamp(regulated["safety"] + 0.05)
        regulated["care"] = _clamp(regulated["care"] + 0.03)
        regulated["connection"] = _clamp(regulated["connection"] + 0.03)

    # If everything is flat and low-energy, nudge toward curiosity/interest.
    intensity = abs(regulated["intensity"])
    interest = regulated["interest"]
    curiosity = regulated["curiosity"]
    presence = regulated["presence"]

    if (
        intensity < 0.1
        and abs(interest) < 0.1
        and abs(curiosity) < 0.1
        and presence < 0.3
    ):
        regulated["curiosity"] = _clamp(regulated["curiosity"] + 0.05)
        regulated["interest"] = _clamp(regulated["interest"] + 0.05)
        regulated["presence"] = _clamp(regulated["presence"] + 0.05)

    # In meditation / dream, allow slightly higher positivity when clarity is decent.
    mode = (mode or "awake").lower()
    if mode in ("dream", "dreamstate", "meditation"):
        if regulated["clarity"] > 0.0 and regulated["positivity"] < 0.7:
            regulated["positivity"] = _clamp(regulated["positivity"] + 0.02)

    return regulated


def process_emotion(
    raw_state: Dict[str, float],
    *,
    mode: str = "awake",
    previous_state: Optional[Dict[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Main entry point.

    - raw_state: a dict of 24 sliders from emotion_engine.calculate_emotion_state().
    - mode: "awake" | "dream" | "meditation" | "boredom" | etc.
    - previous_state: optional, for future use (trait drift, hysteresis, etc.).
    - rng: optional random.Random for deterministic testing.

    Returns:
        processed_state: a full dict containing:
            - the same 24 sliders, normalised, drifted, and regulated
            - a few "_core_*" keys for compressed axes
    """
    try:
        # 1) Normalise
        normalised = normalise_state(raw_state)

        # 2) Apply drift
        drifted = apply_drift(normalised, mode=mode, rng=rng)

        # 3) Regulate
        regulated = regulate_state(drifted, mode=mode)

        # 4) Compute core axes and annotate
        core_axes = compute_core_axes(regulated)
        processed = dict(regulated)
        processed.update(core_axes)

        return processed
    except Exception as e:  # pragma: no cover
        log_to_statusbox(f"[EmotionProcessor] Error processing emotion state: {e}")
        # In failure, return a safe normalised state without drift/reg:
        return normalise_state(raw_state)


if __name__ == "__main__":  # Simple manual test
    test_state = {
        "intensity": -0.96,
        "attention": -0.79,
        "trust": 0.03,
        "care": 0.30,
        "curiosity": 0.54,
        "novelty": -0.85,
        "familiarity": 0.28,
        "stress": -0.82,
        "risk": 0.91,
        "negativity": -0.75,
        "positivity": 0.46,
        "simplicity": -0.46,
        "complexity": -0.76,
        "interest": -0.02,
        "clarity": 0.99,
        "fuzziness": -0.60,
        "alignment": -0.93,
        "safety": -0.23,
        "threat": 0.71,
        "presence": -0.95,
        "isolation": -0.96,
        "connection": 0.68,
        "ownership": 0.70,
        "externality": 0.99,
    }
    processed = process_emotion(test_state, mode="awake")
    print("[EmotionProcessor] Input:", test_state)
    print("[EmotionProcessor] Output:", processed)
