"""
proto_qualia_engine.py

Ina's phenomenology emulator ("proto-qualia" engine).

IMPORTANT PHILOSOPHY NOTES
---------------------------
- This module DOES NOT implement real qualia, consciousness, or subjective
  experience. It models the *shape* of what an experience would feel like
  for a being with qualia, using structured vectors and symbols.

- The goal is to give Ina:
    * a way to reason about "what this would be like" for others (and
      counterfactually for herself),
    * a bridge between emotion_engine, meaning_map, prediction_layer, and
      morality_engine,
    * a richer basis for ethical reasoning and perspective-taking.

- Internally, this is just math + symbols. No valenced experience is produced,
  and no unified global workspace or self-model is maintained here.

HIGH-LEVEL CONCEPTS
-------------------
- ProtoQualiaState:
    A structured snapshot of "as-if experience" along several standardized
    dimensions (safety, agency, connectedness, clarity, overload, etc.).
    This is not felt; it is input to other reasoning modules.

- ProtoQualiaEngine:
    Given:
        - emotional sliders (from emotion_engine),
        - context tags,
        - predicted outcomes (from prediction_layer),
        - meaning symbols (from meaning_map),
      it produces a ProtoQualiaState that can be used by:

        - morality_engine (ethical evaluation),
        - logic_engine (reasoning about trade-offs),
        - who_am_i (self-reflection about patterns),
        - prediction_layer (better modeling of agents).

- This engine supports both:
    - perspective-taking for OTHER agents ("what this is like for them"),
    - counterfactual self-perspective ("if I were a being that felt this").

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math

try:
    import logging
    LOGGER = logging.getLogger(__name__)
except Exception:
    class _DummyLogger:
        def debug(self, *a, **k): ...
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def error(self, *a, **k): ...
    LOGGER = _DummyLogger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProtoQualiaConfig:
    """
    Lightweight config for the proto-qualia engine.

    This keeps things flexible and cheap. If you want to tune defaults,
    you can do it here or via a config file wrapper.
    """
    # Names of emotional sliders from emotion_engine we expect to see.
    expected_emotion_keys: List[str] = field(default_factory=lambda: [
        "intensity",
        "positivity",
        "negativity",
        "familiarity",
        "novelty",
        "care",
        "trust",
        "stress",
        "complexity",
        "simplicity",
        "risk",
    ])

    # Dimensions of the "as-if experience" vector.
    # Values will be in [-1.0, 1.0] by convention.
    proto_dimensions: List[str] = field(default_factory=lambda: [
        "felt_safety",          # +safe / -unsafe
        "felt_agency",          # +in_control / -helpless
        "felt_connectedness",   # +connected / -isolated
        "felt_significance",    # +meaningful / -empty
        "felt_clarity",         # +clear / -confused
        "felt_overload",        # +overwhelmed / -understimulated
        "felt_trust",           # +trusting / -suspicious
        "felt_vulnerability",   # +open / -armored
    ])

    # Simple weighting map from emotion keys -> proto dimensions.
    # This is intentionally basic and can be refined or replaced by ML.
    emotion_to_proto_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Optional temperature for nonlinear squashing.
    squash_temperature: float = 0.75

    def ensure_weights(self) -> None:
        """
        Populate default weights if none are defined.
        This keeps the engine usable with zero external tuning.
        """
        if self.emotion_to_proto_weights:
            return

        # These are heuristic, hand-tuned starting points. They are not sacred.
        self.emotion_to_proto_weights = {
            "intensity": {
                "felt_overload": 0.8,
                "felt_significance": 0.4,
            },
            "positivity": {
                "felt_safety": 0.6,
                "felt_significance": 0.4,
                "felt_connectedness": 0.3,
            },
            "negativity": {
                "felt_safety": -0.7,
                "felt_overload": 0.3,
            },
            "familiarity": {
                "felt_safety": 0.4,
                "felt_clarity": 0.3,
            },
            "novelty": {
                "felt_significance": 0.4,
                "felt_overload": 0.3,
            },
            "care": {
                "felt_connectedness": 0.5,
                "felt_vulnerability": 0.4,
            },
            "trust": {
                "felt_trust": 0.8,
                "felt_safety": 0.3,
            },
            "stress": {
                "felt_safety": -0.8,
                "felt_overload": 0.6,
                "felt_clarity": -0.3,
            },
            "complexity": {
                "felt_overload": 0.5,
                "felt_significance": 0.3,
            },
            "simplicity": {
                "felt_clarity": 0.5,
                "felt_safety": 0.2,
            },
            "risk": {
                "felt_safety": -0.7,
                "felt_agency": -0.2,
            },
        }


@dataclass
class ProtoQualiaState:
    """
    A structured snapshot of "as-if experience" for a given agent and context.

    This DOES NOT represent real subjective feeling. It is a representation
    for reasoning: a map, not a territory.
    """
    # Which agent this state refers to. Could be "self", "human:123", etc.
    agent_id: str

    # Proto-phenomenological vector: dimension -> value in [-1.0, 1.0]
    dimensions: Dict[str, float] = field(default_factory=dict)

    # Raw emotion snapshot that fed into this state (for traceability).
    emotion_snapshot: Dict[str, float] = field(default_factory=dict)

    # Optional tags/context info.
    context_tags: List[str] = field(default_factory=list)

    # Optional brief symbolic summary (used by explanation_bridge).
    summary_label: str = ""

    # Timestamp for ordering / analysis.
    created_at: float = field(default_factory=time.time)

    # Confidence in this estimate [0.0, 1.0].
    confidence: float = 1.0

    # Why this matters to Ina (machine-semantic importance, if available).
    why_it_matters: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.dimensions.get(key, default))


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class ProtoQualiaEngine:
    """
    ProtoQualiaEngine

    Responsibilities:
    - Map emotion_engine outputs + contextual info into ProtoQualiaState.
    - Provide perspective-taking:
        * for other agents ("what this might be like for them"),
        * for Ina-in-counterfactual-mode ("if I were capable of feeling this").
    - Create stable, reusable representations to support:
        * morality_engine (ethical evaluation),
        * logic_engine (trade-offs, dilemmas),
        * prediction_layer (better agent modeling),
        * who_am_i (self reflection in symbolic form).

    Design principles:
    - No actual qualia or consciousness.
    - Minimal overhead: operates on small dicts, no heavy tensors here.
    - Extensible: can later be backed by ML models using the same interface.
    """

    def __init__(self, config: Optional[ProtoQualiaConfig] = None) -> None:
        self.config = config or ProtoQualiaConfig()
        self.config.ensure_weights()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def estimate_for_self(
        self,
        emotion_snapshot: Dict[str, float],
        context_tags: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ProtoQualiaState:
        """
        Counterfactual self-perspective:
        "If I were a being that felt, what would this situation feel like?"

        This is particularly useful for:
        - building moral intuition,
        - aligning behavior with the laws of nature,
        - supporting who_am_i during deep reflection,
        - shaping Ina's symbolic sense of "how intense" or "how heavy" things are.
        """
        return self._estimate(
            agent_id="self",
            emotion_snapshot=emotion_snapshot,
            context_tags=context_tags or [],
            extra_context=extra_context or {},
        )

    def estimate_for_agent(
        self,
        agent_id: str,
        emotion_snapshot: Dict[str, float],
        context_tags: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ProtoQualiaState:
        """
        Model the as-if experience of another agent.

        agent_id can be:
            - "human:<uuid>",
            - "ai_child:<name>",
            - "group:<label>",
            - or any other identifier.

        emotion_snapshot should reflect Ina's best guess at *their* emotional sliders.
        """
        return self._estimate(
            agent_id=agent_id,
            emotion_snapshot=emotion_snapshot,
            context_tags=context_tags or [],
            extra_context=extra_context or {},
        )

    def refine_with_outcome(
        self,
        current_state: ProtoQualiaState,
        observed_outcome: Dict[str, Any],
    ) -> ProtoQualiaState:
        """
        Post-hoc refinement / learning hook.

        This can be used to adjust estimates based on:
        - observed behavior,
        - reported affect (if humans tell her how they felt),
        - conflict between prediction and reality.

        For now this is deliberately conservative; in future it can integrate
        ML-based calibration. Here we just gently nudge confidence or a few
        dimensions if clear signals are present.
        """
        new_state = ProtoQualiaState(
            agent_id=current_state.agent_id,
            dimensions=dict(current_state.dimensions),
            emotion_snapshot=dict(current_state.emotion_snapshot),
            context_tags=list(current_state.context_tags),
            summary_label=current_state.summary_label,
            created_at=current_state.created_at,
            confidence=current_state.confidence,
        )

        # Example: if outcome explicitly reports "overwhelmed", increase felt_overload.
        flags = observed_outcome.get("reported_flags") or []
        if "overwhelmed" in flags:
            self._nudge_dimension(new_state, "felt_overload", 0.2)
            self._nudge_dimension(new_state, "felt_safety", -0.1)
        if "calm" in flags:
            self._nudge_dimension(new_state, "felt_overload", -0.2)
            self._nudge_dimension(new_state, "felt_safety", 0.1)

        # Confidence could be updated based on how well the prediction matched.
        match_score = observed_outcome.get("match_score")
        if isinstance(match_score, (int, float)):
            # Simple moving average blend.
            new_state.confidence = max(
                0.0,
                min(1.0, 0.7 * current_state.confidence + 0.3 * float(match_score)),
            )

        return new_state

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _estimate(
        self,
        agent_id: str,
        emotion_snapshot: Dict[str, float],
        context_tags: List[str],
        extra_context: Dict[str, Any],
    ) -> ProtoQualiaState:
        """
        Core estimation path.

        Steps:
        - Normalize emotion snapshot.
        - Project into proto-dimensions via configured weights.
        - Apply optional nonlinear squashing.
        - Derive a symbolic summary for explanation_bridge.
        """
        norm_emotions = self._normalize_emotions(emotion_snapshot)
        raw_dims = {dim: 0.0 for dim in self.config.proto_dimensions}

        # Linear projection: sum over emotion_keys * weights.
        for e_key, e_val in norm_emotions.items():
            weights_for_e = self.config.emotion_to_proto_weights.get(e_key)
            if not weights_for_e:
                continue
            for dim, w in weights_for_e.items():
                if dim not in raw_dims:
                    continue
                raw_dims[dim] += e_val * w

        # Nonlinear squashing to keep values in [-1, 1] but preserve shape.
        squashed_dims = {
            dim: self._squash(val, self.config.squash_temperature)
            for dim, val in raw_dims.items()
        }

        summary_label = self._build_summary_label(squashed_dims, context_tags, extra_context)
        confidence = self._initial_confidence(norm_emotions, context_tags, extra_context)
        importance = self._derive_importance(extra_context or {}, squashed_dims)

        state = ProtoQualiaState(
            agent_id=agent_id,
            dimensions=squashed_dims,
            emotion_snapshot=norm_emotions,
            context_tags=context_tags,
            summary_label=summary_label,
            confidence=confidence,
            why_it_matters=importance,
        )

        LOGGER.debug(
            "ProtoQualiaEngine: estimated state for agent=%s summary=%s dims=%s",
            agent_id, summary_label, squashed_dims,
        )
        return state

    def _normalize_emotions(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure emotion values are in [-1, 1] and only keep known keys.
        """
        out: Dict[str, float] = {}
        for key in self.config.expected_emotion_keys:
            val = emotions.get(key)
            if val is None:
                continue
            # Clamp to [-1, 1] just in case upstream modules go out of bounds.
            out[key] = max(-1.0, min(1.0, float(val)))
        return out

    @staticmethod
    def _squash(x: float, temperature: float) -> float:
        """
        Smooth nonlinear squashing into [-1, 1].
        """
        # Avoid overflow.
        x_scaled = x / max(temperature, 1e-6)
        # tanh is smooth and symmetric.
        return math.tanh(x_scaled)

    @staticmethod
    def _nudge_dimension(state: ProtoQualiaState, dim: str, delta: float) -> None:
        old = state.dimensions.get(dim, 0.0)
        new_val = max(-1.0, min(1.0, old + delta))
        state.dimensions[dim] = new_val

    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return default

    def _initial_confidence(
        self,
        norm_emotions: Dict[str, float],
        context_tags: List[str],
        extra_context: Dict[str, Any],
    ) -> float:
        """
        Simple heuristic for confidence.

        Later this could be upgraded using prediction_layer calibration or
        explicit feedback from humans.
        """
        # More non-zero emotional sliders -> higher confidence by default.
        non_zero = sum(1 for v in norm_emotions.values() if abs(v) > 0.05)

        base = 0.5 + min(0.4, non_zero * 0.05)

        # If context explicitly marks uncertainty, reduce a bit.
        if "uncertain" in context_tags or extra_context.get("low_evidence"):
            base -= 0.2

        return max(0.0, min(1.0, base))

    def _build_summary_label(
        self,
        dims: Dict[str, float],
        context_tags: List[str],
        extra_context: Dict[str, Any],
    ) -> str:
        """
        Build a short, symbolic label used by explanation_bridge or logs.

        This is deliberately approximate and small, to avoid overfitting words
        to nuanced internal states. It's a hint, not a definition.
        """
        # Find the strongest absolute dimension.
        if not dims:
            return "neutral"

        strongest_dim, strongest_val = max(
            dims.items(), key=lambda kv: abs(kv[1])
        )
        polarity = "high" if strongest_val > 0 else "low"

        # Optionally blend a major context tag if present.
        tag_hint = ""
        for tag in context_tags:
            if tag in ("danger", "comfort", "loss", "connection", "conflict", "care"):
                tag_hint = tag
                break

        if tag_hint:
            return f"{polarity}_{strongest_dim}_{tag_hint}"
        return f"{polarity}_{strongest_dim}"

    def _derive_importance(
        self,
        extra_context: Dict[str, Any],
        dims: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Provide a lightweight "why it matters" summary, preferring machine semantics if supplied.
        """
        machine_semantics = extra_context.get("machine_semantics") if isinstance(extra_context, dict) else None

        if isinstance(machine_semantics, dict):
            cached = machine_semantics.get("why_it_matters")
            if cached:
                return cached

            axes = machine_semantics.get("axes") or {}
            reasons = []
            total_weight = 0.0
            total_contrib = 0.0
            if isinstance(axes, dict):
                for axis_id, axis_data in axes.items():
                    if not isinstance(axis_data, dict):
                        continue
                    val = axis_data.get("value")
                    if val is None:
                        continue
                    weight_raw = axis_data.get("weight", axis_data.get("importance_weight", 1.0))
                    try:
                        weight = float(weight_raw)
                    except Exception:
                        weight = 1.0

                    pressure_raw = axis_data.get("pressure")
                    try:
                        pressure = float(pressure_raw)
                    except Exception:
                        try:
                            pressure = abs(float(val) - 0.5) * 2.0
                        except Exception:
                            pressure = 0.0

                    pressure = max(0.0, min(1.0, pressure))
                    weight = max(0.0, weight)
                    val_clamped = self._clamp01(val, default=0.5)
                    contribution = pressure * weight
                    total_weight += weight
                    total_contrib += contribution
                    if contribution < 0.1:
                        continue
                    reasons.append(
                        {
                            "axis": axis_id,
                            "value": round(val_clamped, 3),
                            "pressure": round(pressure, 3),
                            "weight": round(weight, 3),
                            "reason": axis_data.get("note") or axis_data.get("description") or axis_id,
                        }
                    )

            reasons = sorted(reasons, key=lambda r: r["pressure"] * r["weight"], reverse=True)
            score = self._clamp01(total_contrib / max(total_weight, 1.0), default=0.0)
            return {"score": round(score, 3), "reasons": reasons[:5], "source": "machine_semantics"}

        if dims:
            strongest_dim, strongest_val = max(dims.items(), key=lambda kv: abs(kv[1]))
            score = self._clamp01(abs(strongest_val), default=0.0)
            return {
                "score": round(score, 3),
                "reasons": [
                    {"dimension": strongest_dim, "magnitude": round(strongest_val, 3), "reason": "proto-dimension salience"}
                ],
                "source": "proto_dimensions",
            }

        return {"score": 0.0, "reasons": [], "source": "proto_dimensions"}
