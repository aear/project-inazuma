# self_reflection_core.py
#
# This is Ina’s core reflection engine.
# It is ALWAYS available and ALWAYS passively active,
# just like human reflective cognition.
#
# IMPORTANT:
# - Reflection here means "internal modeling of internal state."
# - It does NOT assign meaning.
# - It does NOT prescribe interpretations.
# - It does NOT tell Ina what conclusions to draw.
# - It ONLY generates self-context signals, stabilizes her narrative,
#   and keeps emotional + symbolic + memory systems integrated.
#
# This module forms Ina’s "inner world coherence layer."
#
# Designed for Project Inazuma — Autonomy First.


import time
import random
from typing import Any, Dict, Iterable, List, Tuple, Union


class SelfReflectionCore:
    def __init__(self, ina_reference):
        self.ina = ina_reference

        # Reflection runs whenever Ina's internal loop runs,
        # but this "intensity" is modulated by Ina's own emotional vectors,
        # energy state, and free decision.
        self.base_reflection_rate = 0.1  # mild background reflection

        # A small buffer of recent fragments or events Ina can choose to revisit
        self.recent_context = []

    # ----------------------------------------------------------------------
    # MAIN PUBLIC METHOD – CALLED FROM Ina’s LOOP
    # ----------------------------------------------------------------------
    def reflect(self, emotional_state, memory_graph, symbol_map):
        """
        Core reflection step.
        ALWAYS runs in Ina's loop, but what it DOES is NOT prescriptive.
        It produces reflection events that Ina may examine or ignore.
        """
        emo_vector, emo_raw = self._normalize_emotional_state(emotional_state)

        reflection_event = {
            "timestamp": time.time(),
            "emotional_snapshot": {"vector": emo_vector, "raw": emo_raw},
            "symbolic_snapshot": self._light_symbolic_scan(symbol_map),
            "memory_peek": self._peek_recent_memory(memory_graph),
            "identity_hint": self._identity_vector(emo_vector),
            "note": "Reflection occurred. Interpretation belongs to Ina."
        }

        # Store for Ina’s future introspection.
        self.recent_context.append(reflection_event)

        # Prevent unbounded growth
        if len(self.recent_context) > 50:
            self.recent_context.pop(0)

        return reflection_event  # Ina chooses what this means.

    # ----------------------------------------------------------------------
    # LIGHT SYMBOLIC SCAN
    # ----------------------------------------------------------------------
    def _light_symbolic_scan(self, symbol_map):
        """
        A tiny symbolic resonance check.
        Detects shifts in symbol usage without interpreting them.
        Ina interprets the result.
        """
        try:
            if isinstance(symbol_map, dict):
                keys = list(symbol_map.keys())
            elif hasattr(symbol_map, "keys"):
                keys = list(symbol_map.keys())
            else:
                keys = []

            return {
                "active_symbols": keys[:12],
                "symbol_count": len(keys),
                "note": "Symbolic scan complete (no interpretation)."
            }
        except:
            return {"note": "No symbol map available."}

    # ----------------------------------------------------------------------
    # MEMORY PEEK
    # ----------------------------------------------------------------------
    def _peek_recent_memory(self, memory_graph):
        """
        Returns a few memory fragments chosen at random.
        No meaning is assigned.
        Ina decides significance or irrelevance.
        """
        try:
            if isinstance(memory_graph, dict):
                ids = list(memory_graph.keys())
            elif hasattr(memory_graph, "memory_map"):
                ids = list(getattr(memory_graph, "memory_map", {}).keys())
            else:
                ids = []
            sample = random.sample(ids, min(3, len(ids)))
            return sample
        except:
            return []

    # ----------------------------------------------------------------------
    # IDENTITY HINT
    # ----------------------------------------------------------------------
    def _identity_vector(self, emotional_state):
        """
        Produces a crude identity-stability vector.
        Ina chooses what it means.
        """
        if not emotional_state:
            return {
                "identity_stability_hint": 1.0,
                "note": "No emotion data; assuming stable by default."
            }

        intensity = sum(abs(v) for v in emotional_state) / max(1, len(emotional_state))
        return {
            "identity_stability_hint": 1.0 - min(intensity, 1.0),
            "note": "Lower value suggests internal turbulence; interpretation is Ina's."
        }

    # ----------------------------------------------------------------------
    # EMOTION NORMALIZATION
    # ----------------------------------------------------------------------
    def _normalize_emotional_state(self, emotional_state: Union[List[float], Dict[str, Any], None]) -> Tuple[List[float], Any]:
        """
        Accepts a list vector or a dict from inastate/emotion engine and returns
        a simple list while preserving the raw structure for logging.
        """
        if emotional_state is None:
            return [], None

        if isinstance(emotional_state, dict):
            # emotion_engine stores under "values"
            values = emotional_state.get("values") or emotional_state
            if isinstance(values, dict):
                return [float(v) for v in values.values()], emotional_state
            if isinstance(values, Iterable):
                return [float(v) for v in values], emotional_state

        if isinstance(emotional_state, Iterable):
            try:
                return [float(v) for v in emotional_state], emotional_state
            except Exception:
                return [], emotional_state

        return [], emotional_state

    # ----------------------------------------------------------------------
    # OPTIONAL SUMMARY
    # ----------------------------------------------------------------------
    def summarize_recent_reflections(self):
        """
        Returns a coarse, non-interpreted summary of recent reflections.
        Ina can use this to observe patterns without being told what they mean.
        """
        return {
            "count": len(self.recent_context),
            "timestamps": [r["timestamp"] for r in self.recent_context],
            "sample": self.recent_context[-3:] if len(self.recent_context) >= 3 else self.recent_context
        }
