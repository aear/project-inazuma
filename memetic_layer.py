"""Memetic Layer handling event superposition and entanglement."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random


@dataclass
class Event:
    actors: List[str]
    symbols: List[str]
    affordances: List[str]
    stakes: str
    phase: str
    amplitude: float = 1.0
    collapse_rule: str = "max"


class MemeticLayer:
    """Simple qualitative event superposition tracker."""

    def __init__(self):
        # context -> list of (Event, weight)
        self.superpositions: Dict[str, List[Tuple[Event, float]]] = {}
        # (id(event_a), id(event_b)) -> correlation strength
        self.entanglements: Dict[Tuple[int, int], float] = {}

    # --- Superposition management -------------------------------------------------
    def add_event(self, context: str, event: Event, weight: float = 1.0) -> None:
        """Insert an event into a context superposition.

        We store raw weights here; normalization happens lazily before
        qualitative operations like ``drift`` or ``collapse``. This allows the
        caller to add events with the intended relative amplitudes without
        earlier entries being rescaled on every insertion.
        """
        self.superpositions.setdefault(context, []).append((event, weight))

    def _normalize(self, context: str) -> None:
        events = self.superpositions.get(context, [])
        total = sum(w for _, w in events)
        if total <= 0:
            return
        self.superpositions[context] = [(e, w / total) for e, w in events]

    def drift(self, context: str, amount: float = 0.1) -> None:
        """Heuristically nudge weights within a context."""
        self._normalize(context)
        events = self.superpositions.get(context, [])
        if not events:
            return
        new_weights = []
        for _, w in events:
            delta = random.uniform(-amount, amount)
            new_weights.append(max(0.0, w + delta))
        total = sum(new_weights)
        if total == 0:
            new_weights = [1.0 / len(events) for _ in events]
        else:
            new_weights = [w / total for w in new_weights]
        self.superpositions[context] = [(ev, w) for (ev, _), w in zip(events, new_weights)]

    def collapse(self, context: str) -> Optional[Event]:
        """Collapse the context to the highest-weighted event."""
        self._normalize(context)
        events = self.superpositions.get(context, [])
        if not events:
            return None
        chosen, _ = max(events, key=lambda ew: ew[1])
        self.superpositions[context] = [(chosen, 1.0)]
        return chosen

    # --- Entanglement management --------------------------------------------------
    def entangle(self, event_a: Event, event_b: Event, strength: float = 1.0) -> None:
        """Track correlation between two events."""
        key = (id(event_a), id(event_b))
        self.entanglements[key] = strength

    def get_entanglement(self, event_a: Event, event_b: Event) -> Optional[float]:
        key = (id(event_a), id(event_b))
        return self.entanglements.get(key)
