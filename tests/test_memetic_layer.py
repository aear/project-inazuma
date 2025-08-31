import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memetic_layer import Event, MemeticLayer


def _sample_event(name: str) -> Event:
    return Event(
        actors=[name],
        symbols=[f"{name}_symbol"],
        affordances=["act"],
        stakes="medium",
        phase="observation",
    )


def test_superposition_collapse():
    layer = MemeticLayer()
    e1 = _sample_event("A")
    e2 = _sample_event("B")
    layer.add_event("story", e1, weight=0.4)
    layer.add_event("story", e2, weight=0.6)
    chosen = layer.collapse("story")
    assert chosen is e2
    ctx = layer.superpositions["story"]
    assert len(ctx) == 1 and ctx[0][0] is e2 and pytest.approx(ctx[0][1], rel=1e-6) == 1.0


def test_entanglement_tracking():
    layer = MemeticLayer()
    e1 = _sample_event("A")
    e2 = _sample_event("B")
    layer.entangle(e1, e2, 0.7)
    assert layer.get_entanglement(e1, e2) == pytest.approx(0.7)


def test_drift_normalizes():
    layer = MemeticLayer()
    e1 = _sample_event("A")
    e2 = _sample_event("B")
    layer.add_event("story", e1, weight=0.5)
    layer.add_event("story", e2, weight=0.5)
    layer.drift("story", amount=0.3)
    weights = [w for _, w in layer.superpositions["story"]]
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0
    assert all(w >= 0 for w in weights)
