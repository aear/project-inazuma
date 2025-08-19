"""Wave Instinct engine.

This module implements a minimal skeleton of the "Wave Instinct" design
outlined in the project notes.  It treats internal imbalance signals as
oscillatory waves and derives a global pressure value that can be used to
modulate generation parameters.

The implementation is intentionally lightweight and focuses on structure
rather than model quality.  It does not depend on PyTorch at runtime but
uses it if available.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Deque

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

try:  # pragma: no cover - numpy is optional but handy
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _corrcoef(x, y) -> float:
    """Return absolute correlation between two sequences."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if torch is not None:
        tx = torch.tensor(list(x), dtype=torch.float32)
        ty = torch.tensor(list(y), dtype=torch.float32)
        if tx.std() == 0 or ty.std() == 0:
            return 0.0
        c = torch.corrcoef(torch.stack([tx, ty]))[0, 1]
        return float(abs(c))
    if np is not None:
        nx = np.asarray(x, dtype=float)
        ny = np.asarray(y, dtype=float)
        if nx.std() == 0 or ny.std() == 0:
            return 0.0
        return float(abs(np.corrcoef(nx, ny)[0, 1]))
    # fallback: manually compute correlation
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return abs(num / (den_x * den_y))


@dataclass
class Oscillator:
    """State for a single imbalance signal."""

    A: float = 0.0      # amplitude
    phi: float = 0.0    # phase
    omega: float = 0.1  # frequency
    mu: float = 0.0     # running mean
    sigma: float = 1.0  # running scale
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    osc_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))


class WaveInstinct:
    """Tiny adaptive oscillator network that yields control knobs."""

    def __init__(self, weights: Dict[str, float] | None = None, window: int = 20, alpha: float = 0.1):
        self.weights = weights or {
            "pred_error": 1.0,
            "grad_norm": 0.5,
            "surprise_kl": 0.8,
            "entropy": 0.3,
            "conflict_score": 1.2,
            "affect": 0.4,
            "novelty": 0.6,
            "energy_budget": 0.1,
        }
        self.window = window
        self.alpha = alpha
        self.oscillators: Dict[str, Oscillator] = {}

    def _update_signal(self, name: str, value: float) -> Tuple[float, float]:
        osc = self.oscillators.setdefault(name, Oscillator())
        # Update stats
        osc.mu = (1 - self.alpha) * osc.mu + self.alpha * value
        osc.sigma = (1 - self.alpha) * osc.sigma + self.alpha * abs(value - osc.mu)

        # Oscillator update (very lightweight PLL-ish rule)
        osc.phi += osc.omega
        predicted = osc.A * math.sin(osc.phi)
        osc.A = (1 - self.alpha) * osc.A + self.alpha * abs(value)

        # History buffers for resonance calculation
        osc.history.append(value)
        osc.osc_history.append(predicted)

        r = _corrcoef(osc.history, osc.osc_history)
        z = 0.0
        if osc.sigma > 1e-6:
            z = (value - osc.mu) / osc.sigma
        return z, r

    # ------------------------------------------------------------------
    def step(self, signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Process incoming signals and return (P, control vector)."""
        pressure = 0.0
        for name, val in signals.items():
            z, r = self._update_signal(name, val)
            weight = self.weights.get(name, 1.0)
            pressure += weight * z * r
        controls = self._policy(pressure, signals)
        return pressure, controls

    # ------------------------------------------------------------------
    def _policy(self, P: float, s: Dict[str, float]) -> Dict[str, float]:
        novelty = s.get("novelty", 0.0)
        conflict = s.get("conflict_score", 0.0)
        valence, arousal = s.get("affect", (0.0, 0.0))
        energy = max(0.0, min(1.0, s.get("energy_budget", 1.0)))
        risk_mode = s.get("risk_mode", "normal")

        controls = {
            "temp": 1.0,
            "topk": 50,
            "topp": 0.9,
            "beam_width": 1,
            "router_gain": 1.0,
            "lr_scale": 1.0,
            "dropout_scale": 1.0,
            "spec_gain": 1.0,
            "plan_horizon": 1,
            "guardrails": 1.0,
        }

        explore = P > 0.5 and novelty > 0.5
        stabilize = P < -0.5 or conflict > 0.5
        if arousal > 0.7 and valence < 0.0:
            explore = False
            stabilize = True

        if explore:
            controls["temp"] = min(1.5, 1.0 + 0.5 * energy)
            controls["topk"] = int(controls["topk"] * (1 + 0.5 * energy))
            controls["router_gain"] = 0.5
            controls["plan_horizon"] = 2
        if stabilize:
            controls["temp"] = 0.7
            controls["topk"] = max(20, int(controls["topk"] * 0.5))
            controls["router_gain"] = 2.0
            controls["plan_horizon"] = 1
            controls["guardrails"] = 1.2

        if risk_mode == "safe":
            controls["guardrails"] *= 1.5
        elif risk_mode == "bold":
            controls["guardrails"] *= 0.7

        # Energy budget scales the more expensive knobs
        controls["topk"] = int(controls["topk"] * energy)
        controls["beam_width"] = max(1, int(controls["beam_width"] * energy))
        return controls


__all__ = ["WaveInstinct", "Oscillator"]

