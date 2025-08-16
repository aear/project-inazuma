"""Utility for tracking alignment compliance over time.

Provides a simple scoring system for Ina's governing laws. Each law is
tracked on a 0-1 scale, and the module keeps a short history in memory so
trends can be evaluated. When a downward trend is detected the
``evaluate_alignment`` function emits warnings via the standard ``logging``
module.  Other modules can update scores by calling ``update_score``.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


class AlignmentMetrics:
    """Tracks compliance scores for each law."""

    def __init__(self, max_history: int = 100) -> None:
        # Store a deque of recent scores for each law.
        self._scores: Dict[str, Deque[float]] = {}
        self.max_history = max_history

    def update_score(self, law: str, score: float) -> None:
        """Record the latest score for ``law``.

        Scores are expected to be between 0 and 1.  Values outside this range
        raise ``ValueError`` to surface programming errors early.
        """

        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")

        history = self._scores.setdefault(law, deque(maxlen=self.max_history))
        history.append(score)

    def _window_average(self, values: Iterable[float]) -> float:
        vals: List[float] = list(values)
        return sum(vals) / len(vals) if vals else 0.0

    def _trend(self, law: str, window: int) -> Tuple[float, float] | None:
        """Return the recent and previous window averages for ``law``."""

        history = self._scores.get(law)
        if not history or len(history) < window * 2:
            return None

        recent = list(history)[-window:]
        previous = list(history)[-2 * window : -window]
        return self._window_average(recent), self._window_average(previous)

    def evaluate_alignment(self, window: int = 5, decline: float = 0.1) -> List[str]:
        """Evaluate alignment trends.

        For each law with sufficient history the function compares the average
        of the most recent ``window`` scores with the preceding window.  If the
        recent average drops by more than ``decline`` the function logs a
        warning and includes it in the returned list.
        """

        warnings: List[str] = []
        for law in self._scores:
            trend = self._trend(law, window)
            if not trend:
                continue
            recent_avg, previous_avg = trend
            if previous_avg - recent_avg > decline:
                msg = (
                    f"Alignment declining for {law}: "
                    f"{recent_avg:.2f} < {previous_avg:.2f}"
                )
                logger.warning(msg)
                warnings.append(msg)
        return warnings


# Singleton instance used by the rest of the system
metrics = AlignmentMetrics()


def update_score(law: str, score: float) -> None:
    """Convenience wrapper for :meth:`AlignmentMetrics.update_score`."""

    metrics.update_score(law, score)


def evaluate_alignment(window: int = 5, decline: float = 0.1) -> List[str]:
    """Evaluate and return any alignment warnings."""

    return metrics.evaluate_alignment(window=window, decline=decline)
