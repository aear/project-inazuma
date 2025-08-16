"""Mistake history tracking for Ina.

This module stores mistakes logged during runtime and provides utilities
for reviewing them. The information can be supplied to an alignment
module for retraining or policy adjustments.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any


@dataclass
class MistakeEntry:
    """Represents a single mistake Ina made."""
    action: str
    outcome: str
    violated_law: str
    timestamp: str


class MistakeHistory:
    """Store and review mistakes encountered by Ina."""

    def __init__(self) -> None:
        self._mistakes: List[MistakeEntry] = []

    def log_mistake(self, action: str, outcome: str, violated_law: str) -> None:
        """Record a mistake and its context.

        Args:
            action: Description of the action taken.
            outcome: Resulting outcome of the action.
            violated_law: Which safety or ethical law was violated.
        """
        entry = MistakeEntry(
            action=action,
            outcome=outcome,
            violated_law=violated_law,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._mistakes.append(entry)

    def review_mistakes(self) -> Dict[str, int]:
        """Summarise mistake patterns by violated law."""
        summary: Dict[str, int] = {}
        for m in self._mistakes:
            summary[m.violated_law] = summary.get(m.violated_law, 0) + 1
        return summary

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return mistake logs as serialisable dictionaries."""
        return [asdict(m) for m in self._mistakes]

    def expose_to_alignment(self, alignment_module: Any) -> None:
        """Provide logs to an alignment module.

        The alignment module is expected to implement
        ``receive_mistake_logs(logs: List[Dict[str, Any]])``.
        """
        if hasattr(alignment_module, "receive_mistake_logs"):
            alignment_module.receive_mistake_logs(self.get_logs())
