from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import atomic_write_json


def _next_month(value: datetime) -> datetime:
    year, month = value.year, value.month + 1
    if month == 13:
        year, month = year + 1, 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


@dataclass
class MonthlyCadence:
    """Persistent due-state; it never starts work by itself."""

    state_path: Path

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def last_completed(self, suite: str, model: str) -> datetime | None:
        raw = self._read().get("completed", {}).get(suite, {}).get(model)
        if not isinstance(raw, str):
            return None
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def is_due(self, suite: str, model: str, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        previous = self.last_completed(suite, model)
        return previous is None or current >= _next_month(previous)

    def mark_completed(
        self, suite: str, model: str, *, completed_at: datetime | None = None
    ) -> None:
        state = self._read()
        completed = state.setdefault("completed", {})
        suite_state = completed.setdefault(suite, {})
        moment = completed_at or datetime.now(timezone.utc)
        suite_state[model] = moment.isoformat()
        atomic_write_json(self.state_path, state)
