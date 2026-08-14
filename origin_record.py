"""Composable, bounded origin records for cognitive module outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import islice
from typing import Any, Iterable, Mapping, Optional

_MAX_ITEMS = 32
_MAX_TEXT = 512


def _bounded(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return str(value)[:_MAX_TEXT]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, Mapping):
        return {
            str(key)[:64]: _bounded(item, depth + 1)
            for key, item in islice(value.items(), _MAX_ITEMS)
            if item not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple, set)):
        return [_bounded(item, depth + 1) for item in islice(iter(value), _MAX_ITEMS)]
    return str(value)[:_MAX_TEXT]


def make_origin(
    module: str, version: str, *, inputs: Optional[Mapping[str, Any]] = None,
    references: Iterable[Any] = (), trigger: Optional[str] = None,
    event_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Build one portable origin record with bounded, JSON-safe content."""
    record = {
        "schema": "ina.origin/V1",
        "module": str(module)[:128],
        "module_version": str(version)[:32],
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "inputs": _bounded(inputs or {}),
        "references": _bounded(list(islice(iter(references), _MAX_ITEMS))),
        "trigger": str(trigger)[:128] if trigger else None,
        "event_id": str(event_id)[:128] if event_id else None,
        "metadata": _bounded(metadata or {}),
    }
    return {key: value for key, value in record.items() if value not in (None, "", [], {})}


def normalize_origins(value: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    """Normalize current and legacy provenance payloads into origin records."""
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[-max(1, int(limit)):]:
        if not isinstance(item, Mapping):
            continue
        if item.get("schema") == "ina.origin/V1" or item.get("module"):
            row = _bounded(item)
        else:
            references = [item[key] for key in ("fragment_id", "event_id") if item.get(key)]
            row = make_origin(
                str(item.get("transformer") or item.get("source") or "unknown"), "legacy",
                inputs={key: item[key] for key in ("symbol", "logic_tag", "context") if item.get(key)},
                references=references, trigger=item.get("source"), event_id=item.get("event_id"),
                timestamp=item.get("timestamp"),
            )
        if isinstance(row, dict) and row:
            rows.append(row)
    return rows


__all__ = ["make_origin", "normalize_origins"]
