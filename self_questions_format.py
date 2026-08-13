"""Display-independent formatting for self-question clipboard exports."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def format_question(entry: Mapping[str, Any]) -> str:
    question = str(entry.get("question") or "unknown")
    first = str(entry.get("first_asked") or entry.get("timestamp") or "")
    updated = str(entry.get("last_updated") or first)
    count = int(entry.get("count", entry.get("times", 1)) or 1)
    lines = [question, f"First asked: {first or 'unknown'}", f"Last updated: {updated or 'unknown'}", f"Asked: {count} time(s)"]
    if entry.get("resolved_at"):
        lines.append(f"Resolved: {entry.get('resolved_at')}")
    if entry.get("resolved_reason"):
        lines.append(f"Reason: {entry.get('resolved_reason')}")
    return "\n".join(lines)


def format_questions(entries: Iterable[Mapping[str, Any]]) -> str:
    return "\n\n---\n\n".join(format_question(entry) for entry in entries)


__all__ = ["format_question", "format_questions"]
