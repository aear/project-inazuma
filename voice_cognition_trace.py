"""Private, bounded diagnostics for Ina's Discord voice cognition path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from discord_runtime import discord_runtime_path


TRACE_VERSION = 1
MAX_TRACE_BYTES = 4 * 1024 * 1024
RETAIN_TRACE_BYTES = 2 * 1024 * 1024


def voice_cognition_trace_path(child: str) -> Path:
    return discord_runtime_path(
        "voice_cognition_trace_path", child=child,
        fallback_name="discord_voice_cognition.jsonl",
    )


def _bounded_payload(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<depth-bounded>"
    if isinstance(value, dict):
        return {
            str(key)[:80]: _bounded_payload(item, depth=depth + 1)
            for key, item in list(value.items())[:48]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_payload(item, depth=depth + 1) for item in value[:32]]
    if isinstance(value, str):
        return value[:512]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:256]


def _compact_trace(path: Path) -> None:
    try:
        size = path.stat().st_size
        if size <= MAX_TRACE_BYTES:
            return
        with path.open("rb") as handle:
            handle.seek(max(0, size - RETAIN_TRACE_BYTES))
            tail = handle.read(RETAIN_TRACE_BYTES)
        newline = tail.find(b"\n")
        if newline >= 0:
            tail = tail[newline + 1:]
        temporary = path.with_suffix(path.suffix + ".compact")
        with temporary.open("wb") as handle:
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        # Diagnostics must never obstruct cognition or voice transport.
        return


def record_voice_cognition(child: str, event: str, payload: dict[str, Any]) -> bool:
    """Append one metadata-only event; raw text/audio is intentionally excluded."""
    path = voice_cognition_trace_path(child)
    row = {
        "version": TRACE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event)[:96],
        "payload": _bounded_payload(payload),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > 16 * 1024:
            return False
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
        _compact_trace(path)
        return True
    except OSError:
        return False


__all__ = ["record_voice_cognition", "voice_cognition_trace_path"]
