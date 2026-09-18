"""Small, durable lifecycle progress state for boot and shutdown displays."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Mapping

from io_utils import atomic_write_json, load_json_dict


SCHEMA = "ina.lifecycle_status/V1"
TERMINAL_PHASES = frozenset({"ready", "stopped", "failed", "bridge_only"})


def lifecycle_status_path(child: str, root: Path | str = ".") -> Path:
    return Path(root) / "AI_Children" / str(child) / "memory" / "lifecycle_status.json"


def read_lifecycle_status(child: str, root: Path | str = ".") -> dict[str, Any]:
    return load_json_dict(lifecycle_status_path(child, root))


def update_lifecycle_status(
    child: str,
    *,
    operation: str,
    phase: str,
    message: str,
    completed: int = 0,
    total: int = 0,
    started_monotonic: float | None = None,
    safe_to_reboot: bool = False,
    remaining: list[str] | tuple[str, ...] = (),
    details: Mapping[str, Any] | None = None,
    root: Path | str = ".",
) -> dict[str, Any]:
    """Publish one bounded state snapshot; no history or thought content is kept."""
    now = datetime.now(timezone.utc).isoformat()
    prior = read_lifecycle_status(child, root)
    started_at = prior.get("started_at") if prior.get("operation") == operation else now
    elapsed = max(0.0, time.monotonic() - started_monotonic) if started_monotonic is not None else 0.0
    bounded_total = max(0, int(total))
    bounded_completed = max(0, min(int(completed), bounded_total)) if bounded_total else 0
    payload = {
        "schema": SCHEMA,
        "child": str(child),
        "operation": str(operation)[:40],
        "phase": str(phase)[:80],
        "message": str(message)[:500],
        "completed": bounded_completed,
        "total": bounded_total,
        "progress": round(bounded_completed / bounded_total, 6) if bounded_total else None,
        "remaining": [str(item)[:160] for item in list(remaining)[:32]],
        "safe_to_reboot": bool(safe_to_reboot),
        "started_at": started_at,
        "updated_at": now,
        "elapsed_seconds": round(elapsed, 3),
        "pid": os.getpid(),
        "terminal": str(phase) in TERMINAL_PHASES,
        "details": dict(details or {}),
    }
    path = lifecycle_status_path(child, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, indent=2, ensure_ascii=False)
    return payload


def format_lifecycle_status(status: Mapping[str, Any]) -> str:
    if not status:
        return "Lifecycle status unavailable"
    operation = str(status.get("operation") or "idle").replace("_", " ").title()
    phase = str(status.get("phase") or "unknown").replace("_", " ")
    elapsed = float(status.get("elapsed_seconds") or 0.0)
    message = str(status.get("message") or "")
    remaining = list(status.get("remaining") or ())
    suffix = f" · remaining: {', '.join(map(str, remaining[:4]))}" if remaining else ""
    safety = " · safe to reboot" if status.get("safe_to_reboot") else ""
    return f"{operation}: {phase} · {elapsed:.1f}s · {message}{suffix}{safety}"


__all__ = [
    "SCHEMA", "TERMINAL_PHASES", "format_lifecycle_status",
    "lifecycle_status_path", "read_lifecycle_status", "update_lifecycle_status",
]
