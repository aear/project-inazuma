"""Explicit append-only message buses for crossing the workspace boundary."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from io_utils import file_lock
from .paths import share_root


def publish_message(child: str, message: Any, *, channel: str = "outbox") -> dict[str, Any]:
    root = share_root(child) / "messages"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("inbox.jsonl" if channel == "inbox" else "outbox.jsonl")
    payload = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
        "message": message,
    }
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"ok": True, "path": str(path), "id": payload["id"]}
