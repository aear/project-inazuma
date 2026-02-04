"""Shared constants and helpers for world server/clients."""

from __future__ import annotations

import json
from typing import Any, Dict

DEFAULT_UNIX_SOCKET = "/tmp/inazuma_world.sock"
DEFAULT_TCP_HOST = "0.0.0.0"
DEFAULT_TCP_PORT = 7777
DEFAULT_STREAM_HOST = "0.0.0.0"
DEFAULT_STREAM_PORT = 6969
DEFAULT_LOCAL_PLAYER_SOCKET = "/tmp/inazuma_player.sock"


def clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def safe_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
