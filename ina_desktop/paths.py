from __future__ import annotations

import hashlib
from pathlib import Path


def memory_root(child: str) -> Path:
    return Path("AI_Children") / str(child) / "memory"


def workspace_root(child: str) -> Path:
    return memory_root(child) / "virtual_workspace"


def status_path(child: str) -> Path:
    return workspace_root(child) / "status.json"


def socket_path(child: str) -> Path:
    return workspace_root(child) / "control.sock"


def lock_path(child: str) -> Path:
    return workspace_root(child) / "service.lock"


def viewer_lock_path(child: str) -> Path:
    return workspace_root(child) / "viewer.lock"


def share_root(child: str) -> Path:
    return workspace_root(child) / "share"


def display_number(child: str, base: int = 100, slots: int = 50) -> int:
    digest = hashlib.sha256(str(child).encode("utf-8")).digest()
    return int(base) + (int.from_bytes(digest[:2], "big") % max(1, int(slots)))
