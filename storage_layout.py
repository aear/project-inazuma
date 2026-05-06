"""Storage layout helpers for durable HDD + fast runtime devices."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_config(path: Path = Path("config.json")) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def format_child_path(raw: Any, child: str) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        text = raw.format(child=child)
    except Exception:
        text = raw.replace("{child}", child)
    return Path(text).expanduser()



def root_is_writable(path: Path) -> bool:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe.exists() and os.access(probe, os.W_OK | os.X_OK)


def storage_layout(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    raw = cfg.get("storage_layout") if isinstance(cfg, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def fast_runtime_path(
    child: str,
    filename: str,
    fallback: Path,
    *,
    subdir: Optional[str] = None,
    root_keys: Iterable[str] = ("fast_runtime_root", "fast_root"),
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Return a fast-device path for rebuildable runtime files when available."""

    cfg = config if isinstance(config, dict) else load_config()
    layout = storage_layout(cfg)
    if not layout or not bool(layout.get("fast_runtime_enabled", True)):
        return fallback

    current_child = cfg.get("current_child") if isinstance(cfg, dict) else None
    if bool(layout.get("fast_runtime_current_child_only", True)):
        if current_child and str(current_child) != str(child):
            return fallback

    mount = format_child_path(layout.get("fast_mount"), child)
    if mount is not None and not mount.is_mount():
        return fallback

    for key in root_keys:
        root = format_child_path(layout.get(key), child)
        if root is None:
            continue
        if key == "fast_root":
            root = root / "AI_Children" / child / "memory" / "fast_runtime"
        if subdir and key in {"fast_runtime_root", "fast_root"}:
            root = root / subdir
        if not root_is_writable(root):
            continue
        return root / filename

    return fallback
