"""Storage layout helpers for durable HDD + fast runtime devices."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config_layers import load_config as load_layered_config


def load_config(path: Path = Path("config.json")) -> Dict[str, Any]:
    if Path(path) == Path("config.json"):
        return load_layered_config()
    from io_utils import load_json_dict
    return load_json_dict(Path(path))


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


def _artifact_class(subdir: Optional[str], root_keys: Iterable[str]) -> str:
    labels = {str(item).lower() for item in root_keys}
    sub = str(subdir or "").lower()
    if sub == "index" or "fast_index_root" in labels:
        return "index"
    if sub == "neural" or "fast_neural_root" in labels:
        return "neural"
    return "runtime"


def fast_runtime_path(
    child: str,
    filename: str,
    fallback: Path,
    *,
    subdir: Optional[str] = None,
    root_keys: Iterable[str] = ("fast_runtime_root", "fast_root"),
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Choose a device for rebuildable runtime data; durable memory is excluded."""

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

    keys = tuple(root_keys)
    fast_candidate = None
    for key in keys:
        root = format_child_path(layout.get(key), child)
        if root is None:
            continue
        if key == "fast_root":
            root = root / "AI_Children" / child / "memory" / "fast_runtime"
        if subdir and key in {"fast_runtime_root", "fast_root"}:
            root = root / subdir
        if root_is_writable(root):
            fast_candidate = root / filename
            break

    if fast_candidate is None:
        return fallback

    try:
        from adaptive_storage import recommend_rebuildable_tier
        tier = recommend_rebuildable_tier(
            child,
            _artifact_class(subdir, keys),
            cfg,
            fast_available=True,
            durable_available=root_is_writable(Path(fallback)),
        )
    except Exception:
        tier = "fast"
    return fallback if tier == "durable" else fast_candidate
