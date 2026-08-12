"""Resolve hot Discord bridge files without moving durable memory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from config_layers import load_config as load_layered_config
from io_utils import load_json_dict


def load_root_config(path: Path = Path("config.json")) -> Dict[str, Any]:
    if Path(path) == Path("config.json"):
        return load_layered_config()
    return load_json_dict(Path(path))


def discord_runtime_path(
    key: str,
    *,
    child: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    fallback_name: Optional[str] = None,
) -> Path:
    cfg = config if isinstance(config, dict) else load_root_config()
    child_name = str(child or cfg.get("current_child") or "Inazuma_Yagami")
    discord_cfg = cfg.get("discord") if isinstance(cfg.get("discord"), dict) else {}
    raw = discord_cfg.get(key)
    current_child = str(cfg.get("current_child") or "Inazuma_Yagami")
    # A concrete fast-runtime path belongs only to the configured child. Explicit
    # operations for another child must not leak into that child's live outbox.
    if child is not None and child_name != current_child and isinstance(raw, str) and "{child}" not in raw:
        raw = None
    if isinstance(raw, str) and raw.strip():
        try:
            return Path(raw.format(child=child_name)).expanduser()
        except Exception:
            return Path(raw.replace("{child}", child_name)).expanduser()

    filename = fallback_name or key
    return Path("AI_Children") / child_name / "memory" / filename


def typed_outbox_path(child: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Path:
    return discord_runtime_path(
        "typed_outbox_path",
        child=child,
        config=config,
        fallback_name="typed_outbox.jsonl",
    )


def typed_outbox_history_path(child: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Path:
    return discord_runtime_path(
        "typed_outbox_history_path",
        child=child,
        config=config,
        fallback_name="typed_outbox_history.jsonl",
    )


def typed_outbox_archive_path(child: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Path:
    return discord_runtime_path(
        "typed_outbox_archive_path",
        child=child,
        config=config,
        fallback_name="typed_outbox_archive.jsonl",
    )


def discord_log_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config if isinstance(config, dict) else load_root_config()
    discord_cfg = cfg.get("discord") if isinstance(cfg.get("discord"), dict) else {}
    raw = discord_cfg.get("runtime_log_dir")
    if isinstance(raw, str) and raw.strip():
        child = str(cfg.get("current_child") or "Inazuma_Yagami")
        try:
            directory = Path(raw.format(child=child)).expanduser()
        except Exception:
            directory = Path(raw.replace("{child}", child)).expanduser()
        return directory / "comms_core.jsonl"
    return Path("logs") / "comms_core.jsonl"
