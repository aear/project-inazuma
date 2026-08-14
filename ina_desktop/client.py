from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from io_utils import atomic_write_json, load_json_dict
from .paths import share_root, socket_path, status_path


def workspace_status(child: str) -> dict[str, Any]:
    return load_json_dict(status_path(child))


def launch_environment(child: str, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return an app environment targeting Ina's display and private buses."""
    env = dict(os.environ if base is None else base)
    status = workspace_status(child)
    if status.get("ready"):
        display = status.get("display")
        if display:
            env["DISPLAY"] = str(display)
        audio = status.get("audio") if isinstance(status.get("audio"), dict) else {}
        if audio.get("output_sink"):
            env["PULSE_SINK"] = str(audio["output_sink"])
        if audio.get("input_source"):
            env["PULSE_SOURCE"] = str(audio["input_source"])
        env["INA_VIRTUAL_WORKSPACE"] = "1"
        env["INA_WORKSPACE_CHILD"] = str(child)
    return env


def send_command(child: str, command: Mapping[str, Any], timeout: float = 2.0) -> dict[str, Any]:
    payload = json.dumps(dict(command), separators=(",", ":")).encode("utf-8") + b"\n"
    path = socket_path(child)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(max(0.1, float(timeout)))
    try:
        client.connect(str(path))
        client.sendall(payload)
        response = b""
        while not response.endswith(b"\n") and len(response) < 1024 * 1024:
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()
    try:
        decoded = json.loads(response.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else {"ok": False, "error": "invalid response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def share_file(child: str, source: Path | str, *, channel: str = "outbox") -> dict[str, Any]:
    """Publish a deliberate copy; source files are never moved or deleted."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        return {"ok": False, "error": "source is not a file"}
    target_root = share_root(child) / ("inbox" if channel == "inbox" else "outbox") / "files"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{source_path.name}"
    shutil.copy2(source_path, target)
    manifest = target.with_suffix(target.suffix + ".json")
    atomic_write_json(manifest, {
        "source": str(source_path), "shared_copy": str(target),
        "channel": channel, "timestamp": time.time(), "size_bytes": target.stat().st_size,
    }, indent=2, ensure_ascii=False)
    return {"ok": True, "path": str(target), "manifest": str(manifest)}


def workspace_command_environment(child: str, command: Sequence[str]) -> Optional[dict[str, str]]:
    names = {Path(str(part)).name for part in command}
    if names & {"paint_runtime.py", "paint_window.py", "daw_window.py", "ina_file_explorer.py"}:
        return launch_environment(child)
    return None
