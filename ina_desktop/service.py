"""Supervised owner of Ina's virtual display, input socket and audio buses."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config_layers import load_config
from io_utils import atomic_write_json
from .audio import ensure_audio_buses, unload_audio_buses
from .paths import display_number, lock_path, share_root, socket_path, status_path, workspace_root
from .x11 import X11Desktop

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


def workspace_control_api_payload() -> dict[str, Any]:
    """Describe the bounded virtual-desktop controls available to Ina."""
    return {
        "version": 1,
        "commands": [
            {"action": "windows", "arguments": {}},
            {
                "action": "focus_tool",
                "aliases": ["select_tool"],
                "arguments": {"tool": "paint, daw, music, or a visible window-title fragment"},
            },
            {
                "action": "cycle_window",
                "aliases": ["next_window", "previous_window"],
                "arguments": {"direction": "1 for next, -1 for previous"},
            },
            {"action": "tile", "arguments": {}},
            {"action": "capture", "arguments": {}},
            {"action": "open_file_explorer", "arguments": {}, "execution_policy": "launches only the fixed data-only explorer"},
        ],
        "examples": [
            {"action": "focus_tool", "tool": "paint"},
            {"action": "focus_tool", "tool": "daw"},
            {"action": "next_window"},
        ],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _xvfb_prefix() -> list[str]:
    direct = shutil.which("Xvfb")
    if direct:
        return [direct]
    host_xvfb = Path("/run/host/usr/bin/Xvfb")
    if host_xvfb.is_file() and os.access(host_xvfb, os.X_OK):
        # Running the visible host binary directly makes it a real child of the
        # service, so it inherits Ina's aggregate cgroup instead of escaping via
        # flatpak-session-helper.
        return [str(host_xvfb)]
    host_spawn = shutil.which("flatpak-spawn")
    if host_spawn:
        probe = subprocess.run(
            [host_spawn, "--host", "sh", "-lc", "command -v Xvfb"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return [host_spawn, "--host", "Xvfb"]
    return []


class VirtualWorkspaceService:
    def __init__(
        self, child: str, width: int = 1920, height: int = 1080, *,
        input_enabled: bool = True, share_bus_enabled: bool = True,
    ) -> None:
        self.child = str(child)
        self.width = max(800, int(width))
        self.height = max(600, int(height))
        self.input_enabled = bool(input_enabled)
        self.share_bus_enabled = bool(share_bus_enabled)
        self.root = workspace_root(self.child)
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            share_root(self.child) / "inbox" / "files",
            share_root(self.child) / "outbox" / "files",
            share_root(self.child) / "messages",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.server: socket.socket | None = None
        self.xvfb: subprocess.Popen | None = None
        self.desktop: X11Desktop | None = None
        self.audio: dict[str, Any] = {}
        self.display = ""
        self.state: dict[str, Any] = {
            "service_pid": os.getpid(), "status": "starting", "ready": False,
            "started_at": _now(), "child": self.child,
        }

    def _publish(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = _now()
        atomic_write_json(status_path(self.child), self.state, indent=2, ensure_ascii=False)

    def _start_display(self) -> None:
        prefix = _xvfb_prefix()
        if not prefix:
            raise RuntimeError(
                "Xvfb is not installed; install Fedora package xorg-x11-server-Xvfb"
            )
        base = display_number(self.child)
        errors = []
        for number in range(base, base + 10):
            display = f":{number}"
            command = [
                *prefix, display, "-screen", "0", f"{self.width}x{self.height}x24",
                "-nolisten", "tcp", "-noreset", "+extension", "RANDR",
                "+extension", "RENDER", "+extension", "XTEST",
            ]
            env = os.environ.copy()
            if prefix[0].startswith("/run/host/"):
                host_lib = "/run/host/usr/lib64"
                env["LD_LIBRARY_PATH"] = host_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            process = subprocess.Popen(
                command, cwd=str(Path(__file__).resolve().parents[1]), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    desktop = X11Desktop(display)
                except Exception:
                    time.sleep(0.05)
                    continue
                self.xvfb = process
                self.desktop = desktop
                self.display = display
                return
            error = ""
            if process.stderr is not None:
                try:
                    error = process.stderr.read(1000)
                except Exception:
                    pass
            errors.append(f"{display}: {error.strip() or 'not ready'}")
            if process.poll() is None:
                process.terminate()
        raise RuntimeError("; ".join(errors[-3:]))

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        desktop = self.desktop
        if desktop is None:
            return {"ok": False, "error": "display unavailable"}
        action = str(request.get("action") or "").strip().lower()
        input_actions = {
            "mouse_move", "mouse_button", "key", "type_text", "focus",
            "focus_tool", "select_tool", "cycle_window", "next_window", "previous_window",
        }
        if action in input_actions and not self.input_enabled:
            return {"ok": False, "error": "workspace input is disabled by policy"}
        if action == "status":
            return {"ok": True, **self.state}
        if action == "mouse_move":
            desktop.mouse_move(int(request.get("x", 0)), int(request.get("y", 0)))
        elif action == "mouse_button":
            desktop.mouse_button(int(request.get("button", 1)), bool(request.get("pressed", True)))
        elif action == "key":
            desktop.key(str(request.get("keysym") or ""), bool(request.get("pressed", True)))
        elif action == "type_text":
            text = str(request.get("text") or "")
            if len(text) > 16_384:
                return {"ok": False, "error": "text exceeds 16384 characters"}
            desktop.type_text(text)
        elif action == "focus":
            desktop.focus(int(request.get("window_id", 0)))
        elif action in {"focus_tool", "select_tool"}:
            tool = str(request.get("tool") or request.get("name") or request.get("title") or "")
            selected = desktop.focus_tool(tool)
            if selected is None:
                return {
                    "ok": False,
                    "error": f"no open window matches tool: {tool or 'missing'}",
                    "windows": [item.__dict__ for item in desktop.windows()],
                }
            return {"ok": True, "window": selected.__dict__}
        elif action in {"cycle_window", "next_window", "previous_window"}:
            if action == "previous_window":
                direction = -1
            elif action == "next_window":
                direction = 1
            else:
                direction = int(request.get("direction", 1) or 1)
            selected = desktop.cycle_window(direction)
            if selected is None:
                return {"ok": False, "error": "no titled windows are open"}
            return {"ok": True, "window": selected.__dict__}
        elif action == "tile":
            return {"ok": True, "windows": desktop.tile()}
        elif action == "windows":
            return {"ok": True, "windows": [item.__dict__ for item in desktop.windows()]}
        elif action == "capture":
            path = desktop.save_ppm(self.root / "latest.ppm")
            return {"ok": True, "path": str(path)}
        elif action == "open_file_explorer":
            project_root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env.update({"DISPLAY": self.display, "INA_VIRTUAL_WORKSPACE": "1", "INA_WORKSPACE_CHILD": self.child})
            process = subprocess.Popen(
                [sys.executable, str(project_root / "ina_file_explorer.py"), "--child", self.child],
                cwd=str(project_root), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return {"ok": True, "pid": process.pid, "execution_allowed": False}
        else:
            return {"ok": False, "error": f"unknown action: {action}"}
        return {"ok": True}

    def _serve(self) -> None:
        path = socket_path(self.child)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(8)
        server.settimeout(1.0)
        self.server = server
        while not self.stop_event.is_set():
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    raw = b""
                    while not raw.endswith(b"\n") and len(raw) <= 1024 * 1024:
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        raw += chunk
                    request = json.loads(raw.decode("utf-8"))
                    response = self._dispatch(request if isinstance(request, dict) else {})
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                connection.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")

    def run(self) -> int:
        try:
            self._start_display()
            self.audio = ensure_audio_buses()
            self._publish(
                status="running", ready=True, display=self.display,
                resolution=[self.width, self.height], audio=self.audio,
                display_process_pid=self.xvfb.pid if self.xvfb is not None else None,
                input_enabled=self.input_enabled, share_bus_enabled=self.share_bus_enabled,
                control_socket=str(socket_path(self.child)), share_root=str(share_root(self.child)),
                control_api=workspace_control_api_payload(),
            )
            self._serve()
            return 0
        except Exception as exc:
            message = str(exc)
            if "Xvfb is not installed" in message:
                self._publish(status="blocked", ready=False, error=message, audio=self.audio)
                self.stop_event.wait()
                return 0
            self._publish(status="failed", ready=False, error=message, audio=self.audio)
            return 1
        finally:
            self.stop_event.set()
            if self.server is not None:
                try:
                    self.server.close()
                except Exception:
                    pass
            try:
                socket_path(self.child).unlink()
            except FileNotFoundError:
                pass
            if self.desktop is not None:
                self.desktop.close()
            if self.xvfb is not None and self.xvfb.poll() is None:
                self.xvfb.terminate()
                try:
                    self.xvfb.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.xvfb.kill()
            unload_audio_buses(list(self.audio.get("module_ids") or []))
            previous_status = self.state.get("status")
            self._publish(status="stopped", ready=False, previous_status=previous_status)

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass


def _lock(child: str):
    path = lock_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)
    workspace_policy = config.get("virtual_workspace")
    workspace_policy = workspace_policy if isinstance(workspace_policy, dict) else {}
    width = args.width if args.width is not None else workspace_policy.get("width", 1920)
    height = args.height if args.height is not None else workspace_policy.get("height", 1080)
    lock = _lock(args.child)
    if lock is None:
        return 0
    service = VirtualWorkspaceService(
        args.child, width, height,
        input_enabled=workspace_policy.get("input_enabled", True),
        share_bus_enabled=workspace_policy.get("share_bus_enabled", True),
    )
    signal.signal(signal.SIGTERM, lambda *_args: service.stop())
    signal.signal(signal.SIGINT, lambda *_args: service.stop())
    try:
        return service.run()
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
