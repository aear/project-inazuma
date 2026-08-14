"""Event-driven supervisor for Ina's world and Discord bridge processes."""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ina_process import psutil

from config_layers import load_config
from gui_hook import log_to_statusbox
from io_utils import atomic_write_json
from runtime_lifecycle import stop_runtime_services
from safe_popen import safe_popen

try:
    import fcntl
except Exception:  # pragma: no cover - Linux runtime has fcntl
    fcntl = None


SERVICE_COMMANDS = {
    "world_server": [sys.executable, "world_server.py"],
    "discord_bridge": [sys.executable, "discord_bridge.py"],
    "virtual_workspace": [sys.executable, "virtual_workspace.py"],
}
SERVICE_SIGNALS = {
    "world_server": getattr(signal, "SIGUSR1", signal.SIGTERM),
    "discord_bridge": getattr(signal, "SIGUSR2", signal.SIGTERM),
    "virtual_workspace": getattr(signal, "SIGHUP", signal.SIGTERM),
}
_START_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory"


def supervisor_status_path(child: str) -> Path:
    return _memory_root(child) / "runtime_services.json"


def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _is_supervisor_process(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        return any(Path(str(part)).name == "runtime_services.py" for part in process.cmdline())
    except (psutil.Error, OSError):
        return False


def ensure_runtime_service_supervisor(child: str) -> Optional[int]:
    """Replace legacy detached bridges, then launch one inherited supervisor."""
    with _START_LOCK:
        status_path = supervisor_status_path(child)
        status = _safe_json(status_path)
        existing_pid = int(status.get("supervisor_pid", 0) or 0)
        if _is_supervisor_process(existing_pid):
            return existing_pid

        # Exact project-script matching prevents unrelated Discord/world programs
        # from being touched. This one-time handover moves legacy instances under
        # the GUI -> supervisor process tree and therefore the cgroup envelope.
        stopped = stop_runtime_services(Path(__file__).resolve().parent)
        if stopped.get("matched"):
            log_to_statusbox(
                f"[Services] Replaced detached runtime services: {', '.join(stopped['matched'])}."
            )
        process = safe_popen(
            [sys.executable, "runtime_services.py", "--supervise", "--child", str(child)],
            cwd=str(Path(__file__).resolve().parent),
        )
        if process is None:
            log_to_statusbox("[Services] Failed to launch runtime service supervisor.")
            return None

        # Verify the child acquired its singleton lock and published state.
        # This closes the race where rapid GUI actions could replace a supervisor
        # that was alive but had not written its PID yet.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            published = _safe_json(status_path)
            if int(published.get("supervisor_pid", 0) or 0) == int(process.pid):
                return int(process.pid)
            if process.poll() is not None:
                break
            time.sleep(0.05)
        log_to_statusbox("[Services] Supervisor did not publish a verified ready state.")
        try:
            process.terminate()
        except Exception:
            pass
        return None


def request_service_restart(child: str, service: str) -> Dict[str, Any]:
    service = str(service or "").strip().lower()
    if service not in SERVICE_SIGNALS:
        return {"ok": False, "reason": "unknown_service", "service": service}
    status = _safe_json(supervisor_status_path(child))
    pid = int(status.get("supervisor_pid", 0) or 0)
    if not _is_supervisor_process(pid):
        return {"ok": False, "reason": "supervisor_not_running", "service": service}
    try:
        os.kill(pid, SERVICE_SIGNALS[service])
    except OSError as exc:
        return {"ok": False, "reason": str(exc), "service": service, "supervisor_pid": pid}
    return {"ok": True, "service": service, "supervisor_pid": pid}


class RuntimeServiceSupervisor:
    def __init__(self, child: str) -> None:
        self.child = str(child)
        self.project_root = Path(__file__).resolve().parent
        self.status_path = supervisor_status_path(self.child)
        self.events: queue.Queue[tuple[str, str, Any]] = queue.Queue()
        self.processes: Dict[str, Any] = {}
        self.started_at: Dict[str, float] = {}
        self.failures = {name: 0 for name in SERVICE_COMMANDS}
        self.restart_requested: set[str] = set()
        self.timers: Dict[str, threading.Timer] = {}
        self.stopping = False
        self.state: Dict[str, Any] = {
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "started_at": _now(),
            "services": {},
        }
        cfg = load_config()
        raw = cfg.get("runtime_services") if isinstance(cfg, dict) else None
        policy = raw if isinstance(raw, dict) else {}
        self.enabled = {
            "world_server": bool(policy.get("world_server_enabled", True)),
            "discord_bridge": bool(policy.get("discord_bridge_enabled", True)),
            "virtual_workspace": bool(policy.get("virtual_workspace_enabled", True)),
        }
        discord_cfg = cfg.get("discord") if isinstance(cfg, dict) else None
        if isinstance(discord_cfg, dict) and discord_cfg.get("enabled") is False:
            self.enabled["discord_bridge"] = False
        self.restart_base = max(0.25, float(policy.get("restart_base_delay_sec", 1.0) or 1.0))
        self.restart_max = max(self.restart_base, float(policy.get("restart_max_delay_sec", 30.0) or 30.0))
        self.stable_seconds = max(5.0, float(policy.get("stable_reset_sec", 60.0) or 60.0))

    def _publish(self) -> None:
        self.state["updated_at"] = _now()
        atomic_write_json(self.status_path, self.state, indent=2, ensure_ascii=False)

    def _service_state(self, name: str, **updates: Any) -> None:
        current = self.state.setdefault("services", {}).setdefault(name, {})
        current.update(updates)
        current["updated_at"] = _now()
        self._publish()

    def _watch(self, name: str, process: Any) -> None:
        try:
            returncode = process.wait()
        except Exception as exc:
            self.events.put(("exit", name, {"process": process, "returncode": None, "error": str(exc)}))
            return
        self.events.put(("exit", name, {"process": process, "returncode": int(returncode)}))

    def _launch(self, name: str) -> None:
        self.timers.pop(name, None)
        if self.stopping:
            return
        if not self.enabled.get(name, False):
            self._service_state(name, status="disabled", pid=None)
            return
        current = self.processes.get(name)
        if current is not None and current.poll() is None:
            return
        command = list(SERVICE_COMMANDS[name])
        if name == "virtual_workspace":
            command.extend(["--child", self.child])
        process = safe_popen(
            command,
            cwd=str(self.project_root),
            governor_module=name,
        )
        if process is None:
            self.events.put(("exit", name, {"process": None, "returncode": None, "error": "launch_failed"}))
            return
        self.processes[name] = process
        self.started_at[name] = time.monotonic()
        self._service_state(
            name,
            status="running",
            pid=int(process.pid),
            command=command,
            started_at=_now(),
            restart_count=int(self.failures[name]),
            last_error=None,
        )
        threading.Thread(target=self._watch, args=(name, process), daemon=True).start()
        log_to_statusbox(f"[Services] {name.replace('_', ' ')} started (pid={process.pid}).")

    def _schedule_restart(self, name: str, *, immediate: bool = False) -> None:
        if self.stopping or not self.enabled.get(name, False):
            return
        old_timer = self.timers.pop(name, None)
        if old_timer is not None:
            old_timer.cancel()
        delay = 0.0 if immediate else min(self.restart_max, self.restart_base * (2 ** max(0, self.failures[name] - 1)))
        self._service_state(name, status="restart_wait", pid=None, restart_in_sec=round(delay, 2))
        timer = threading.Timer(delay, lambda: self.events.put(("launch", name, None)))
        timer.daemon = True
        self.timers[name] = timer
        timer.start()

    def _restart(self, name: str) -> None:
        if not self.enabled.get(name, False):
            self._service_state(name, status="disabled", pid=None)
            return
        process = self.processes.get(name)
        if process is None or process.poll() is not None:
            self._schedule_restart(name, immediate=True)
            return
        self.restart_requested.add(name)
        self._service_state(name, status="restart_requested", pid=int(process.pid))
        try:
            process.terminate()
        except Exception as exc:
            self.restart_requested.discard(name)
            self._service_state(name, status="restart_failed", last_error=str(exc))

    def _handle_exit(self, name: str, detail: Dict[str, Any]) -> None:
        process = detail.get("process")
        if process is not None and self.processes.get(name) is not process:
            return
        self.processes.pop(name, None)
        runtime = max(0.0, time.monotonic() - self.started_at.pop(name, time.monotonic()))
        requested = name in self.restart_requested
        self.restart_requested.discard(name)
        if runtime >= self.stable_seconds:
            self.failures[name] = 0
        if not requested:
            self.failures[name] += 1
        self._service_state(
            name,
            status="stopped" if self.stopping else "exited",
            pid=None,
            returncode=detail.get("returncode"),
            last_error=detail.get("error"),
            last_runtime_sec=round(runtime, 3),
            restart_count=int(self.failures[name]),
        )
        if not self.stopping:
            self._schedule_restart(name, immediate=requested)

    def request_stop(self) -> None:
        self.events.put(("stop", "", None))

    def request_restart(self, name: str) -> None:
        self.events.put(("restart", name, None))

    def run(self) -> None:
        self.state["status"] = "running"
        self._publish()
        for name in SERVICE_COMMANDS:
            self._launch(name)
        while not self.stopping:
            action, name, detail = self.events.get()
            if action == "stop":
                self.stopping = True
                break
            if action == "launch":
                self._launch(name)
            elif action == "restart":
                self._restart(name)
            elif action == "exit" and isinstance(detail, dict):
                self._handle_exit(name, detail)

        for timer in self.timers.values():
            timer.cancel()
        for name, process in list(self.processes.items()):
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    continue
        for name, process in list(self.processes.items()):
            try:
                process.wait(timeout=5.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self._service_state(name, status="stopped", pid=None)
        self.state["status"] = "stopped"
        self._publish()


def _acquire_supervisor_lock(child: str):
    lock_path = _memory_root(child) / "runtime_services.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervise Ina runtime bridge services.")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--child", default=None)
    args = parser.parse_args()
    cfg = load_config()
    child = str(args.child or cfg.get("current_child") or "Inazuma_Yagami")
    lock_handle = _acquire_supervisor_lock(child)
    if lock_handle is None:
        return
    supervisor = RuntimeServiceSupervisor(child)
    signal.signal(signal.SIGTERM, lambda *_args: supervisor.request_stop())
    signal.signal(signal.SIGINT, lambda *_args: supervisor.request_stop())
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_args: supervisor.request_restart("world_server"))
    if hasattr(signal, "SIGUSR2"):
        signal.signal(signal.SIGUSR2, lambda *_args: supervisor.request_restart("discord_bridge"))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_args: supervisor.request_restart("virtual_workspace"))
    supervisor.run()
    lock_handle.close()


if __name__ == "__main__":
    main()
