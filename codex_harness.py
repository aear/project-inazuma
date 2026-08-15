"""Lightweight standalone GUI harness for Codex's local app-server protocol."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse
import webbrowser

MAX_EVENTS = 600
MAX_EVENT_CHARS = 65536
MAX_SUMMARY_CHARS = 12000
MAX_PROMPT_CHARS = 100000
REQUEST_TIMEOUT_SECONDS = 30.0
BLOCKED_BILLING_ENV = {
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL",
}
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "2", "TOKIO_WORKER_THREADS": "2",
    "UV_THREADPOOL_SIZE": "2", "MALLOC_ARENA_MAX": "2",
}
APPROVAL_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "execCommandApproval",
    "applyPatchApproval",
    "item/permissions/requestApproval",
}

TERMINAL_TURN_STATUSES = {"completed", "interrupted", "failed"}


class SubscriptionAuthError(RuntimeError):
    pass


def subscription_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(source if source is not None else os.environ)
    for name in BLOCKED_BILLING_ENV:
        env.pop(name, None)
    env.update(THREAD_LIMIT_ENV)
    env["INA_CODEX_HARNESS"] = "1"
    return env


def discover_codex(explicit: str | None = None) -> str:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    located = shutil.which("codex")
    if located:
        candidates.append(Path(located))
    candidates.append(Path.home() / ".local" / "bin" / "codex")
    extension_root = Path.home() / ".var" / "app" / "com.visualstudio.code" / "data" / "vscode" / "extensions"
    if extension_root.is_dir():
        candidates.extend(sorted(
            extension_root.glob("openai.chatgpt-*/bin/linux-x86_64/codex"),
            reverse=True,
        ))
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        except OSError:
            continue
    raise FileNotFoundError(
        "Codex CLI was not found. Install the standalone CLI or pass --codex-binary; "
        "the VS Code process does not need to be running."
    )


class BoundedEvents:
    def __init__(self, maximum: int = MAX_EVENTS) -> None:
        self.maximum = max(20, min(MAX_EVENTS, int(maximum)))
        self._items: deque[dict[str, Any]] = deque(maxlen=self.maximum)
        self._sequence = 0
        self._condition = threading.Condition()

    def append(self, kind: str, payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            payload = payload[:MAX_EVENT_CHARS]
        else:
            try:
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                encoded = repr(payload)
            if len(encoded) > MAX_EVENT_CHARS:
                payload = {
                    "truncated": True,
                    "preview": encoded[:MAX_EVENT_CHARS],
                    "original_characters": len(encoded),
                }
        event = {
            "sequence": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
            "payload": payload,
        }
        with self._condition:
            self._sequence += 1
            event["sequence"] = self._sequence
            self._items.append(event)
            self._condition.notify_all()
        return event

    def wait_after(self, sequence: int, timeout: float = 20.0) -> list[dict[str, Any]]:
        with self._condition:
            if not any(item["sequence"] > sequence for item in self._items):
                self._condition.wait(timeout=max(0.0, min(25.0, timeout)))
            return [dict(item) for item in self._items if item["sequence"] > sequence]


@dataclass(frozen=True)
class HarnessConfig:
    root: Path
    codex_binary: str


class AppServerClient:
    def __init__(self, config: HarnessConfig, events: BoundedEvents | None = None) -> None:
        self.config = config
        self.events = events or BoundedEvents()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._server_requests: dict[int, dict[str, Any]] = {}
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.active_model: str | None = None
        self.running_turn = False
        self.thread_status = "notLoaded"
        self.turn_status = "idle"
        self.work_status = "Ready"
        self.started_monotonic = time.monotonic()
        self.last_test_status = "not observed"
        self.last_benchmark_status = "not observed"
        self.diff_seen = False
        self.process = subprocess.Popen(
            [
                config.codex_binary,
                "-c", 'forced_login_method="chatgpt"',
                "app-server", "--stdio",
            ],
            cwd=str(config.root),
            env=subscription_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.request("initialize", {
            "clientInfo": {"name": "ina-codex-harness", "title": "Ina Codex Harness", "version": "0.1"},
            "capabilities": {"experimentalApi": True},
        })
        self.notify("initialized", {})
        self.events.append("status", "Codex app server ready.")

    def _write(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("Codex app server is not running.")
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._write_lock:
            self.process.stdin.write(line)
            self.process.stdin.flush()

    def notify(self, method: str, params: Any) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._state_lock:
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = response_queue
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            response = response_queue.get(timeout=max(0.1, timeout))
        except queue.Empty as exc:
            raise TimeoutError(f"Codex app-server request timed out: {method}") from exc
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return response.get("result")

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.events.append("diagnostic", line.rstrip())
                continue
            request_id = message.get("id")
            if request_id is not None and ("result" in message or "error" in message):
                with self._state_lock:
                    waiting = self._pending.get(int(request_id))
                if waiting is not None:
                    waiting.put(message)
                continue
            method = str(message.get("method") or "")
            if request_id is not None:
                with self._state_lock:
                    self._server_requests[int(request_id)] = message
                self.events.append(
                    "approval" if method in APPROVAL_METHODS else "request",
                    {"request_id": int(request_id), "method": method, "params": message.get("params")},
                )
                continue
            self._handle_notification(method, message.get("params"))

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            normalized = line.strip()
            if normalized:
                self.events.append("diagnostic", normalized)

    def _handle_notification(self, method: str, params: Any) -> None:
        if method == "turn/started" and isinstance(params, dict):
            turn = params.get("turn") or {}
            self.turn_id = str(turn.get("id") or self.turn_id or "") or None
            self.running_turn = True
            self.turn_status = str(turn.get("status") or "inProgress")
        elif method == "turn/completed" and isinstance(params, dict):
            turn = params.get("turn") or {}
            status = str(turn.get("status") or "completed")
            self.turn_status = status
            self.running_turn = status not in TERMINAL_TURN_STATUSES
            self.work_status = status.capitalize()
        elif method == "thread/status/changed" and isinstance(params, dict):
            status = params.get("status") or {}
            self.thread_status = str(status.get("type") or self.thread_status)
            self.running_turn = self.thread_status == "active"
        elif method == "account/login/completed":
            self.events.append("auth", params)
        self._append_notification_event(method, params)

    def _append_notification_event(self, method: str, params: Any) -> None:
        data = params if isinstance(params, dict) else {}
        raw = {"method": method, "params": params}
        if method in {"item/agentMessage/delta", "agentMessage/delta"}:
            kind, summary = "assistant", data.get("delta") or ""
        elif method in {"item/reasoning/summaryTextDelta", "item/reasoning/summaryPartAdded"}:
            summary = data.get("delta") or "Reasoning summary updated"
            self.work_status = str(summary)[-MAX_SUMMARY_CHARS:]
            kind = "work_status"
        elif method == "turn/diff/updated":
            kind, summary = "diff", data.get("diff") or "Workspace diff updated"
            self.diff_seen = True
        elif method in {"item/commandExecution/outputDelta", "item/mcpToolCall/progress"}:
            kind, summary = "tool_output", data.get("delta") or data.get("message") or "Tool output updated"
        elif method in {"item/started", "item/completed"}:
            item = data.get("item") or {}
            item_type = str(item.get("type") or "item")
            status = str(item.get("status") or ("completed" if method.endswith("completed") else "started"))
            kind, summary = "tool", f"{item_type}: {status}"
            command = str(item.get("command") or "").lower()
            if "test" in command or "pytest" in command:
                self.last_test_status = status
            if "benchmark" in command:
                self.last_benchmark_status = status
        elif method in {"turn/started", "turn/completed", "thread/status/changed"}:
            kind, summary = "state", f"{method}: {self.turn_status if method.startswith('turn/') else self.thread_status}"
        elif method in {"turn/plan/updated", "item/plan/delta"}:
            kind, summary = "plan", data.get("delta") or data.get("plan") or "Plan updated"
        elif method in {"error", "warning"} or method.endswith("/error"):
            kind, summary = "diagnostic", data.get("message") or method
        else:
            kind, summary = "protocol", method or "notification"
        self.events.append(kind, {"summary": summary, "raw": raw})

    def account(self, refresh: bool = False) -> dict[str, Any]:
        result = self.request("account/read", {"refreshToken": bool(refresh)})
        account = result.get("account") if isinstance(result, dict) else None
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            raise SubscriptionAuthError("ChatGPT subscription login is required; API-key mode is disabled.")
        return {
            "type": "chatgpt",
            "plan_type": account.get("planType"),
            "requires_openai_auth": bool(result.get("requiresOpenaiAuth")),
        }

    def start_login(self, device_code: bool = False) -> dict[str, Any]:
        login_type = "chatgptDeviceCode" if device_code else "chatgpt"
        result = self.request("account/login/start", {"type": login_type})
        if not isinstance(result, dict) or result.get("type") not in {"chatgpt", "chatgptDeviceCode"}:
            raise SubscriptionAuthError("Codex returned a non-ChatGPT login method.")
        return {key: value for key, value in result.items() if key in {
            "type", "authUrl", "loginId", "userCode", "verificationUrl",
        }}

    def new_thread(self, *, model: str | None = None) -> dict[str, Any]:
        self.account(refresh=False)
        result = self.request("thread/start", {
            "cwd": str(self.config.root),
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandbox": "workspace-write",
            "model": model,
            "runtimeWorkspaceRoots": [str(self.config.root)],
            "threadSource": "appServer",
        })
        thread = result.get("thread") if isinstance(result, dict) else {}
        self.thread_id = str(thread.get("id") or "") or None
        self.turn_id = None
        self.active_model = str(result.get("model") or model or "") or None
        self.events.append("status", {"new_thread": self.thread_id})
        return {"thread_id": self.thread_id, "model": self.active_model}

    def send_prompt(
        self, prompt: str, *, model: str | None = None,
        effort: str | None = None, collaboration_mode: str = "default",
        steering: bool = True,
    ) -> dict[str, Any]:
        prompt = str(prompt or "")
        if not prompt.strip():
            raise ValueError("Prompt is empty.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt exceeds {MAX_PROMPT_CHARS} characters.")
        if self.running_turn:
            raise RuntimeError("A Codex turn is already running.")
        if not self.thread_id:
            self.new_thread(model=model)
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "cwd": str(self.config.root),
        }
        if model:
            params["model"] = model
        if effort:
            params["effort"] = effort
        collaboration_model = model or self.active_model
        if steering and collaboration_mode in {"default", "plan"} and collaboration_model:
            params["collaborationMode"] = {
                "mode": collaboration_mode,
                "settings": {
                    "model": collaboration_model,
                    "reasoning_effort": effort,
                    "developer_instructions": None,
                },
            }
        self.events.append("user", {"summary": prompt, "raw": {"steering": bool(steering)}})
        result = self.request("turn/start", params)
        turn = result.get("turn") if isinstance(result, dict) else {}
        self.turn_id = str(turn.get("id") or "") or None
        self.turn_status = str(turn.get("status") or "inProgress")
        self.running_turn = self.turn_status not in TERMINAL_TURN_STATUSES
        return self.status()

    def status(self) -> dict[str, Any]:
        result = {
            "server_running": self.process.poll() is None,
            "turn_running": self.running_turn,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "root": str(self.config.root),
            "pid": self.process.pid if self.process.poll() is None else None,
            "thread_status": self.thread_status,
            "turn_status": self.turn_status,
            "work_status": self.work_status,
            "git": self._git_status(),
            "tests": self.last_test_status,
            "benchmarks": self.last_benchmark_status,
            "diff": "changed" if self.diff_seen else "clean in this session",
        }
        app_resources = self._resource_status(result["pid"])
        harness_resources = self._resource_status(os.getpid())
        result["resources"] = {
            "rss_mib": f"{harness_resources['rss_mib'] or '?'} / {app_resources['rss_mib'] or '?'}",
            "threads": f"{harness_resources['threads'] or '?'} / {app_resources['threads'] or '?'}",
            "harness": harness_resources,
            "app_server": app_resources,
        }
        return result

    def _git_status(self) -> dict[str, Any]:
        head = self.config.root / ".git" / "HEAD"
        try:
            value = head.read_text(encoding="utf-8").strip()
            branch = value.rsplit("/", 1)[-1] if value.startswith("ref:") else value[:12]
        except OSError:
            branch = None
        return {"branch": branch, "session_diff": self.diff_seen}

    def _resource_status(self, pid: int | None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "uptime_seconds": round(time.monotonic() - self.started_monotonic, 1),
            "rss_mib": None, "threads": None,
        }
        if not pid:
            return result
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    result["rss_mib"] = round(int(line.split()[1]) / 1024.0, 1)
                elif line.startswith("Threads:"):
                    result["threads"] = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return result

    def interrupt(self) -> bool:
        if not self.thread_id or not self.turn_id or not self.running_turn:
            return False
        self.request("turn/interrupt", {"threadId": self.thread_id, "turnId": self.turn_id})
        self.events.append("status", "Interrupt requested.")
        return True

    def respond(self, request_id: int, decision: str) -> None:
        with self._state_lock:
            message = self._server_requests.pop(int(request_id), None)
        if message is None:
            raise KeyError("Approval request is no longer pending.")
        method = str(message.get("method") or "")
        if method not in APPROVAL_METHODS:
            raise ValueError("This request requires a richer response than an approval decision.")
        allowed = {"accept", "acceptForSession", "decline", "cancel"}
        if decision not in allowed:
            raise ValueError("Unsupported approval decision.")
        self._write({"jsonrpc": "2.0", "id": int(request_id), "result": {"decision": decision}})
        self.events.append("status", {"approval": decision, "request_id": int(request_id)})

    def capabilities(self) -> dict[str, Any]:
        methods = {
            "models": ("model/list", {"limit": 100}),
            "skills": ("skills/list", {"cwds": [str(self.config.root)], "forceReload": False}),
            "plugins": ("plugin/list", {"cwds": [str(self.config.root)], "forceRefetch": False}),
            "apps": ("app/list", {}),
            "mcp_servers": ("mcpServerStatus/list", {}),
            "collaboration_modes": ("collaborationMode/list", {}),
        }
        result: dict[str, Any] = {}
        for name, (method, params) in methods.items():
            try:
                result[name] = self.request(method, params)
            except Exception as exc:
                result[name] = {"unavailable": str(exc)}
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except OSError:
                pass


class HarnessHandler(BaseHTTPRequestHandler):
    server: "HarnessServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _authorized(self) -> bool:
        query = parse_qs(urlparse(self.path).query)
        supplied = self.headers.get("X-Harness-Token") or (query.get("token") or [""])[0]
        return secrets.compare_digest(str(supplied), self.server.access_token)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not self._authorized():
            self._json({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        if parsed.path == "/":
            body = self.server.ui_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if parsed.path == "/api/status":
                payload = self.server.client.status()
                try:
                    payload["account"] = self.server.client.account()
                except SubscriptionAuthError:
                    payload["account"] = None
                self._json(payload)
                return
            if parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                after = max(0, int((query.get("after") or ["0"])[0]))
                wait = max(0.0, min(20.0, float((query.get("wait") or ["0"])[0])))
                self._json(self.server.client.events.wait_after(after, wait))
                return
            if parsed.path == "/api/capabilities":
                self._json(self.server.client.capabilities())
                return
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        parsed = urlparse(self.path)
        try:
            length = min(MAX_PROMPT_CHARS + 8192, int(self.headers.get("Content-Length", "0")))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/run":
                self._json(self.server.client.send_prompt(
                    payload.get("prompt", ""), model=payload.get("model"),
                    effort=payload.get("effort"),
                    collaboration_mode=str(payload.get("mode") or "default"),
                    steering=bool(payload.get("steering", True)),
                ))
                return
            if parsed.path == "/api/new":
                self._json(self.server.client.new_thread(model=payload.get("model")))
                return
            if parsed.path == "/api/interrupt":
                self._json({"interrupted": self.server.client.interrupt()})
                return
            if parsed.path == "/api/approval":
                self.server.client.respond(int(payload.get("request_id")), str(payload.get("decision")))
                self._json({"ok": True})
                return
            if parsed.path == "/api/login":
                self._json(self.server.client.start_login(bool(payload.get("device_code"))))
                return
        except (ValueError, RuntimeError, TimeoutError, KeyError, SubscriptionAuthError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


class HarnessServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], client: AppServerClient,
        access_token: str, ui_path: Path,
    ) -> None:
        super().__init__(address, HarnessHandler)
        self.client = client
        self.access_token = access_token
        self.ui_path = ui_path


def build_config(root: Path | str, codex_binary: str | None = None) -> HarnessConfig:
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Workspace is not a directory: {resolved_root}")
    return HarnessConfig(resolved_root, discover_codex(codex_binary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--codex-binary")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    config = build_config(args.root, args.codex_binary)
    try:
        os.nice(5)
    except OSError:
        pass
    client = AppServerClient(config)
    token = secrets.token_urlsafe(24)
    ui_path = Path(__file__).with_name("codex_harness_ui.html")
    server = HarnessServer(("127.0.0.1", max(0, min(65535, args.port))), client, token, ui_path)
    host, port = server.server_address
    url = f"http://{host}:{port}/?token={token}"
    print(f"Codex Harness: {url}")
    if not args.no_browser:
        webbrowser.open(url, new=1)
    try:
        server.serve_forever(poll_interval=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVAL_METHODS", "AppServerClient", "BLOCKED_BILLING_ENV", "BoundedEvents",
    "HarnessConfig", "HarnessServer", "MAX_EVENTS", "MAX_PROMPT_CHARS",
    "SubscriptionAuthError", "build_config", "discover_codex",
    "subscription_environment",
]
