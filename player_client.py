#!/usr/bin/env python3
"""Player bridge client: local unix socket <-> TCP world server."""

from __future__ import annotations

import argparse
import copy
import json
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from world_protocol import (
    DEFAULT_LOCAL_PLAYER_SOCKET,
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    safe_json_dumps,
)

try:
    from player_viewer import run_viewer
except Exception:
    run_viewer = None

try:
    from player_arch_viewer import run_arch_viewer
except Exception:
    run_arch_viewer = None

try:
    from player_menu import run_player_menu
except Exception:
    run_player_menu = None


class PlayerClient:
    def __init__(
        self,
        *,
        tcp_host: str,
        tcp_port: int,
        local_socket: str,
        speech_enabled: Optional[bool] = None,
        speech_device: Optional[object] = None,
        speech_config: Optional[dict] = None,
        player_name: Optional[str] = None,
    ) -> None:
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.local_socket = local_socket
        self.server_sock: Optional[socket.socket] = None
        self.server_file = None
        self.server_lock = threading.Lock()
        self.local_server: Optional[socket.socket] = None
        self.local_clients: dict[socket.socket, Any] = {}
        self.local_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_state: Optional[Dict[str, Any]] = None
        self._comms_lock = threading.Lock()
        self._comms_seq = 0
        self._comms_log: Deque[Tuple[int, Dict[str, Any]]] = deque(maxlen=200)
        self.player_name = player_name or _load_player_name()
        self._speech_config = speech_config if speech_config is not None else _load_speech_config()
        if speech_enabled is None:
            speech_enabled = bool(self._speech_config.get("enabled", True))
        self._speech_enabled = bool(speech_enabled)
        self._speech_device = speech_device if speech_device is not None else _resolve_speech_device(self._speech_config)
        self._speech_monitor = None
        self._speech_last_send = 0.0
        self._speech_last_active: Optional[bool] = None

    def start(self) -> None:
        self._connect_server()
        self._start_local_server()
        threading.Thread(target=self._server_read_loop, daemon=True).start()
        threading.Thread(target=self._accept_local_loop, daemon=True).start()
        self._start_speech_monitor()

    def close(self) -> None:
        self.stop_event.set()
        if self._speech_monitor is not None:
            try:
                self._speech_monitor.stop()
            except Exception:
                pass
            self._speech_monitor = None
        if self.server_sock:
            self.server_sock.close()
        if self.local_server:
            self.local_server.close()
        with self.local_lock:
            for client in list(self.local_clients.values()):
                try:
                    client.close()
                except Exception:
                    pass
            self.local_clients.clear()
        if self.local_socket and socket_path_exists(self.local_socket):
            try:
                os_remove(self.local_socket)
            except OSError:
                pass

    def send(self, payload: Dict[str, Any]) -> None:
        data = safe_json_dumps(payload).encode("utf-8") + b"\n"
        with self.server_lock:
            if not self.server_file:
                return
            self.server_file.write(data)
            self.server_file.flush()

    def _connect_server(self) -> None:
        sock = socket.create_connection((self.tcp_host, self.tcp_port), timeout=5)
        sock.settimeout(None)
        self.server_sock = sock
        self.server_file = sock.makefile("rwb")
        self.send({"type": "hello", "role": "player", "name": self.player_name})
        self.send({"type": "subscribe"})

    def _start_local_server(self) -> None:
        if self.local_socket and socket_path_exists(self.local_socket):
            os_remove(self.local_socket)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.local_socket)
        server.listen(5)
        self.local_server = server

    def _accept_local_loop(self) -> None:
        if not self.local_server:
            return
        while not self.stop_event.is_set():
            try:
                client, _ = self.local_server.accept()
            except OSError:
                break
            with self.local_lock:
                self.local_clients[client] = client
            threading.Thread(target=self._handle_local_client, args=(client,), daemon=True).start()

    def _handle_local_client(self, client: socket.socket) -> None:
        file = client.makefile("rb")
        while not self.stop_event.is_set():
            line = file.readline()
            if not line:
                break
            self._forward_to_server(line)
        with self.local_lock:
            self.local_clients.pop(client, None)
        try:
            client.close()
        except Exception:
            pass

    def _forward_to_server(self, line: bytes) -> None:
        if not line.endswith(b"\n"):
            line += b"\n"
        with self.server_lock:
            if not self.server_file:
                return
            self.server_file.write(line)
            self.server_file.flush()

    def _server_read_loop(self) -> None:
        if not self.server_file:
            return
        while not self.stop_event.is_set():
            try:
                line = self.server_file.readline()
            except TimeoutError:
                continue
            if not line:
                break
            self._broadcast_to_locals(line)
            payload = self._decode_payload(line)
            if payload is None:
                continue
            self._handle_payload(payload)

    def _broadcast_to_locals(self, line: bytes) -> None:
        with self.local_lock:
            dead = []
            for client in self.local_clients.values():
                try:
                    client.sendall(line)
                except OSError:
                    dead.append(client)
            for client in dead:
                self.local_clients.pop(client, None)

    @staticmethod
    def _decode_payload(line: bytes) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _handle_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("type") == "state":
            state = payload.get("state")
            if isinstance(state, dict):
                self._update_state(state)
            entities = (state or {}).get("entities", {})
            player = entities.get("player")
            if player:
                print(f"[state] Player pos={player.get('position')} yaw={player.get('yaw_deg')}")
                return
        if payload.get("type") == "comms":
            name = payload.get("name") or payload.get("entity_id") or "unknown"
            text = payload.get("text") or ""
            entry = {
                "name": name,
                "text": text,
                "role": payload.get("role") or "unknown",
                "timestamp": payload.get("timestamp") or time.time(),
            }
            with self._comms_lock:
                self._comms_seq += 1
                self._comms_log.append((self._comms_seq, entry))
            print(f"[comms] {name}: {text}")
            return
        print(f"[server] {payload}")

    def _update_state(self, state: Dict[str, Any]) -> None:
        with self._state_lock:
            self._last_state = state

    def get_state_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._last_state is None:
                return None
            return copy.deepcopy(self._last_state)

    def get_comms_since(self, last_seq: int) -> Tuple[int, List[Dict[str, Any]]]:
        with self._comms_lock:
            if not self._comms_log:
                return last_seq, []
            oldest_seq = self._comms_log[0][0]
            if last_seq < oldest_seq:
                last_seq = oldest_seq - 1
            messages = [copy.deepcopy(msg) for seq, msg in self._comms_log if seq > last_seq]
            latest_seq = self._comms_log[-1][0] if self._comms_log else last_seq
            return latest_seq, messages

    def _start_speech_monitor(self) -> None:
        if not self._speech_enabled:
            return
        try:
            from speech_activity import SpeechActivityConfig, SpeechActivityMonitor
        except Exception as exc:
            print(f"[speech] monitor unavailable: {exc}")
            return
        config = SpeechActivityConfig.from_config(self._speech_config)
        monitor = SpeechActivityMonitor(config=config, device=self._speech_device)
        if not monitor.is_available:
            print("[speech] sounddevice missing; speech detection disabled.")
            return
        if not monitor.start(self._handle_speech_result):
            print("[speech] failed to start speech detection.")
            return
        self._speech_monitor = monitor

    def _handle_speech_result(self, result) -> None:
        now = time.time()
        active = bool(result.active)
        if (
            self._speech_last_active is None
            or active != self._speech_last_active
            or (now - self._speech_last_send) >= 2.0
        ):
            payload = {
                "type": "speech",
                "active": active,
                "level": float(result.rms),
                "snr": float(result.snr),
                "ratio": float(result.band_ratio),
                "timestamp": float(result.timestamp),
            }
            self.send(payload)
            self._speech_last_send = now
            self._speech_last_active = active


# ---------------------------- CLI helpers ---------------------------- #


def socket_path_exists(path: str) -> bool:
    try:
        return bool(path) and os_path_exists(path)
    except Exception:
        return False


def os_path_exists(path: str) -> bool:
    import os

    return os.path.exists(path)


def os_remove(path: str) -> None:
    import os

    os.remove(path)


def _load_speech_config() -> dict:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    block = data.get("speech_activity")
    if isinstance(block, dict):
        return dict(block)
    return {}


def _load_player_name() -> str:
    path = Path("config.json")
    if not path.exists():
        return "Player"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "Player"
    name = data.get("player_name")
    if not name:
        return "Player"
    return str(name)


def _resolve_speech_device(config: dict) -> Optional[object]:
    if not isinstance(config, dict):
        return None
    device = config.get("device")
    if device is None:
        return None
    try:
        if isinstance(device, (int, float)):
            return int(device)
        text = str(device).strip()
        if text.isdigit():
            return int(text)
        return text
    except Exception:
        return None


def _parse_device_arg(raw: Optional[str]) -> Optional[object]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def parse_move(tokens: list[str]) -> Dict[str, Any]:
    if not tokens:
        return {}
    cmd = tokens[0]
    if cmd in ("w", "s", "a", "d", "q", "e"):
        mapping = {
            "w": (1.0, 0.0, 0.0),
            "s": (-1.0, 0.0, 0.0),
            "a": (0.0, -1.0, 0.0),
            "d": (0.0, 1.0, 0.0),
            "q": (0.0, 0.0, -1.0),
            "e": (0.0, 0.0, 1.0),
        }
        forward, strafe, turn = mapping[cmd]
        return {"forward": forward, "strafe": strafe, "turn": turn}
    if cmd == "move" and len(tokens) >= 3:
        forward = float(tokens[1])
        strafe = float(tokens[2])
        turn = 0.0
        run = False
        duration = 0.0
        for token in tokens[3:]:
            if token.lower() == "run":
                run = True
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if turn == 0.0:
                turn = value
            else:
                duration = value
        return {
            "forward": forward,
            "strafe": strafe,
            "turn": turn,
            "run": run,
            "duration": duration,
        }
    if cmd == "turn" and len(tokens) >= 2:
        turn = float(tokens[1])
        duration = float(tokens[2]) if len(tokens) > 2 else 0.0
        return {"forward": 0.0, "strafe": 0.0, "turn": turn, "duration": duration}
    return {}


def interactive_loop(client: PlayerClient) -> None:
    print("Player client ready. Commands: move f s [turn] [run] [duration], stop, state, channel <name>, say <text>, quit")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        tokens = raw.split()
        cmd = tokens[0].lower()
        if cmd in ("quit", "exit"):
            break
        if cmd == "help":
            print("move f s [turn] [run] [duration] | stop | state | channel <name> | quit")
            continue
        if cmd == "stop":
            client.send({"type": "stop"})
            continue
        if cmd == "state":
            client.send({"type": "state"})
            continue
        if cmd == "channel" and len(tokens) > 1:
            client.send({"type": "set_channel", "channel": " ".join(tokens[1:])})
            continue
        if cmd == "say" and len(tokens) > 1:
            client.send({"type": "comms", "text": " ".join(tokens[1:])})
            continue

        move_args = parse_move(tokens)
        if move_args:
            run = bool(move_args.pop("run", False))
            duration = float(move_args.pop("duration", 0.0))
            client.send(
                {
                    "type": "move",
                    "input": move_args,
                    "run": run,
                }
            )
            if duration > 0:
                time.sleep(duration)
                client.send({"type": "stop"})
            continue

        print("Unknown command.")


def menu_loop(
    *,
    tcp_host: str,
    tcp_port: int,
    local_socket: str,
    speech_enabled: Optional[bool],
    speech_device: Optional[object],
) -> tuple[Optional[PlayerClient], str]:
    current_host = tcp_host
    current_port = tcp_port
    current_socket = local_socket

    while True:
        print("\nInazuma Player Client")
        print("---------------------")
        print(f"[1] Connect + headless bridge   ({current_host}:{current_port})")
        print(f"[2] Connect + interactive       ({current_host}:{current_port})")
        print(f"[3] Connect + map viewer         ({current_host}:{current_port})")
        print(f"[4] Connect + house viewer       ({current_host}:{current_port})")
        print(f"[5] Set TCP host                (current: {current_host})")
        print(f"[6] Set TCP port                (current: {current_port})")
        print(f"[7] Set local socket            (current: {current_socket})")
        print("[Q] Quit")

        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None, "quit"
        if choice == "1":
            client = PlayerClient(
                tcp_host=current_host,
                tcp_port=current_port,
                local_socket=current_socket,
                speech_enabled=speech_enabled,
                speech_device=speech_device,
            )
            client.start()
            return client, "headless"
        if choice == "2":
            client = PlayerClient(
                tcp_host=current_host,
                tcp_port=current_port,
                local_socket=current_socket,
                speech_enabled=speech_enabled,
                speech_device=speech_device,
            )
            client.start()
            return client, "interactive"
        if choice == "3":
            client = PlayerClient(
                tcp_host=current_host,
                tcp_port=current_port,
                local_socket=current_socket,
                speech_enabled=speech_enabled,
                speech_device=speech_device,
            )
            client.start()
            return client, "viewer"
        if choice == "4":
            client = PlayerClient(
                tcp_host=current_host,
                tcp_port=current_port,
                local_socket=current_socket,
                speech_enabled=speech_enabled,
                speech_device=speech_device,
            )
            client.start()
            return client, "house"
        if choice == "5":
            new_host = input("TCP host: ").strip()
            if new_host:
                current_host = new_host
            continue
        if choice == "6":
            new_port = input("TCP port: ").strip()
            try:
                current_port = int(new_port)
            except ValueError:
                print("Invalid port.")
            continue
        if choice == "7":
            new_socket = input("Local socket path: ").strip()
            if new_socket:
                current_socket = new_socket
            continue
        print("Unknown option.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Player bridge client (tcp + local unix).")
    parser.add_argument("--tcp-host", default=DEFAULT_TCP_HOST)
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--local-socket", default=DEFAULT_LOCAL_PLAYER_SOCKET)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--house", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--bordered", action="store_true")
    parser.add_argument("--no-menu", action="store_true")
    parser.add_argument("--speech", dest="speech_enabled", action="store_true")
    parser.add_argument("--no-speech", dest="speech_enabled", action="store_false")
    parser.set_defaults(speech_enabled=None)
    parser.add_argument("--speech-device", default=None)
    args = parser.parse_args()

    client: Optional[PlayerClient] = None
    mode = "headless"
    speech_device = _parse_device_arg(args.speech_device)
    player_name: Optional[str] = None

    if args.house and not args.no_menu and run_player_menu is not None:
        mode, player_name = run_player_menu()
        if mode == "quit":
            return
        client = PlayerClient(
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            local_socket=args.local_socket,
            speech_enabled=args.speech_enabled,
            speech_device=speech_device,
            player_name=player_name,
        )
        client.start()
    elif args.no_menu or args.interactive or args.view or args.house:
        client = PlayerClient(
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            local_socket=args.local_socket,
            speech_enabled=args.speech_enabled,
            speech_device=speech_device,
            player_name=player_name,
        )
        client.start()
        if args.house:
            mode = "house"
        elif args.view:
            mode = "viewer"
        elif args.interactive:
            mode = "interactive"
    else:
        client, mode = menu_loop(
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            local_socket=args.local_socket,
            speech_enabled=args.speech_enabled,
            speech_device=speech_device,
        )

    if client:
        if mode == "interactive":
            interactive_loop(client)
        elif mode == "viewer":
            if run_viewer is None:
                print("Viewer unavailable (PyQt5 missing or import error).")
            else:
                try:
                    run_viewer(
                        client=client,
                        fullscreen=not args.windowed,
                        borderless=not args.bordered,
                    )
                except Exception as exc:
                    print(f"Viewer failed: {exc}")
        elif mode == "house":
            if run_arch_viewer is None:
                print("House viewer unavailable (PyQt5 missing or import error).")
            else:
                try:
                    run_arch_viewer(
                        client=client,
                        fullscreen=not args.windowed,
                        borderless=not args.bordered,
                    )
                except Exception as exc:
                    print(f"House viewer failed: {exc}")
        else:
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        client.close()


if __name__ == "__main__":
    main()
