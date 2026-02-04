#!/usr/bin/env python3
"""Local Ina client to drive movement over the unix socket."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from world_protocol import DEFAULT_UNIX_SOCKET, safe_json_dumps

try:
    from model_manager import update_inastate
except Exception:  # pragma: no cover - optional dependency
    update_inastate = None

try:
    from motor_controls import MotorController, MotorInput
except Exception:  # pragma: no cover - optional dependency
    MotorController = None
    MotorInput = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> dict:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_inastate_path() -> Path:
    config = _load_config()
    child = config.get("current_child", "Inazuma_Yagami")
    return Path("AI_Children") / str(child) / "memory" / "inastate.json"


def _load_motor_poll_interval() -> float:
    config = _load_config()
    try:
        value = float(config.get("ina_motor_poll_interval", 0.5))
    except Exception:
        value = 0.5
    return max(0.2, min(value, 5.0))


class InaClient:
    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self.sock: Optional[socket.socket] = None
        self.file = None
        self._reader_thread: Optional[threading.Thread] = None
        self._manager_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._intent_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._send_lock = threading.Lock()
        self._motor = MotorController() if MotorController is not None else None
        self._intent_poll_interval = _load_motor_poll_interval()
        self._inastate_path = _resolve_inastate_path()
        self._intent_last_mtime: Optional[float] = None
        self._intent_last_key: Optional[tuple] = None
        self._intent_pending_stop: Optional[float] = None
        self._world_bounds: Optional[tuple[float, float, float, float]] = None
        self._touch_last_key: Optional[tuple] = None
        self._touch_last_ts = 0.0

    def connect(self) -> None:
        self._connect_once()

    def start(self, retry_interval: float = 300.0) -> None:
        if self._manager_thread and self._manager_thread.is_alive():
            return
        self._manager_thread = threading.Thread(
            target=self._connection_loop,
            args=(retry_interval,),
            daemon=True,
        )
        self._manager_thread.start()

    def _connection_loop(self, retry_interval: float) -> None:
        retry_interval = max(1.0, float(retry_interval))
        while not self._stop_event.is_set():
            try:
                self._connect_once()
            except OSError:
                self._mark_disconnected()
                self._sleep_with_stop(retry_interval)
                continue

            if self._reader_thread:
                self._reader_thread.join()

            self._mark_disconnected()
            if self._stop_event.is_set():
                break
            self._sleep_with_stop(retry_interval)

    def _connect_once(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self.sock = sock
        self.file = sock.makefile("rwb")
        self.send({"type": "hello", "role": "ina", "name": "Ina"})
        self.send({"type": "subscribe"})
        self._mark_connected()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._start_heartbeat()
        self._start_intent_loop()

    def close(self) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None
            self.file = None
        self._mark_disconnected()

    def send(self, payload: Dict[str, Any]) -> None:
        data = safe_json_dumps(payload).encode("utf-8") + b"\n"
        with self._send_lock:
            if not self.file:
                return
            try:
                self.file.write(data)
                self.file.flush()
            except OSError:
                self._handle_disconnect()

    def move(
        self,
        *,
        forward: float = 0.0,
        strafe: float = 0.0,
        up: float = 0.0,
        turn: float = 0.0,
        run: bool = False,
        duration: float = 0.0,
    ) -> None:
        if self._motor is not None and MotorInput is not None:
            try:
                motor_input = MotorInput(
                    forward=forward,
                    strafe=strafe,
                    up=up,
                    turn=turn,
                    run=run,
                    jump=up > 0.5,
                )
                self._motor.step(motor_input)
            except Exception:
                pass
        self.send(
            {
                "type": "move",
                "input": {
                    "forward": forward,
                    "strafe": strafe,
                    "up": up,
                    "turn": turn,
                },
                "run": run,
            }
        )
        if duration > 0:
            time.sleep(duration)
            self.stop()

    def stop(self) -> None:
        self.send({"type": "stop"})

    def request_state(self) -> None:
        self.send({"type": "state"})

    def set_channel(self, channel: str) -> None:
        self.send({"type": "set_channel", "channel": channel})

    def send_comms(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.send({"type": "comms", "text": text})

    def _read_loop(self) -> None:
        if not self.file:
            return
        while not self._stop_event.is_set():
            try:
                line = self.file.readline()
            except OSError:
                break
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "state":
                state = payload.get("state")
                if isinstance(state, dict):
                    self._handle_state(state)
            self._print_payload(payload)
        self._handle_disconnect()

    @staticmethod
    def _print_payload(payload: Dict[str, Any]) -> None:
        msg_type = payload.get("type")
        if msg_type == "state":
            entities = payload.get("state", {}).get("entities", {})
            ina = entities.get("ina")
            if ina:
                pos = ina.get("position")
                yaw = ina.get("yaw_deg")
                print(f"[state] Ina pos={pos} yaw={yaw}")
                return
        if msg_type == "comms":
            name = payload.get("name") or payload.get("entity_id") or "unknown"
            text = payload.get("text") or ""
            print(f"[comms] {name}: {text}")
            return
        print(f"[server] {payload}")

    def _handle_state(self, state: Dict[str, Any]) -> None:
        bounds = state.get("bounds")
        if self._motor is not None and isinstance(bounds, dict):
            try:
                min_x = float(bounds.get("min_x", self._motor.bounds[0]))
                max_x = float(bounds.get("max_x", self._motor.bounds[1]))
                min_y = float(bounds.get("min_y", self._motor.bounds[2]))
                max_y = float(bounds.get("max_y", self._motor.bounds[3]))
                self._motor.bounds = (min_x, max_x, min_y, max_y)
                self._world_bounds = (min_x, max_x, min_y, max_y)
            except Exception:
                pass

        entities = state.get("entities") or {}
        ina = entities.get("ina")
        if not ina:
            return
        pos = ina.get("position")
        vel = ina.get("velocity")
        yaw = ina.get("yaw_deg")
        if self._motor is not None and isinstance(pos, (list, tuple)) and len(pos) >= 3:
            try:
                velocity = (0.0, 0.0, 0.0)
                if isinstance(vel, (list, tuple)) and len(vel) >= 3:
                    velocity = (float(vel[0]), float(vel[1]), float(vel[2]))
                self._motor.observe_state(
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    velocity=velocity,
                    yaw_deg=float(yaw) if yaw is not None else None,
                )
            except Exception:
                pass
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            self._update_world_touch(pos)

    def _mark_connected(self) -> None:
        self._connected_event.set()
        self._set_inastate("world_connected", True)
        self._set_inastate("last_world_heartbeat", _now_iso())

    def _mark_disconnected(self) -> None:
        self._connected_event.clear()
        self._set_inastate("world_connected", False)
        self._set_inastate("last_world_disconnect", _now_iso())

    def _handle_disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.file = None
        self._mark_disconnected()

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _start_intent_loop(self) -> None:
        if self._intent_thread and self._intent_thread.is_alive():
            return
        self._intent_thread = threading.Thread(target=self._intent_loop, daemon=True)
        self._intent_thread.start()

    def _intent_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._intent_pending_stop is not None and now >= self._intent_pending_stop:
                self.stop()
                self._intent_pending_stop = None

            if not self._connected_event.is_set():
                self._sleep_with_stop(self._intent_poll_interval)
                continue

            intent = self._read_motor_intent()
            if intent:
                self._apply_motor_intent(intent)

            self._sleep_with_stop(self._intent_poll_interval)

    def _read_motor_intent(self) -> Optional[Dict[str, Any]]:
        path = self._inastate_path
        try:
            stat = path.stat()
        except OSError:
            return None
        mtime = stat.st_mtime
        if self._intent_last_mtime is not None and mtime <= self._intent_last_mtime:
            return None
        self._intent_last_mtime = mtime
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        intent = state.get("motor_intent")
        if not isinstance(intent, dict):
            return None
        seq = intent.get("seq")
        try:
            seq_val = int(seq) if seq is not None else None
        except Exception:
            seq_val = None
        key = (
            seq_val,
            intent.get("forward"),
            intent.get("strafe"),
            intent.get("turn"),
            intent.get("up"),
            intent.get("run"),
            intent.get("duration"),
            intent.get("stop"),
            intent.get("action"),
        )
        if self._intent_last_key == key:
            return None
        self._intent_last_key = key
        return dict(intent)

    def _apply_motor_intent(self, intent: Dict[str, Any]) -> None:
        stop_flag = bool(intent.get("stop")) or str(intent.get("action") or "").lower() == "stop"
        try:
            forward = float(intent.get("forward", 0.0))
            strafe = float(intent.get("strafe", 0.0))
            turn = float(intent.get("turn", 0.0))
            up = float(intent.get("up", 0.0))
        except Exception:
            forward = strafe = turn = up = 0.0
        run = bool(intent.get("run", False))
        try:
            duration = float(intent.get("duration", 0.0))
        except Exception:
            duration = 0.0
        intent_snapshot = {
            "forward": forward,
            "strafe": strafe,
            "turn": turn,
            "up": up,
            "run": run,
            "duration": duration,
            "stop": stop_flag,
            "seq": intent.get("seq"),
            "timestamp": _now_iso(),
        }
        self._set_inastate("last_motor_intent", intent_snapshot)

        if stop_flag or (abs(forward) < 1e-3 and abs(strafe) < 1e-3 and abs(turn) < 1e-3 and abs(up) < 1e-3):
            self.stop()
            self._intent_pending_stop = None
            return

        if self._motor is not None and MotorInput is not None:
            try:
                self._motor.step(
                    MotorInput(
                        forward=forward,
                        strafe=strafe,
                        up=up,
                        turn=turn,
                        run=run,
                        jump=up > 0.5,
                    )
                )
            except Exception:
                pass

        self.send(
            {
                "type": "move",
                "input": {
                    "forward": forward,
                    "strafe": strafe,
                    "up": up,
                    "turn": turn,
                },
                "run": run,
            }
        )
        if duration > 0:
            self._intent_pending_stop = time.monotonic() + duration
        else:
            self._intent_pending_stop = None

    def _update_world_touch(self, pos: list[Any]) -> None:
        if update_inastate is None:
            return
        if self._world_bounds is None:
            return
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            return
        min_x, max_x, min_y, max_y = self._world_bounds
        threshold = 0.08

        contacts = []

        def add_contact(surface: str, distance: float) -> None:
            pressure = 1.0 - max(0.0, min(distance / threshold, 1.0))
            contacts.append({"surface": surface, "pressure": round(pressure, 4)})

        ground_dist = z - 0.0
        if ground_dist <= threshold:
            add_contact("ground", ground_dist)
        if (x - min_x) <= threshold:
            add_contact("bounds_min_x", x - min_x)
        if (max_x - x) <= threshold:
            add_contact("bounds_max_x", max_x - x)
        if (y - min_y) <= threshold:
            add_contact("bounds_min_y", y - min_y)
        if (max_y - y) <= threshold:
            add_contact("bounds_max_y", max_y - y)

        grounded = z <= threshold
        key = (grounded, tuple((c["surface"], c["pressure"]) for c in contacts))
        now = time.monotonic()
        if self._touch_last_key == key and (now - self._touch_last_ts) < 5.0:
            return
        self._touch_last_key = key
        self._touch_last_ts = now
        self._set_inastate(
            "touch_world",
            {
                "grounded": grounded,
                "contacts": contacts,
                "timestamp": _now_iso(),
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._connected_event.is_set():
                return
            self.send({"type": "ping"})
            self._set_inastate("last_world_heartbeat", _now_iso())
            self._sleep_with_stop(15.0)

    def _sleep_with_stop(self, duration: float) -> None:
        end = time.monotonic() + duration
        while not self._stop_event.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))

    def _set_inastate(self, key: str, value: Any) -> None:
        if update_inastate is None:
            return
        try:
            update_inastate(key, value)
        except Exception:
            pass


def _parse_move_command(tokens: list[str]) -> Dict[str, Any]:
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


def _interactive_loop(client: InaClient) -> None:
    print("Ina client ready. Commands: move f s [turn] [run] [duration], stop, state, channel <name>, say <text>, quit")
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
            client.stop()
            continue
        if cmd == "state":
            client.request_state()
            continue
        if cmd == "channel" and len(tokens) > 1:
            client.set_channel(" ".join(tokens[1:]))
            continue
        if cmd == "say" and len(tokens) > 1:
            client.send_comms(" ".join(tokens[1:]))
            continue

        move_args = _parse_move_command(tokens)
        if move_args:
            run = bool(move_args.pop("run", False))
            duration = float(move_args.pop("duration", 0.0))
            client.move(run=run, duration=duration, **move_args)
            continue

        print("Unknown command.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ina local client (unix socket).")
    parser.add_argument("--socket", default=DEFAULT_UNIX_SOCKET)
    parser.add_argument("--forward", type=float)
    parser.add_argument("--strafe", type=float)
    parser.add_argument("--turn", type=float)
    parser.add_argument("--up", type=float)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--retry-interval", type=float, default=300.0)
    args = parser.parse_args()

    client = InaClient(args.socket)

    if args.daemon:
        client.start(retry_interval=args.retry_interval)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    elif args.interactive or all(val is None for val in (args.forward, args.strafe, args.turn, args.up)):
        client.start(retry_interval=args.retry_interval)
        _interactive_loop(client)
    else:
        client.connect()
        client.move(
            forward=args.forward or 0.0,
            strafe=args.strafe or 0.0,
            turn=args.turn or 0.0,
            up=args.up or 0.0,
            run=args.run,
            duration=args.duration,
        )

    client.close()


if __name__ == "__main__":
    main()
