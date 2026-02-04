#!/usr/bin/env python3
"""World server with dual sockets + lightweight streaming for OBS."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from world_protocol import (
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    DEFAULT_UNIX_SOCKET,
    clamp,
    safe_json_dumps,
)

try:
    from obs_bridge import OBSWebSocketBridge
except Exception:  # pragma: no cover - optional dependency
    OBSWebSocketBridge = None


LOGGER = logging.getLogger("world_server")


@dataclass
class MotionLimits:
    walk_speed: float = 1.4
    run_speed: float = 2.4
    accel: float = 2.8
    decel: float = 3.2
    turn_rate_deg: float = 110.0
    vertical_speed: float = 0.6
    max_z: float = 2.0


@dataclass
class EntityState:
    entity_id: str
    role: str
    name: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0
    input_forward: float = 0.0
    input_strafe: float = 0.0
    input_up: float = 0.0
    input_turn: float = 0.0
    run: bool = False
    last_seen: float = 0.0
    speech_active: bool = False
    speech_level: float = 0.0
    speech_ratio: float = 0.0
    speech_ts: float = 0.0


class PositionStore:
    def __init__(self, path: str, *, flush_interval: float = 5.0) -> None:
        self.path = Path(path)
        self.flush_interval = float(flush_interval)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"entities": {}, "doors": {}}
        self._dirty = False
        self._last_flush = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {"entities": {}, "doors": {}}
        if not isinstance(self._data, dict):
            self._data = {"entities": {}, "doors": {}}
        self._data.setdefault("entities", {})
        self._data.setdefault("doors", {})

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entities = self._data.get("entities", {})
            record = entities.get(entity_id)
            if isinstance(record, dict):
                return dict(record)
            return None

    def get_door_states(self) -> Dict[str, bool]:
        with self._lock:
            doors = self._data.get("doors", {})
            if not isinstance(doors, dict):
                return {}
            parsed = {}
            for key, value in doors.items():
                state = _parse_door_state(value)
                if state is None:
                    continue
                parsed[str(key)] = state
            return parsed

    def update_entity(
        self,
        *,
        entity_id: str,
        name: str,
        role: str,
        position: Tuple[float, float, float],
        yaw_deg: float,
        last_seen: float,
    ) -> None:
        with self._lock:
            entities = self._data.setdefault("entities", {})
            entities[entity_id] = {
                "entity_id": entity_id,
                "name": name,
                "role": role,
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "yaw_deg": float(yaw_deg),
                "last_seen": float(last_seen),
            }
            self._dirty = True

    def update_door_state(self, door_id: str, open_state: bool) -> None:
        if not door_id:
            return
        with self._lock:
            doors = self._data.setdefault("doors", {})
            if not isinstance(doors, dict):
                doors = {}
                self._data["doors"] = doors
            door_id = str(door_id)
            open_state = bool(open_state)
            if doors.get(door_id) == open_state:
                return
            doors[door_id] = open_state
            self._dirty = True

    def update_from_state(self, state: Dict[str, Any]) -> None:
        entities = state.get("entities", {})
        if isinstance(entities, dict):
            for entity_id, entity in entities.items():
                if not isinstance(entity, dict):
                    continue
                position = entity.get("position")
                if not isinstance(position, (list, tuple)) or len(position) < 3:
                    continue
                try:
                    yaw = float(entity.get("yaw_deg", 0.0))
                except Exception:
                    yaw = 0.0
                self.update_entity(
                    entity_id=str(entity_id),
                    name=str(entity.get("name") or entity_id),
                    role=str(entity.get("role") or "observer"),
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    yaw_deg=yaw,
                    last_seen=float(entity.get("last_seen") or time.time()),
                )
        doors = state.get("doors")
        if isinstance(doors, dict):
            for door_id, raw_state in doors.items():
                parsed = _parse_door_state(raw_state)
                if parsed is None:
                    continue
                self.update_door_state(str(door_id), parsed)

    def maybe_flush(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._dirty:
                return
            if not force and (now - self._last_flush) < self.flush_interval:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
                self._dirty = False
                self._last_flush = now
            except Exception:
                pass


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "player"


def _model_to_server_pos(pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert model coords (x, y-up, z-depth) to server coords (x, y-depth, z-up)."""
    x, y, z = pos
    return (float(x), float(z), float(y))


def _sanitize_position(
    position: Tuple[float, float, float],
    *,
    bounds: Tuple[float, float, float, float],
    max_z: float,
) -> Optional[Tuple[float, float, float]]:
    try:
        x, y, z = float(position[0]), float(position[1]), float(position[2])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    min_x, max_x, min_y, max_y = bounds
    if x < min_x or x > max_x or y < min_y or y > max_y or z < 0.0 or z > max_z:
        return None
    return (x, y, z)


def _parse_door_state(raw_state: Any) -> Optional[bool]:
    state = raw_state
    if isinstance(state, dict):
        state = state.get("open")
    if isinstance(state, bool):
        return state
    if isinstance(state, (int, float)):
        return bool(state)
    if isinstance(state, str):
        normalized = state.strip().lower()
        if normalized in ("open", "opened", "true", "yes", "1"):
            return True
        if normalized in ("closed", "close", "false", "no", "0"):
            return False
    return None


def _parse_spawn_points(value: Any) -> Dict[str, Tuple[float, float, float]]:
    if not isinstance(value, dict):
        return {}
    points: Dict[str, Tuple[float, float, float]] = {}
    for key, raw in value.items():
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue
        try:
            points[str(key)] = (float(raw[0]), float(raw[1]), float(raw[2]))
        except Exception:
            continue
    return points


def _load_spawn_points_from_plan(path: str) -> Dict[str, Tuple[float, float, float]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}

    building = data.get("building", {})
    storeys = building.get("storeys", [])
    storey = storeys[0] if storeys else {}
    spawn_points = _parse_spawn_points(storey.get("spawns"))
    spawn_points = {
        key: _model_to_server_pos(value)
        for key, value in spawn_points.items()
    }

    if "player" not in spawn_points:
        spawn = storey.get("spawn", {})
        position = spawn.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            spawn_points["player"] = _model_to_server_pos(
                (float(position[0]), float(position[1]), float(position[2]))
            )

    if "ina" not in spawn_points:
        ina_spawn = storey.get("ina_spawn")
        if isinstance(ina_spawn, (list, tuple)) and len(ina_spawn) >= 3:
            spawn_points["ina"] = _model_to_server_pos(
                (float(ina_spawn[0]), float(ina_spawn[1]), float(ina_spawn[2]))
            )

    return spawn_points


def _load_bounds_from_plan(path: str) -> Optional[Tuple[float, float, float, float]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    ground = data.get("site", {}).get("ground", {})
    size = ground.get("size")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            half_x = float(size[0]) / 2.0
            half_y = float(size[1]) / 2.0
        except Exception:
            return None
        return (-half_x, half_x, -half_y, half_y)
    return None


class WorldState:
    def __init__(
        self,
        *,
        bounds: Tuple[float, float, float, float],
        tv_channel: str,
        door_states: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._bounds = bounds
        self._tv_channel = tv_channel
        self._entities: Dict[str, EntityState] = {}
        self._doors: Dict[str, bool] = {}
        if door_states:
            self._doors = {str(key): bool(value) for key, value in door_states.items()}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._version = 0

    def ensure_entity(
        self,
        entity_id: str,
        role: str,
        name: str,
        *,
        position: Optional[Tuple[float, float, float]] = None,
        yaw_deg: Optional[float] = None,
    ) -> EntityState:
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                entity = EntityState(entity_id=entity_id, role=role, name=name)
                if position is not None:
                    entity.position = [float(position[0]), float(position[1]), float(position[2])]
                if yaw_deg is not None:
                    entity.yaw_deg = float(yaw_deg) % 360.0
                self._entities[entity_id] = entity
                self._version += 1
                self._condition.notify_all()
            return entity

    def update_input(
        self,
        entity_id: str,
        *,
        forward: float,
        strafe: float,
        up: float,
        turn: float,
        run: bool,
        now: float,
    ) -> None:
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return
            entity.input_forward = clamp(forward, -1.0, 1.0)
            entity.input_strafe = clamp(strafe, -1.0, 1.0)
            entity.input_up = clamp(up, -1.0, 1.0)
            entity.input_turn = clamp(turn, -1.0, 1.0)
            entity.run = bool(run)
            entity.last_seen = now

    def set_pose(
        self,
        entity_id: str,
        *,
        position: Tuple[float, float, float],
        yaw_deg: Optional[float],
        now: float,
        bounds: Tuple[float, float, float, float],
        max_z: float,
    ) -> None:
        min_x, max_x, min_y, max_y = bounds
        x, y, z = position
        x = clamp(float(x), min_x, max_x)
        y = clamp(float(y), min_y, max_y)
        z = clamp(float(z), 0.0, max_z)
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return
            changed = (
                abs(entity.position[0] - x) > 1e-4
                or abs(entity.position[1] - y) > 1e-4
                or abs(entity.position[2] - z) > 1e-4
            )
            entity.position = [x, y, z]
            entity.velocity = [0.0, 0.0, 0.0]
            entity.input_forward = 0.0
            entity.input_strafe = 0.0
            entity.input_up = 0.0
            entity.input_turn = 0.0
            entity.run = False
            if yaw_deg is not None:
                yaw = float(yaw_deg) % 360.0
                if abs(entity.yaw_deg - yaw) > 1e-3:
                    changed = True
                entity.yaw_deg = yaw
            entity.last_seen = now
            if changed:
                self._version += 1
                self._condition.notify_all()

    def set_speech_activity(
        self,
        entity_id: str,
        *,
        active: bool,
        level: float,
        ratio: float,
        now: float,
    ) -> None:
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return
            changed = False
            active = bool(active)
            if entity.speech_active != active:
                entity.speech_active = active
                changed = True
            level = max(0.0, float(level))
            ratio = max(0.0, float(ratio))
            if abs(entity.speech_level - level) > 1e-4:
                entity.speech_level = level
                changed = True
            if abs(entity.speech_ratio - ratio) > 1e-4:
                entity.speech_ratio = ratio
                changed = True
            entity.speech_ts = float(now)
            entity.last_seen = now
            if changed:
                self._version += 1
                self._condition.notify_all()

    def set_tv_channel(self, channel: str) -> None:
        normalized = (channel or "").strip().lower()
        if not normalized:
            return
        with self._lock:
            if normalized == self._tv_channel:
                return
            self._tv_channel = normalized
            self._version += 1
            self._condition.notify_all()

    def set_door_state(self, door_id: str, *, open_state: Optional[bool] = None) -> Optional[Tuple[bool, bool]]:
        if not door_id:
            return None
        door_id = str(door_id)
        with self._lock:
            current = self._doors.get(door_id, False)
            if open_state is None:
                open_state = not current
            open_state = bool(open_state)
            if current == open_state:
                return current, False
            self._doors[door_id] = open_state
            self._version += 1
            self._condition.notify_all()
            return open_state, True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def wait_for_update(self, last_version: int, timeout: float) -> Tuple[Optional[Dict[str, Any]], int]:
        with self._condition:
            if not self._condition.wait_for(lambda: self._version > last_version, timeout=timeout):
                return None, last_version
            return self._snapshot_locked(), self._version

    def step(self, dt: float, limits: MotionLimits) -> Tuple[bool, bool]:
        changed = False
        active = False
        with self._lock:
            for entity in self._entities.values():
                entity_changed, entity_active = _step_entity(entity, dt, limits, self._bounds)
                changed = changed or entity_changed
                active = active or entity_active
            if changed:
                self._version += 1
                self._condition.notify_all()
        return changed, active

    def _snapshot_locked(self) -> Dict[str, Any]:
        return {
            "tv_channel": self._tv_channel,
            "doors": dict(self._doors),
            "bounds": {
                "min_x": self._bounds[0],
                "max_x": self._bounds[1],
                "min_y": self._bounds[2],
                "max_y": self._bounds[3],
            },
            "entities": {
                entity_id: {
                    "role": entity.role,
                    "name": entity.name,
                    "position": [round(val, 4) for val in entity.position],
                    "velocity": [round(val, 4) for val in entity.velocity],
                    "yaw_deg": round(entity.yaw_deg, 2),
                    "run": entity.run,
                    "last_seen": entity.last_seen,
                    "speech_active": entity.speech_active,
                    "speech_level": round(entity.speech_level, 6),
                    "speech_ratio": round(entity.speech_ratio, 4),
                    "speech_ts": entity.speech_ts,
                }
                for entity_id, entity in self._entities.items()
            },
        }


@dataclass
class ClientSession:
    writer: asyncio.StreamWriter
    remote: str
    entity_id: Optional[str] = None
    subscribed: bool = False
    last_state_sent: float = 0.0


class WorldServer:
    def __init__(
        self,
        *,
        unix_socket: str,
        tcp_host: str,
        tcp_port: int,
        stream_host: str,
        stream_port: int,
        bounds: Tuple[float, float, float, float],
        tv_channel: str,
        door_states: Optional[Dict[str, bool]] = None,
        scene_map: Dict[str, str],
        obs_bridge: Optional[Any],
        stream_enabled: bool,
        obs_capture_fps: float = 0.0,
        position_store: Optional[PositionStore] = None,
        spawn_points: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        self._unix_socket = unix_socket
        self._tcp_host = tcp_host
        self._tcp_port = int(tcp_port)
        self._stream_host = stream_host
        self._stream_port = int(stream_port)
        self._clients: Dict[str, ClientSession] = {}
        self._clients_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._tcp_server: Optional[asyncio.AbstractServer] = None
        self._unix_server: Optional[asyncio.AbstractServer] = None
        self._limits = MotionLimits()
        self._state = WorldState(bounds=bounds, tv_channel=tv_channel, door_states=door_states)
        self._scene_map = scene_map
        self._obs_bridge = obs_bridge
        self._simulation_task: Optional[asyncio.Task] = None
        self._state_interval = 0.5
        self._stream_enabled = stream_enabled
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._bounds = bounds
        self._obs_capture_fps = max(0.0, float(obs_capture_fps))
        self._obs_frame_bytes: Optional[bytes] = None
        self._obs_frame_version = 0
        self._obs_frame_lock = threading.Lock()
        self._obs_frame_condition = threading.Condition(self._obs_frame_lock)
        self._obs_capture_stop = threading.Event()
        self._obs_capture_thread: Optional[threading.Thread] = None
        self._position_store = position_store
        self._spawn_points = spawn_points or {}

    @property
    def state(self) -> WorldState:
        return self._state

    async def start(self) -> None:
        await self._start_tcp_server()
        await self._start_unix_server()
        self._start_obs_capture()
        if self._stream_enabled:
            self._start_http_server()
        LOGGER.info("World server ready (tcp=%s:%s unix=%s)", self._tcp_host, self._tcp_port, self._unix_socket)

    async def wait(self) -> None:
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        self._obs_capture_stop.set()
        if self._tcp_server:
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        if self._unix_server:
            self._unix_server.close()
            await self._unix_server.wait_closed()
        if self._http_server:
            self._http_server.shutdown()
        if self._http_thread:
            self._http_thread.join(timeout=2.0)
        if self._obs_capture_thread:
            self._obs_capture_thread.join(timeout=2.0)
        if self._position_store:
            self._position_store.update_from_state(self._state.snapshot())
            self._position_store.maybe_flush(force=True)
        if self._unix_socket and os.path.exists(self._unix_socket):
            try:
                os.remove(self._unix_socket)
            except OSError:
                pass

    async def _start_tcp_server(self) -> None:
        self._tcp_server = await asyncio.start_server(self._handle_client, self._tcp_host, self._tcp_port)

    async def _start_unix_server(self) -> None:
        if self._unix_socket and os.path.exists(self._unix_socket):
            os.remove(self._unix_socket)
        self._unix_server = await asyncio.start_unix_server(self._handle_client, path=self._unix_socket)

    def _start_http_server(self) -> None:
        handler = self._make_http_handler()
        server = ThreadingHTTPServer((self._stream_host, self._stream_port), handler)
        self._http_server = server

        thread = threading.Thread(target=server.serve_forever, name="world_stream", daemon=True)
        thread.start()
        self._http_thread = thread
        LOGGER.info("Streaming server ready (http=%s:%s)", self._stream_host, self._stream_port)

    def _start_obs_capture(self) -> None:
        if self._obs_capture_thread and self._obs_capture_thread.is_alive():
            return
        if self._obs_capture_fps <= 0:
            return
        if not self._obs_bridge or not getattr(self._obs_bridge, "is_available", False):
            return
        self._obs_capture_stop.clear()
        thread = threading.Thread(target=self._capture_obs_loop, name="obs_capture", daemon=True)
        thread.start()
        self._obs_capture_thread = thread

    def _capture_obs_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            LOGGER.warning("OBS capture disabled (cv2 missing): %s", exc)
            return
        interval = 1.0 / max(self._obs_capture_fps, 0.1)
        while not self._obs_capture_stop.is_set():
            frame = None
            if self._obs_bridge:
                try:
                    frame = self._obs_bridge.capture_frame()
                except Exception:
                    frame = None
            if frame is not None:
                try:
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                except Exception:
                    ok = False
                if ok:
                    with self._obs_frame_condition:
                        self._obs_frame_bytes = encoded.tobytes()
                        self._obs_frame_version += 1
                        self._obs_frame_condition.notify_all()
            self._obs_capture_stop.wait(interval)

    def _get_obs_frame(self) -> Tuple[int, Optional[bytes]]:
        with self._obs_frame_lock:
            return self._obs_frame_version, self._obs_frame_bytes

    def _wait_obs_frame(self, last_version: int, timeout: float) -> Tuple[int, Optional[bytes]]:
        with self._obs_frame_condition:
            if self._obs_frame_version <= last_version:
                self._obs_frame_condition.wait(timeout=timeout)
            return self._obs_frame_version, self._obs_frame_bytes

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        remote = self._format_remote(writer)
        session_id = uuid.uuid4().hex
        session = ClientSession(writer=writer, remote=remote)
        async with self._clients_lock:
            self._clients[session_id] = session

        LOGGER.info("Client connected: %s", remote)
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    await self._send(session, {"type": "error", "error": "invalid_json"})
                    continue
                await self._handle_message(session, payload)
        except Exception as exc:
            LOGGER.warning("Client error (%s): %s", remote, exc)
        finally:
            async with self._clients_lock:
                self._clients.pop(session_id, None)
            if self._position_store:
                self._position_store.update_from_state(self._state.snapshot())
                self._position_store.maybe_flush()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            LOGGER.info("Client disconnected: %s", remote)

    async def _handle_message(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        msg_type = (payload.get("type") or "").strip().lower()
        if msg_type == "hello":
            await self._handle_hello(session, payload)
            return
        if msg_type == "subscribe":
            session.subscribed = True
            await self._send_state(session, force=True)
            return
        if msg_type == "state":
            await self._send_state(session, force=True)
            return
        if msg_type == "move":
            await self._handle_move(session, payload)
            return
        if msg_type == "stop":
            await self._handle_stop(session, payload)
            return
        if msg_type == "pose":
            await self._handle_pose(session, payload)
            return
        if msg_type == "set_channel":
            await self._handle_set_channel(session, payload)
            return
        if msg_type == "door":
            await self._handle_door(session, payload)
            return
        if msg_type == "comms":
            await self._handle_comms(session, payload)
            return
        if msg_type == "speech":
            await self._handle_speech(session, payload)
            return
        if msg_type == "ping":
            await self._send(session, {"type": "pong", "ts": time.time()})
            return
        await self._send(session, {"type": "error", "error": "unknown_type", "detail": msg_type})

    async def _handle_hello(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        role = (payload.get("role") or "observer").strip().lower()
        name = payload.get("name") or role
        entity_id = payload.get("entity_id")
        if not entity_id:
            if role == "ina":
                entity_id = "ina"
            elif role == "player":
                entity_id = f"player:{_slugify_name(str(name))}"
            else:
                entity_id = uuid.uuid4().hex[:8]

        position = None
        yaw = None
        if self._position_store:
            record = self._position_store.get(entity_id)
            if record:
                pos = record.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    position = (float(pos[0]), float(pos[1]), float(pos[2]))
                try:
                    yaw = float(record.get("yaw_deg"))
                except Exception:
                    yaw = None
        if position is not None:
            position = _sanitize_position(position, bounds=self._bounds, max_z=self._limits.max_z)
            if position is None:
                yaw = None
        if position is None:
            spawn_key = entity_id
            spawn = self._spawn_points.get(spawn_key)
            if spawn is None and role in self._spawn_points:
                spawn = self._spawn_points.get(role)
            if spawn is None and role == "player":
                spawn = self._spawn_points.get("player")
            if spawn is None and role == "ina":
                spawn = self._spawn_points.get("ina")
            if spawn is not None:
                position = (float(spawn[0]), float(spawn[1]), float(spawn[2]))

        self._state.ensure_entity(entity_id, role=role, name=name, position=position, yaw_deg=yaw)
        session.entity_id = entity_id

        await self._send(
            session,
            {
                "type": "welcome",
                "entity_id": entity_id,
                "state": self._state.snapshot(),
            },
        )
        await self._broadcast_state(force=True)

    async def _handle_move(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        entity_id = payload.get("entity_id") or session.entity_id
        if not entity_id:
            await self._send(session, {"type": "error", "error": "missing_entity"})
            return
        role = payload.get("role") or "observer"
        name = payload.get("name") or entity_id
        self._state.ensure_entity(entity_id, role=role, name=name)
        input_vec = payload.get("input") or {}
        now = time.time()
        self._state.update_input(
            entity_id,
            forward=float(input_vec.get("forward", 0.0)),
            strafe=float(input_vec.get("strafe", 0.0)),
            up=float(input_vec.get("up", 0.0)),
            turn=float(input_vec.get("turn", 0.0)),
            run=bool(payload.get("run", False)),
            now=now,
        )
        await self._send(session, {"type": "ack", "action": "move", "entity_id": entity_id})
        await self._ensure_simulation()

    async def _handle_stop(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        entity_id = payload.get("entity_id") or session.entity_id
        if not entity_id:
            await self._send(session, {"type": "error", "error": "missing_entity"})
            return
        now = time.time()
        self._state.update_input(
            entity_id,
            forward=0.0,
            strafe=0.0,
            up=0.0,
            turn=0.0,
            run=False,
            now=now,
        )
        await self._send(session, {"type": "ack", "action": "stop", "entity_id": entity_id})
        await self._ensure_simulation()

    async def _handle_set_channel(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        channel = (payload.get("channel") or "").strip().lower()
        if not channel:
            await self._send(session, {"type": "error", "error": "missing_channel"})
            return
        self._state.set_tv_channel(channel)
        await self._send(session, {"type": "ack", "action": "set_channel", "channel": channel})
        self._switch_obs_scene(channel)

    async def _handle_door(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        door_id = (payload.get("door_id") or payload.get("id") or "").strip()
        if not door_id:
            await self._send(session, {"type": "error", "error": "missing_door"})
            return
        open_state = payload.get("open")
        if open_state is not None:
            parsed = _parse_door_state(open_state)
            if parsed is None:
                await self._send(session, {"type": "error", "error": "invalid_door_state"})
                return
            open_state = parsed
        result = self._state.set_door_state(door_id, open_state=open_state)
        if result is None:
            await self._send(session, {"type": "error", "error": "invalid_door"})
            return
        open_state, changed = result
        await self._send(
            session,
            {
                "type": "ack",
                "action": "door",
                "door_id": door_id,
                "open": open_state,
            },
        )
        if self._position_store:
            self._position_store.update_door_state(door_id, open_state)
            self._position_store.maybe_flush()
        if changed:
            await self._broadcast_state(force=True)

    async def _handle_comms(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        text = (payload.get("text") or "").strip()
        if not text:
            await self._send(session, {"type": "error", "error": "missing_text"})
            return
        entity_id = payload.get("entity_id") or session.entity_id
        if not entity_id:
            await self._send(session, {"type": "error", "error": "missing_entity"})
            return
        role = payload.get("role") or "observer"
        name = payload.get("name") or entity_id
        self._state.ensure_entity(entity_id, role=role, name=name)
        message = {
            "type": "comms",
            "entity_id": entity_id,
            "name": name,
            "role": role,
            "text": text,
            "timestamp": time.time(),
        }
        await self._broadcast(message)

    async def _handle_speech(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        entity_id = payload.get("entity_id") or session.entity_id
        if not entity_id:
            await self._send(session, {"type": "error", "error": "missing_entity"})
            return
        role = payload.get("role") or "observer"
        name = payload.get("name") or entity_id
        self._state.ensure_entity(entity_id, role=role, name=name)
        now = time.time()
        self._state.set_speech_activity(
            entity_id,
            active=bool(payload.get("active", False)),
            level=float(payload.get("level", 0.0)),
            ratio=float(payload.get("ratio", 0.0)),
            now=now,
        )
        await self._send(session, {"type": "ack", "action": "speech", "entity_id": entity_id})

    async def _handle_pose(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        entity_id = payload.get("entity_id") or session.entity_id
        if not entity_id:
            await self._send(session, {"type": "error", "error": "missing_entity"})
            return
        position = payload.get("position") or payload.get("pos")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            await self._send(session, {"type": "error", "error": "missing_position"})
            return
        role = payload.get("role") or "observer"
        name = payload.get("name") or entity_id
        self._state.ensure_entity(entity_id, role=role, name=name)

        now = time.time()
        self._state.set_pose(
            entity_id,
            position=(float(position[0]), float(position[1]), float(position[2])),
            yaw_deg=payload.get("yaw_deg"),
            now=now,
            bounds=self._bounds,
            max_z=self._limits.max_z,
        )
        if self._position_store:
            min_x, max_x, min_y, max_y = self._bounds
            x = clamp(float(position[0]), min_x, max_x)
            y = clamp(float(position[1]), min_y, max_y)
            z = clamp(float(position[2]), 0.0, self._limits.max_z)
            yaw = payload.get("yaw_deg")
            if yaw is None:
                record = self._position_store.get(entity_id)
                if record:
                    yaw = record.get("yaw_deg", 0.0)
            try:
                yaw = float(yaw) % 360.0
            except Exception:
                yaw = 0.0
            self._position_store.update_entity(
                entity_id=entity_id,
                name=str(name),
                role=str(role),
                position=(x, y, z),
                yaw_deg=yaw,
                last_seen=now,
            )
            self._position_store.maybe_flush()

    async def _send(self, session: ClientSession, payload: Dict[str, Any]) -> None:
        data = safe_json_dumps(payload) + "\n"
        session.writer.write(data.encode("utf-8"))
        await session.writer.drain()

    async def _send_state(self, session: ClientSession, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - session.last_state_sent) < self._state_interval:
            return
        session.last_state_sent = now
        await self._send(session, {"type": "state", "state": self._state.snapshot()})

    async def _broadcast_state(self, *, force: bool = False) -> None:
        async with self._clients_lock:
            sessions = list(self._clients.values())
        for session in sessions:
            if session.subscribed:
                try:
                    await self._send_state(session, force=force)
                except Exception:
                    continue

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        async with self._clients_lock:
            sessions = list(self._clients.values())
        for session in sessions:
            try:
                await self._send(session, payload)
            except Exception:
                continue

    async def _ensure_simulation(self) -> None:
        if self._simulation_task and not self._simulation_task.done():
            return
        self._simulation_task = asyncio.create_task(self._simulation_loop())

    async def _simulation_loop(self) -> None:
        tick_interval = 0.1
        last_ts = time.monotonic()
        while True:
            await asyncio.sleep(tick_interval)
            now = time.monotonic()
            dt = max(0.01, min(0.2, now - last_ts))
            last_ts = now
            changed, active = self._state.step(dt, self._limits)
            if changed:
                if self._position_store:
                    self._position_store.update_from_state(self._state.snapshot())
                    self._position_store.maybe_flush()
                await self._broadcast_state()
            if not active:
                break

    def _switch_obs_scene(self, channel: str) -> None:
        if not self._obs_bridge:
            return
        scene_name = self._scene_map.get(channel, channel)
        if not scene_name:
            return
        try:
            ok = self._obs_bridge.set_program_scene(scene_name)
            if ok:
                LOGGER.info("OBS scene switched to: %s", scene_name)
            else:
                LOGGER.warning("OBS scene switch failed: %s", scene_name)
        except Exception as exc:
            LOGGER.warning("OBS scene switch error: %s", exc)

    def _make_http_handler(self):
        state = self._state
        obs_get = self._get_obs_frame
        obs_wait = self._wait_obs_frame

        class StreamHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                LOGGER.info("stream %s - %s", self.address_string(), fmt % args)

            def do_GET(self) -> None:  # noqa: N802 - signature required by BaseHTTPRequestHandler
                parsed = urlparse(self.path)
                if parsed.path == "/state":
                    payload = safe_json_dumps(state.snapshot()).encode("utf-8")
                    self._send_bytes(payload, content_type="application/json")
                    return
                if parsed.path.startswith("/channel/"):
                    channel = parsed.path.split("/channel/", 1)[1]
                    self._send_channel_html(channel)
                    return
                if parsed.path == "/stream":
                    params = parse_qs(parsed.query)
                    channel = (params.get("channel") or [""])[0]
                    self._handle_stream(channel)
                    return
                if parsed.path == "/obs.jpg":
                    self._send_obs_frame()
                    return
                if parsed.path == "/obs.mjpeg":
                    self._stream_obs_mjpeg()
                    return
                if parsed.path == "/":
                    self._send_channel_html("world")
                    return
                self.send_error(404, "Not found")

            def _send_bytes(self, data: bytes, *, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_channel_html(self, channel: str) -> None:
                html = _render_channel_html(channel)
                self._send_bytes(html.encode("utf-8"), content_type="text/html; charset=utf-8")

            def _handle_stream(self, channel: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                last_version = 0
                while True:
                    snapshot, last_version = state.wait_for_update(last_version, timeout=5.0)
                    if snapshot is None:
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except BrokenPipeError:
                            break
                        continue
                    payload = _filter_state_for_channel(snapshot, channel)
                    data = safe_json_dumps(payload)
                    msg = f"event: state\ndata: {data}\n\n".encode("utf-8")
                    try:
                        self.wfile.write(msg)
                        self.wfile.flush()
                    except BrokenPipeError:
                        break

            def _send_obs_frame(self) -> None:
                _version, data = obs_get()
                if not data:
                    self.send_error(503, "OBS frame unavailable")
                    return
                self._send_bytes(data, content_type="image/jpeg")

            def _stream_obs_mjpeg(self) -> None:
                boundary = "obsframe"
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                last_version = 0
                while True:
                    version, data = obs_wait(last_version, timeout=5.0)
                    if not data or version == last_version:
                        continue
                    last_version = version
                    header = (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(data)}\r\n\r\n"
                    )
                    try:
                        self.wfile.write(header.encode("utf-8"))
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except BrokenPipeError:
                        break

        return StreamHandler

    @staticmethod
    def _format_remote(writer: asyncio.StreamWriter) -> str:
        peer = writer.get_extra_info("peername")
        if not peer:
            return "unix"
        if isinstance(peer, tuple):
            return f"{peer[0]}:{peer[1]}"
        return str(peer)


# ------------------------------ Helpers ------------------------------ #


def _step_entity(
    entity: EntityState,
    dt: float,
    limits: MotionLimits,
    bounds: Tuple[float, float, float, float],
) -> Tuple[bool, bool]:
    min_x, max_x, min_y, max_y = bounds

    forward = clamp(entity.input_forward, -1.0, 1.0)
    strafe = clamp(entity.input_strafe, -1.0, 1.0)
    up = clamp(entity.input_up, -1.0, 1.0)
    turn = clamp(entity.input_turn, -1.0, 1.0)

    input_mag = math.sqrt(forward * forward + strafe * strafe)
    if input_mag > 1.0:
        forward /= input_mag
        strafe /= input_mag
        input_mag = 1.0

    yaw_delta = turn * limits.turn_rate_deg * dt
    entity.yaw_deg = (entity.yaw_deg + yaw_delta) % 360.0

    speed_target = limits.run_speed if entity.run else limits.walk_speed
    local_x = strafe * speed_target
    local_y = forward * speed_target
    yaw_rad = math.radians(entity.yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    target_vx = local_x * cos_y - local_y * sin_y
    target_vy = local_x * sin_y + local_y * cos_y
    target_vz = up * limits.vertical_speed

    accel = limits.accel if input_mag > 0.001 or abs(up) > 0.001 else limits.decel
    max_delta = accel * dt

    delta_x = target_vx - entity.velocity[0]
    delta_y = target_vy - entity.velocity[1]
    delta_z = target_vz - entity.velocity[2]
    delta_mag = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

    if delta_mag > max_delta and delta_mag > 0:
        scale = max_delta / delta_mag
        delta_x *= scale
        delta_y *= scale
        delta_z *= scale

    entity.velocity[0] += delta_x
    entity.velocity[1] += delta_y
    entity.velocity[2] += delta_z

    # Clamp horizontal speed to limit.
    horiz_speed = math.sqrt(entity.velocity[0] ** 2 + entity.velocity[1] ** 2)
    if horiz_speed > speed_target > 0:
        scale = speed_target / horiz_speed
        entity.velocity[0] *= scale
        entity.velocity[1] *= scale

    # Snap tiny velocities to zero for stability.
    if input_mag < 0.01 and horiz_speed < 0.02:
        entity.velocity[0] = 0.0
        entity.velocity[1] = 0.0
    if abs(up) < 0.01 and abs(entity.velocity[2]) < 0.01:
        entity.velocity[2] = 0.0

    old_pos = tuple(entity.position)
    entity.position[0] += entity.velocity[0] * dt
    entity.position[1] += entity.velocity[1] * dt
    entity.position[2] += entity.velocity[2] * dt

    entity.position[0] = clamp(entity.position[0], min_x, max_x)
    entity.position[1] = clamp(entity.position[1], min_y, max_y)
    entity.position[2] = clamp(entity.position[2], 0.0, limits.max_z)

    if entity.position[0] in (min_x, max_x):
        entity.velocity[0] = 0.0
    if entity.position[1] in (min_y, max_y):
        entity.velocity[1] = 0.0
    if entity.position[2] in (0.0, limits.max_z):
        entity.velocity[2] = 0.0

    changed = (
        abs(entity.position[0] - old_pos[0]) > 1e-4
        or abs(entity.position[1] - old_pos[1]) > 1e-4
        or abs(entity.position[2] - old_pos[2]) > 1e-4
        or abs(yaw_delta) > 1e-4
    )
    active = (
        input_mag > 0.001
        or abs(up) > 0.001
        or abs(turn) > 0.001
        or abs(entity.velocity[0]) > 0.001
        or abs(entity.velocity[1]) > 0.001
        or abs(entity.velocity[2]) > 0.001
    )
    return changed, active


def _filter_state_for_channel(state: Dict[str, Any], channel: str) -> Dict[str, Any]:
    if not channel or channel == "world":
        return state
    if channel == "tv":
        return {"tv_channel": state.get("tv_channel")}
    entities = state.get("entities", {})
    if channel in entities:
        return {"entity": entities[channel], "entity_id": channel}
    return state


def _render_channel_html(channel: str) -> str:
    channel = channel.strip() or "world"
    safe_channel = channel.replace("\"", "")
    if safe_channel == "obs":
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Inazuma Channel - OBS</title>
  <style>
    :root {{
      color-scheme: only light;
    }}
    body {{
      margin: 0;
      background: #0a0a0f;
      color: #f5f2e9;
      font-family: "Cascadia Mono", "Consolas", monospace;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
    }}
    .frame {{
      max-width: 96vw;
      max-height: 96vh;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 12px;
      box-shadow: 0 12px 26px rgba(0, 0, 0, 0.35);
      background: #101318;
    }}
    .hint {{
      position: absolute;
      bottom: 16px;
      left: 18px;
      color: rgba(245, 242, 233, 0.6);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <img class="frame" id="obsFrame" src="/obs.mjpeg" alt="OBS feed" />
  <div class="hint">OBS feed via websocket (/{safe_channel})</div>
  <script>
    const img = document.getElementById("obsFrame");
    img.onerror = () => {{
      img.src = "/obs.jpg?ts=" + Date.now();
    }};
  </script>
</body>
</html>"""
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Inazuma Channel - {safe_channel}</title>
  <style>
    :root {{
      color-scheme: only light;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #1b1d27, #09090c 70%);
      color: #f5f2e9;
      font-family: "Cascadia Mono", "Consolas", monospace;
    }}
    .wrap {{
      padding: 24px 32px;
      min-height: 100vh;
      box-sizing: border-box;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .card {{
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 12px;
      padding: 16px 18px;
      background: rgba(16, 18, 24, 0.7);
      box-shadow: 0 12px 26px rgba(0, 0, 0, 0.35);
      max-width: 520px;
    }}
    .stat {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .stat:last-child {{
      border-bottom: none;
    }}
    .label {{
      color: rgba(245, 242, 233, 0.7);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
    }}
    .value {{
      font-size: 14px;
      text-align: right;
    }}
    .hint {{
      margin-top: 14px;
      font-size: 12px;
      color: rgba(245, 242, 233, 0.5);
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Channel: {safe_channel}</h1>
    <div class=\"card\" id=\"card\">
      <div class=\"stat\"><div class=\"label\">status</div><div class=\"value\" id=\"status\">waiting for data...</div></div>
    </div>
    <div class=\"hint\">Stream: /stream?channel={safe_channel}</div>
  </div>
  <script>
    const channel = "{safe_channel}";
    const statusEl = document.getElementById("status");
    const cardEl = document.getElementById("card");

    function renderState(data) {{
      if (!data) {{
        statusEl.textContent = "no data";
        return;
      }}
      if (channel === "tv") {{
        statusEl.textContent = data.tv_channel || "unknown";
        return;
      }}
      if (data.entity) {{
        statusEl.textContent = data.entity.name || channel;
        renderEntity(data.entity);
        return;
      }}
      if (data.entities) {{
        statusEl.textContent = "world";
        renderWorld(data.entities, data.tv_channel);
        return;
      }}
      statusEl.textContent = "unknown payload";
    }}

    function clearStats() {{
      const stats = cardEl.querySelectorAll(".stat");
      stats.forEach((item, index) => {{
        if (index > 0) item.remove();
      }});
    }}

    function addStat(label, value) {{
      const row = document.createElement("div");
      row.className = "stat";
      row.innerHTML = "<div class=\\\"label\\\">" + label + "</div><div class=\\\"value\\\">" + value + "</div>";
      cardEl.appendChild(row);
    }}

    function renderEntity(entity) {{
      clearStats();
      addStat("position", entity.position ? entity.position.join(", ") : "-");
      addStat("velocity", entity.velocity ? entity.velocity.join(", ") : "-");
      addStat("yaw", entity.yaw_deg ?? "-");
      addStat("run", entity.run ? "yes" : "no");
      addStat("speech", entity.speech_active ? "talking" : "silent");
      addStat("speech_lvl", entity.speech_level ?? "-");
    }}

    function renderWorld(entities, tvChannel) {{
      clearStats();
      addStat("tv", tvChannel || "unknown");
      Object.keys(entities).forEach((key) => {{
        const entity = entities[key];
        const label = (entity.name || key) + " (" + key + ")";
        const pos = entity.position ? entity.position.join(", ") : "-";
        addStat(label, pos);
      }});
    }}

    const evt = new EventSource(`/stream?channel=${encodeURIComponent(channel)}`);
    evt.addEventListener("state", (event) => {{
      try {{
        renderState(JSON.parse(event.data));
      }} catch (err) {{
        statusEl.textContent = "bad data";
      }}
    }});
    evt.onerror = () => {{
      statusEl.textContent = "stream offline";
    }};
  </script>
</body>
</html>"""


def _load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _parse_scene_map(items: Optional[list[str]]) -> Dict[str, str]:
    scene_map: Dict[str, str] = {}
    if not items:
        return scene_map
    for item in items:
        if "=" not in item:
            continue
        channel, scene = item.split("=", 1)
        channel = channel.strip().lower()
        scene = scene.strip()
        if channel and scene:
            scene_map[channel] = scene
    return scene_map


async def _run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    world_cfg = config.get("world")
    if not isinstance(world_cfg, dict):
        world_cfg = {}
    tv_channel = (
        config.get("tv_settings", {}).get("active_channel")
        if isinstance(config.get("tv_settings"), dict)
        else None
    )
    tv_channel = tv_channel or "spotify"

    positions_path = world_cfg.get("positions_path", "world_positions.json")
    if positions_path:
        try:
            flush_interval = float(world_cfg.get("positions_flush_interval", 5.0))
        except Exception:
            flush_interval = 5.0
        position_store = PositionStore(str(positions_path), flush_interval=flush_interval)
    else:
        position_store = None
    door_states = position_store.get_door_states() if position_store else {}

    spawn_points = {}
    plan_path = world_cfg.get("house_plan_path", "ina_house_plan.json")
    if plan_path:
        spawn_points.update(_load_spawn_points_from_plan(str(plan_path)))
    spawn_points.update(_parse_spawn_points(world_cfg.get("spawn_points")))

    obs_bridge = None
    obs_capture_fps = 0.0
    if OBSWebSocketBridge and args.obs_enabled:
        obs_cfg = config.get("obs_websocket")
        if isinstance(obs_cfg, dict):
            obs_cfg = dict(obs_cfg)
            obs_cfg["enabled"] = True
        else:
            obs_cfg = {"enabled": True}
        obs_bridge = OBSWebSocketBridge.from_config(obs_cfg, logger=LOGGER.info)
        if isinstance(obs_cfg, dict):
            try:
                obs_capture_fps = float(obs_cfg.get("capture_fps", 2.0))
            except Exception:
                obs_capture_fps = 2.0

    scene_map = _parse_scene_map(args.obs_scene)
    bounds_cfg = world_cfg.get("bounds")
    if isinstance(bounds_cfg, (list, tuple)) and len(bounds_cfg) >= 4:
        try:
            state_bounds = (
                float(bounds_cfg[0]),
                float(bounds_cfg[1]),
                float(bounds_cfg[2]),
                float(bounds_cfg[3]),
            )
        except Exception:
            state_bounds = None
    else:
        state_bounds = None
    if state_bounds is None:
        state_bounds = _load_bounds_from_plan(str(plan_path)) or (-10.0, 10.0, -10.0, 10.0)

    server = WorldServer(
        unix_socket=args.unix_socket,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
        bounds=state_bounds,
        tv_channel=tv_channel,
        door_states=door_states,
        scene_map=scene_map,
        obs_bridge=obs_bridge,
        stream_enabled=args.stream_enabled,
        obs_capture_fps=obs_capture_fps,
        position_store=position_store,
        spawn_points=spawn_points,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    await server.start()
    await server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inazuma world server (unix + tcp + streaming).")
    parser.add_argument("--unix-socket", default=DEFAULT_UNIX_SOCKET)
    parser.add_argument("--tcp-host", default=DEFAULT_TCP_HOST)
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--stream-host", default=DEFAULT_STREAM_HOST)
    parser.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT)
    parser.add_argument("--stream", dest="stream_enabled", action="store_true")
    parser.add_argument("--no-stream", dest="stream_enabled", action="store_false")
    parser.set_defaults(stream_enabled=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--obs-enabled", action="store_true", default=False)
    parser.add_argument(
        "--obs-scene",
        action="append",
        help="Map channel to OBS scene: channel=Scene Name (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
