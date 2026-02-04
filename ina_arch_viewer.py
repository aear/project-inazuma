#!/usr/bin/env python3
"""House viewer first-person camera for Ina's vision feed."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover - optional dependency
    QtCore = None
    QtWidgets = None
    _QT_IMPORT_ERROR = exc
else:
    _QT_IMPORT_ERROR = None

try:
    from house_viewer import HouseViewer
except Exception as exc:  # pragma: no cover - optional dependency
    HouseViewer = None
    _HOUSE_IMPORT_ERROR = exc
else:
    _HOUSE_IMPORT_ERROR = None

try:
    from model_manager import update_inastate
except Exception:  # pragma: no cover - optional dependency
    update_inastate = None

from world_protocol import DEFAULT_TCP_HOST, DEFAULT_TCP_PORT, safe_json_dumps


class WorldObserver:
    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self.sock: Optional[socket.socket] = None
        self.file = None
        self._reader_thread: Optional[threading.Thread] = None
        self._manager_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_state: Optional[Dict[str, Any]] = None

    def start(self, retry_interval: float = 300.0) -> None:
        if self._manager_thread and self._manager_thread.is_alive():
            return
        self._manager_thread = threading.Thread(
            target=self._connection_loop,
            args=(retry_interval,),
            daemon=True,
        )
        self._manager_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.file = None

    def get_state_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._last_state is None:
                return None
            return json.loads(json.dumps(self._last_state))

    def _connection_loop(self, retry_interval: float) -> None:
        retry_interval = max(1.0, float(retry_interval))
        while not self._stop_event.is_set():
            try:
                self._connect_once()
            except OSError:
                self._sleep_with_stop(retry_interval)
                continue
            if self._reader_thread:
                self._reader_thread.join()
            if self._stop_event.is_set():
                break
            self._sleep_with_stop(retry_interval)

    def _connect_once(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=5)
        sock.settimeout(None)
        self.sock = sock
        self.file = sock.makefile("rwb")
        self._send({"type": "hello", "role": "observer", "name": "InaVision"})
        self._send({"type": "subscribe"})
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self.file:
            return
        data = safe_json_dumps(payload).encode("utf-8") + b"\n"
        try:
            self.file.write(data)
            self.file.flush()
        except OSError:
            pass

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
                    with self._state_lock:
                        self._last_state = state

    def _sleep_with_stop(self, duration: float) -> None:
        end = time.monotonic() + duration
        while not self._stop_event.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))


class InaVisionBridge(QtCore.QObject):
    def __init__(
        self,
        *,
        observer: WorldObserver,
        fullscreen: bool = True,
        borderless: bool = True,
        tick_ms: int = 100,
    ) -> None:
        super().__init__()
        if HouseViewer is None:
            raise RuntimeError(f"HouseViewer unavailable: {_HOUSE_IMPORT_ERROR}")
        self.observer = observer
        self.viewer = HouseViewer()
        self.fullscreen = fullscreen
        self.borderless = borderless
        self.tick_ms = tick_ms
        self._last_sync = 0.0
        self._last_frame_ts = 0.0
        self._frame_interval = 1.0
        self._vision_path = _resolve_vision_path()
        self._door_state_synced = False

        self._init_window()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.tick_ms)
        self.viewer.installEventFilter(self)
        self.viewer.view.installEventFilter(self)

    def _init_window(self) -> None:
        if self.borderless:
            self.viewer.setWindowFlags(self.viewer.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.viewer.statusBar().setVisible(False)
        for toolbar in self.viewer.findChildren(QtWidgets.QToolBar):
            toolbar.hide()
        for dock in self.viewer.findChildren(QtWidgets.QDockWidget):
            dock.hide()
        self.viewer.setFocusPolicy(QtCore.Qt.NoFocus)
        self.viewer.view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.viewer.view.setMouseTracking(False)
        if self.fullscreen:
            self.viewer.showFullScreen()
        else:
            self.viewer.resize(1280, 720)
            self.viewer.show()
        self.viewer.set_first_person_enabled(True)
        self.viewer.set_player_avatar_enabled(False)
        if hasattr(self.viewer, "ina_anim_use_inastate"):
            self.viewer.ina_anim_use_inastate = False
        self.viewer.mouse_capture_supported = False
        if hasattr(self.viewer, "_set_mouse_capture"):
            self.viewer._set_mouse_capture(False)
        if hasattr(self.viewer, "keys_down"):
            self.viewer.keys_down.clear()
        self._publish_vision_path()

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self._last_sync < (self.tick_ms / 1000.0):
            return
        self._last_sync = now
        self._sync_ina_pose()
        self._export_frame(now)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (
            QtCore.QEvent.KeyPress,
            QtCore.QEvent.KeyRelease,
            QtCore.QEvent.MouseButtonPress,
            QtCore.QEvent.MouseButtonRelease,
            QtCore.QEvent.MouseMove,
            QtCore.QEvent.Wheel,
        ):
            return True
        return super().eventFilter(obj, event)

    def _sync_ina_pose(self) -> None:
        snapshot = self.observer.get_state_snapshot()
        if not snapshot:
            return
        self._sync_door_states(snapshot)
        entities = snapshot.get("entities") or {}
        ina = entities.get("ina")
        if not ina:
            return
        pos = ina.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            return
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        self.viewer.player_pos = np.array([x, y, z], dtype=float)
        yaw = ina.get("yaw_deg")
        if yaw is not None:
            self.viewer.player_yaw = float(yaw)
        if hasattr(self.viewer, "_sync_first_person_camera"):
            self.viewer._sync_first_person_camera()

        player = entities.get("player")
        if player:
            pos = player.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                self.viewer.ina_pos = np.array(
                    [float(pos[0]), float(pos[1]), float(pos[2])], dtype=float
                )
            velocity = player.get("velocity")
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                try:
                    self.viewer.ina_velocity = np.array(
                        [float(velocity[0]), float(velocity[1]), float(velocity[2])],
                        dtype=float,
                    )
                except Exception:
                    pass
            if hasattr(self.viewer, "_update_ina_avatar_mesh"):
                self.viewer._update_ina_avatar_mesh()

    def _sync_door_states(self, snapshot: Dict[str, Any]) -> None:
        door_states = snapshot.get("doors")
        if not isinstance(door_states, dict):
            return
        if hasattr(self.viewer, "apply_door_states"):
            self.viewer.apply_door_states(door_states, snap=not self._door_state_synced)
            self._door_state_synced = True

    def _publish_vision_path(self) -> None:
        if update_inastate is None or self._vision_path is None:
            return
        try:
            update_inastate("vision_frame_path", str(self._vision_path))
            update_inastate("vision_frame_source", "ina_viewer")
        except Exception:
            pass

    def _export_frame(self, now: float) -> None:
        if self._vision_path is None:
            return
        if now - self._last_frame_ts < self._frame_interval:
            return
        self._last_frame_ts = now
        try:
            pixmap = self.viewer.view.grab()
        except Exception:
            return
        if pixmap.isNull():
            return
        try:
            self._vision_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap.save(str(self._vision_path), "PNG")
        except Exception:
            return
        if update_inastate is None:
            return
        try:
            update_inastate("vision_frame_ts", time.time())
        except Exception:
            pass


def _resolve_vision_path() -> Optional[Path]:
    config_path = Path("config.json")
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    child = config.get("current_child") or "default_child"
    base = Path("AI_Children") / child / "memory" / "vision_session"
    return base / "world_view_ina.png"


def run_ina_viewer(
    *,
    host: str,
    port: int,
    fullscreen: bool = True,
    borderless: bool = True,
    retry_interval: float = 300.0,
) -> None:
    if QtWidgets is None:
        raise RuntimeError(f"PyQt5 unavailable: {_QT_IMPORT_ERROR}")
    if HouseViewer is None:
        raise RuntimeError(f"HouseViewer unavailable: {_HOUSE_IMPORT_ERROR}")

    observer = WorldObserver(host=host, port=port)
    observer.start(retry_interval=retry_interval)

    app = QtWidgets.QApplication([])
    app.setApplicationName("Inazuma Ina Vision")
    bridge = InaVisionBridge(observer=observer, fullscreen=fullscreen, borderless=borderless)
    bridge.viewer.show()
    app.exec_()
    observer.stop()


def main() -> None:
    if QtWidgets is None:
        raise RuntimeError(f"PyQt5 unavailable: {_QT_IMPORT_ERROR}")
    parser = argparse.ArgumentParser(description="Ina first-person house viewer (vision feed).")
    parser.add_argument("--tcp-host", default=DEFAULT_TCP_HOST)
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--bordered", action="store_true")
    parser.add_argument("--retry-interval", type=float, default=300.0)
    args = parser.parse_args()
    run_ina_viewer(
        host=args.tcp_host,
        port=args.tcp_port,
        fullscreen=not args.windowed,
        borderless=not args.bordered,
        retry_interval=args.retry_interval,
    )


if __name__ == "__main__":
    main()
