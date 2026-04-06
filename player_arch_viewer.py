#!/usr/bin/env python3
"""House viewer first-person bridge for the player client."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

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

try:
    from audio_settings_panel import AudioSettingsPanel
except Exception:  # pragma: no cover - optional dependency
    AudioSettingsPanel = None

class PauseMenu(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Paused")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #f0f0f0;")

        audio_box = self._build_audio_controls()

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(12)

        continue_btn = QtWidgets.QPushButton("Continue")
        quit_btn = QtWidgets.QPushButton("Quit")
        continue_btn.clicked.connect(self.accept)
        quit_btn.clicked.connect(self.reject)

        button_row.addWidget(continue_btn)
        button_row.addWidget(quit_btn)

        layout.addWidget(title)
        if audio_box is not None:
            layout.addWidget(audio_box)
        layout.addLayout(button_row)

        self.setStyleSheet(
            "QDialog { background-color: rgba(18, 20, 26, 0.95); border-radius: 10px; }"
            "QPushButton { padding: 6px 14px; font-size: 13px; }"
        )

    def _build_audio_controls(self) -> Optional[QtWidgets.QWidget]:
        if AudioSettingsPanel is None:
            group = QtWidgets.QGroupBox("Audio Devices")
            layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel("Audio settings unavailable.")
            label.setStyleSheet("color: #b0b0b0; font-size: 11px;")
            layout.addWidget(label)
            return group
        panel = AudioSettingsPanel()
        return panel


class ChatInput(QtWidgets.QLineEdit):
    def __init__(
        self,
        *,
        focus_callback: Optional[Callable[[], None]],
        blur_callback: Optional[Callable[[], None]],
    ) -> None:
        super().__init__()
        self._focus_callback = focus_callback
        self._blur_callback = blur_callback

    def focusInEvent(self, event) -> None:
        if self._focus_callback is not None:
            self._focus_callback()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if self._blur_callback is not None:
            self._blur_callback()

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)


class ChatDock(QtWidgets.QDockWidget):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        send_world: Callable[[str], None],
        send_discord: Callable[[str], None],
        check_discord: Callable[[], None],
        focus_callback: Optional[Callable[[], None]],
        blur_callback: Optional[Callable[[], None]],
    ) -> None:
        super().__init__("Chat", parent)
        self._send_world = send_world
        self._send_discord = send_discord
        self._check_discord = check_discord
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self._build_ui(focus_callback=focus_callback, blur_callback=blur_callback)

    def _build_ui(
        self,
        *,
        focus_callback: Optional[Callable[[], None]],
        blur_callback: Optional[Callable[[], None]],
    ) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._presence_label = QtWidgets.QLabel("Connected: -")
        self._presence_label.setStyleSheet("color: #c8d6e8; font-size: 12px;")

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(240)
        self._log.setStyleSheet("background-color: #101318; color: #e0e0e0;")

        input_row = QtWidgets.QHBoxLayout()
        self._input = ChatInput(focus_callback=focus_callback, blur_callback=blur_callback)
        self._input.setPlaceholderText("Say something...")
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self._on_send_world)
        self._input.returnPressed.connect(self._on_send_world)
        input_row.addWidget(self._input)
        input_row.addWidget(send_btn)

        discord_row = QtWidgets.QHBoxLayout()
        self._discord_btn = QtWidgets.QPushButton("Send to Discord")
        self._discord_btn.clicked.connect(self._on_send_discord)
        self._discord_check_btn = QtWidgets.QPushButton("Check Discord")
        self._discord_check_btn.clicked.connect(self._on_check_discord)
        discord_row.addWidget(self._discord_btn)
        discord_row.addWidget(self._discord_check_btn)

        self._desk_label = QtWidgets.QLabel("Desk: use the computer to unlock Discord.")
        self._desk_label.setStyleSheet("color: #b8b8b8; font-size: 11px;")

        layout.addWidget(self._presence_label)
        layout.addWidget(self._log)
        layout.addLayout(input_row)
        layout.addLayout(discord_row)
        layout.addWidget(self._desk_label)

        self.setWidget(root)
        self.set_discord_enabled(False)

    def _consume_input(self) -> str:
        text = self._input.text().strip()
        if text:
            self._input.clear()
        return text

    def _on_send_world(self) -> None:
        text = self._consume_input()
        if text:
            self._send_world(text)

    def _on_send_discord(self) -> None:
        text = self._consume_input()
        if text:
            self._send_discord(text)

    def _on_check_discord(self) -> None:
        self._check_discord()

    def append_lines(self, lines: List[str]) -> None:
        if not lines:
            return
        for line in lines:
            self._log.appendPlainText(line)
        scrollbar = self._log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def append_system(self, text: str) -> None:
        self.append_lines([f"[system] {text}"])

    def set_discord_enabled(self, enabled: bool) -> None:
        self._discord_btn.setEnabled(enabled)
        self._discord_check_btn.setEnabled(enabled)
        if enabled:
            self._desk_label.setText("Desk: connected.")
        else:
            self._desk_label.setText("Desk: use the computer to unlock Discord.")

    def set_presence(self, names: List[str]) -> None:
        if not names:
            self._presence_label.setText("Connected: -")
            return
        self._presence_label.setText("Connected: " + ", ".join(names))

    def focus_input(self) -> None:
        self._input.setFocus(QtCore.Qt.OtherFocusReason)

    def clear_input_focus(self) -> None:
        self._input.clearFocus()


def _format_world_line(entry: dict) -> str:
    name = entry.get("name") or "unknown"
    text = entry.get("text") or ""
    timestamp = entry.get("timestamp")
    if timestamp is not None:
        try:
            stamp = time.strftime("%H:%M", time.localtime(float(timestamp)))
            return f"[{stamp}] {name}: {text}"
        except Exception:
            pass
    return f"{name}: {text}"


def _tail_lines(path: Path, *, max_lines: int = 200) -> List[str]:
    if max_lines <= 0:
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            data = b""
            block_size = 4096
            while end > 0 and data.count(b"\n") <= max_lines:
                read_size = min(block_size, end)
                end -= read_size
                handle.seek(end)
                data = handle.read(read_size) + data
            lines = data.splitlines()[-max_lines:]
        return [line.decode("utf-8", errors="ignore") for line in lines]
    except Exception:
        return []


def _load_recent_discord_messages(limit: int = 6) -> List[str]:
    path = Path("logs") / "comms_core.jsonl"
    if not path.exists():
        return []
    lines = _tail_lines(path, max_lines=max(limit * 20, 80))
    results: List[str] = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("backend") != "discord":
            continue
        if record.get("direction") != "inbound":
            continue
        sender = record.get("sender") or {}
        name = sender.get("display_name") or sender.get("internal_id") or "discord"
        text = record.get("text") or ""
        created_at = record.get("created_at") or ""
        if created_at:
            msg = f"[discord {created_at}] {name}: {text}"
        else:
            msg = f"[discord] {name}: {text}"
        results.append(msg)
        if len(results) >= limit:
            break
    return list(reversed(results))


class HouseViewerBridge(QtCore.QObject):
    def __init__(
        self,
        *,
        client,
        fullscreen: bool = True,
        borderless: bool = True,
        tick_ms: int = 100,
    ) -> None:
        super().__init__()
        if HouseViewer is None:
            raise RuntimeError(f"HouseViewer unavailable: {_HOUSE_IMPORT_ERROR}")
        self.client = client
        self.viewer = HouseViewer()
        self.fullscreen = fullscreen
        self.borderless = borderless
        self.tick_ms = tick_ms
        self._last_pose = None
        self._last_sync = 0.0
        self._door_state_synced = False
        self._pause_dialog: Optional[PauseMenu] = None
        self._pause_active = False
        self._last_frame_ts = 0.0
        self._frame_interval = 1.0
        self._vision_path = _resolve_player_path()
        self._chat_dock: Optional[ChatDock] = None
        self._chat_seq = 0
        self._last_desk_check = 0.0
        self._chat_active = False
        self._last_presence_check = 0.0
        self._presence_interval = 1.0

        self._init_window()
        self._init_chat_panel()
        if hasattr(self.viewer, "set_desk_use_callback"):
            self.viewer.set_desk_use_callback(self._on_desk_use)
        if hasattr(self.viewer, "set_door_state_callback"):
            self.viewer.set_door_state_callback(self._send_door_state)
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
        if self.fullscreen:
            self.viewer.showFullScreen()
        else:
            self.viewer.resize(1280, 720)
            self.viewer.show()
        self.viewer.set_first_person_enabled(True)
        self.viewer.set_player_avatar_enabled(False)
        self._publish_player_path()

    def _init_chat_panel(self) -> None:
        self._chat_dock = ChatDock(
            self.viewer,
            send_world=self._send_world_chat,
            send_discord=self._send_discord_chat,
            check_discord=self._check_discord_messages,
            focus_callback=self._on_chat_focus,
            blur_callback=self._on_chat_blur,
        )
        self.viewer.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._chat_dock)
        self._chat_dock.show()

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self._last_sync < (self.tick_ms / 1000.0):
            return
        self._last_sync = now
        snapshot = self.client.get_state_view() if hasattr(self.client, "get_state_view") else self.client.get_state_snapshot()
        self._push_player_pose()
        self._pull_ina_pose(snapshot)
        self._update_chat_log()
        self._update_desk_status(now)
        self._update_presence(now, snapshot)
        self._export_frame(now)

    def _on_chat_focus(self) -> None:
        self._chat_active = True
        if hasattr(self.viewer, "keys_down"):
            self.viewer.keys_down.clear()
        if hasattr(self.viewer, "_set_mouse_capture"):
            self.viewer._set_mouse_capture(False)
        if hasattr(self.viewer, "_radial_menu_active"):
            self.viewer._radial_menu_active = True
        if hasattr(self.viewer, "_update_crosshair_visibility"):
            self.viewer._update_crosshair_visibility()

    def _on_chat_blur(self) -> None:
        self._chat_active = False
        if self._pause_active:
            return
        if hasattr(self.viewer, "_radial_menu_active"):
            self.viewer._radial_menu_active = False
        if hasattr(self.viewer, "_update_crosshair_visibility"):
            self.viewer._update_crosshair_visibility()
        if getattr(self.viewer, "first_person_enabled", False):
            if hasattr(self.viewer, "_set_mouse_capture"):
                self.viewer._set_mouse_capture(True)

    def _focus_chat_input(self) -> None:
        if self._chat_dock is None:
            return
        self._chat_dock.focus_input()

    def _send_world_chat(self, text: str) -> None:
        payload = {"type": "comms", "text": text}
        self.client.send(payload)

    def _can_use_discord(self) -> bool:
        if hasattr(self.viewer, "is_using_desk"):
            try:
                return bool(self.viewer.is_using_desk())
            except Exception:
                return False
        return False

    def _send_discord_chat(self, text: str) -> None:
        if self._chat_dock is None:
            return
        if not self._can_use_discord():
            self._chat_dock.append_system("Use the desk computer to send Discord messages.")
            return
        try:
            from model_manager import append_typed_outbox_entry
        except Exception:
            self._chat_dock.append_system("Discord bridge unavailable.")
            return
        entry_id = append_typed_outbox_entry(
            text,
            metadata={"source": "player_chat_panel"},
        )
        if entry_id:
            self._chat_dock.append_system("Queued Discord message.")
        else:
            self._chat_dock.append_system("Discord message not queued.")

    def _check_discord_messages(self) -> None:
        if self._chat_dock is None:
            return
        if not self._can_use_discord():
            self._chat_dock.append_system("Use the desk computer to check Discord.")
            return
        lines = _load_recent_discord_messages()
        if not lines:
            self._chat_dock.append_system("No recent Discord messages.")
            return
        self._chat_dock.append_lines(lines)

    def _update_chat_log(self) -> None:
        if self._chat_dock is None:
            return
        next_seq, messages = self.client.get_comms_since(self._chat_seq)
        self._chat_seq = max(self._chat_seq, next_seq)
        if not messages:
            return
        lines = [_format_world_line(entry) for entry in messages]
        self._chat_dock.append_lines(lines)

    def _update_desk_status(self, now: float) -> None:
        if self._chat_dock is None:
            return
        if now - self._last_desk_check < 0.5:
            return
        self._last_desk_check = now
        self._chat_dock.set_discord_enabled(self._can_use_discord())

    def _update_presence(self, now: float, snapshot: Optional[dict]) -> None:
        if self._chat_dock is None:
            return
        if now - self._last_presence_check < self._presence_interval:
            return
        self._last_presence_check = now
        if not snapshot:
            self._chat_dock.set_presence([])
            return
        entities = snapshot.get("entities") or {}
        if not isinstance(entities, dict):
            self._chat_dock.set_presence([])
            return
        cutoff = time.time() - 600.0
        names = []
        for entity_id, entity in entities.items():
            if not isinstance(entity, dict):
                continue
            last_seen = entity.get("last_seen")
            try:
                last_seen = float(last_seen)
            except Exception:
                last_seen = None
            if last_seen is not None and last_seen < cutoff:
                continue
            name = entity.get("name") or entity_id
            name = str(name)
            if name not in names:
                names.append(name)
        local_name = getattr(self.client, "player_name", None)
        if local_name:
            local_name = str(local_name)
            if local_name in names:
                names.remove(local_name)
                names.insert(0, local_name)
        self._chat_dock.set_presence(names)

    def _on_desk_use(self, _desk) -> None:
        if self._chat_dock is None:
            return
        self._chat_dock.append_system("Desk online. Discord is available.")
        self._focus_chat_input()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.KeyPress:
            if hasattr(event, "key"):
                key = event.key()
                if key == QtCore.Qt.Key_T and not self._pause_active:
                    self._focus_chat_input()
                    return True
                if key == QtCore.Qt.Key_Escape and not self._chat_active:
                    if getattr(self.viewer, "first_person_enabled", False):
                        self._toggle_pause_menu()
                        return True
        return super().eventFilter(obj, event)

    def _toggle_pause_menu(self) -> None:
        if self._pause_active:
            self._close_pause_menu(continue_play=True)
            return
        self._open_pause_menu()

    def _open_pause_menu(self) -> None:
        if self._pause_dialog is not None:
            return
        self._pause_active = True
        if self._chat_dock is not None:
            self._chat_dock.clear_input_focus()
        if hasattr(self.viewer, "keys_down"):
            self.viewer.keys_down.clear()
        if hasattr(self.viewer, "_set_mouse_capture"):
            self.viewer._set_mouse_capture(False)
        if hasattr(self.viewer, "_radial_menu_active"):
            self.viewer._radial_menu_active = True
        if hasattr(self.viewer, "_update_crosshair_visibility"):
            self.viewer._update_crosshair_visibility()

        dialog = PauseMenu(self.viewer)
        dialog.accepted.connect(lambda: self._close_pause_menu(continue_play=True))
        dialog.rejected.connect(lambda: self._close_pause_menu(continue_play=False))
        dialog.adjustSize()
        center = self.viewer.geometry().center()
        dialog.move(center.x() - dialog.width() // 2, center.y() - dialog.height() // 2)
        self._pause_dialog = dialog
        dialog.show()

    def _close_pause_menu(self, *, continue_play: bool) -> None:
        if self._pause_dialog is not None:
            self._pause_dialog.close()
        self._pause_dialog = None
        self._pause_active = False
        if hasattr(self.viewer, "_radial_menu_active"):
            self.viewer._radial_menu_active = False
        if hasattr(self.viewer, "_update_crosshair_visibility"):
            self.viewer._update_crosshair_visibility()
        if continue_play and getattr(self.viewer, "first_person_enabled", False) and not self._chat_active:
            if hasattr(self.viewer, "_set_mouse_capture"):
                self.viewer._set_mouse_capture(True)
        if not continue_play:
            self.viewer.close()

    def _push_player_pose(self) -> None:
        if self.viewer.player_pos is None:
            return
        pos = self.viewer.player_pos
        payload = {
            "type": "pose",
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "yaw_deg": float(self.viewer.player_yaw),
        }
        if payload != self._last_pose:
            self.client.send(payload)
            self._last_pose = payload

    def _pull_ina_pose(self, snapshot: Optional[dict]) -> None:
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
        self.viewer.ina_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
        velocity = ina.get("velocity")
        if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            try:
                self.viewer.ina_velocity = np.array(
                    [float(velocity[0]), float(velocity[1]), float(velocity[2])], dtype=float
                )
            except Exception:
                pass
        self.viewer._update_ina_avatar_mesh()

    def _send_door_state(self, door_id: str, open_state: bool) -> None:
        self.client.send({"type": "door", "door_id": door_id, "open": bool(open_state)})

    def _sync_door_states(self, snapshot: dict) -> None:
        door_states = snapshot.get("doors")
        if not isinstance(door_states, dict):
            return
        if hasattr(self.viewer, "apply_door_states"):
            self.viewer.apply_door_states(door_states, snap=not self._door_state_synced)
            self._door_state_synced = True

    def _publish_player_path(self) -> None:
        if update_inastate is None or self._vision_path is None:
            return
        try:
            update_inastate("player_frame_path", str(self._vision_path))
            update_inastate("player_frame_source", "player_viewer")
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
            update_inastate("player_frame_ts", time.time())
        except Exception:
            pass


def _resolve_player_path() -> Optional[Path]:
    config_path = Path("config.json")
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    child = config.get("current_child") or "default_child"
    base = Path("AI_Children") / child / "memory" / "vision_session"
    return base / "world_view_player.png"


def run_arch_viewer(*, client, fullscreen: bool = True, borderless: bool = True) -> None:
    if QtWidgets is None:
        raise RuntimeError(f"PyQt5 unavailable: {_QT_IMPORT_ERROR}")
    if HouseViewer is None:
        raise RuntimeError(f"HouseViewer unavailable: {_HOUSE_IMPORT_ERROR}")
    app = QtWidgets.QApplication([])
    app.setApplicationName("Inazuma Player House View")
    bridge = HouseViewerBridge(client=client, fullscreen=fullscreen, borderless=borderless)
    bridge.viewer.show()
    app.exec_()
