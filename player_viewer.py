#!/usr/bin/env python3
"""Borderless fullscreen world viewer for the player client."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from world_protocol import clamp

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover - optional dependency
    QtCore = None
    QtGui = None
    QtWidgets = None
    _QT_IMPORT_ERROR = exc
else:
    _QT_IMPORT_ERROR = None


if QtWidgets is not None:
    class WorldViewer(QtWidgets.QWidget):
        def __init__(
            self,
            *,
            client,
            fullscreen: bool = True,
            borderless: bool = True,
            tick_ms: int = 100,
        ) -> None:
            super().__init__()
            self.client = client
            self.fullscreen = fullscreen
            self.borderless = borderless
            self.tick_ms = tick_ms

            self._forward = 0.0
            self._strafe = 0.0
            self._up = 0.0
            self._turn = 0.0
            self._run = False
            self._last_input = (0.0, 0.0, 0.0, 0.0, False)
            self._last_mouse_ts = 0.0
            self._mouse_look = True
            self._recenter_mouse = True
            self._sensitivity = 0.01
            self._last_mouse_pos = None

            self._init_window()
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(self.tick_ms)

        def _init_window(self) -> None:
            if self.borderless:
                self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
            self.setWindowTitle("Inazuma Player View")
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setMouseTracking(True)
            self.setCursor(QtCore.Qt.BlankCursor)
            if self.fullscreen:
                self.showFullScreen()
            else:
                self.resize(1280, 720)
                self.show()

        def _tick(self) -> None:
            now = time.monotonic()
            if self._mouse_look and (now - self._last_mouse_ts) > 0.2:
                self._turn = 0.0

            forward = self._forward
            strafe = self._strafe
            up = self._up
            turn = clamp(self._turn, -1.0, 1.0)
            run = self._run

            if abs(forward) < 1e-3:
                forward = 0.0
            if abs(strafe) < 1e-3:
                strafe = 0.0
            if abs(up) < 1e-3:
                up = 0.0
            if abs(turn) < 1e-3:
                turn = 0.0

            current = (forward, strafe, up, turn, run)
            has_input = any(abs(val) > 1e-3 for val in current[:4]) or run

            if has_input:
                if current != self._last_input:
                    self.client.send(
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
                    self._last_input = current
            else:
                if self._last_input != (0.0, 0.0, 0.0, 0.0, False):
                    self.client.send({"type": "stop"})
                    self._last_input = (0.0, 0.0, 0.0, 0.0, False)

            self.update()

        def _toggle_mouse_look(self) -> None:
            self._mouse_look = not self._mouse_look
            if self._mouse_look:
                self.setCursor(QtCore.Qt.BlankCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)

        def _toggle_fullscreen(self) -> None:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
            key = event.key()
            if key == QtCore.Qt.Key_W:
                self._forward = 1.0
            elif key == QtCore.Qt.Key_S:
                self._forward = -1.0
            elif key == QtCore.Qt.Key_A:
                self._strafe = -1.0
            elif key == QtCore.Qt.Key_D:
                self._strafe = 1.0
            elif key == QtCore.Qt.Key_Q:
                self._up = -1.0
            elif key == QtCore.Qt.Key_E:
                self._up = 1.0
            elif key == QtCore.Qt.Key_Shift:
                self._run = True
            elif key == QtCore.Qt.Key_M:
                self._toggle_mouse_look()
            elif key == QtCore.Qt.Key_R:
                self._recenter_mouse = not self._recenter_mouse
            elif key == QtCore.Qt.Key_F11:
                self._toggle_fullscreen()
            elif key == QtCore.Qt.Key_Escape:
                if self._mouse_look:
                    self._toggle_mouse_look()
                else:
                    self.close()
            else:
                super().keyPressEvent(event)

        def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
            key = event.key()
            if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_S):
                self._forward = 0.0
            elif key in (QtCore.Qt.Key_A, QtCore.Qt.Key_D):
                self._strafe = 0.0
            elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_E):
                self._up = 0.0
            elif key == QtCore.Qt.Key_Shift:
                self._run = False
            else:
                super().keyReleaseEvent(event)

        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
            if not self._mouse_look:
                return
            if self._last_mouse_pos is None:
                self._last_mouse_pos = event.pos()
                return
            dx = event.pos().x() - self._last_mouse_pos.x()
            self._turn = clamp(dx * self._sensitivity, -1.0, 1.0)
            self._last_mouse_ts = time.monotonic()

            if self._recenter_mouse and self.isVisible():
                center = self.rect().center()
                QtGui.QCursor.setPos(self.mapToGlobal(center))
                self._last_mouse_pos = center
            else:
                self._last_mouse_pos = event.pos()

        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
            if event.button() == QtCore.Qt.RightButton:
                self._toggle_mouse_look()
            else:
                super().mousePressEvent(event)

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.fillRect(self.rect(), QtGui.QColor(8, 10, 16))

            state = self.client.get_state_snapshot()
            if not state:
                painter.setPen(QtGui.QColor(200, 200, 200))
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Waiting for world state...")
                return

            bounds = state.get("bounds") or {}
            min_x = float(bounds.get("min_x", -10.0))
            max_x = float(bounds.get("max_x", 10.0))
            min_y = float(bounds.get("min_y", -10.0))
            max_y = float(bounds.get("max_y", 10.0))
            tv_channel = state.get("tv_channel") or "-"
            entities = state.get("entities") or {}

            rect = self.rect()
            padding = 40
            width = max_x - min_x
            height = max_y - min_y
            usable_w = max(1.0, rect.width() - padding * 2)
            usable_h = max(1.0, rect.height() - padding * 2)
            scale = min(usable_w / width, usable_h / height)
            offset_x = padding + (usable_w - width * scale) / 2
            offset_y = padding + (usable_h - height * scale) / 2

            def world_to_screen(x: float, y: float) -> QtCore.QPointF:
                sx = offset_x + (x - min_x) * scale
                sy = offset_y + (max_y - y) * scale
                return QtCore.QPointF(sx, sy)

            border_rect = QtCore.QRectF(
                offset_x,
                offset_y,
                width * scale,
                height * scale,
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 90, 120), 2))
            painter.drawRect(border_rect)

            grid_pen = QtGui.QPen(QtGui.QColor(30, 40, 60), 1)
            painter.setPen(grid_pen)
            grid_step = max(1.0, math.floor(width / 10))
            x = math.ceil(min_x / grid_step) * grid_step
            while x <= max_x:
                start = world_to_screen(x, min_y)
                end = world_to_screen(x, max_y)
                painter.drawLine(start, end)
                x += grid_step
            y = math.ceil(min_y / grid_step) * grid_step
            while y <= max_y:
                start = world_to_screen(min_x, y)
                end = world_to_screen(max_x, y)
                painter.drawLine(start, end)
                y += grid_step

            for entity_id, entity in entities.items():
                pos = entity.get("position") or [0.0, 0.0, 0.0]
                yaw = float(entity.get("yaw_deg") or 0.0)
                role = (entity.get("role") or "").lower()
                name = entity.get("name") or entity_id

                center = world_to_screen(float(pos[0]), float(pos[1]))
                radius = max(5, int(scale * 0.12))
                if role == "ina":
                    color = QtGui.QColor(90, 200, 200)
                elif role == "player":
                    color = QtGui.QColor(230, 170, 70)
                else:
                    color = QtGui.QColor(200, 200, 200)

                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QtGui.QPen(QtGui.QColor(15, 15, 20), 1))
                painter.drawEllipse(center, radius, radius)

                yaw_rad = math.radians(yaw)
                line_len = radius * 2.6
                line_end = QtCore.QPointF(
                    center.x() + math.cos(yaw_rad) * line_len,
                    center.y() - math.sin(yaw_rad) * line_len,
                )
                painter.setPen(QtGui.QPen(color, 2))
                painter.drawLine(center, line_end)

                painter.setPen(QtGui.QColor(220, 220, 220))
                painter.drawText(center + QtCore.QPointF(radius + 6, -radius - 4), name)

            painter.setPen(QtGui.QColor(180, 180, 200))
            painter.drawText(
                QtCore.QRect(16, 16, rect.width() - 32, 20),
                QtCore.Qt.AlignLeft,
                f"TV: {tv_channel}",
            )
            painter.setPen(QtGui.QColor(140, 140, 150))
            hint = "WASD move | mouse look (RMB/M) | shift run | F11 fullscreen | Esc release/exit"
            painter.drawText(
                QtCore.QRect(16, rect.height() - 32, rect.width() - 32, 20),
                QtCore.Qt.AlignLeft,
                hint,
            )
else:
    WorldViewer = None


def run_viewer(*, client, fullscreen: bool = True, borderless: bool = True) -> None:
    if QtWidgets is None:
        raise RuntimeError(f"PyQt5 unavailable: {_QT_IMPORT_ERROR}")
    if WorldViewer is None:
        raise RuntimeError("WorldViewer unavailable.")
    app = QtWidgets.QApplication([])
    app.setApplicationName("Inazuma Player View")
    view = WorldViewer(client=client, fullscreen=fullscreen, borderless=borderless)
    app.exec_()
