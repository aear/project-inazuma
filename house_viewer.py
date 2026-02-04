# house_viewer.py

import copy
import json
from datetime import date, datetime, timedelta, timezone
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import urllib.error
import urllib.request

from geometry_utils import (
    get_unit_cube_meshdata,
    get_unit_cylinder_meshdata,
    get_unit_sphere_meshdata,
)
from house_model import (
    create_prototype_house,
    create_prototype_exterior,
    load_house_from_plan,
    Opening,
    Room,
    House,
    ExteriorModel,
    FenceSegment,
    WallSegment,
)

try:
    from model_manager import get_inastate
except Exception:  # pragma: no cover - optional dependency
    get_inastate = None
try:
    from safe_popen import safe_popen
except Exception:  # pragma: no cover - optional dependency
    safe_popen = None

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass
class ArchitectWall:
    start: Vec2
    end: Vec2
    height: float
    thickness: float
    color: Tuple[float, float, float, float]
    openings: List[Opening] = field(default_factory=list)


@dataclass
class ArchitectState:
    footprint_points: List[Vec2] = field(default_factory=list)
    walls: List[ArchitectWall] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    fences: List[FenceSegment] = field(default_factory=list)
    ceiling_lights: List["ArchitectCeilingLight"] = field(default_factory=list)
    light_switches: List["ArchitectLightSwitch"] = field(default_factory=list)
    spawn_point: Optional[Vec3] = None
    ina_spawn_point: Optional[Vec3] = None


@dataclass
class ArchitectCeilingLight:
    position: Vec2
    color: Tuple[float, float, float, float]
    room: Optional[str] = None


@dataclass
class ArchitectLightSwitch:
    position: Vec2
    height: float
    room: Optional[str] = None


@dataclass
class ArchitectSettings:
    tool: str = "select"
    grid_size: float = 0.5
    snap_enabled: bool = True
    snap_existing: bool = True
    axis_lock: bool = True
    view_scale: float = 40.0
    wall_height: float = 2.7
    wall_thickness: float = 0.2
    wall_color: Tuple[float, float, float, float] = (0.85, 0.85, 0.9, 1.0)
    room_height: float = 2.7
    room_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.35)
    room_name_prefix: str = "room"
    fence_height: float = 1.0
    fence_thickness: float = 0.08
    fence_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    door_width: float = 0.9
    door_height: float = 2.0
    door_sill: float = 0.0
    window_width: float = 1.0
    window_height: float = 1.2
    window_sill: float = 0.9
    roof_overhang: float = 0.3
    roof_thickness: float = 0.15
    roof_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    ground_size: Tuple[float, float] = (20.0, 20.0)
    ground_color: Tuple[float, float, float, float] = (0.15, 0.22, 0.15, 1.0)
    spawn_height: float = 0.0
    ceiling_light_color: Tuple[float, float, float, float] = (1.0, 0.95, 0.7, 1.0)
    ceiling_light_radius: float = 0.25
    switch_color: Tuple[float, float, float, float] = (0.85, 0.85, 0.9, 1.0)
    switch_height: float = 1.2


@dataclass
class DoorInstance:
    item: gl.GLMeshItem
    hinge: np.ndarray
    width: float
    thickness: float
    height: float
    base_angle: float
    open_angle: float
    wall_ref: Optional[WallSegment] = None
    offset_along_wall: float = 0.0
    door_id: Optional[str] = None
    current_angle: float = 0.0
    target_angle: float = 0.0


@dataclass
class FurniturePartDef:
    size: Vec3
    offset: Vec3
    color: Tuple[float, float, float, float]
    gl_options: Optional[str] = None
    shape: str = "box"
    emissive: float = 0.0


@dataclass
class FurniturePrototype:
    key: str
    label: str
    parts: List[FurniturePartDef]
    seat_offset: Optional[Vec3] = None
    interact_radius: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class FurniturePart:
    mesh: gl.GLMeshItem
    size: Vec3
    offset: Vec3
    base_size: Vec3
    base_offset: Vec3


@dataclass
class FurnitureInstance:
    instance_id: int
    key: str
    label: str
    position: np.ndarray
    rotation: float
    parts: List[FurniturePart]
    collision_radius: float
    seat_offset: Optional[Vec3] = None
    interact_radius: float = 0.0
    tags: List[str] = field(default_factory=list)
    pose_offset: Vec3 = (0.0, 0.0, 0.0)
    tilt_deg: float = 0.0
    knocked: bool = False


class ArchitectCanvas(QtWidgets.QGraphicsView):
    def __init__(self, viewer: "HouseViewer"):
        super().__init__()
        self.viewer = viewer
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setBackgroundBrush(QtGui.QColor(18, 18, 22))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._view_scale = None
        self.set_view_scale(self.viewer.architect_settings.view_scale)

        self._panning = False
        self._pan_start = QtCore.QPoint()
        self._dragging = False
        self._drag_start = QtCore.QPointF()
        self._preview_item = None
        self._footprint_points: List[Vec2] = []
        self._footprint_preview = None
        self._last_mouse_scene_pos: Optional[QtCore.QPointF] = None

        self.refresh_scene()

    def refresh_scene(self):
        self._scene.clear()
        self._preview_item = None
        self._footprint_preview = None

        state = self.viewer.architect_state
        settings = self.viewer.architect_settings

        ground_w = settings.ground_size[0]
        ground_d = settings.ground_size[1]
        ground_rect = QtCore.QRectF(
            -ground_w / 2.0,
            -ground_d / 2.0,
            ground_w,
            ground_d,
        )
        ground_item = QtWidgets.QGraphicsRectItem(ground_rect)
        ground_item.setPen(QtGui.QPen(QtGui.QColor(40, 60, 40, 120)))
        ground_item.setBrush(QtGui.QBrush(self._to_qcolor(settings.ground_color, alpha_override=80)))
        ground_item.setZValue(-10)
        self._scene.addItem(ground_item)

        if state.footprint_points:
            polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in state.footprint_points])
            footprint = QtWidgets.QGraphicsPolygonItem(polygon)
            footprint_pen = QtGui.QPen(QtGui.QColor(120, 180, 220, 200))
            footprint_pen.setStyle(QtCore.Qt.DashLine)
            footprint.setPen(footprint_pen)
            footprint.setBrush(QtGui.QBrush(QtGui.QColor(80, 120, 160, 40)))
            self._scene.addItem(footprint)

        for wall in state.walls:
            wall_item = self._make_thick_segment(
                wall.start,
                wall.end,
                wall.thickness,
                wall.color,
            )
            self._scene.addItem(wall_item)
            for op in wall.openings:
                opening_item = self._make_opening_marker(wall, op)
                if opening_item is not None:
                    self._scene.addItem(opening_item)

        for fence in state.fences:
            fence_item = self._make_thick_segment(
                fence.start,
                fence.end,
                fence.thickness,
                fence.color,
            )
            fence_item.setOpacity(0.75)
            self._scene.addItem(fence_item)

        for room in state.rooms:
            center_x, _, center_z = room.center
            width, _, depth = room.size
            rect = QtCore.QRectF(
                center_x - width / 2.0,
                center_z - depth / 2.0,
                width,
                depth,
            )
            room_item = QtWidgets.QGraphicsRectItem(rect)
            room_item.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220, 140)))
            room_item.setBrush(QtGui.QBrush(self._to_qcolor(room.color, alpha_override=140)))
            self._scene.addItem(room_item)

        for light in state.ceiling_lights:
            radius = max(0.08, settings.ceiling_light_radius)
            circle = QtWidgets.QGraphicsEllipseItem(
                light.position[0] - radius,
                light.position[1] - radius,
                radius * 2.0,
                radius * 2.0,
            )
            pen = QtGui.QPen(self._to_qcolor(light.color, alpha_override=220))
            pen.setWidthF(0.05)
            circle.setPen(pen)
            circle.setBrush(QtGui.QBrush(self._to_qcolor(light.color, alpha_override=90)))
            self._scene.addItem(circle)

        for switch in state.light_switches:
            size = max(0.12, settings.grid_size * 0.3)
            rect = QtCore.QRectF(
                switch.position[0] - size / 2.0,
                switch.position[1] - size / 2.0,
                size,
                size,
            )
            item = QtWidgets.QGraphicsRectItem(rect)
            pen = QtGui.QPen(self._to_qcolor(settings.switch_color, alpha_override=220))
            pen.setWidthF(0.05)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(self._to_qcolor(settings.switch_color, alpha_override=120)))
            self._scene.addItem(item)

        if state.spawn_point is not None:
            self._draw_spawn_marker(
                state.spawn_point,
                QtGui.QColor(255, 180, 60, 220),
                QtGui.QColor(255, 180, 60, 60),
            )

        if state.ina_spawn_point is not None:
            self._draw_spawn_marker(
                state.ina_spawn_point,
                QtGui.QColor(120, 200, 255, 220),
                QtGui.QColor(120, 200, 255, 60),
            )

        if not self._scene.items():
            self._scene.setSceneRect(QtCore.QRectF(-12, -12, 24, 24))
        else:
            self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-2, -2, 2, 2))

    def _to_qcolor(self, color, alpha_override: Optional[int] = None) -> QtGui.QColor:
        r, g, b, a = color
        alpha = int(a * 255)
        if alpha_override is not None:
            alpha = alpha_override
        return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), alpha)

    def _draw_spawn_marker(self, spawn: Vec3, pen_color: QtGui.QColor, fill_color: QtGui.QColor):
        spawn_x = spawn[0]
        spawn_y = spawn[2]
        radius = max(0.15, self.viewer.architect_settings.grid_size * 0.4)
        pen = QtGui.QPen(pen_color)
        pen.setWidthF(0.05)
        circle = QtWidgets.QGraphicsEllipseItem(
            spawn_x - radius,
            spawn_y - radius,
            radius * 2.0,
            radius * 2.0,
        )
        circle.setPen(pen)
        circle.setBrush(QtGui.QBrush(fill_color))
        self._scene.addItem(circle)
        self._scene.addLine(
            QtCore.QLineF(spawn_x - radius * 1.5, spawn_y, spawn_x + radius * 1.5, spawn_y),
            pen,
        )
        self._scene.addLine(
            QtCore.QLineF(spawn_x, spawn_y - radius * 1.5, spawn_x, spawn_y + radius * 1.5),
            pen,
        )

    def _make_thick_segment(self, start, end, thickness, color):
        poly = self._segment_polygon(start, end, thickness)
        if poly is None:
            return QtWidgets.QGraphicsLineItem()
        item = QtWidgets.QGraphicsPolygonItem(poly)
        pen = QtGui.QPen(self._to_qcolor(color, alpha_override=210))
        pen.setWidthF(0.0)
        item.setPen(pen)
        item.setBrush(QtGui.QBrush(self._to_qcolor(color, alpha_override=180)))
        return item

    def _make_opening_marker(self, wall: ArchitectWall, opening: Opening):
        start = np.array(wall.start, dtype=float)
        end = np.array(wall.end, dtype=float)
        seg = end - start
        length = float(np.hypot(seg[0], seg[1]))
        if length < 1e-5:
            return None
        direction = seg / length
        offset = float(np.clip(opening.offset_along_wall, 0.0, length))
        center = start + direction * offset

        marker_thickness = max(0.04, wall.thickness * 0.6)
        marker_length = max(0.2, opening.width)
        poly = self._segment_polygon(
            (center[0] - direction[0] * marker_length / 2.0, center[1] - direction[1] * marker_length / 2.0),
            (center[0] + direction[0] * marker_length / 2.0, center[1] + direction[1] * marker_length / 2.0),
            marker_thickness,
        )
        if poly is None:
            return None
        color = (0.5, 0.3, 0.2, 1.0) if opening.type == "door" else (0.6, 0.8, 1.0, 1.0)
        item = QtWidgets.QGraphicsPolygonItem(poly)
        item.setPen(QtGui.QPen(self._to_qcolor(color, alpha_override=220)))
        item.setBrush(QtGui.QBrush(self._to_qcolor(color, alpha_override=200)))
        return item

    def _segment_polygon(self, start, end, thickness):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-5:
            return None
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        half = thickness / 2.0
        p1 = QtCore.QPointF(x1 + px * half, y1 + py * half)
        p2 = QtCore.QPointF(x2 + px * half, y2 + py * half)
        p3 = QtCore.QPointF(x2 - px * half, y2 - py * half)
        p4 = QtCore.QPointF(x1 - px * half, y1 - py * half)
        return QtGui.QPolygonF([p1, p2, p3, p4])

    def set_view_scale(self, scale: float):
        try:
            scale_value = float(scale)
        except (TypeError, ValueError):
            return
        if scale_value <= 1e-3:
            return
        center = None
        if self._view_scale is None:
            center = QtCore.QPointF(0.0, 0.0)
        else:
            center = self.mapToScene(self.viewport().rect().center())
        self._view_scale = scale_value
        self.resetTransform()
        self.scale(scale_value, -scale_value)
        self.centerOn(center)

    def _snap_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        settings = self.viewer.architect_settings
        point = self._snap_to_existing(point)
        if not settings.snap_enabled or settings.grid_size <= 1e-6:
            return point
        grid = settings.grid_size
        snapped_x = round(point.x() / grid) * grid
        snapped_y = round(point.y() / grid) * grid
        snapped = QtCore.QPointF(snapped_x, snapped_y)
        return self._snap_to_existing(snapped)

    def _scene_pos(self, event):
        return self._snap_point(self.mapToScene(event.pos()))

    def _snap_to_existing(self, point: QtCore.QPointF) -> QtCore.QPointF:
        settings = self.viewer.architect_settings
        if not settings.snap_existing:
            return point

        snap_radius = max(settings.grid_size * 0.75, 0.2)
        best_point = None
        best_dist = None

        for target in self._snap_targets():
            dx = point.x() - target.x()
            dy = point.y() - target.y()
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_point = target

        if best_dist is not None and best_dist <= snap_radius:
            return best_point
        return point

    def _snap_targets(self):
        state = self.viewer.architect_state
        points = []
        for wall in state.walls:
            points.append(QtCore.QPointF(wall.start[0], wall.start[1]))
            points.append(QtCore.QPointF(wall.end[0], wall.end[1]))
        for fence in state.fences:
            points.append(QtCore.QPointF(fence.start[0], fence.start[1]))
            points.append(QtCore.QPointF(fence.end[0], fence.end[1]))
        for fx, fy in state.footprint_points:
            points.append(QtCore.QPointF(fx, fy))
        for fx, fy in self._footprint_points:
            points.append(QtCore.QPointF(fx, fy))
        return points

    def _axis_lock_point(self, start: QtCore.QPointF, end: QtCore.QPointF) -> QtCore.QPointF:
        if not self.viewer.architect_settings.axis_lock:
            return end
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        if abs(dx) >= abs(dy):
            return QtCore.QPointF(end.x(), start.y())
        return QtCore.QPointF(start.x(), end.y())

    def _start_drag(self, pos: QtCore.QPointF):
        self._dragging = True
        self._drag_start = pos

    def _finish_drag(self):
        self._dragging = False
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def _update_preview_line(self, start: QtCore.QPointF, end: QtCore.QPointF, color: QtGui.QColor):
        if self._preview_item is None:
            pen = QtGui.QPen(color)
            pen.setStyle(QtCore.Qt.DashLine)
            self._preview_item = self._scene.addLine(QtCore.QLineF(start, end), pen)
        else:
            if isinstance(self._preview_item, QtWidgets.QGraphicsLineItem):
                self._preview_item.setLine(QtCore.QLineF(start, end))

    def _update_preview_rect(self, start: QtCore.QPointF, end: QtCore.QPointF, color: QtGui.QColor):
        rect = QtCore.QRectF(start, end).normalized()
        if self._preview_item is None:
            pen = QtGui.QPen(color)
            pen.setStyle(QtCore.Qt.DashLine)
            self._preview_item = self._scene.addRect(rect, pen)
        else:
            if isinstance(self._preview_item, QtWidgets.QGraphicsRectItem):
                self._preview_item.setRect(rect)

    def _update_footprint_preview(self, current: Optional[QtCore.QPointF] = None):
        if not self._footprint_points:
            if self._footprint_preview is not None:
                self._scene.removeItem(self._footprint_preview)
                self._footprint_preview = None
            return

        points = list(self._footprint_points)
        if current is not None:
            points.append((current.x(), current.y()))
        path = QtGui.QPainterPath()
        first = points[0]
        path.moveTo(first[0], first[1])
        for x, y in points[1:]:
            path.lineTo(x, y)
        if self._footprint_preview is None:
            pen = QtGui.QPen(QtGui.QColor(120, 200, 240, 200))
            pen.setStyle(QtCore.Qt.DashLine)
            self._footprint_preview = self._scene.addPath(path, pen)
        else:
            self._footprint_preview.setPath(path)

    def _close_footprint(self):
        if len(self._footprint_points) < 3:
            self.viewer._set_architect_status("Footprint needs at least 3 points.")
            return
        self.viewer.architect_set_footprint(list(self._footprint_points))
        self._footprint_points = []
        if self._footprint_preview is not None:
            self._scene.removeItem(self._footprint_preview)
            self._footprint_preview = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        if event.button() != QtCore.Qt.LeftButton:
            if (
                event.button() == QtCore.Qt.RightButton
                and self.viewer.architect_settings.tool == "footprint"
            ):
                self._close_footprint()
                return
            return super().mousePressEvent(event)

        tool = self.viewer.architect_settings.tool
        pos = self._scene_pos(event)

        if tool == "select":
            self.viewer.architect_delete_at((pos.x(), pos.y()))
            return

        if tool in ("wall", "fence", "room"):
            self._start_drag(pos)
            preview_color = QtGui.QColor(220, 220, 220, 180)
            if tool == "room":
                self._update_preview_rect(pos, pos, preview_color)
            else:
                self._update_preview_line(pos, pos, preview_color)
            return

        if tool == "footprint":
            if self.viewer.architect_settings.axis_lock and self._footprint_points:
                last_point = QtCore.QPointF(self._footprint_points[-1][0], self._footprint_points[-1][1])
                pos = self._axis_lock_point(last_point, pos)
            self._footprint_points.append((pos.x(), pos.y()))
            self._update_footprint_preview()
            return

        if tool in ("door", "window"):
            self.viewer.architect_add_opening(tool, (pos.x(), pos.y()))
            return

        if tool == "spawn":
            self.viewer.architect_set_spawn((pos.x(), pos.y()))
            return

        if tool == "ina_spawn":
            self.viewer.architect_set_ina_spawn((pos.x(), pos.y()))
            return

        if tool == "ceiling_light":
            self.viewer.architect_add_ceiling_light((pos.x(), pos.y()))
            return

        if tool == "switch":
            self.viewer.architect_add_light_switch((pos.x(), pos.y()))
            return

        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        tool = self.viewer.architect_settings.tool
        pos = self._scene_pos(event)
        self._last_mouse_scene_pos = pos

        if tool == "footprint":
            if self.viewer.architect_settings.axis_lock and self._footprint_points:
                last_point = QtCore.QPointF(self._footprint_points[-1][0], self._footprint_points[-1][1])
                pos = self._axis_lock_point(last_point, pos)
            self._update_footprint_preview(current=pos)
            return

        if self._dragging:
            preview_color = QtGui.QColor(220, 220, 220, 180)
            if tool == "room":
                self._update_preview_rect(self._drag_start, pos, preview_color)
            else:
                pos = self._axis_lock_point(self._drag_start, pos)
                pos = self._snap_to_existing(pos)
                self._update_preview_line(self._drag_start, pos, preview_color)
            return

        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            return

        if event.button() != QtCore.Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        if not self._dragging:
            return super().mouseReleaseEvent(event)

        tool = self.viewer.architect_settings.tool
        end_pos = self._scene_pos(event)
        if tool in ("wall", "fence"):
            end_pos = self._axis_lock_point(self._drag_start, end_pos)
            end_pos = self._snap_to_existing(end_pos)
        start_pos = self._drag_start
        self._finish_drag()

        if tool == "wall":
            self.viewer.architect_add_wall((start_pos.x(), start_pos.y()), (end_pos.x(), end_pos.y()))
        elif tool == "fence":
            self.viewer.architect_add_fence((start_pos.x(), start_pos.y()), (end_pos.x(), end_pos.y()))
        elif tool == "room":
            self.viewer.architect_add_room(start_pos, end_pos)

        return super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.viewer.architect_settings.tool == "footprint":
            self._close_footprint()
            return
        return super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if (
            event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace)
            and self.viewer.architect_settings.tool == "select"
            and self._last_mouse_scene_pos is not None
        ):
            pos = self._last_mouse_scene_pos
            self.viewer.architect_delete_at((pos.x(), pos.y()))
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        if event.angleDelta().y() > 0:
            factor = zoom_in
        else:
            factor = zoom_out
        self.scale(factor, factor)

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        super().drawBackground(painter, rect)
        settings = self.viewer.architect_settings
        grid = settings.grid_size
        if grid <= 1e-6:
            return

        left = int(np.floor(rect.left() / grid)) * grid
        right = int(np.ceil(rect.right() / grid)) * grid
        top = int(np.floor(rect.top() / grid)) * grid
        bottom = int(np.ceil(rect.bottom() / grid)) * grid

        minor_pen = QtGui.QPen(QtGui.QColor(40, 40, 45))
        major_pen = QtGui.QPen(QtGui.QColor(65, 65, 75))
        major_step = grid * 5

        x = left
        while x <= right:
            painter.setPen(major_pen if abs(x / major_step - round(x / major_step)) < 1e-6 else minor_pen)
            painter.drawLine(QtCore.QLineF(x, top, x, bottom))
            x += grid

        y = top
        while y <= bottom:
            painter.setPen(major_pen if abs(y / major_step - round(y / major_step)) < 1e-6 else minor_pen)
            painter.drawLine(QtCore.QLineF(left, y, right, y))
            y += grid


class RadialMenu(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.98)
        self._buttons = []
        self._radius = 90
        self._button_size = QtCore.QSize(120, 30)

    def show_menu(self, global_center: QtCore.QPoint, actions: List[Tuple[str, callable]]):
        self._clear_buttons()
        if not actions:
            return
        count = len(actions)
        angle_step = (2.0 * math.pi) / max(count, 1)
        diameter = (self._radius * 2) + self._button_size.width()
        self.setFixedSize(diameter, diameter)
        center = QtCore.QPoint(diameter // 2, diameter // 2)

        for idx, (label, callback) in enumerate(actions):
            angle = angle_step * idx - (math.pi / 2.0)
            dx = math.cos(angle) * self._radius
            dy = math.sin(angle) * self._radius
            btn = QtWidgets.QToolButton(self)
            btn.setText(label)
            btn.setFixedSize(self._button_size)
            btn.move(
                int(center.x() + dx - self._button_size.width() / 2),
                int(center.y() + dy - self._button_size.height() / 2),
            )
            btn.clicked.connect(self._wrap_callback(callback))
            self._buttons.append(btn)

        self.move(global_center.x() - diameter // 2, global_center.y() - diameter // 2)
        self.show()

    def _wrap_callback(self, callback):
        def _handler():
            self.close()
            if callback is not None:
                callback()
        return _handler

    def _clear_buttons(self):
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons = []

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Backspace):
            self.close()
            return
        super().keyPressEvent(event)


class CrosshairWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._size = 12
        self._gap = 4
        self._thickness = 2
        self._color = QtGui.QColor(235, 235, 235, 190)
        diameter = (self._size * 2) + (self._thickness * 2)
        self.setFixedSize(diameter, diameter)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(self._color)
        pen.setWidth(self._thickness)
        painter.setPen(pen)
        center = self.rect().center()
        painter.drawLine(center.x() - self._size, center.y(), center.x() - self._gap, center.y())
        painter.drawLine(center.x() + self._gap, center.y(), center.x() + self._size, center.y())
        painter.drawLine(center.x(), center.y() - self._size, center.x(), center.y() - self._gap)
        painter.drawLine(center.x(), center.y() + self._gap, center.x(), center.y() + self._size)


class HouseViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ina House Viewer (Prototype)")

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((10, 10, 20))
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.crosshair = CrosshairWidget(self.view)
        self.crosshair.hide()
        self.architect_state = ArchitectState()
        self.architect_settings = ArchitectSettings()
        self.architect_dirty = False
        self.architect_loaded = False
        self.architect_plan_cache = None
        self.architect_room_counter = 1
        self._runtime_rooms = []
        self.doors: List[DoorInstance] = []
        self.door_items: List[gl.GLGraphicsItem] = []
        self.door_open_speed = 120.0
        self.door_interact_distance = 2.0
        self._door_state_callback = None
        self._door_state_cache = {}
        self.architect_canvas = ArchitectCanvas(self)
        self.central_stack = QtWidgets.QStackedWidget()
        self.central_stack.addWidget(self.view)
        self.central_stack.addWidget(self.architect_canvas)
        self.setCentralWidget(self.central_stack)

        # Camera defaults
        self.default_distance = 35
        self.default_elevation = 20
        self.default_azimuth = 45
        self.scene_center = pg.Vector(0.0, 0.0, 0.0)
        self.scene_distance = self.default_distance
        self.cam_center = self.scene_center
        self.cam_distance = self.default_distance
        self.cam_azimuth = self.default_azimuth
        self.cam_elevation = self.default_elevation
        self.view.opts["distance"] = self.default_distance
        self.view.opts["elevation"] = self.default_elevation
        self.view.opts["azimuth"] = self.default_azimuth

        # Player / first-person state
        self.first_person_enabled = False
        self.player_eye_height = 1.6
        self.player_height = 1.7
        self.player_width = 0.4
        self.player_depth = 0.4
        self.player_eye_height_stand = self.player_eye_height
        self.player_crouch_factor = 0.55
        self.player_crouch_speed_multiplier = 0.55
        self.player_crouched = False
        self.player_pos = None  # GL coords: x, y, z
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self.player_speed = 4.0
        self.player_sprint_multiplier = 2.0
        self.player_turn_speed = 90.0
        self.player_look_distance = 0.6
        self.mouse_sensitivity = 0.15
        self.ground_snap_enabled = True
        self.player_vertical_velocity = 0.0
        self.player_jump_speed = 4.8
        self.player_gravity = 9.8
        self.player_respawn_depth = 6.0
        self._jump_consumed = False
        self.door_block_angle = 8.0
        self.max_light_intensity = 2.0
        self.max_light_value = 1.5
        self._climb_surface_instance = None
        self._climb_surface_z = None
        self._climb_surface_radius = 0.0
        self._climb_surface_margin = 0.05
        self.player_items: List[gl.GLGraphicsItem] = []
        self.player_item = None
        self.player_avatar_enabled = True
        self.player_avatar_parts: List[FurniturePart] = []
        self.ina_items: List[gl.GLGraphicsItem] = []
        self.ina_avatar_parts: List[FurniturePart] = []
        self.ina_pos = None
        self.ina_velocity = np.zeros(3, dtype=float)
        self.ina_anim_use_inastate = True
        self.ina_schema_path = "body_schema.json"
        self._schema_cache = {}
        self.seated = False
        self.seated_return_pos = None
        self.seated_return_yaw = 0.0
        self.seated_eye_height_factor = 0.72
        self._desk_in_use = False
        self._desk_use_callback = None
        self.room_light_state = {}
        self.player_schema_path = "player_schema.json"
        self.exterior_model = None
        self._interaction_highlight = None
        self._interaction_target = None
        self._player_anim_phase = 0.0
        self._player_anim_last_pos = None
        self._player_anim_last_ts = time.perf_counter()
        self._ina_anim_phase = 0.0
        self._ina_anim_last_pos = None
        self._ina_anim_last_ts = time.perf_counter()
        self._hifi_player_cache = None
        self._hifi_player_cache_ts = 0.0
        self.tv_channel = self._load_tv_channel()
        self._tv_stream_last_update = 0.0
        self._tv_stream_interval = 1.0
        self._tv_stream_bounds = None
        self._tv_stream_config_mtime = None
        self._tv_stream_items = {}
        self._tv_stream_size = (320, 180)
        self.radial_menu = RadialMenu(self)
        self._radial_menu_active = False
        self._radial_menu_target = None
        self.radial_menu.finished.connect(self._on_radial_menu_closed)
        self._light_dimmer_dialog = None
        self.interact_hold_threshold = 0.35
        self._interact_hold_timer = QtCore.QTimer(self)
        self._interact_hold_timer.setSingleShot(True)
        self._interact_hold_timer.timeout.connect(self._on_interact_hold_timeout)
        self._interact_press_active = False
        self._interact_press_target = None
        self._interact_hold_opened = False
        self._load_player_schema()
        self.keys_down = set()
        self._orbit_camera_state = None
        self._last_mouse_pos = None
        self._mouse_captured = False
        self._recentering = False
        self.mouse_capture_supported = self._detect_mouse_capture_support()
        self.mouse_look_requires_button = not self.mouse_capture_supported
        self.mouse_look_button = QtCore.Qt.RightButton
        self._mouse_look_active = False
        self._mouse_cursor_hidden = False
        self._last_tick = time.perf_counter()
        self._tick_timer = QtCore.QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(16)
        self._player_control_keys = {
            QtCore.Qt.Key_W,
            QtCore.Qt.Key_A,
            QtCore.Qt.Key_S,
            QtCore.Qt.Key_D,
            QtCore.Qt.Key_Space,
            QtCore.Qt.Key_C,
            QtCore.Qt.Key_Shift,
            QtCore.Qt.Key_Control,
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
        }

        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)

        # Reference grid
        self.grid_item = gl.GLGridItem()
        self.grid_item.scale(2, 2, 1)
        self.view.addItem(self.grid_item)

        # Toolbar
        toolbar = self.addToolBar("Main")
        reset_act = QtWidgets.QAction("Reset Camera", self)
        reset_act.triggered.connect(self.reset_camera)
        toolbar.addAction(reset_act)

        reload_act = QtWidgets.QAction("Reload Scene", self)
        reload_act.triggered.connect(self.reload_scene)
        toolbar.addAction(reload_act)

        fps_act = QtWidgets.QAction("First-Person", self)
        fps_act.setCheckable(True)
        fps_act.toggled.connect(self.set_first_person_enabled)
        toolbar.addAction(fps_act)
        self.first_person_action = fps_act

        avatar_act = QtWidgets.QAction("Player Avatar", self)
        avatar_act.setCheckable(True)
        avatar_act.setChecked(self.player_avatar_enabled)
        avatar_act.toggled.connect(self.set_player_avatar_enabled)
        toolbar.addAction(avatar_act)
        self.player_avatar_action = avatar_act

        architect_act = QtWidgets.QAction("Architect", self)
        architect_act.setCheckable(True)
        architect_act.toggled.connect(self.set_architect_mode)
        toolbar.addAction(architect_act)
        self.architect_action = architect_act

        furnish_act = QtWidgets.QAction("Furnish", self)
        furnish_act.setCheckable(True)
        furnish_act.toggled.connect(self.set_furnish_mode)
        toolbar.addAction(furnish_act)
        self.furnish_action = furnish_act

        # Store GL items so we can remove them on reload
        self.exterior_items: List[gl.GLGraphicsItem] = []
        self.interior_items: List[gl.GLGraphicsItem] = []
        self.furniture_items: List[gl.GLGraphicsItem] = []
        self.light_items: List[gl.GLGraphicsItem] = []
        self.sky_items: List[gl.GLGraphicsItem] = []
        self.sky_radius = 90.0
        self._sky_last_update = 0.0
        self.sun_location_name = "SE14"
        self.sun_location = (51.48, -0.03)
        self._sun_times_cache = {
            "date": None,
            "sunrise": None,
            "sunset": None,
            "fetched_at": 0.0,
            "source": None,
        }
        self._lit_items = []
        self._lighting_last_update = 0.0

        # For now: allow toggling interior debug rendering if you want it
        self.show_interior = True

        self.furnish_mode_enabled = False
        self.furniture_catalog = self._build_furniture_catalog()
        self.furniture_active_key = next(iter(self.furniture_catalog), None)
        self.furniture_rotation = 0.0
        self.furniture_snap_enabled = True
        self.furniture_grid_size = self.architect_settings.grid_size
        self.furniture_instances: List[FurnitureInstance] = []
        self.furniture_instances_by_id = {}
        self.furniture_preview: Optional[FurnitureInstance] = None
        self._furniture_preview_items: List[gl.GLGraphicsItem] = []
        self._furniture_instance_counter = 1
        self._furnish_last_mouse_pos = None
        self._furnish_press_pos = None

        self.architect_dock = self._build_architect_dock()
        self.architect_dock.hide()
        self.furnish_dock = self._build_furnish_dock()
        self.furnish_dock.hide()

        self.reload_scene()

    # ---- Coordinate helpers ----

    @staticmethod
    def _to_gl_pos(pos):
        """Map model coords (x, y-up, z-depth) to GL coords (x, y-depth, z-up)."""
        x, y, z = pos
        return (x, z, y)

    @staticmethod
    def _to_model_pos(pos):
        """Map GL coords (x, y-depth, z-up) to model coords (x, y-up, z-depth)."""
        x, y, z = pos
        return (x, z, y)

    @staticmethod
    def _to_gl_size(size):
        """Swap model height/depth so height becomes the GL z extent."""
        w, h, d = size
        return (w, d, h)

    @staticmethod
    def _to_gl_normal(normal):
        nx, ny, nz = normal
        return (nx, nz, ny)

    @staticmethod
    def _normalize_tags(tags, fallback: Optional[List[str]] = None) -> List[str]:
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags, (list, tuple)):
            cleaned = []
            for tag in tags:
                if not isinstance(tag, str):
                    continue
                text = tag.strip()
                if text:
                    cleaned.append(text)
            if cleaned:
                return cleaned
        if fallback:
            return list(fallback)
        return []

    def _is_sofa_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key in ("sofa", "couch"):
            return True
        tags = self._normalize_tags(instance.tags)
        return "sofa" in tags or "couch" in tags

    def _is_desk_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "desk":
            return True
        tags = self._normalize_tags(instance.tags)
        return "desk" in tags or "computer" in tags

    def _is_counter_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "counter":
            return True
        tags = self._normalize_tags(instance.tags)
        return "counter" in tags

    def _is_hifi_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "hifi":
            return True
        tags = self._normalize_tags(instance.tags)
        return "hifi" in tags or "hi-fi" in tags

    def _is_tv_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "tv":
            return True
        tags = self._normalize_tags(instance.tags)
        return "tv" in tags or "screen" in tags

    def _is_bookshelf_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "bookshelf":
            return True
        tags = self._normalize_tags(instance.tags)
        return "bookshelf" in tags or "books" in tags or "reading" in tags

    def _is_living_room_unit_instance(self, instance: FurnitureInstance) -> bool:
        if instance.key == "living_room_unit":
            return True
        tags = self._normalize_tags(instance.tags)
        return "living_room_unit" in tags or ("living_room" in tags and "cabinet" in tags)

    def _kitchen_appliance_kind(self, instance: FurnitureInstance) -> Optional[str]:
        if instance.key == "fridge":
            return "fridge"
        if instance.key == "microwave":
            return "microwave"
        if instance.key in ("cooker", "stove", "oven"):
            return "cooker"
        tags = self._normalize_tags(instance.tags)
        if "fridge" in tags:
            return "fridge"
        if "microwave" in tags:
            return "microwave"
        if "cooker" in tags or "stove" in tags or "oven" in tags:
            return "cooker"
        return None

    def _local_bounds_from_defs(self, defs: List[FurniturePartDef]):
        min_x = min_y = min_z = None
        max_x = max_y = max_z = None
        for part_def in defs:
            size = part_def.size
            offset = part_def.offset
            half_x = float(size[0]) * 0.5
            half_y = float(size[1]) * 0.5
            half_z = float(size[2]) * 0.5
            x0 = float(offset[0]) - half_x
            x1 = float(offset[0]) + half_x
            y0 = float(offset[1]) - half_y
            y1 = float(offset[1]) + half_y
            z0 = float(offset[2]) - half_z
            z1 = float(offset[2]) + half_z
            min_x = x0 if min_x is None else min(min_x, x0)
            max_x = x1 if max_x is None else max(max_x, x1)
            min_y = y0 if min_y is None else min(min_y, y0)
            max_y = y1 if max_y is None else max(max_y, y1)
            min_z = z0 if min_z is None else min(min_z, z0)
            max_z = z1 if max_z is None else max(max_z, z1)
        return min_x, max_x, min_y, max_y, min_z, max_z

    def _local_bounds_from_parts(self, parts: List[FurniturePart], use_base: bool = True):
        min_x = min_y = min_z = None
        max_x = max_y = max_z = None
        for part in parts:
            size = part.base_size if use_base else part.size
            offset = part.base_offset if use_base else part.offset
            half_x = float(size[0]) * 0.5
            half_y = float(size[1]) * 0.5
            half_z = float(size[2]) * 0.5
            x0 = float(offset[0]) - half_x
            x1 = float(offset[0]) + half_x
            y0 = float(offset[1]) - half_y
            y1 = float(offset[1]) + half_y
            z0 = float(offset[2]) - half_z
            z1 = float(offset[2]) + half_z
            min_x = x0 if min_x is None else min(min_x, x0)
            max_x = x1 if max_x is None else max(max_x, x1)
            min_y = y0 if min_y is None else min(min_y, y0)
            max_y = y1 if max_y is None else max(max_y, y1)
            min_z = z0 if min_z is None else min(min_z, z0)
            max_z = z1 if max_z is None else max(max_z, z1)
        return min_x, max_x, min_y, max_y, min_z, max_z

    def _clear_climb_surface(self) -> None:
        self._climb_surface_instance = None
        self._climb_surface_z = None
        self._climb_surface_radius = 0.0

    def _furniture_vertical_bounds(self, instance: FurnitureInstance, *, use_base: bool = True):
        bounds = self._local_bounds_from_parts(instance.parts, use_base=use_base)
        min_z = bounds[4]
        max_z = bounds[5]
        if min_z is None or max_z is None:
            return None
        base_z = float(instance.position[2])
        return base_z + float(min_z), base_z + float(max_z)

    def _furniture_surface_z(self, instance: FurnitureInstance) -> Optional[float]:
        if instance.seat_offset:
            try:
                return float(instance.position[2]) + float(instance.seat_offset[2])
            except Exception:
                pass
        bounds = self._furniture_vertical_bounds(instance, use_base=False)
        if bounds is None:
            return None
        return bounds[1]

    def _furniture_surface_position(self, instance: FurnitureInstance) -> np.ndarray:
        pos = np.array(instance.position, dtype=float)
        surface_z = self._furniture_surface_z(instance)
        if surface_z is not None:
            pos[2] = surface_z
        return pos

    def _is_climbable_instance(self, instance: FurnitureInstance) -> bool:
        tag_set = set(self._normalize_tags(instance.tags))
        blocked = {"tv", "screen", "hifi", "lamp", "microwave", "fridge", "cooker", "stove", "appliance"}
        if any(tag in tag_set for tag in blocked):
            return False
        climb_tags = {"surface", "seat", "bed", "sofa", "desk", "counter", "cabinet", "bookshelf"}
        return bool(tag_set.intersection(climb_tags))

    def _nearest_climbable_surface(self, used_ids: set[int]):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if instance.instance_id in used_ids:
                continue
            if not self._is_climbable_instance(instance):
                continue
            radius = max(instance.interact_radius, instance.collision_radius + 0.4)
            surface_pos = self._furniture_surface_position(instance)
            surface_xy = np.array(surface_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - surface_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, surface_pos)
        if best is None or best_dist is None:
            return None
        instance, surface_pos = best
        radius = max(instance.interact_radius, instance.collision_radius + 0.4)
        if best_dist > radius:
            return None
        return instance, surface_pos, best_dist

    def _update_climb_surface(self) -> Optional[float]:
        if self._climb_surface_instance is None or self.player_pos is None:
            return None
        instance = self._climb_surface_instance
        if instance not in self.furniture_instances:
            self._clear_climb_surface()
            return None
        surface_z = self._furniture_surface_z(instance)
        if surface_z is None:
            self._clear_climb_surface()
            return None
        dx = float(self.player_pos[0]) - float(instance.position[0])
        dy = float(self.player_pos[1]) - float(instance.position[1])
        radius = self._climb_surface_radius or max(instance.collision_radius, 0.5)
        if (dx * dx + dy * dy) > (radius * radius):
            self._clear_climb_surface()
            return None
        self._climb_surface_z = surface_z
        return surface_z

    # ---- Camera helpers ----

    def reset_camera(self):
        if self.first_person_enabled:
            self._reset_player()
            self._sync_first_person_camera()
            return
        self._update_camera_position(recenter=True)

    # ---- Scene management ----

    def clear_items(self, items: List[gl.GLGraphicsItem]):
        for item in items:
            self.view.removeItem(item)
            if item in self._lit_items:
                self._lit_items.remove(item)
        items.clear()

    def reload_scene(self):
        self.clear_items(self.exterior_items)
        self.clear_items(self.interior_items)
        self.clear_items(self.player_items)
        self.clear_items(self.door_items)
        self.clear_items(self.furniture_items)
        self.clear_items(getattr(self, "light_items", []))
        self.clear_items(getattr(self, "sky_items", []))
        self.clear_items(self.ina_items)
        self._clear_furnish_preview()
        self._lit_items = []
        self.doors.clear()
        self.player_item = None
        self.player_avatar_parts = []
        self.player_pos = None
        self.ina_avatar_parts = []
        self.ina_pos = None
        self.room_light_state = {}
        self.furniture_instances.clear()
        self.furniture_instances_by_id.clear()
        self._furniture_instance_counter = 1
        self._refresh_furniture_list()

        # Exterior
        try:
            house, exterior = load_house_from_plan()
        except FileNotFoundError:
            house = create_prototype_house()
            exterior = create_prototype_exterior()
        except Exception:
            house = create_prototype_house()
            exterior = create_prototype_exterior()
        self.exterior_model = exterior
        self._runtime_rooms = [room for room in house.rooms if room.name != "garden"]
        self._build_exterior(exterior)

        self._build_sky()

        # Interior boxes (optional debug)
        if self.show_interior:
            self._build_interior(house)

        self._load_furniture_from_plan()
        self._load_lighting_from_plan()

        self._load_player_schema()
        self._ensure_player()
        self._update_player_visibility()
        self._ensure_ina_avatar()
        if self.first_person_enabled:
            self._sync_first_person_camera()
        if self.furnish_mode_enabled:
            self._ensure_furnish_preview()

    # ---- Architect mode ----

    def _build_architect_dock(self) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget("Architect", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        actions_box = QtWidgets.QGroupBox("Plan Actions")
        actions_layout = QtWidgets.QGridLayout(actions_box)
        load_btn = QtWidgets.QPushButton("Load Plan")
        save_btn = QtWidgets.QPushButton("Save Plan")
        save_reload_btn = QtWidgets.QPushButton("Save + Reload 3D")
        clear_btn = QtWidgets.QPushButton("Clear All")
        actions_layout.addWidget(load_btn, 0, 0)
        actions_layout.addWidget(save_btn, 0, 1)
        actions_layout.addWidget(save_reload_btn, 1, 0, 1, 2)
        actions_layout.addWidget(clear_btn, 2, 0, 1, 2)
        layout.addWidget(actions_box)

        load_btn.clicked.connect(self.load_architect_from_plan)
        save_btn.clicked.connect(self.save_architect_plan)
        save_reload_btn.clicked.connect(lambda: self.save_architect_plan(reload_after=True))
        clear_btn.clicked.connect(self.architect_clear)

        tool_box = QtWidgets.QGroupBox("Tool")
        tool_layout = QtWidgets.QGridLayout(tool_box)
        self.architect_tool_group = QtWidgets.QButtonGroup(self)
        self.architect_tool_buttons = {}
        tools = [
            ("select", "Select"),
            ("footprint", "Footprint"),
            ("wall", "Wall"),
            ("room", "Room"),
            ("door", "Door"),
            ("window", "Window"),
            ("spawn", "Player Spawn"),
            ("ina_spawn", "Ina Spawn"),
            ("fence", "Fence"),
            ("ceiling_light", "Ceiling Light"),
            ("switch", "Switch"),
        ]
        for index, (tool_key, label) in enumerate(tools):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            self.architect_tool_group.addButton(button)
            self.architect_tool_buttons[tool_key] = button
            row = index // 2
            col = index % 2
            tool_layout.addWidget(button, row, col)
            button.clicked.connect(lambda checked, key=tool_key: self._set_architect_tool(key))
        layout.addWidget(tool_box)

        self.architect_controls = {}
        tabs = QtWidgets.QTabWidget()

        wall_tab = QtWidgets.QWidget()
        wall_layout = QtWidgets.QFormLayout(wall_tab)
        wall_height = self._make_float_spin(0.1, 20.0, 0.05, self.architect_settings.wall_height)
        wall_thickness = self._make_float_spin(0.02, 2.0, 0.02, self.architect_settings.wall_thickness)
        wall_color_widget, wall_swatch, wall_color_button = self._make_color_picker(
            self.architect_settings.wall_color
        )
        wall_layout.addRow("Height (m)", wall_height)
        wall_layout.addRow("Thickness (m)", wall_thickness)
        wall_layout.addRow("Color", wall_color_widget)
        tabs.addTab(wall_tab, "Walls")
        self.architect_controls["wall_height"] = wall_height
        self.architect_controls["wall_thickness"] = wall_thickness
        self.architect_controls["wall_color_swatch"] = wall_swatch

        room_tab = QtWidgets.QWidget()
        room_layout = QtWidgets.QFormLayout(room_tab)
        room_height = self._make_float_spin(0.1, 10.0, 0.05, self.architect_settings.room_height)
        room_color_widget, room_swatch, room_color_button = self._make_color_picker(
            self.architect_settings.room_color
        )
        room_prefix = QtWidgets.QLineEdit(self.architect_settings.room_name_prefix)
        room_layout.addRow("Height (m)", room_height)
        room_layout.addRow("Color", room_color_widget)
        room_layout.addRow("Name Prefix", room_prefix)
        tabs.addTab(room_tab, "Rooms")
        self.architect_controls["room_height"] = room_height
        self.architect_controls["room_color_swatch"] = room_swatch
        self.architect_controls["room_prefix"] = room_prefix

        openings_tab = QtWidgets.QWidget()
        openings_layout = QtWidgets.QFormLayout(openings_tab)
        door_width = self._make_float_spin(0.4, 3.0, 0.05, self.architect_settings.door_width)
        door_height = self._make_float_spin(0.5, 4.0, 0.05, self.architect_settings.door_height)
        door_sill = self._make_float_spin(0.0, 2.0, 0.05, self.architect_settings.door_sill)
        window_width = self._make_float_spin(0.4, 4.0, 0.05, self.architect_settings.window_width)
        window_height = self._make_float_spin(0.4, 3.0, 0.05, self.architect_settings.window_height)
        window_sill = self._make_float_spin(0.0, 2.0, 0.05, self.architect_settings.window_sill)
        openings_layout.addRow("Door Width (m)", door_width)
        openings_layout.addRow("Door Height (m)", door_height)
        openings_layout.addRow("Door Sill (m)", door_sill)
        openings_layout.addRow("Window Width (m)", window_width)
        openings_layout.addRow("Window Height (m)", window_height)
        openings_layout.addRow("Window Sill (m)", window_sill)
        tabs.addTab(openings_tab, "Openings")
        self.architect_controls["door_width"] = door_width
        self.architect_controls["door_height"] = door_height
        self.architect_controls["door_sill"] = door_sill
        self.architect_controls["window_width"] = window_width
        self.architect_controls["window_height"] = window_height
        self.architect_controls["window_sill"] = window_sill

        fence_tab = QtWidgets.QWidget()
        fence_layout = QtWidgets.QFormLayout(fence_tab)
        fence_height = self._make_float_spin(0.1, 4.0, 0.05, self.architect_settings.fence_height)
        fence_thickness = self._make_float_spin(0.02, 1.0, 0.02, self.architect_settings.fence_thickness)
        fence_color_widget, fence_swatch, fence_color_button = self._make_color_picker(
            self.architect_settings.fence_color
        )
        fence_layout.addRow("Height (m)", fence_height)
        fence_layout.addRow("Thickness (m)", fence_thickness)
        fence_layout.addRow("Color", fence_color_widget)
        tabs.addTab(fence_tab, "Fences")
        self.architect_controls["fence_height"] = fence_height
        self.architect_controls["fence_thickness"] = fence_thickness
        self.architect_controls["fence_color_swatch"] = fence_swatch

        site_tab = QtWidgets.QWidget()
        site_layout = QtWidgets.QFormLayout(site_tab)
        roof_overhang = self._make_float_spin(0.0, 2.0, 0.05, self.architect_settings.roof_overhang)
        roof_thickness = self._make_float_spin(0.05, 1.0, 0.05, self.architect_settings.roof_thickness)
        roof_color_widget, roof_swatch, roof_color_button = self._make_color_picker(
            self.architect_settings.roof_color
        )
        ground_size_x = self._make_float_spin(5.0, 200.0, 0.5, self.architect_settings.ground_size[0])
        ground_size_y = self._make_float_spin(5.0, 200.0, 0.5, self.architect_settings.ground_size[1])
        ground_color_widget, ground_swatch, ground_color_button = self._make_color_picker(
            self.architect_settings.ground_color
        )
        grid_size = self._make_float_spin(0.1, 5.0, 0.1, self.architect_settings.grid_size)
        spawn_height = self._make_float_spin(-1.0, 5.0, 0.05, self.architect_settings.spawn_height)
        view_scale = self._make_float_spin(5.0, 200.0, 1.0, self.architect_settings.view_scale)
        snap_enabled = QtWidgets.QCheckBox("Snap To Grid")
        snap_enabled.setChecked(self.architect_settings.snap_enabled)
        snap_existing = QtWidgets.QCheckBox("Snap To Existing")
        snap_existing.setChecked(self.architect_settings.snap_existing)
        axis_lock = QtWidgets.QCheckBox("Axis Lock")
        axis_lock.setChecked(self.architect_settings.axis_lock)
        site_layout.addRow("Roof Overhang (m)", roof_overhang)
        site_layout.addRow("Roof Thickness (m)", roof_thickness)
        site_layout.addRow("Roof Color", roof_color_widget)
        site_layout.addRow("Ground Size X (m)", ground_size_x)
        site_layout.addRow("Ground Size Z (m)", ground_size_y)
        site_layout.addRow("Ground Color", ground_color_widget)
        site_layout.addRow("Grid Size (m)", grid_size)
        site_layout.addRow("View Scale (px/m)", view_scale)
        site_layout.addRow("Spawn Height (m)", spawn_height)
        site_layout.addRow("", snap_enabled)
        site_layout.addRow("", snap_existing)
        site_layout.addRow("", axis_lock)
        tabs.addTab(site_tab, "Site/Grid")
        self.architect_controls["roof_overhang"] = roof_overhang
        self.architect_controls["roof_thickness"] = roof_thickness
        self.architect_controls["roof_color_swatch"] = roof_swatch
        self.architect_controls["ground_size_x"] = ground_size_x
        self.architect_controls["ground_size_y"] = ground_size_y
        self.architect_controls["ground_color_swatch"] = ground_swatch
        self.architect_controls["grid_size"] = grid_size
        self.architect_controls["view_scale"] = view_scale
        self.architect_controls["spawn_height"] = spawn_height
        self.architect_controls["snap_enabled"] = snap_enabled
        self.architect_controls["snap_existing"] = snap_existing
        self.architect_controls["axis_lock"] = axis_lock

        layout.addWidget(tabs)
        layout.addStretch(1)

        dock.setWidget(container)

        wall_height.valueChanged.connect(lambda v: self._set_architect_setting("wall_height", v))
        wall_thickness.valueChanged.connect(lambda v: self._set_architect_setting("wall_thickness", v))
        room_height.valueChanged.connect(lambda v: self._set_architect_setting("room_height", v))
        room_prefix.textChanged.connect(lambda v: self._set_architect_setting("room_name_prefix", v.strip() or "room"))
        door_width.valueChanged.connect(lambda v: self._set_architect_setting("door_width", v))
        door_height.valueChanged.connect(lambda v: self._set_architect_setting("door_height", v))
        door_sill.valueChanged.connect(lambda v: self._set_architect_setting("door_sill", v))
        window_width.valueChanged.connect(lambda v: self._set_architect_setting("window_width", v))
        window_height.valueChanged.connect(lambda v: self._set_architect_setting("window_height", v))
        window_sill.valueChanged.connect(lambda v: self._set_architect_setting("window_sill", v))
        fence_height.valueChanged.connect(lambda v: self._set_architect_setting("fence_height", v))
        fence_thickness.valueChanged.connect(lambda v: self._set_architect_setting("fence_thickness", v))
        roof_overhang.valueChanged.connect(lambda v: self._set_architect_setting("roof_overhang", v))
        roof_thickness.valueChanged.connect(lambda v: self._set_architect_setting("roof_thickness", v))
        ground_size_x.valueChanged.connect(lambda v: self._update_ground_size(x=v, y=None))
        ground_size_y.valueChanged.connect(lambda v: self._update_ground_size(x=None, y=v))
        grid_size.valueChanged.connect(lambda v: self._set_architect_setting("grid_size", v))
        view_scale.valueChanged.connect(lambda v: self._set_architect_setting("view_scale", v))
        spawn_height.valueChanged.connect(lambda v: self._set_architect_setting("spawn_height", v))
        snap_enabled.toggled.connect(lambda v: self._set_architect_setting("snap_enabled", v))
        snap_existing.toggled.connect(lambda v: self._set_architect_setting("snap_existing", v))
        axis_lock.toggled.connect(lambda v: self._set_architect_setting("axis_lock", v))

        wall_color_button.clicked.connect(lambda: self._pick_architect_color("wall_color"))
        room_color_button.clicked.connect(lambda: self._pick_architect_color("room_color"))
        fence_color_button.clicked.connect(lambda: self._pick_architect_color("fence_color"))
        roof_color_button.clicked.connect(lambda: self._pick_architect_color("roof_color"))
        ground_color_button.clicked.connect(lambda: self._pick_architect_color("ground_color"))

        self.architect_tool_buttons[self.architect_settings.tool].setChecked(True)
        self._set_architect_tool(self.architect_settings.tool)

        return dock

    def _build_furnish_dock(self) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget("Furnish", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        palette_box = QtWidgets.QGroupBox("Furniture Palette")
        palette_layout = QtWidgets.QVBoxLayout(palette_box)
        palette_list = QtWidgets.QListWidget()
        for proto in self.furniture_catalog.values():
            item = QtWidgets.QListWidgetItem(proto.label)
            item.setData(QtCore.Qt.UserRole, proto.key)
            palette_list.addItem(item)
        palette_layout.addWidget(palette_list)
        layout.addWidget(palette_box)

        placement_box = QtWidgets.QGroupBox("Placement")
        placement_layout = QtWidgets.QGridLayout(placement_box)
        rotation_spin = QtWidgets.QSpinBox()
        rotation_spin.setRange(0, 359)
        rotation_spin.setSingleStep(5)
        rotation_spin.setValue(int(self.furniture_rotation))
        rotate_left = QtWidgets.QToolButton()
        rotate_left.setText("Rotate -")
        rotate_right = QtWidgets.QToolButton()
        rotate_right.setText("Rotate +")
        snap_enabled = QtWidgets.QCheckBox("Snap To Grid")
        snap_enabled.setChecked(self.furniture_snap_enabled)
        grid_size = self._make_float_spin(0.1, 5.0, 0.1, self.furniture_grid_size)

        placement_layout.addWidget(QtWidgets.QLabel("Rotation (deg)"), 0, 0)
        placement_layout.addWidget(rotation_spin, 0, 1)
        placement_layout.addWidget(rotate_left, 1, 0)
        placement_layout.addWidget(rotate_right, 1, 1)
        placement_layout.addWidget(snap_enabled, 2, 0, 1, 2)
        placement_layout.addWidget(QtWidgets.QLabel("Grid Size (m)"), 3, 0)
        placement_layout.addWidget(grid_size, 3, 1)
        layout.addWidget(placement_box)

        plan_box = QtWidgets.QGroupBox("Plan Actions")
        plan_layout = QtWidgets.QGridLayout(plan_box)
        save_btn = QtWidgets.QPushButton("Save Furniture")
        spawn_btn = QtWidgets.QPushButton("Ina Spawn At Bed")
        plan_layout.addWidget(save_btn, 0, 0)
        plan_layout.addWidget(spawn_btn, 0, 1)
        layout.addWidget(plan_box)

        placed_box = QtWidgets.QGroupBox("Placed Items")
        placed_layout = QtWidgets.QVBoxLayout(placed_box)
        placed_list = QtWidgets.QListWidget()
        placed_layout.addWidget(placed_list)
        placed_btns = QtWidgets.QHBoxLayout()
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        clear_btn = QtWidgets.QPushButton("Clear All")
        placed_btns.addWidget(remove_btn)
        placed_btns.addWidget(clear_btn)
        placed_layout.addLayout(placed_btns)
        layout.addWidget(placed_box)

        hint = QtWidgets.QLabel("Tip: click in the scene to place furniture.")
        hint.setStyleSheet("color: rgb(180, 180, 180);")
        layout.addWidget(hint)
        layout.addStretch(1)

        dock.setWidget(container)

        self.furniture_palette_list = palette_list
        self.furniture_list = placed_list
        self.furnish_rotation_spin = rotation_spin
        self.furnish_snap_checkbox = snap_enabled
        self.furnish_grid_spin = grid_size

        palette_list.currentItemChanged.connect(self._on_furnish_palette_changed)
        rotation_spin.valueChanged.connect(self._set_furniture_rotation)
        rotate_left.clicked.connect(lambda: self._adjust_furniture_rotation(-15.0))
        rotate_right.clicked.connect(lambda: self._adjust_furniture_rotation(15.0))
        snap_enabled.toggled.connect(self._toggle_furniture_snap)
        grid_size.valueChanged.connect(self._update_furniture_grid_size)
        remove_btn.clicked.connect(self._remove_selected_furniture)
        clear_btn.clicked.connect(self._clear_all_furniture)
        save_btn.clicked.connect(self._save_furniture_plan)
        spawn_btn.clicked.connect(self._set_spawn_to_bed)

        if self.furniture_active_key:
            for row in range(palette_list.count()):
                item = palette_list.item(row)
                if item.data(QtCore.Qt.UserRole) == self.furniture_active_key:
                    palette_list.setCurrentRow(row)
                    break

        return dock

    def _build_furniture_catalog(self) -> dict:
        catalog = {}

        bed_width = 2.1
        bed_depth = 1.6
        frame_height = 0.25
        mattress_height = 0.22
        headboard_height = 0.8
        headboard_thickness = 0.08
        pillow_width = 0.6
        pillow_depth = 0.4
        pillow_height = 0.12

        bed_parts = [
            FurniturePartDef(
                size=(bed_width, bed_depth, frame_height),
                offset=(0.0, 0.0, frame_height / 2.0),
                color=(0.45, 0.32, 0.2, 1.0),
            ),
            FurniturePartDef(
                size=(bed_width - 0.1, bed_depth - 0.1, mattress_height),
                offset=(0.0, 0.0, frame_height + mattress_height / 2.0),
                color=(0.86, 0.86, 0.9, 1.0),
            ),
            FurniturePartDef(
                size=(bed_width - 0.1, headboard_thickness, headboard_height),
                offset=(0.0, bed_depth / 2.0 - headboard_thickness / 2.0, headboard_height / 2.0),
                color=(0.4, 0.28, 0.18, 1.0),
            ),
            FurniturePartDef(
                size=(pillow_width, pillow_depth, pillow_height),
                offset=(
                    bed_width * 0.25,
                    bed_depth / 2.0 - headboard_thickness - pillow_depth / 2.0 - 0.05,
                    frame_height + mattress_height + pillow_height / 2.0,
                ),
                color=(0.95, 0.95, 0.96, 1.0),
                shape="sphere",
            ),
            FurniturePartDef(
                size=(pillow_width, pillow_depth, pillow_height),
                offset=(
                    -bed_width * 0.25,
                    bed_depth / 2.0 - headboard_thickness - pillow_depth / 2.0 - 0.05,
                    frame_height + mattress_height + pillow_height / 2.0,
                ),
                color=(0.95, 0.95, 0.96, 1.0),
                shape="sphere",
            ),
        ]
        bed_seat_height = frame_height + mattress_height * 0.85
        catalog["bed"] = FurniturePrototype(
            key="bed",
            label="Bed",
            parts=bed_parts,
            seat_offset=(0.0, 0.0, bed_seat_height),
            interact_radius=1.8,
            tags=["bed", "sleep", "rest", "furniture"],
        )

        table_width = 1.2
        table_depth = 0.8
        table_height = 0.75
        top_thickness = 0.06
        leg_thickness = 0.08
        leg_height = table_height - top_thickness

        table_parts = [
            FurniturePartDef(
                size=(table_width, table_depth, top_thickness),
                offset=(0.0, 0.0, table_height - top_thickness / 2.0),
                color=(0.55, 0.38, 0.22, 1.0),
            ),
        ]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                table_parts.append(
                    FurniturePartDef(
                        size=(leg_thickness, leg_thickness, leg_height),
                        offset=(
                            x_sign * (table_width / 2.0 - leg_thickness / 2.0),
                            y_sign * (table_depth / 2.0 - leg_thickness / 2.0),
                            leg_height / 2.0,
                        ),
                        color=(0.5, 0.35, 0.2, 1.0),
                        shape="cylinder",
                    )
                )
        catalog["table"] = FurniturePrototype(
            key="table",
            label="Table",
            parts=table_parts,
            tags=["table", "surface", "furniture"],
        )

        chair_width = 0.5
        chair_depth = 0.5
        seat_height = 0.45
        seat_thickness = 0.05
        chair_leg_thickness = 0.05
        chair_leg_height = seat_height - seat_thickness
        back_height = 0.5
        back_thickness = 0.06

        chair_parts = [
            FurniturePartDef(
                size=(chair_width, chair_depth, seat_thickness),
                offset=(0.0, 0.0, seat_height - seat_thickness / 2.0),
                color=(0.48, 0.32, 0.18, 1.0),
            ),
            FurniturePartDef(
                size=(chair_width, back_thickness, back_height),
                offset=(0.0, chair_depth / 2.0 - back_thickness / 2.0, seat_height + back_height / 2.0),
                color=(0.46, 0.3, 0.18, 1.0),
            ),
        ]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                chair_parts.append(
                    FurniturePartDef(
                        size=(chair_leg_thickness, chair_leg_thickness, chair_leg_height),
                        offset=(
                            x_sign * (chair_width / 2.0 - chair_leg_thickness / 2.0),
                            y_sign * (chair_depth / 2.0 - chair_leg_thickness / 2.0),
                            chair_leg_height / 2.0,
                        ),
                        color=(0.4, 0.26, 0.16, 1.0),
                        shape="cylinder",
                    )
                )
        catalog["chair"] = FurniturePrototype(
            key="chair",
            label="Chair",
            parts=chair_parts,
            seat_offset=(0.0, 0.0, seat_height),
            interact_radius=0.9,
            tags=["chair", "seat", "furniture"],
        )

        sofa_width = 2.0
        sofa_depth = 0.9
        sofa_base_height = 0.22
        sofa_cushion_height = 0.18
        sofa_back_height = 0.7
        sofa_back_thickness = 0.12
        sofa_arm_width = 0.15
        sofa_arm_height = 0.55
        sofa_seat_height = sofa_base_height + sofa_cushion_height

        sofa_parts = [
            FurniturePartDef(
                size=(sofa_width, sofa_depth, sofa_base_height),
                offset=(0.0, 0.0, sofa_base_height / 2.0),
                color=(0.28, 0.24, 0.22, 1.0),
            ),
            FurniturePartDef(
                size=(sofa_width - 0.12, sofa_depth - 0.1, sofa_cushion_height),
                offset=(
                    0.0,
                    0.0,
                    sofa_base_height + sofa_cushion_height / 2.0,
                ),
                color=(0.55, 0.5, 0.46, 1.0),
                shape="sphere",
            ),
            FurniturePartDef(
                size=(sofa_width, sofa_back_thickness, sofa_back_height),
                offset=(
                    0.0,
                    sofa_depth / 2.0 - sofa_back_thickness / 2.0,
                    sofa_base_height + sofa_cushion_height + sofa_back_height / 2.0,
                ),
                color=(0.32, 0.28, 0.26, 1.0),
            ),
            FurniturePartDef(
                size=(sofa_arm_width, sofa_depth, sofa_arm_height),
                offset=(
                    -(sofa_width / 2.0 - sofa_arm_width / 2.0),
                    0.0,
                    sofa_arm_height / 2.0,
                ),
                color=(0.3, 0.26, 0.24, 1.0),
            ),
            FurniturePartDef(
                size=(sofa_arm_width, sofa_depth, sofa_arm_height),
                offset=(
                    sofa_width / 2.0 - sofa_arm_width / 2.0,
                    0.0,
                    sofa_arm_height / 2.0,
                ),
                color=(0.3, 0.26, 0.24, 1.0),
            ),
        ]
        catalog["sofa"] = FurniturePrototype(
            key="sofa",
            label="Sofa",
            parts=sofa_parts,
            seat_offset=(0.0, 0.0, sofa_seat_height),
            interact_radius=1.8,
            tags=["sofa", "couch", "seat", "sleep", "lounge", "furniture"],
        )

        unit_width = 1.6
        unit_depth = 0.45
        unit_height = 0.65
        unit_top_thickness = 0.04
        unit_body_height = unit_height - unit_top_thickness
        unit_door_height = 0.45
        unit_door_thickness = 0.02
        unit_door_width = (unit_width - 0.12) / 2.0
        unit_door_offset_y = unit_depth / 2.0 - unit_door_thickness / 2.0
        unit_door_offset_z = unit_door_height / 2.0 + 0.08

        unit_parts = [
            FurniturePartDef(
                size=(unit_width, unit_depth, unit_body_height),
                offset=(0.0, 0.0, unit_body_height / 2.0),
                color=(0.52, 0.4, 0.28, 1.0),
            ),
            FurniturePartDef(
                size=(unit_width, unit_depth, unit_top_thickness),
                offset=(0.0, 0.0, unit_body_height + unit_top_thickness / 2.0),
                color=(0.62, 0.5, 0.36, 1.0),
            ),
        ]
        for x_sign in (-1.0, 1.0):
            unit_parts.append(
                FurniturePartDef(
                    size=(unit_door_width, unit_door_thickness, unit_door_height),
                    offset=(
                        x_sign * (unit_door_width / 2.0 + 0.03),
                        unit_door_offset_y,
                        unit_door_offset_z,
                    ),
                    color=(0.46, 0.34, 0.24, 1.0),
                )
            )
        catalog["living_room_unit"] = FurniturePrototype(
            key="living_room_unit",
            label="Living Room Unit",
            parts=unit_parts,
            tags=["living_room", "cabinet", "surface", "furniture"],
        )

        hifi_width = 0.55
        hifi_depth = 0.3
        hifi_height = 0.2
        hifi_panel_height = 0.12
        hifi_panel_depth = 0.02

        hifi_parts = [
            FurniturePartDef(
                size=(hifi_width, hifi_depth, hifi_height),
                offset=(0.0, 0.0, hifi_height / 2.0),
                color=(0.16, 0.17, 0.2, 1.0),
            ),
            FurniturePartDef(
                size=(hifi_width - 0.06, hifi_panel_depth, hifi_panel_height),
                offset=(
                    0.0,
                    hifi_depth / 2.0 - hifi_panel_depth / 2.0,
                    hifi_panel_height / 2.0 + 0.03,
                ),
                color=(0.05, 0.05, 0.07, 1.0),
            ),
        ]
        catalog["hifi"] = FurniturePrototype(
            key="hifi",
            label="Hi-Fi Stack",
            parts=hifi_parts,
            interact_radius=1.15,
            tags=["hifi", "audio", "music", "electronics", "furniture"],
        )

        tv_width = 2.2
        tv_height = 1.2
        tv_depth = 0.08
        tv_base_width = 0.9
        tv_base_depth = 0.35
        tv_base_height = 0.04
        tv_neck_width = 0.12
        tv_neck_depth = 0.1
        tv_neck_height = 0.28

        tv_parts = [
            FurniturePartDef(
                size=(tv_base_width, tv_base_depth, tv_base_height),
                offset=(0.0, 0.0, tv_base_height / 2.0),
                color=(0.18, 0.18, 0.2, 1.0),
            ),
            FurniturePartDef(
                size=(tv_neck_width, tv_neck_depth, tv_neck_height),
                offset=(0.0, 0.0, tv_base_height + tv_neck_height / 2.0),
                color=(0.2, 0.2, 0.22, 1.0),
            ),
            FurniturePartDef(
                size=(tv_width, tv_depth, tv_height),
                offset=(
                    0.0,
                    0.0,
                    tv_base_height + tv_neck_height + tv_height / 2.0,
                ),
                color=(0.08, 0.1, 0.16, 1.0),
                emissive=0.35,
            ),
        ]
        catalog["tv"] = FurniturePrototype(
            key="tv",
            label="Big TV",
            parts=tv_parts,
            interact_radius=1.6,
            tags=["tv", "screen", "media", "furniture"],
        )

        desk_width = 1.4
        desk_depth = 0.7
        desk_height = 0.75
        desk_top_thickness = 0.05
        desk_leg_thickness = 0.08
        desk_leg_height = desk_height - desk_top_thickness
        desk_seat_height = 0.46
        desk_monitor_height = 0.35
        desk_monitor_width = 0.5
        desk_monitor_depth = 0.08

        desk_parts = [
            FurniturePartDef(
                size=(desk_width, desk_depth, desk_top_thickness),
                offset=(0.0, 0.0, desk_height - desk_top_thickness / 2.0),
                color=(0.5, 0.36, 0.22, 1.0),
            ),
            FurniturePartDef(
                size=(desk_monitor_width, desk_monitor_depth, desk_monitor_height),
                offset=(
                    0.0,
                    desk_depth / 2.0 - desk_monitor_depth / 2.0 - 0.05,
                    desk_height + desk_monitor_height / 2.0,
                ),
                color=(0.15, 0.16, 0.18, 1.0),
            ),
        ]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                desk_parts.append(
                    FurniturePartDef(
                        size=(desk_leg_thickness, desk_leg_thickness, desk_leg_height),
                        offset=(
                            x_sign * (desk_width / 2.0 - desk_leg_thickness / 2.0),
                            y_sign * (desk_depth / 2.0 - desk_leg_thickness / 2.0),
                            desk_leg_height / 2.0,
                        ),
                        color=(0.45, 0.32, 0.2, 1.0),
                        shape="cylinder",
                    )
                )
        catalog["desk"] = FurniturePrototype(
            key="desk",
            label="Computer Desk",
            parts=desk_parts,
            seat_offset=(0.0, -(desk_depth / 2.0 + 0.35), desk_seat_height),
            interact_radius=1.3,
            tags=["desk", "computer", "work", "discord", "furniture"],
        )

        shelf_width = 1.4
        shelf_depth = 0.35
        shelf_height = 1.8
        shelf_thickness = 0.04
        shelf_side_thickness = 0.06
        shelf_back_thickness = 0.03
        shelf_color = (0.5, 0.36, 0.22, 1.0)
        shelf_accent = (0.42, 0.3, 0.2, 1.0)

        shelf_parts = [
            FurniturePartDef(
                size=(shelf_width, shelf_depth, shelf_thickness),
                offset=(0.0, 0.0, shelf_thickness / 2.0),
                color=shelf_color,
            ),
            FurniturePartDef(
                size=(shelf_width, shelf_depth, shelf_thickness),
                offset=(0.0, 0.0, shelf_height - shelf_thickness / 2.0),
                color=shelf_color,
            ),
            FurniturePartDef(
                size=(shelf_width, shelf_back_thickness, shelf_height),
                offset=(0.0, -shelf_depth / 2.0 + shelf_back_thickness / 2.0, shelf_height / 2.0),
                color=shelf_accent,
            ),
            FurniturePartDef(
                size=(shelf_side_thickness, shelf_depth, shelf_height),
                offset=(
                    -(shelf_width / 2.0 - shelf_side_thickness / 2.0),
                    0.0,
                    shelf_height / 2.0,
                ),
                color=shelf_accent,
            ),
            FurniturePartDef(
                size=(shelf_side_thickness, shelf_depth, shelf_height),
                offset=(
                    shelf_width / 2.0 - shelf_side_thickness / 2.0,
                    0.0,
                    shelf_height / 2.0,
                ),
                color=shelf_accent,
            ),
        ]
        for shelf_idx in range(1, 4):
            z = shelf_height * (shelf_idx / 4.0)
            shelf_parts.append(
                FurniturePartDef(
                    size=(shelf_width - shelf_side_thickness * 2.0, shelf_depth - shelf_back_thickness, shelf_thickness),
                    offset=(0.0, shelf_back_thickness / 2.0, z),
                    color=shelf_color,
                )
            )
        catalog["bookshelf"] = FurniturePrototype(
            key="bookshelf",
            label="Bookshelf",
            parts=shelf_parts,
            interact_radius=1.4,
            tags=["bookshelf", "books", "reading", "furniture", "surface"],
        )

        fridge_width = 0.9
        fridge_depth = 0.8
        fridge_height = 1.9
        fridge_door_thickness = 0.05
        fridge_handle_width = 0.06
        fridge_handle_depth = 0.04
        fridge_handle_height = 0.6

        fridge_parts = [
            FurniturePartDef(
                size=(fridge_width, fridge_depth, fridge_height),
                offset=(0.0, 0.0, fridge_height / 2.0),
                color=(0.92, 0.93, 0.95, 1.0),
            ),
            FurniturePartDef(
                size=(fridge_width - 0.04, fridge_door_thickness, fridge_height - 0.04),
                offset=(
                    0.0,
                    fridge_depth / 2.0 - fridge_door_thickness / 2.0,
                    fridge_height / 2.0,
                ),
                color=(0.88, 0.9, 0.92, 1.0),
            ),
            FurniturePartDef(
                size=(fridge_handle_width, fridge_handle_depth, fridge_handle_height),
                offset=(
                    fridge_width / 2.0 - fridge_handle_width / 2.0 - 0.05,
                    fridge_depth / 2.0 + fridge_handle_depth / 2.0 - 0.01,
                    fridge_height * 0.55,
                ),
                color=(0.72, 0.74, 0.76, 1.0),
                shape="cylinder",
            ),
        ]
        catalog["fridge"] = FurniturePrototype(
            key="fridge",
            label="Fridge",
            parts=fridge_parts,
            interact_radius=1.4,
            tags=["kitchen", "appliance", "fridge", "food", "snack", "furniture"],
        )

        microwave_width = 0.6
        microwave_depth = 0.45
        microwave_height = 0.35
        microwave_door_thickness = 0.03
        microwave_window_height = 0.18
        microwave_panel_width = 0.12

        microwave_parts = [
            FurniturePartDef(
                size=(microwave_width, microwave_depth, microwave_height),
                offset=(0.0, 0.0, microwave_height / 2.0),
                color=(0.28, 0.29, 0.32, 1.0),
            ),
            FurniturePartDef(
                size=(microwave_width - 0.08, microwave_door_thickness, microwave_window_height),
                offset=(
                    -microwave_panel_width * 0.45,
                    microwave_depth / 2.0 - microwave_door_thickness / 2.0,
                    microwave_height * 0.55,
                ),
                color=(0.08, 0.09, 0.11, 0.6),
                gl_options="translucent",
                emissive=0.18,
            ),
            FurniturePartDef(
                size=(microwave_panel_width, microwave_door_thickness, microwave_height * 0.6),
                offset=(
                    microwave_width / 2.0 - microwave_panel_width / 2.0 - 0.04,
                    microwave_depth / 2.0 - microwave_door_thickness / 2.0,
                    microwave_height * 0.5,
                ),
                color=(0.2, 0.21, 0.24, 1.0),
            ),
        ]
        catalog["microwave"] = FurniturePrototype(
            key="microwave",
            label="Microwave",
            parts=microwave_parts,
            interact_radius=1.1,
            tags=["kitchen", "appliance", "microwave", "food", "furniture"],
        )

        cooker_width = 0.8
        cooker_depth = 0.6
        cooker_height = 0.9
        cooker_top_height = 0.06
        cooker_body_height = cooker_height - cooker_top_height
        cooker_door_thickness = 0.04
        cooker_door_height = 0.38

        cooker_parts = [
            FurniturePartDef(
                size=(cooker_width, cooker_depth, cooker_body_height),
                offset=(0.0, 0.0, cooker_body_height / 2.0),
                color=(0.25, 0.26, 0.28, 1.0),
            ),
            FurniturePartDef(
                size=(cooker_width, cooker_depth, cooker_top_height),
                offset=(0.0, 0.0, cooker_body_height + cooker_top_height / 2.0),
                color=(0.15, 0.16, 0.18, 1.0),
            ),
            FurniturePartDef(
                size=(cooker_width - 0.1, cooker_door_thickness, cooker_door_height),
                offset=(
                    0.0,
                    cooker_depth / 2.0 - cooker_door_thickness / 2.0,
                    cooker_door_height / 2.0 + 0.08,
                ),
                color=(0.1, 0.1, 0.12, 0.7),
                gl_options="translucent",
            ),
            FurniturePartDef(
                size=(cooker_width - 0.2, 0.03, 0.03),
                offset=(
                    0.0,
                    cooker_depth / 2.0 + 0.02,
                    cooker_door_height + 0.14,
                ),
                color=(0.6, 0.6, 0.62, 1.0),
            ),
        ]
        burner_size = 0.14
        burner_height = 0.02
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                cooker_parts.append(
                    FurniturePartDef(
                        size=(burner_size, burner_size, burner_height),
                        offset=(
                            x_sign * cooker_width * 0.22,
                            y_sign * cooker_depth * 0.2,
                            cooker_body_height + cooker_top_height + burner_height / 2.0,
                        ),
                        color=(0.08, 0.08, 0.1, 1.0),
                        shape="cylinder",
                    )
                )
        catalog["cooker"] = FurniturePrototype(
            key="cooker",
            label="Cooker",
            parts=cooker_parts,
            interact_radius=1.25,
            tags=["kitchen", "appliance", "cooker", "stove", "food", "furniture"],
        )

        counter_width = 1.6
        counter_depth = 0.6
        counter_height = 0.9
        counter_top_thickness = 0.05
        counter_body_height = counter_height - counter_top_thickness

        counter_parts = [
            FurniturePartDef(
                size=(counter_width, counter_depth, counter_body_height),
                offset=(0.0, 0.0, counter_body_height / 2.0),
                color=(0.62, 0.46, 0.32, 1.0),
            ),
            FurniturePartDef(
                size=(counter_width, counter_depth, counter_top_thickness),
                offset=(0.0, 0.0, counter_body_height + counter_top_thickness / 2.0),
                color=(0.82, 0.78, 0.72, 1.0),
            ),
        ]
        catalog["counter"] = FurniturePrototype(
            key="counter",
            label="Kitchen Counter",
            parts=counter_parts,
            tags=["kitchen", "counter", "surface", "furniture"],
        )

        lamp_base = 0.3
        lamp_base_height = 0.05
        lamp_stem_height = 1.1
        lamp_stem_width = 0.06
        lamp_shade_height = 0.35
        lamp_shade_width = 0.5
        lamp_shade_depth = 0.5

        lamp_parts = [
            FurniturePartDef(
                size=(lamp_base, lamp_base, lamp_base_height),
                offset=(0.0, 0.0, lamp_base_height / 2.0),
                color=(0.2, 0.2, 0.22, 1.0),
                shape="cylinder",
            ),
            FurniturePartDef(
                size=(lamp_stem_width, lamp_stem_width, lamp_stem_height),
                offset=(0.0, 0.0, lamp_base_height + lamp_stem_height / 2.0),
                color=(0.35, 0.35, 0.38, 1.0),
                shape="cylinder",
            ),
            FurniturePartDef(
                size=(lamp_shade_width, lamp_shade_depth, lamp_shade_height),
                offset=(
                    0.0,
                    0.0,
                    lamp_base_height + lamp_stem_height + lamp_shade_height / 2.0,
                ),
                color=(1.0, 0.95, 0.7, 0.8),
                gl_options="translucent",
                shape="cylinder",
                emissive=0.25,
            ),
        ]
        catalog["lamp"] = FurniturePrototype(
            key="lamp",
            label="Floor Lamp",
            parts=lamp_parts,
            tags=["lamp", "light", "furniture"],
        )

        return catalog

    def _make_float_spin(self, minimum, maximum, step, value):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _make_color_picker(self, color):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(32, 16)
        swatch.setFrameShape(QtWidgets.QFrame.Box)
        self._set_color_swatch(swatch, color)
        button = QtWidgets.QPushButton("Pick")
        layout.addWidget(swatch)
        layout.addWidget(button)
        return widget, swatch, button

    def _set_color_swatch(self, swatch: QtWidgets.QLabel, color):
        r, g, b, a = color
        qcolor = QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        swatch.setStyleSheet(
            "background-color: rgba(%d, %d, %d, %d);" % (qcolor.red(), qcolor.green(), qcolor.blue(), qcolor.alpha())
        )

    def _pick_architect_color(self, setting_key: str):
        current = getattr(self.architect_settings, setting_key)
        qcolor = QtGui.QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
        picked = QtWidgets.QColorDialog.getColor(qcolor, self, "Pick Color")
        if not picked.isValid():
            return
        rgba = (picked.red() / 255.0, picked.green() / 255.0, picked.blue() / 255.0, current[3])
        setattr(self.architect_settings, setting_key, rgba)
        swatch_key = f"{setting_key}_swatch"
        swatch = self.architect_controls.get(swatch_key)
        if swatch is not None:
            self._set_color_swatch(swatch, rgba)

    def _normalize_color(self, value, fallback):
        if value is None:
            return fallback
        if len(value) >= 4:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]), float(fallback[3]))
        return fallback

    def _update_ground_size(self, x: Optional[float], y: Optional[float]):
        gx, gy = self.architect_settings.ground_size
        if x is None:
            x = gx
        if y is None:
            y = gy
        self.architect_settings.ground_size = (float(x), float(y))

    def _set_architect_setting(self, key: str, value):
        setattr(self.architect_settings, key, value)
        if key in ("grid_size", "snap_enabled"):
            self.architect_canvas.viewport().update()
        if key == "view_scale":
            self.architect_canvas.set_view_scale(value)
        if key == "spawn_height" and self.architect_state.spawn_point is not None:
            x, _, z = self.architect_state.spawn_point
            self.architect_state.spawn_point = (x, float(value), z)
            self.architect_canvas.refresh_scene()
        if key == "spawn_height" and self.architect_state.ina_spawn_point is not None:
            x, _, z = self.architect_state.ina_spawn_point
            self.architect_state.ina_spawn_point = (x, float(value), z)
            self.architect_canvas.refresh_scene()

    def _set_architect_tool(self, tool: str):
        self.architect_settings.tool = tool
        if tool == "footprint":
            message = "Footprint: click to add points, double-click or right-click to close."
        elif tool == "wall":
            message = "Wall: click-drag to draw exterior wall segments."
        elif tool == "room":
            message = "Room: click-drag to draw an interior room."
        elif tool == "door":
            message = "Door: click a wall to place a door opening."
        elif tool == "window":
            message = "Window: click a wall to place a window opening."
        elif tool == "spawn":
            message = "Player Spawn: click to set the player spawn point."
        elif tool == "ina_spawn":
            message = "Ina Spawn: click to set Ina's spawn point."
        elif tool == "fence":
            message = "Fence: click-drag to draw fence segments."
        elif tool == "ceiling_light":
            message = "Ceiling Light: click to place a ceiling light."
        elif tool == "switch":
            message = "Switch: click to place a light switch (binds to room if inside)."
        else:
            message = "Select: click to delete nearest item."
        self._set_architect_status(message)

    def _set_architect_status(self, message: str):
        if message:
            self.statusBar().showMessage(message, 5000)
        else:
            self.statusBar().clearMessage()

    def set_architect_mode(self, enabled: bool):
        if enabled:
            if self.first_person_enabled:
                self.first_person_action.setChecked(False)
            if self.furnish_action.isChecked():
                self.furnish_action.setChecked(False)
            self.central_stack.setCurrentWidget(self.architect_canvas)
            self.architect_dock.show()
            self.architect_canvas.setFocus()
            if not self.architect_loaded:
                self.load_architect_from_plan()
            self._set_architect_status("Architect mode enabled.")
        else:
            self.architect_dock.hide()
            self.central_stack.setCurrentWidget(self.view)
            self.view.setFocus()
            self._set_architect_status("")
            if self.architect_loaded:
                self._refresh_light_items_from_state()

    def set_furnish_mode(self, enabled: bool):
        if enabled:
            if self.first_person_enabled:
                self.first_person_action.setChecked(False)
            if self.architect_action.isChecked():
                self.architect_action.setChecked(False)
            self.central_stack.setCurrentWidget(self.view)
            self.furnish_dock.show()
            self.view.setFocus()
            self.furnish_mode_enabled = True
            self._set_furnish_status("Furnish mode enabled.")
            self._ensure_furnish_preview()
        else:
            self.furnish_mode_enabled = False
            self._clear_furnish_preview()
            self._furnish_press_pos = None
            self.furnish_dock.hide()
            self._set_furnish_status("")

    def _set_furnish_status(self, message: str):
        if message:
            self.statusBar().showMessage(message, 5000)
        else:
            self.statusBar().clearMessage()

    def _on_furnish_palette_changed(self, current, previous):
        if current is None:
            return
        key = current.data(QtCore.Qt.UserRole)
        if key:
            self._set_active_furniture(key)

    def _set_active_furniture(self, key: str):
        if key == self.furniture_active_key:
            return
        if key not in self.furniture_catalog:
            return
        self.furniture_active_key = key
        self._clear_furnish_preview()
        if self.furnish_mode_enabled:
            self._ensure_furnish_preview()

    def _set_furniture_rotation(self, value: float):
        rotation = float(value) % 360.0
        if abs(rotation - self.furniture_rotation) < 1e-4:
            return
        self.furniture_rotation = rotation
        if self.furnish_mode_enabled:
            self._update_furnish_preview_rotation()

    def _adjust_furniture_rotation(self, delta: float):
        rotation = (self.furniture_rotation + delta) % 360.0
        self.furniture_rotation = rotation
        if hasattr(self, "furnish_rotation_spin"):
            self.furnish_rotation_spin.blockSignals(True)
            self.furnish_rotation_spin.setValue(int(round(rotation)) % 360)
            self.furnish_rotation_spin.blockSignals(False)
        if self.furnish_mode_enabled:
            self._update_furnish_preview_rotation()

    def _toggle_furniture_snap(self, enabled: bool):
        self.furniture_snap_enabled = bool(enabled)

    def _update_furniture_grid_size(self, value: float):
        self.furniture_grid_size = float(value)

    def _ensure_furnish_preview(self):
        if not self.furnish_mode_enabled:
            return
        if self.furniture_active_key not in self.furniture_catalog:
            return
        if self.furniture_preview is not None:
            return
        base_pos = self._default_furnish_position()
        proto = self.furniture_catalog[self.furniture_active_key]
        self.furniture_preview = self._create_furniture_instance(
            proto,
            base_pos,
            self.furniture_rotation,
            preview=True,
            instance_id=0,
            tags=None,
        )
        if self._furnish_last_mouse_pos is not None:
            gl_pos = self._screen_pos_to_ground(self._furnish_last_mouse_pos)
            if gl_pos is not None:
                gl_pos = self._snap_furnish_position(gl_pos)
                self._update_furnish_preview_position(gl_pos)

    def _clear_furnish_preview(self):
        for item in self._furniture_preview_items:
            self.view.removeItem(item)
        self._furniture_preview_items.clear()
        self.furniture_preview = None

    def _default_furnish_position(self):
        ground_z = self._ground_z()
        return np.array([0.0, 0.0, ground_z], dtype=float)

    def _update_furnish_preview_position(self, gl_pos: np.ndarray):
        if self.furniture_preview is None:
            return
        proto = self.furniture_catalog.get(self.furniture_active_key)
        if proto is not None:
            gl_pos = self._snap_furniture_to_counter(proto, gl_pos)
        self.furniture_preview.position = np.array(gl_pos, dtype=float)
        self._update_furniture_instance_transform(self.furniture_preview)

    def _update_furnish_preview_rotation(self):
        if self.furniture_preview is None:
            return
        self.furniture_preview.rotation = self.furniture_rotation
        self._update_furniture_instance_transform(self.furniture_preview)

    def _point_over_counter(self, gl_pos: np.ndarray, counter: FurnitureInstance) -> bool:
        bounds = self._local_bounds_from_parts(counter.parts, use_base=True)
        if bounds[0] is None:
            return False
        margin = 0.08
        dx = float(gl_pos[0]) - float(counter.position[0])
        dy = float(gl_pos[1]) - float(counter.position[1])
        angle = -np.radians(counter.rotation)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        local_x = cos_a * dx - sin_a * dy
        local_y = sin_a * dx + cos_a * dy
        min_x, max_x, min_y, max_y, _, _ = bounds
        if min_x is None or max_x is None or min_y is None or max_y is None:
            return False
        return (
            min_x - margin <= local_x <= max_x + margin
            and min_y - margin <= local_y <= max_y + margin
        )

    def _snap_furniture_to_counter(self, proto: FurniturePrototype, gl_pos: np.ndarray) -> np.ndarray:
        tags = self._normalize_tags(proto.tags)
        wants_counter = proto.key == "microwave" or "microwave" in tags
        wants_unit = (
            proto.key in ("hifi", "tv")
            or "hifi" in tags
            or "tv" in tags
            or "screen" in tags
        )
        if not (wants_counter or wants_unit):
            return gl_pos
        room = self._room_for_point((float(gl_pos[0]), float(gl_pos[1])))
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if wants_counter and self._is_counter_instance(instance):
                pass
            elif wants_unit and self._is_living_room_unit_instance(instance):
                pass
            else:
                continue
            if room is not None:
                counter_room = self._room_for_point(
                    (float(instance.position[0]), float(instance.position[1]))
                )
                if counter_room != room:
                    continue
            if not self._point_over_counter(gl_pos, instance):
                continue
            dist = float(np.linalg.norm(np.array(gl_pos[:2], dtype=float) - instance.position[:2]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = instance
        if best is None:
            return gl_pos
        counter_bounds = self._local_bounds_from_parts(best.parts, use_base=True)
        proto_bounds = self._local_bounds_from_defs(proto.parts)
        if counter_bounds[5] is None:
            return gl_pos
        top_z = float(best.position[2]) + float(counter_bounds[5])
        min_z = float(proto_bounds[4] or 0.0)
        snapped = np.array(gl_pos, dtype=float)
        snapped[2] = top_z - min_z
        return snapped

    def _mesh_for_furniture_part(self, part_def: FurniturePartDef) -> Tuple[gl.MeshData, bool]:
        shape = (part_def.shape or "box").lower()
        if shape == "sphere":
            return get_unit_sphere_meshdata(), True
        if shape == "cylinder":
            return get_unit_cylinder_meshdata(), True
        return get_unit_cube_meshdata(), False

    def _create_furniture_instance(
        self,
        proto: FurniturePrototype,
        position: np.ndarray,
        rotation: float,
        preview: bool,
        instance_id: int,
        tags: Optional[List[str]] = None,
    ) -> FurnitureInstance:
        parts: List[FurniturePart] = []
        room = None
        if not preview:
            room = self._room_for_point((float(position[0]), float(position[1])))
        for part_def in proto.parts:
            color = part_def.color
            if preview:
                alpha = min(0.35, color[3] * 0.35)
                color = (color[0], color[1], color[2], alpha)
            gl_options = part_def.gl_options
            if gl_options is None and color[3] < 0.99:
                gl_options = "translucent"
            meshdata, smooth = self._mesh_for_furniture_part(part_def)
            mesh = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=smooth,
                color=color,
                shader="shaded",
                drawEdges=preview,
            )
            if gl_options is not None:
                mesh.setGLOptions(gl_options)
            self.view.addItem(mesh)
            if preview:
                self._furniture_preview_items.append(mesh)
            else:
                self.furniture_items.append(mesh)
                self._register_lit_item(mesh, part_def.color, room=room, emissive=part_def.emissive)
            parts.append(
                FurniturePart(
                    mesh=mesh,
                    size=part_def.size,
                    offset=part_def.offset,
                    base_size=part_def.size,
                    base_offset=part_def.offset,
                )
            )

        collision_radius = self._compute_furniture_collision_radius(proto)
        instance_tags = list(tags) if tags is not None else list(proto.tags)
        instance = FurnitureInstance(
            instance_id=instance_id,
            key=proto.key,
            label=proto.label,
            position=np.array(position, dtype=float),
            rotation=rotation,
            parts=parts,
            collision_radius=collision_radius,
            seat_offset=proto.seat_offset,
            interact_radius=proto.interact_radius,
            tags=instance_tags,
        )
        self._update_furniture_instance_transform(instance)
        return instance

    def _update_furniture_instance_transform(self, instance: FurnitureInstance):
        angle = np.radians(instance.rotation)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        pose_offset = np.array(getattr(instance, "pose_offset", (0.0, 0.0, 0.0)), dtype=float)
        tilt_deg = float(getattr(instance, "tilt_deg", 0.0))
        rot = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        for part in instance.parts:
            offset = np.array(part.offset, dtype=float)
            rotated_offset = rot @ offset
            center = instance.position + rotated_offset + pose_offset
            part.mesh.resetTransform()
            part.mesh.scale(part.size[0], part.size[1], part.size[2])
            if abs(instance.rotation) > 1e-3:
                part.mesh.rotate(instance.rotation, 0, 0, 1)
            if abs(tilt_deg) > 1e-3:
                part.mesh.rotate(tilt_deg, 1, 0, 0)
            part.mesh.translate(center[0], center[1], center[2])

    def _compute_furniture_collision_radius(self, proto: FurniturePrototype) -> float:
        max_radius = 0.0
        for part in proto.parts:
            half_x = float(part.size[0]) * 0.5
            half_y = float(part.size[1]) * 0.5
            radius = math.hypot(half_x, half_y)
            max_radius = max(max_radius, radius)
        return max(max_radius, 0.2)

    def _place_furniture_at(self, gl_pos: np.ndarray):
        proto = self.furniture_catalog.get(self.furniture_active_key)
        if proto is None:
            return
        gl_pos = self._snap_furniture_to_counter(proto, gl_pos)
        instance_id = self._furniture_instance_counter
        self._furniture_instance_counter += 1
        instance = self._create_furniture_instance(
            proto,
            gl_pos,
            self.furniture_rotation,
            preview=False,
            instance_id=instance_id,
            tags=None,
        )
        self.furniture_instances.append(instance)
        self.furniture_instances_by_id[instance_id] = instance
        self._add_furniture_list_item(instance)

    def _add_furniture_list_item(self, instance: FurnitureInstance):
        if not hasattr(self, "furniture_list"):
            return
        item = QtWidgets.QListWidgetItem(f"{instance.label} #{instance.instance_id}")
        item.setData(QtCore.Qt.UserRole, instance.instance_id)
        self.furniture_list.addItem(item)

    def _refresh_furniture_list(self):
        if not hasattr(self, "furniture_list"):
            return
        self.furniture_list.clear()
        for instance in self.furniture_instances:
            self._add_furniture_list_item(instance)

    def _remove_selected_furniture(self):
        if not hasattr(self, "furniture_list"):
            return
        for item in list(self.furniture_list.selectedItems()):
            instance_id = item.data(QtCore.Qt.UserRole)
            instance = self.furniture_instances_by_id.get(instance_id)
            if instance is not None:
                self._remove_furniture_instance(instance)
            row = self.furniture_list.row(item)
            self.furniture_list.takeItem(row)

    def _clear_all_furniture(self):
        for instance in list(self.furniture_instances):
            self._remove_furniture_instance(instance)
        self._refresh_furniture_list()

    def _load_furniture_from_plan(self, plan_path: str = "ina_house_plan.json"):
        data = self._load_plan_data(plan_path)
        building = data.get("building", {})
        storeys = building.get("storeys", [])
        storey = storeys[0] if storeys else {}
        for entry in storey.get("furniture", []):
            key = entry.get("key")
            if key not in self.furniture_catalog:
                continue
            pos = entry.get("position", [0.0, 0.0, 0.0])
            if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                continue
            model_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            gl_pos = np.array(self._to_gl_pos(model_pos), dtype=float)
            rotation = float(entry.get("rotation", 0.0))
            instance_id = self._allocate_furniture_id(entry)
            instance = self._create_furniture_instance(
                self.furniture_catalog[key],
                gl_pos,
                rotation,
                preview=False,
                instance_id=instance_id,
                tags=self._normalize_tags(
                    entry.get("tags"),
                    fallback=self.furniture_catalog[key].tags,
                ),
            )
            self.furniture_instances.append(instance)
            self.furniture_instances_by_id[instance_id] = instance
        self._refresh_furniture_list()

    def _allocate_furniture_id(self, entry: dict) -> int:
        candidate = None
        entry_id = entry.get("id")
        if isinstance(entry_id, int):
            candidate = entry_id
        elif isinstance(entry_id, str):
            tail = entry_id.split("_")[-1]
            if tail.isdigit():
                candidate = int(tail)
        if candidate is None or candidate <= 0 or candidate in self.furniture_instances_by_id:
            candidate = self._furniture_instance_counter
        self._furniture_instance_counter = max(self._furniture_instance_counter, candidate + 1)
        return candidate

    def _load_lighting_from_plan(self, plan_path: str = "ina_house_plan.json"):
        data = self._load_plan_data(plan_path)
        building = data.get("building", {})
        storeys = building.get("storeys", [])
        storey = storeys[0] if storeys else {}
        lighting = storey.get("lighting", {})
        wall_height = float(storey.get("height", self.architect_settings.wall_height))
        ceiling_y = wall_height - 0.05
        self.architect_settings.wall_height = wall_height
        self.architect_state.ceiling_lights = []
        self.architect_state.light_switches = []

        radius = max(0.08, float(self.architect_settings.ceiling_light_radius))
        ceiling_size = (radius * 2.0, 0.05, radius * 2.0)
        for light in lighting.get("ceiling_lights", []):
            pos = light.get("position", [0.0, 0.0])
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            color = self._normalize_color(
                light.get("color", self.architect_settings.ceiling_light_color),
                self.architect_settings.ceiling_light_color,
            )
            room = light.get("room")
            if not room:
                room = self._room_for_point((float(pos[0]), float(pos[1])))
            self.architect_state.ceiling_lights.append(
                ArchitectCeilingLight(
                    position=(float(pos[0]), float(pos[1])),
                    color=color,
                    room=room,
                )
            )
            color = self._light_color_for_room(color, room)
            center = (float(pos[0]), ceiling_y, float(pos[1]))
            mesh = self._make_box(
                center=center,
                size=ceiling_size,
                color=color,
                draw_edges=True,
                room=room,
            )
            mesh.setGLOptions("translucent")
            self.view.addItem(mesh)
            self.light_items.append(mesh)

        switch_size = (0.16, 0.22, 0.05)
        for switch in lighting.get("switches", []):
            pos = switch.get("position", [0.0, 0.0])
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            height = float(switch.get("height", self.architect_settings.switch_height))
            color = self._normalize_color(
                switch.get("color", self.architect_settings.switch_color),
                self.architect_settings.switch_color,
            )
            room = switch.get("room")
            if not room:
                room = self._room_for_point((float(pos[0]), float(pos[1])))
            self.architect_state.light_switches.append(
                ArchitectLightSwitch(
                    position=(float(pos[0]), float(pos[1])),
                    height=height,
                    room=room,
                )
            )
            center = self._switch_world_center((float(pos[0]), float(pos[1])), height, room)
            mesh = self._make_box(
                center=center,
                size=switch_size,
                color=color,
                draw_edges=True,
                room=room,
            )
            self.view.addItem(mesh)
            self.light_items.append(mesh)

    def _refresh_light_items_from_state(self):
        self.clear_items(self.light_items)
        wall_height = float(self.architect_settings.wall_height)
        ceiling_y = wall_height - 0.05

        radius = max(0.08, float(self.architect_settings.ceiling_light_radius))
        ceiling_size = (radius * 2.0, 0.05, radius * 2.0)
        for light in self.architect_state.ceiling_lights:
            center = (float(light.position[0]), ceiling_y, float(light.position[1]))
            color = self._light_color_for_room(light.color, light.room)
            mesh = self._make_box(
                center=center,
                size=ceiling_size,
                color=color,
                draw_edges=True,
                room=light.room,
            )
            if color[3] < 0.99:
                mesh.setGLOptions("translucent")
            self.view.addItem(mesh)
            self.light_items.append(mesh)

        switch_size = (0.16, 0.22, 0.05)
        for switch in self.architect_state.light_switches:
            center = self._switch_world_center(
                (float(switch.position[0]), float(switch.position[1])),
                float(switch.height),
                switch.room,
            )
            mesh = self._make_box(
                center=center,
                size=switch_size,
                color=self.architect_settings.switch_color,
                draw_edges=True,
                room=switch.room,
            )
            self.view.addItem(mesh)
            self.light_items.append(mesh)

        self._update_lighting(force=True)

    def _light_color_for_room(self, color, room: Optional[str]):
        key = room or "global"
        state = self._get_room_light_state(key)
        if not state["enabled"]:
            return (color[0] * 0.25, color[1] * 0.25, color[2] * 0.25, color[3] * 0.6)
        intensity = max(0.05, min(self.max_light_intensity, float(state["intensity"])))
        tint = state["color"]
        return (
            min(self.max_light_value, color[0] * intensity * tint[0]),
            min(self.max_light_value, color[1] * intensity * tint[1]),
            min(self.max_light_value, color[2] * intensity * tint[2]),
            color[3],
        )

    def _switch_world_center(self, pos: Vec2, height: float, room_name: Optional[str]) -> Vec3:
        offset = self._switch_offset_from_wall(pos, room_name)
        return (float(pos[0] + offset[0]), float(height), float(pos[1] + offset[1]))

    def _switch_offset_from_wall(self, pos: Vec2, room_name: Optional[str]) -> Vec2:
        if self.exterior_model is None or not self.exterior_model.walls:
            return (0.0, 0.0)
        best_wall = None
        best_dist = None
        best_mid = None
        for wall in self.exterior_model.walls:
            dist, _ = self._distance_to_segment(pos, wall.start, wall.end)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_wall = wall
        if best_wall is None:
            return (0.0, 0.0)

        x1, z1 = best_wall.start
        x2, z2 = best_wall.end
        dx = x2 - x1
        dz = z2 - z1
        length = math.hypot(dx, dz)
        if length < 1e-6:
            return (0.0, 0.0)
        nx, nz = -dz / length, dx / length
        mid_x = (x1 + x2) / 2.0
        mid_z = (z1 + z2) / 2.0

        target_vec = None
        if room_name:
            for room in self.architect_state.rooms:
                if room.name == room_name:
                    target_vec = (room.center[0] - mid_x, room.center[2] - mid_z)
                    break
        if target_vec is None:
            target_vec = (pos[0] - mid_x, pos[1] - mid_z)

        if (nx * target_vec[0] + nz * target_vec[1]) < 0.0:
            nx, nz = -nx, -nz

        offset_dist = (best_wall.thickness * 0.5) + 0.04
        return (nx * offset_dist, nz * offset_dist)

    def _save_furniture_plan(self):
        plan_path = "ina_house_plan.json"
        data = self._load_plan_data(plan_path)
        storey = self._ensure_plan_storey(data)
        self._update_plan_with_furniture(storey)
        if self._save_plan_data(data, plan_path, status_callback=self._set_furnish_status):
            self._set_furnish_status("Furniture saved.")

    def _find_bed_instance(self) -> Optional[FurnitureInstance]:
        if hasattr(self, "furniture_list"):
            for item in self.furniture_list.selectedItems():
                instance_id = item.data(QtCore.Qt.UserRole)
                instance = self.furniture_instances_by_id.get(instance_id)
                if instance is not None and instance.key == "bed":
                    return instance
        for instance in self.furniture_instances:
            if instance.key == "bed":
                return instance
        return None

    def _set_spawn_to_bed(self):
        bed_instance = self._find_bed_instance()
        if bed_instance is None:
            self._set_furnish_status("Place a bed to set spawn.")
            return

        model_pos = self._to_model_pos(bed_instance.position)
        spawn = (float(model_pos[0]), float(model_pos[1]), float(model_pos[2]))
        plan_path = "ina_house_plan.json"
        data = self._load_plan_data(plan_path)
        storey = self._ensure_plan_storey(data)
        spawns = storey.setdefault("spawns", {})
        spawns["ina"] = [float(v) for v in spawn]
        if self._save_plan_data(data, plan_path, status_callback=self._set_furnish_status):
            self._set_furnish_status("Ina spawn set to bed.")

        self.architect_state.ina_spawn_point = spawn
        if self.architect_loaded:
            self.architect_canvas.refresh_scene()
        self.ina_pos = np.array(self._to_gl_pos(spawn), dtype=float)
        self._ensure_ina_avatar()

    def _remove_furniture_instance(self, instance: FurnitureInstance):
        for part in instance.parts:
            self.view.removeItem(part.mesh)
            if part.mesh in self.furniture_items:
                self.furniture_items.remove(part.mesh)
            if part.mesh in self._lit_items:
                self._lit_items.remove(part.mesh)
        entry = self._tv_stream_items.pop(instance.instance_id, None)
        if entry and entry.get("item") is not None:
            self.view.removeItem(entry["item"])
        if instance in self.furniture_instances:
            self.furniture_instances.remove(instance)
        self.furniture_instances_by_id.pop(instance.instance_id, None)

    def _handle_furnish_event(self, event) -> bool:
        if not self.furnish_mode_enabled or self.first_person_enabled:
            return False
        if event.type() == QtCore.QEvent.MouseMove:
            self._furnish_last_mouse_pos = event.pos()
            self._ensure_furnish_preview()
            if self.furniture_preview is not None:
                gl_pos = self._screen_pos_to_ground(event.pos())
                if gl_pos is not None:
                    gl_pos = self._snap_furnish_position(gl_pos)
                    self._update_furnish_preview_position(gl_pos)
            return False
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self._furnish_press_pos = event.pos()
            return False
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == QtCore.Qt.LeftButton:
                if self._furnish_press_pos is None:
                    return False
                delta = event.pos() - self._furnish_press_pos
                self._furnish_press_pos = None
                if delta.manhattanLength() <= 4:
                    gl_pos = self._screen_pos_to_ground(event.pos())
                    if gl_pos is not None:
                        gl_pos = self._snap_furnish_position(gl_pos)
                        self._place_furniture_at(gl_pos)
            return False
        return False

    def _snap_furnish_position(self, gl_pos: np.ndarray) -> np.ndarray:
        if not self.furniture_snap_enabled:
            return gl_pos
        grid = self.furniture_grid_size
        if grid <= 1e-6:
            return gl_pos
        return np.array(
            [
                round(gl_pos[0] / grid) * grid,
                round(gl_pos[1] / grid) * grid,
                gl_pos[2],
            ],
            dtype=float,
        )

    def _vector_to_np(self, value) -> np.ndarray:
        if isinstance(value, pg.Vector):
            return np.array([value.x(), value.y(), value.z()], dtype=float)
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            try:
                return np.array([value.x(), value.y(), value.z()], dtype=float)
            except Exception:
                pass
        try:
            x, y, z = value
            return np.array([float(x), float(y), float(z)], dtype=float)
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=float)

    def _screen_pos_to_ground(self, pos: QtCore.QPoint) -> Optional[np.ndarray]:
        width = self.view.width()
        height = self.view.height()
        if width <= 1 or height <= 1:
            return None

        center = self._vector_to_np(self.view.opts.get("center", self.cam_center))
        distance = float(self.view.opts.get("distance", self.cam_distance))
        azimuth = float(self.view.opts.get("azimuth", self.cam_azimuth))
        elevation = float(self.view.opts.get("elevation", self.cam_elevation))

        az = math.radians(azimuth)
        el = math.radians(elevation)
        cam_offset = np.array(
            [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ],
            dtype=float,
        )
        cam_pos = center + cam_offset * distance
        forward = center - cam_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return None
        forward /= forward_norm

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(forward, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return None
        right /= right_norm
        up = np.cross(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            return None
        up /= up_norm

        x_ndc = (2.0 * pos.x() / width) - 1.0
        y_ndc = 1.0 - (2.0 * pos.y() / height)
        fov = float(self.view.opts.get("fov", 60.0))
        scale = math.tan(math.radians(fov) / 2.0)
        aspect = width / float(height)
        ray_dir = forward + right * x_ndc * aspect * scale + up * y_ndc * scale
        ray_norm = np.linalg.norm(ray_dir)
        if ray_norm < 1e-6:
            return None
        ray_dir /= ray_norm

        ground_z = self._ground_z()
        if abs(ray_dir[2]) < 1e-6:
            return None
        t = (ground_z - cam_pos[2]) / ray_dir[2]
        if t < 0.0:
            return None
        return cam_pos + ray_dir * t

    def architect_set_footprint(self, points: List[Vec2]):
        self.architect_state.footprint_points = points
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_wall(self, start: Vec2, end: Vec2):
        if self._segment_length(start, end) < 1e-3:
            return
        wall = ArchitectWall(
            start=start,
            end=end,
            height=self.architect_settings.wall_height,
            thickness=self.architect_settings.wall_thickness,
            color=self.architect_settings.wall_color,
            openings=[],
        )
        self.architect_state.walls.append(wall)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_fence(self, start: Vec2, end: Vec2):
        if self._segment_length(start, end) < 1e-3:
            return
        fence = FenceSegment(
            start=start,
            end=end,
            height=self.architect_settings.fence_height,
            thickness=self.architect_settings.fence_thickness,
            color=self.architect_settings.fence_color,
        )
        self.architect_state.fences.append(fence)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_room(self, start: QtCore.QPointF, end: QtCore.QPointF):
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        width = abs(x2 - x1)
        depth = abs(y2 - y1)
        if width < 0.1 or depth < 0.1:
            return
        center_x = (x1 + x2) / 2.0
        center_z = (y1 + y2) / 2.0
        height = self.architect_settings.room_height
        name = f"{self.architect_settings.room_name_prefix}_{self.architect_room_counter}"
        self.architect_room_counter += 1
        room = Room(
            name=name,
            center=(center_x, height / 2.0, center_z),
            size=(width, height, depth),
            color=self.architect_settings.room_color,
        )
        self.architect_state.rooms.append(room)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_opening(self, opening_type: str, pos: Vec2):
        if not self.architect_state.walls:
            self._set_architect_status("Add walls before placing openings.")
            return
        best_wall = None
        best_dist = None
        best_offset = 0.0
        for wall in self.architect_state.walls:
            dist, offset = self._distance_to_segment(pos, wall.start, wall.end)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_offset = offset
                best_wall = wall
        if best_wall is None:
            return
        threshold = max(self.architect_settings.grid_size * 1.5, best_wall.thickness * 2.0)
        if best_dist is not None and best_dist > threshold:
            self._set_architect_status("Openings must be placed near a wall.")
            return
        wall_idx = self.architect_state.walls.index(best_wall) + 1
        op_idx = len(best_wall.openings) + 1
        opening_id = f"{opening_type}_{wall_idx}_{op_idx}"
        if opening_type == "door":
            opening = Opening(
                type="door",
                width=self.architect_settings.door_width,
                height=self.architect_settings.door_height,
                sill_height=self.architect_settings.door_sill,
                offset_along_wall=best_offset,
                id=opening_id,
            )
        else:
            opening = Opening(
                type="window",
                width=self.architect_settings.window_width,
                height=self.architect_settings.window_height,
                sill_height=self.architect_settings.window_sill,
                offset_along_wall=best_offset,
                id=opening_id,
            )
        best_wall.openings.append(opening)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def _room_for_point(self, pos: Vec2) -> Optional[str]:
        x, z = float(pos[0]), float(pos[1])
        rooms = self.architect_state.rooms if self.architect_state.rooms else self._runtime_rooms
        for room in rooms:
            center_x, _, center_z = room.center
            width, _, depth = room.size
            if abs(x - center_x) <= width / 2.0 and abs(z - center_z) <= depth / 2.0:
                return room.name
        return None

    def architect_set_spawn(self, pos: Vec2):
        spawn = (float(pos[0]), float(self.architect_settings.spawn_height), float(pos[1]))
        self.architect_state.spawn_point = spawn
        if self.exterior_model is not None:
            self.exterior_model.spawn_point = spawn
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_set_ina_spawn(self, pos: Vec2):
        spawn = (float(pos[0]), float(self.architect_settings.spawn_height), float(pos[1]))
        self.architect_state.ina_spawn_point = spawn
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()
        self.ina_pos = np.array(self._to_gl_pos(spawn), dtype=float)
        self._ensure_ina_avatar()

    def architect_add_ceiling_light(self, pos: Vec2):
        light = ArchitectCeilingLight(
            position=(float(pos[0]), float(pos[1])),
            color=self.architect_settings.ceiling_light_color,
            room=self._room_for_point(pos),
        )
        self.architect_state.ceiling_lights.append(light)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()
        self._refresh_light_items_from_state()

    def architect_add_light_switch(self, pos: Vec2):
        switch = ArchitectLightSwitch(
            position=(float(pos[0]), float(pos[1])),
            height=float(self.architect_settings.switch_height),
            room=self._room_for_point(pos),
        )
        self.architect_state.light_switches.append(switch)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()
        self._refresh_light_items_from_state()

    def architect_clear(self):
        self.architect_state = ArchitectState()
        self.architect_room_counter = 1
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()
        self._refresh_light_items_from_state()

    def architect_delete_at(self, pos: Vec2):
        px, py = pos
        threshold = max(self.architect_settings.grid_size * 0.8, 0.3)

        best_opening = None
        best_opening_dist = None
        for wall in self.architect_state.walls:
            length = self._segment_length(wall.start, wall.end)
            if length < 1e-4:
                continue
            dx = (wall.end[0] - wall.start[0]) / length
            dy = (wall.end[1] - wall.start[1]) / length
            for op in wall.openings:
                center = max(0.0, min(op.offset_along_wall, length))
                cx = wall.start[0] + dx * center
                cy = wall.start[1] + dy * center
                dist = math.hypot(px - cx, py - cy)
                radius = max(op.width * 0.6, threshold)
                if dist <= radius and (best_opening_dist is None or dist < best_opening_dist):
                    best_opening = (wall, op)
                    best_opening_dist = dist

        if best_opening is not None:
            wall, opening = best_opening
            try:
                wall.openings.remove(opening)
            except ValueError:
                return
            self._set_architect_dirty()
            self.architect_canvas.refresh_scene()
            self._set_architect_status("Opening deleted.")
            return

        if self.architect_state.spawn_point is not None:
            sx, _, sz = self.architect_state.spawn_point
            if math.hypot(px - sx, py - sz) <= threshold:
                self.architect_state.spawn_point = None
                if self.exterior_model is not None:
                    self.exterior_model.spawn_point = None
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Spawn point deleted.")
                return

        if self.architect_state.ina_spawn_point is not None:
            sx, _, sz = self.architect_state.ina_spawn_point
            if math.hypot(px - sx, py - sz) <= threshold:
                self.architect_state.ina_spawn_point = None
                self.clear_items(self.ina_items)
                self.ina_avatar_parts = []
                self.ina_pos = None
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Ina spawn deleted.")
                return

        if self.architect_state.ceiling_lights:
            best_idx = None
            best_dist = None
            for idx, light in enumerate(self.architect_state.ceiling_lights):
                dist = math.hypot(px - light.position[0], py - light.position[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.ceiling_lights.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Ceiling light deleted.")
                self._refresh_light_items_from_state()
                return

        if self.architect_state.light_switches:
            best_idx = None
            best_dist = None
            for idx, switch in enumerate(self.architect_state.light_switches):
                dist = math.hypot(px - switch.position[0], py - switch.position[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.light_switches.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Light switch deleted.")
                self._refresh_light_items_from_state()
                return

        if self.architect_state.footprint_points:
            best_idx = None
            best_dist = None
            for idx, (fx, fy) in enumerate(self.architect_state.footprint_points):
                dist = math.hypot(px - fx, py - fy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.footprint_points.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Footprint point deleted.")
                return

        if self.architect_state.rooms:
            best_idx = None
            best_dist = None
            for idx, room in enumerate(self.architect_state.rooms):
                cx, _, cz = room.center
                width, _, depth = room.size
                dx = max(abs(px - cx) - width / 2.0, 0.0)
                dy = max(abs(py - cz) - depth / 2.0, 0.0)
                dist = math.hypot(dx, dy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.rooms.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Room deleted.")
                return

        if self.architect_state.walls:
            best_idx = None
            best_dist = None
            for idx, wall in enumerate(self.architect_state.walls):
                dist, _ = self._distance_to_segment(pos, wall.start, wall.end)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.walls.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Wall deleted.")
                return

        if self.architect_state.fences:
            best_idx = None
            best_dist = None
            for idx, fence in enumerate(self.architect_state.fences):
                dist, _ = self._distance_to_segment(pos, fence.start, fence.end)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= threshold:
                self.architect_state.fences.pop(best_idx)
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Fence deleted.")
                return

    def load_architect_from_plan(self):
        data = None
        plan_path = "ina_house_plan.json"
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = None
        except Exception:
            data = None

        player_spawn_override = None
        ina_spawn_point = None
        if data is not None:
            self.architect_plan_cache = data
            site = data.get("site", {})
            ground = site.get("ground", {})
            ground_size = ground.get("size", list(self.architect_settings.ground_size))
            ground_color = ground.get("color", list(self.architect_settings.ground_color))
            self.architect_settings.ground_size = (float(ground_size[0]), float(ground_size[1]))
            self.architect_settings.ground_color = self._normalize_color(
                ground_color,
                self.architect_settings.ground_color,
            )

            building = data.get("building", {})
            storeys = building.get("storeys", [])
            storey = storeys[0] if storeys else {}
            self.architect_settings.wall_height = float(storey.get("height", self.architect_settings.wall_height))
            self.architect_settings.room_height = self.architect_settings.wall_height

            wall_defaults = storey.get("exterior_walls", {})
            self.architect_settings.wall_thickness = float(
                wall_defaults.get("default_thickness", self.architect_settings.wall_thickness)
            )
            self.architect_settings.wall_color = self._normalize_color(
                wall_defaults.get("default_color", self.architect_settings.wall_color),
                self.architect_settings.wall_color,
            )

            roof = storey.get("roof", {})
            self.architect_settings.roof_overhang = float(roof.get("overhang", self.architect_settings.roof_overhang))
            self.architect_settings.roof_thickness = float(
                roof.get("thickness", self.architect_settings.roof_thickness)
            )
            self.architect_settings.roof_color = self._normalize_color(
                roof.get("color", self.architect_settings.roof_color),
                self.architect_settings.roof_color,
            )

            lighting = storey.get("lighting", {})
            self.architect_settings.ceiling_light_color = self._normalize_color(
                lighting.get("default_ceiling_color", self.architect_settings.ceiling_light_color),
                self.architect_settings.ceiling_light_color,
            )
            self.architect_settings.switch_color = self._normalize_color(
                lighting.get("default_switch_color", self.architect_settings.switch_color),
                self.architect_settings.switch_color,
            )
            self.architect_settings.switch_height = float(
                lighting.get("default_switch_height", self.architect_settings.switch_height)
            )
            self.architect_settings.ceiling_light_radius = float(
                lighting.get("default_ceiling_radius", self.architect_settings.ceiling_light_radius)
            )

            spawns = storey.get("spawns", {})
            player_spawn = spawns.get("player")
            if isinstance(player_spawn, (list, tuple)) and len(player_spawn) >= 3:
                player_spawn_override = (
                    float(player_spawn[0]),
                    float(player_spawn[1]),
                    float(player_spawn[2]),
                )
            ina_spawn = spawns.get("ina") or storey.get("ina_spawn")
            if isinstance(ina_spawn, (list, tuple)) and len(ina_spawn) >= 3:
                ina_spawn_point = (
                    float(ina_spawn[0]),
                    float(ina_spawn[1]),
                    float(ina_spawn[2]),
                )

        try:
            house, exterior = load_house_from_plan(plan_path)
        except Exception:
            house = create_prototype_house()
            exterior = create_prototype_exterior()

        ceiling_lights = []
        light_switches = []
        if data is not None:
            building = data.get("building", {})
            storeys = building.get("storeys", [])
            storey = storeys[0] if storeys else {}
            lighting = storey.get("lighting", {})
            for light in lighting.get("ceiling_lights", []):
                pos = light.get("position", [0.0, 0.0])
                color = self._normalize_color(
                    light.get("color", self.architect_settings.ceiling_light_color),
                    self.architect_settings.ceiling_light_color,
                )
                room = light.get("room")
                ceiling_lights.append(
                    ArchitectCeilingLight(
                        position=(float(pos[0]), float(pos[1])),
                        color=color,
                        room=room,
                    )
                )
            for switch in lighting.get("switches", []):
                pos = switch.get("position", [0.0, 0.0])
                room = switch.get("room")
                height = float(switch.get("height", self.architect_settings.switch_height))
                light_switches.append(
                    ArchitectLightSwitch(
                        position=(float(pos[0]), float(pos[1])),
                        height=height,
                        room=room,
                    )
                )

        self.architect_state = ArchitectState(
            footprint_points=list(exterior.footprint.outline),
            walls=[
                ArchitectWall(
                    start=wall.start,
                    end=wall.end,
                    height=wall.height,
                    thickness=wall.thickness,
                    color=wall.color,
                    openings=[
                        Opening(
                            type=op.type,
                            width=op.width,
                            height=op.height,
                            sill_height=op.sill_height,
                            offset_along_wall=op.offset_along_wall,
                            id=op.id,
                        )
                        for op in wall.openings
                    ],
                )
                for wall in exterior.walls
            ],
            rooms=[room for room in house.rooms if room.name != "garden"],
            fences=[
                FenceSegment(
                    start=fence.start,
                    end=fence.end,
                    height=fence.height,
                    thickness=fence.thickness,
                    color=fence.color,
                )
                for fence in exterior.fences
            ],
            ceiling_lights=ceiling_lights,
            light_switches=light_switches,
            spawn_point=player_spawn_override if player_spawn_override is not None else exterior.spawn_point,
            ina_spawn_point=ina_spawn_point,
        )
        self.architect_room_counter = len(self.architect_state.rooms) + 1
        self.architect_dirty = False
        self.architect_loaded = True
        if self.architect_state.spawn_point is not None:
            self.architect_settings.spawn_height = float(self.architect_state.spawn_point[1])
        self._sync_architect_controls()
        self.architect_canvas.refresh_scene()
        self._refresh_light_items_from_state()
        self._set_architect_status("Plan loaded into architect mode.")

    def save_architect_plan(self, reload_after: bool = False):
        data = copy.deepcopy(self.architect_plan_cache) if self.architect_plan_cache else {}
        self._update_plan_from_architect(data)
        plan_path = "ina_house_plan.json"
        if not self._save_plan_data(data, plan_path, status_callback=self._set_architect_status):
            return

        self.architect_plan_cache = data
        self.architect_dirty = False
        if reload_after:
            self.reload_scene()
        self._set_architect_status("Plan saved.")

    def _load_plan_data(self, plan_path: str = "ina_house_plan.json") -> dict:
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_plan_data(self, data, plan_path: str, status_callback=None) -> bool:
        try:
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            if status_callback is not None:
                status_callback(f"Failed to save plan: {exc}")
            return False
        return True

    def _ensure_plan_storey(self, data):
        building = data.setdefault("building", {})
        building.setdefault("name", "Ina_House_Prototype")
        storeys = building.setdefault("storeys", [])
        if not storeys:
            storeys.append({})
        storey = storeys[0]
        storey.setdefault("id", "ground_floor")
        return storey

    def _update_plan_from_architect(self, data):
        if "version" not in data:
            data["version"] = "1.0"
        if "units" not in data:
            data["units"] = "meters"

        site = data.setdefault("site", {})
        site.setdefault("origin", [0.0, 0.0, 0.0])
        ground = site.setdefault("ground", {})
        ground["size"] = [float(self.architect_settings.ground_size[0]), float(self.architect_settings.ground_size[1])]
        ground["color"] = list(self.architect_settings.ground_color)

        building = data.setdefault("building", {})
        building.setdefault("name", "Ina_House_Prototype")
        storeys = building.setdefault("storeys", [])
        if not storeys:
            storeys.append({})
        storey = storeys[0]
        storey.setdefault("id", "ground_floor")
        wall_heights = [wall.height for wall in self.architect_state.walls]
        storey["height"] = max(wall_heights) if wall_heights else self.architect_settings.wall_height

        footprint_points = list(self.architect_state.footprint_points)
        if not footprint_points and self.architect_state.walls:
            points = []
            for wall in self.architect_state.walls:
                points.append(wall.start)
                points.append(wall.end)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            footprint_points = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ]

        storey["footprint"] = {
            "type": "polygon",
            "points": [[float(p[0]), float(p[1])] for p in footprint_points],
        }

        exterior_walls = storey.setdefault("exterior_walls", {})
        exterior_walls["default_thickness"] = float(self.architect_settings.wall_thickness)
        exterior_walls["default_color"] = list(self.architect_settings.wall_color)
        segments = []
        for idx, wall in enumerate(self.architect_state.walls, start=1):
            openings = []
            for op_idx, op in enumerate(wall.openings, start=1):
                opening_id = op.id or f"{op.type}_{idx}_{op_idx}"
                op.id = opening_id
                openings.append(
                    {
                        "type": op.type,
                        "id": opening_id,
                        "offset_along_wall": float(op.offset_along_wall),
                        "width": float(op.width),
                        "height": float(op.height),
                        "sill_height": float(op.sill_height),
                    }
                )
            segments.append(
                {
                    "id": f"wall_{idx}",
                    "kind": "exterior",
                    "from": [float(wall.start[0]), float(wall.start[1])],
                    "to": [float(wall.end[0]), float(wall.end[1])],
                    "height": float(wall.height),
                    "thickness": float(wall.thickness),
                    "color": list(wall.color),
                    "openings": openings,
                }
            )
        exterior_walls["segments"] = segments

        rooms = []
        for idx, room in enumerate(self.architect_state.rooms, start=1):
            rooms.append(
                {
                    "id": f"room_{idx}",
                    "name": room.name,
                    "center": [float(c) for c in room.center],
                    "size": [float(s) for s in room.size],
                    "color": list(room.color),
                }
            )
        storey["interior_rooms"] = rooms

        storey["roof"] = {
            "type": "flat",
            "overhang": float(self.architect_settings.roof_overhang),
            "thickness": float(self.architect_settings.roof_thickness),
            "color": list(self.architect_settings.roof_color),
        }

        fences = []
        for idx, fence in enumerate(self.architect_state.fences, start=1):
            fences.append(
                {
                    "id": f"fence_{idx}",
                    "from": [float(fence.start[0]), float(fence.start[1])],
                    "to": [float(fence.end[0]), float(fence.end[1])],
                    "height": float(fence.height),
                    "thickness": float(fence.thickness),
                    "color": list(fence.color),
                }
            )
        storey["fences"] = fences

        spawns = storey.setdefault("spawns", {})
        if self.architect_state.spawn_point is not None:
            spawn_pos = [float(v) for v in self.architect_state.spawn_point]
            storey["spawn"] = {"position": list(spawn_pos)}
            spawns["player"] = list(spawn_pos)
        else:
            storey.pop("spawn", None)
            spawns.pop("player", None)
        if self.architect_state.ina_spawn_point is not None:
            spawns["ina"] = [float(v) for v in self.architect_state.ina_spawn_point]
        else:
            spawns.pop("ina", None)
        if not spawns:
            storey.pop("spawns", None)

        self._update_plan_with_lighting(storey)
        self._update_plan_with_furniture(storey)

    def _update_plan_with_lighting(self, storey):
        lighting = storey.setdefault("lighting", {})
        lighting["default_ceiling_color"] = list(self.architect_settings.ceiling_light_color)
        lighting["default_switch_color"] = list(self.architect_settings.switch_color)
        lighting["default_switch_height"] = float(self.architect_settings.switch_height)
        lighting["default_ceiling_radius"] = float(self.architect_settings.ceiling_light_radius)
        lights = []
        for idx, light in enumerate(self.architect_state.ceiling_lights, start=1):
            payload = {
                "id": f"ceiling_light_{idx}",
                "position": [float(light.position[0]), float(light.position[1])],
                "color": list(light.color),
            }
            if light.room:
                payload["room"] = light.room
            lights.append(payload)
        lighting["ceiling_lights"] = lights

        switches = []
        for idx, switch in enumerate(self.architect_state.light_switches, start=1):
            payload = {
                "id": f"switch_{idx}",
                "position": [float(switch.position[0]), float(switch.position[1])],
                "height": float(switch.height),
                "color": list(self.architect_settings.switch_color),
            }
            if switch.room:
                payload["room"] = switch.room
            switches.append(payload)
        lighting["switches"] = switches

    def _update_plan_with_furniture(self, storey):
        furniture_entries = []
        for instance in self.furniture_instances:
            model_pos = self._to_model_pos(instance.position)
            payload = {
                "id": f"furniture_{instance.instance_id}",
                "key": instance.key,
                "position": [float(model_pos[0]), float(model_pos[1]), float(model_pos[2])],
                "rotation": float(instance.rotation),
            }
            tags = self._normalize_tags(instance.tags)
            if tags:
                payload["tags"] = tags
            furniture_entries.append(payload)
        storey["furniture"] = furniture_entries

    def _sync_architect_controls(self):
        if not self.architect_controls:
            return
        self.architect_controls["wall_height"].blockSignals(True)
        self.architect_controls["wall_thickness"].blockSignals(True)
        self.architect_controls["room_height"].blockSignals(True)
        self.architect_controls["room_prefix"].blockSignals(True)
        self.architect_controls["door_width"].blockSignals(True)
        self.architect_controls["door_height"].blockSignals(True)
        self.architect_controls["door_sill"].blockSignals(True)
        self.architect_controls["window_width"].blockSignals(True)
        self.architect_controls["window_height"].blockSignals(True)
        self.architect_controls["window_sill"].blockSignals(True)
        self.architect_controls["fence_height"].blockSignals(True)
        self.architect_controls["fence_thickness"].blockSignals(True)
        self.architect_controls["roof_overhang"].blockSignals(True)
        self.architect_controls["roof_thickness"].blockSignals(True)
        self.architect_controls["ground_size_x"].blockSignals(True)
        self.architect_controls["ground_size_y"].blockSignals(True)
        self.architect_controls["grid_size"].blockSignals(True)
        self.architect_controls["view_scale"].blockSignals(True)
        self.architect_controls["spawn_height"].blockSignals(True)
        self.architect_controls["snap_enabled"].blockSignals(True)
        self.architect_controls["snap_existing"].blockSignals(True)
        self.architect_controls["axis_lock"].blockSignals(True)

        self.architect_controls["wall_height"].setValue(self.architect_settings.wall_height)
        self.architect_controls["wall_thickness"].setValue(self.architect_settings.wall_thickness)
        self.architect_controls["room_height"].setValue(self.architect_settings.room_height)
        self.architect_controls["room_prefix"].setText(self.architect_settings.room_name_prefix)
        self.architect_controls["door_width"].setValue(self.architect_settings.door_width)
        self.architect_controls["door_height"].setValue(self.architect_settings.door_height)
        self.architect_controls["door_sill"].setValue(self.architect_settings.door_sill)
        self.architect_controls["window_width"].setValue(self.architect_settings.window_width)
        self.architect_controls["window_height"].setValue(self.architect_settings.window_height)
        self.architect_controls["window_sill"].setValue(self.architect_settings.window_sill)
        self.architect_controls["fence_height"].setValue(self.architect_settings.fence_height)
        self.architect_controls["fence_thickness"].setValue(self.architect_settings.fence_thickness)
        self.architect_controls["roof_overhang"].setValue(self.architect_settings.roof_overhang)
        self.architect_controls["roof_thickness"].setValue(self.architect_settings.roof_thickness)
        self.architect_controls["ground_size_x"].setValue(self.architect_settings.ground_size[0])
        self.architect_controls["ground_size_y"].setValue(self.architect_settings.ground_size[1])
        self.architect_controls["grid_size"].setValue(self.architect_settings.grid_size)
        self.architect_controls["view_scale"].setValue(self.architect_settings.view_scale)
        self.architect_controls["spawn_height"].setValue(self.architect_settings.spawn_height)
        self.architect_controls["snap_enabled"].setChecked(self.architect_settings.snap_enabled)
        self.architect_controls["snap_existing"].setChecked(self.architect_settings.snap_existing)
        self.architect_controls["axis_lock"].setChecked(self.architect_settings.axis_lock)

        self.architect_controls["wall_height"].blockSignals(False)
        self.architect_controls["wall_thickness"].blockSignals(False)
        self.architect_controls["room_height"].blockSignals(False)
        self.architect_controls["room_prefix"].blockSignals(False)
        self.architect_controls["door_width"].blockSignals(False)
        self.architect_controls["door_height"].blockSignals(False)
        self.architect_controls["door_sill"].blockSignals(False)
        self.architect_controls["window_width"].blockSignals(False)
        self.architect_controls["window_height"].blockSignals(False)
        self.architect_controls["window_sill"].blockSignals(False)
        self.architect_controls["fence_height"].blockSignals(False)
        self.architect_controls["fence_thickness"].blockSignals(False)
        self.architect_controls["roof_overhang"].blockSignals(False)
        self.architect_controls["roof_thickness"].blockSignals(False)
        self.architect_controls["ground_size_x"].blockSignals(False)
        self.architect_controls["ground_size_y"].blockSignals(False)
        self.architect_controls["grid_size"].blockSignals(False)
        self.architect_controls["view_scale"].blockSignals(False)
        self.architect_controls["spawn_height"].blockSignals(False)
        self.architect_controls["snap_enabled"].blockSignals(False)
        self.architect_controls["snap_existing"].blockSignals(False)
        self.architect_controls["axis_lock"].blockSignals(False)

        self._set_color_swatch(self.architect_controls["wall_color_swatch"], self.architect_settings.wall_color)
        self._set_color_swatch(self.architect_controls["room_color_swatch"], self.architect_settings.room_color)
        self._set_color_swatch(self.architect_controls["fence_color_swatch"], self.architect_settings.fence_color)
        self._set_color_swatch(self.architect_controls["roof_color_swatch"], self.architect_settings.roof_color)
        self._set_color_swatch(self.architect_controls["ground_color_swatch"], self.architect_settings.ground_color)

    def _segment_length(self, start: Vec2, end: Vec2) -> float:
        return math.hypot(end[0] - start[0], end[1] - start[1])

    def _distance_to_segment(self, point: Vec2, start: Vec2, end: Vec2):
        px, py = point
        sx, sy = start
        ex, ey = end
        vx = ex - sx
        vy = ey - sy
        length_sq = vx * vx + vy * vy
        if length_sq < 1e-8:
            return math.hypot(px - sx, py - sy), 0.0
        t = ((px - sx) * vx + (py - sy) * vy) / length_sq
        t = max(0.0, min(1.0, t))
        proj_x = sx + t * vx
        proj_y = sy + t * vy
        dist = math.hypot(px - proj_x, py - proj_y)
        offset = math.hypot(vx, vy) * t
        return dist, offset

    def _set_architect_dirty(self):
        self.architect_dirty = True

    # ---- Exterior rendering ----

    def _build_exterior(self, exterior: ExteriorModel):
        """
        Build garden, walls, and a simple flat roof.
        """
        bounds_points: List[Vec2] = []
        bounds_points.extend(exterior.footprint.outline)
        if not bounds_points:
            for wall in exterior.walls:
                bounds_points.append(wall.start)
                bounds_points.append(wall.end)

        if bounds_points:
            xs = [p[0] for p in bounds_points]
            zs = [p[1] for p in bounds_points]
            min_x, max_x = min(xs), max(xs)
            min_z, max_z = min(zs), max(zs)
        else:
            half_w = exterior.garden_size[0] / 2.0
            half_d = exterior.garden_size[2] / 2.0
            min_x = exterior.garden_center[0] - half_w
            max_x = exterior.garden_center[0] + half_w
            min_z = exterior.garden_center[2] - half_d
            max_z = exterior.garden_center[2] + half_d

        base_min_x = min_x
        base_max_x = max_x
        base_min_z = min_z
        base_max_z = max_z

        if exterior.roof_overhang > 1e-4:
            min_x -= exterior.roof_overhang
            max_x += exterior.roof_overhang
            min_z -= exterior.roof_overhang
            max_z += exterior.roof_overhang

        # Update camera target based on footprint
        center_model = (
            (min_x + max_x) / 2.0,
            exterior.footprint.wall_height / 2.0,
            (min_z + max_z) / 2.0,
        )
        self.scene_center = pg.Vector(*self._to_gl_pos(center_model))
        span = max(exterior.garden_size[0], exterior.garden_size[2])
        self.scene_distance = max(12.0, span * 0.6)
        self.cam_center = self.scene_center
        self.cam_distance = self.scene_distance
        self.cam_azimuth = self.default_azimuth
        self.cam_elevation = self.default_elevation
        self.view.opts["center"] = self.scene_center

        # Garden
        garden_height = min(exterior.garden_size[1], 0.05)
        garden_size = (
            exterior.garden_size[0],
            garden_height,
            exterior.garden_size[2],
        )
        garden_mesh = self._make_box(
            center=exterior.garden_center,
            size=garden_size,
            color=exterior.garden_color,
            draw_edges=False,
            normal=exterior.garden_normal,
        )
        self.view.addItem(garden_mesh)
        self.exterior_items.append(garden_mesh)

        # Fences: use plan-specified segments if provided; otherwise fallback perimeter
        if getattr(exterior, "fences", None):
            for fence in exterior.fences:
                fence_items = self._build_fence_with_posts(fence)
                for item in fence_items:
                    self.view.addItem(item)
                    self.exterior_items.append(item)
        else:
            fence_color = (1.0, 1.0, 1.0, 1.0)
            fence_height = 1.0
            fence_thickness = 0.05
            half_w = garden_size[0] / 2.0
            half_d = garden_size[2] / 2.0
            ground_y = exterior.garden_center[1] + garden_height / 2.0
            fence_y = ground_y + fence_height / 2.0

            fence_segments = [
                # Front/back
                ((0.0, fence_y, -half_d), (garden_size[0], fence_height, fence_thickness)),
                ((0.0, fence_y, half_d), (garden_size[0], fence_height, fence_thickness)),
                # Left/right
                ((-half_w, fence_y, 0.0), (fence_thickness, fence_height, garden_size[2])),
                ((half_w, fence_y, 0.0), (fence_thickness, fence_height, garden_size[2])),
            ]
            for center, size in fence_segments:
                fence_mesh = self._make_box(
                    center=center,
                    size=size,
                    color=fence_color,
                    draw_edges=False,
                    normal=(0, 1, 0),
                )
                self.view.addItem(fence_mesh)
                self.exterior_items.append(fence_mesh)

        # Walls with openings
        for wall in exterior.walls:
            wall_items = self._build_wall_with_openings(wall)
            for item in wall_items:
                self.view.addItem(item)
                self.exterior_items.append(item)

        # Roof: single flat box slightly above wall height
        # Use footprint polygon to avoid unsupported cut-throughs.
        if exterior.footprint.outline or exterior.walls:
            roof_outline = list(exterior.footprint.outline)
            if not roof_outline:
                roof_outline = [
                    (base_min_x, base_min_z),
                    (base_max_x, base_min_z),
                    (base_max_x, base_max_z),
                    (base_min_x, base_max_z),
                ]
            roof_mesh = self._make_roof_mesh(
                roof_outline,
                exterior.roof_height,
                exterior.roof_thickness,
                exterior.roof_overhang,
                exterior.roof_color,
            )
            if roof_mesh is not None:
                self.view.addItem(roof_mesh)
                self.exterior_items.append(roof_mesh)
        # With updated center, re-seat the camera to frame the scene
        self._update_camera_position(recenter=True)

    def _make_wall_box(self, wall: WallSegment) -> gl.GLMeshItem:
        """
        Creates a stretched cube for a wall segment, given start/end in (x,z).
        """
        (x1, z1), (x2, z2) = wall.start, wall.end
        dx = x2 - x1
        dy = z2 - z1

        # Midpoint of the segment in ground plane (x, y)
        mx = (x1 + x2) / 2.0
        my = (z1 + z2) / 2.0

        # Length in the ground plane (x, y)
        length = (dx ** 2 + dy ** 2) ** 0.5
        height = wall.height
        thickness = wall.thickness  # depth/thickness along GL y-axis

        # Base cube
        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=wall.color,
            shader="shaded",
            drawEdges=True,
        )
        # Scale then rotate, translate last to avoid rotating the translation offset.
        mesh.scale(length, thickness, height)
        angle = np.degrees(np.arctan2(dy, dx))
        mesh.rotate(-angle, 0, 0, 1)
        mesh.translate(mx, my, height / 2.0)
        self._register_lit_item(mesh, wall.color, room=None)

        return mesh

    def _make_fence_box(self, fence) -> gl.GLMeshItem:
        (x1, z1), (x2, z2) = fence.start, fence.end
        dx = x2 - x1
        dy = z2 - z1
        length = (dx ** 2 + dy ** 2) ** 0.5
        height = fence.height
        thickness = fence.thickness

        mx = (x1 + x2) / 2.0
        my = (z1 + z2) / 2.0

        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=fence.color,
            shader="shaded",
            drawEdges=False,
        )
        mesh.scale(length, thickness, height)
        angle = np.degrees(np.arctan2(dy, dx))
        mesh.rotate(-angle, 0, 0, 1)
        mesh.translate(mx, my, height / 2.0)
        self._register_lit_item(mesh, fence.color, room=None)
        return mesh

    def _build_fence_with_posts(self, fence) -> List[gl.GLMeshItem]:
        items: List[gl.GLMeshItem] = []
        (x1, z1), (x2, z2) = fence.start, fence.end
        dx = x2 - x1
        dy = z2 - z1
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 1e-4:
            return items

        direction = (dx / length, dy / length)
        rail_height = max(0.06, fence.height * 0.12)
        rail_thickness = max(0.04, fence.thickness * 0.6)

        top_rail = self._make_wall_segment_box(
            fence.start,
            fence.end,
            rail_height,
            rail_thickness,
            fence.color,
            bottom=fence.height - rail_height,
            draw_edges=False,
        )
        if top_rail is not None:
            items.append(top_rail)

        mid_rail = self._make_wall_segment_box(
            fence.start,
            fence.end,
            rail_height,
            rail_thickness,
            fence.color,
            bottom=max(0.0, fence.height * 0.5 - rail_height / 2.0),
            draw_edges=False,
        )
        if mid_rail is not None:
            items.append(mid_rail)

        post_size = max(0.08, fence.thickness * 1.4)
        post_color = fence.color
        post_spacing = max(1.5, fence.height * 0.8)
        count = max(1, int(length // post_spacing))

        for i in range(count + 1):
            t = i / max(1, count)
            px = x1 + direction[0] * length * t
            pz = z1 + direction[1] * length * t
            post = self._make_box(
                center=(px, fence.height / 2.0, pz),
                size=(post_size, fence.height, post_size),
                color=post_color,
                draw_edges=False,
                normal=(0, 1, 0),
            )
            items.append(post)

        return items

    def _make_roof_mesh(
        self,
        outline: List[Vec2],
        roof_height: float,
        roof_thickness: float,
        roof_overhang: float,
        color,
    ) -> Optional[gl.GLMeshItem]:
        if len(outline) < 3:
            return None
        cx = sum(p[0] for p in outline) / len(outline)
        cz = sum(p[1] for p in outline) / len(outline)
        min_dist = min(math.hypot(p[0] - cx, p[1] - cz) for p in outline)
        if min_dist < 1e-3:
            return None

        inner_points = list(outline)
        outer_points = (
            self._expand_polygon_radial(outline, roof_overhang)
            if roof_overhang > 1e-4
            else list(outline)
        )
        core_shrink = max(roof_overhang * 0.6, roof_thickness * 0.5, 0.15)
        core_shrink = min(core_shrink, min_dist * 0.45)
        core_points = (
            self._expand_polygon_radial(outline, -core_shrink)
            if core_shrink > 0.05
            else list(outline)
        )

        core_triangles = self._triangulate_polygon(core_points)
        outer_triangles = self._triangulate_polygon(outer_points)
        if not core_triangles or not outer_triangles:
            return None

        inner_y = roof_height
        core_y = roof_height + max(0.22, roof_thickness * 0.8)
        upturn = max(0.06, roof_overhang * 0.2) if roof_overhang > 1e-4 else 0.0
        outer_y = roof_height + upturn
        bottom_y = roof_height - roof_thickness

        verts = []
        for x, z in core_points:
            verts.append(self._to_gl_pos((x, core_y, z)))
        core_start = 0
        for x, z in inner_points:
            verts.append(self._to_gl_pos((x, inner_y, z)))
        inner_start = len(core_points)
        for x, z in outer_points:
            verts.append(self._to_gl_pos((x, outer_y, z)))
        outer_start = inner_start + len(inner_points)
        for x, z in outer_points:
            verts.append(self._to_gl_pos((x, bottom_y, z)))
        bottom_start = outer_start + len(outer_points)

        faces = []
        for a, b, c in core_triangles:
            faces.append([core_start + a, core_start + b, core_start + c])
        for a, b, c in outer_triangles:
            faces.append([bottom_start + a, bottom_start + c, bottom_start + b])

        n = len(inner_points)
        for i in range(n):
            j = (i + 1) % n
            faces.append([core_start + i, core_start + j, inner_start + j])
            faces.append([core_start + i, inner_start + j, inner_start + i])
            faces.append([inner_start + i, inner_start + j, outer_start + j])
            faces.append([inner_start + i, outer_start + j, outer_start + i])
            faces.append([outer_start + i, outer_start + j, bottom_start + j])
            faces.append([outer_start + i, bottom_start + j, bottom_start + i])

        mesh_data = gl.MeshData(vertexes=np.array(verts, dtype=float), faces=np.array(faces, dtype=int))
        mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            color=color,
            shader="shaded",
            drawEdges=False,
        )
        self._register_lit_item(mesh, color, room=None)
        return mesh

    def _expand_polygon_radial(self, points: List[Vec2], offset: float) -> List[Vec2]:
        if abs(offset) < 1e-4:
            return list(points)
        cx = sum(p[0] for p in points) / len(points)
        cz = sum(p[1] for p in points) / len(points)
        expanded = []
        for x, z in points:
            dx = x - cx
            dz = z - cz
            dist = (dx ** 2 + dz ** 2) ** 0.5
            if dist < 1e-6:
                expanded.append((x, z))
            else:
                scale = (dist + offset) / dist
                expanded.append((cx + dx * scale, cz + dz * scale))
        return expanded

    def _triangulate_polygon(self, points: List[Vec2]) -> List[Tuple[int, int, int]]:
        if len(points) < 3:
            return []

        def area(poly):
            total = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                total += x1 * y2 - x2 * y1
            return total * 0.5

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def point_in_triangle(p, a, b, c):
            eps = 1e-6
            c1 = cross(a, b, p)
            c2 = cross(b, c, p)
            c3 = cross(c, a, p)
            return c1 >= -eps and c2 >= -eps and c3 >= -eps

        indices = list(range(len(points)))
        if area(points) < 0.0:
            indices.reverse()

        triangles = []
        guard = 0
        while len(indices) > 2 and guard < len(points) * len(points):
            ear_found = False
            for idx in range(len(indices)):
                i_prev = indices[idx - 1]
                i_curr = indices[idx]
                i_next = indices[(idx + 1) % len(indices)]

                if cross(points[i_prev], points[i_curr], points[i_next]) <= 1e-6:
                    continue

                is_ear = True
                for j in indices:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if point_in_triangle(points[j], points[i_prev], points[i_curr], points[i_next]):
                        is_ear = False
                        break

                if not is_ear:
                    continue

                triangles.append((i_prev, i_curr, i_next))
                indices.pop(idx)
                ear_found = True
                break

            if not ear_found:
                break
            guard += 1

        return triangles

    def _make_wall_segment_box(
        self,
        start: Vec2,
        end: Vec2,
        height: float,
        thickness: float,
        color,
        bottom: float = 0.0,
        normal: Optional[Vec2] = None,
        offset_normal: float = 0.0,
        draw_edges: bool = True,
    ) -> Optional[gl.GLMeshItem]:
        (x1, z1), (x2, z2) = start, end
        dx = x2 - x1
        dy = z2 - z1
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 1e-4 or height < 1e-4:
            return None

        mx = (x1 + x2) / 2.0
        my = (z1 + z2) / 2.0
        if normal is not None and abs(offset_normal) > 1e-6:
            mx += normal[0] * offset_normal
            my += normal[1] * offset_normal
        z_center = bottom + height / 2.0

        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=color,
            shader="shaded",
            drawEdges=draw_edges,
        )
        mesh.scale(length, thickness, height)
        angle = np.degrees(np.arctan2(dy, dx))
        mesh.rotate(-angle, 0, 0, 1)
        mesh.translate(mx, my, z_center)
        self._register_lit_item(mesh, color, room=None)
        return mesh

    def _build_wall_with_openings(self, wall: WallSegment) -> List[gl.GLGraphicsItem]:
        items: List[gl.GLGraphicsItem] = []
        (x1, z1), (x2, z2) = wall.start, wall.end
        dx = x2 - x1
        dy = z2 - z1
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 1e-4:
            return items

        direction = (dx / length, dy / length)
        normal = (-direction[1], direction[0])

        gap_ranges = []
        openings = [op for op in wall.openings if op.width > 1e-4]
        for op in openings:
            center = max(0.0, min(op.offset_along_wall, length))
            half = max(0.0, op.width / 2.0)
            start = max(0.0, center - half)
            end = min(length, center + half)
            if end - start < 1e-4:
                continue
            gap_ranges.append((start, end))

        gap_ranges.sort(key=lambda g: g[0])
        merged_gaps = []
        for start, end in gap_ranges:
            if not merged_gaps or start > merged_gaps[-1][1] + 1e-4:
                merged_gaps.append([start, end])
            else:
                merged_gaps[-1][1] = max(merged_gaps[-1][1], end)

        prev = 0.0
        for start, end in merged_gaps:
            if start - prev > 1e-4:
                seg_start = (x1 + direction[0] * prev, z1 + direction[1] * prev)
                seg_end = (x1 + direction[0] * start, z1 + direction[1] * start)
                mesh = self._make_wall_segment_box(
                    seg_start,
                    seg_end,
                    wall.height,
                    wall.thickness,
                    wall.color,
                    bottom=0.0,
                    normal=None,
                    draw_edges=True,
                )
                if mesh is not None:
                    items.append(mesh)
            prev = max(prev, end)
        if length - prev > 1e-4:
            seg_start = (x1 + direction[0] * prev, z1 + direction[1] * prev)
            seg_end = (x1 + direction[0] * length, z1 + direction[1] * length)
            mesh = self._make_wall_segment_box(
                seg_start,
                seg_end,
                wall.height,
                wall.thickness,
                wall.color,
                bottom=0.0,
                normal=None,
                draw_edges=True,
            )
            if mesh is not None:
                items.append(mesh)

        for op in openings:
            center = max(0.0, min(op.offset_along_wall, length))
            half = max(0.0, op.width / 2.0)
            start = max(0.0, center - half)
            end = min(length, center + half)
            if end - start < 1e-4:
                continue

            bottom = max(0.0, op.sill_height)
            top = min(wall.height, op.sill_height + op.height)

            if bottom > 1e-4:
                seg_start = (x1 + direction[0] * start, z1 + direction[1] * start)
                seg_end = (x1 + direction[0] * end, z1 + direction[1] * end)
                mesh = self._make_wall_segment_box(
                    seg_start,
                    seg_end,
                    bottom,
                    wall.thickness,
                    wall.color,
                    bottom=0.0,
                    normal=None,
                    draw_edges=True,
                )
                if mesh is not None:
                    items.append(mesh)

            if top < wall.height - 1e-4:
                seg_start = (x1 + direction[0] * start, z1 + direction[1] * start)
                seg_end = (x1 + direction[0] * end, z1 + direction[1] * end)
                mesh = self._make_wall_segment_box(
                    seg_start,
                    seg_end,
                    wall.height - top,
                    wall.thickness,
                    wall.color,
                    bottom=top,
                    normal=None,
                    draw_edges=True,
                )
                if mesh is not None:
                    items.append(mesh)

            if op.type == "window":
                self._add_window_panels(wall, op, direction, normal)
            elif op.type == "door":
                self._add_door_instance(wall, op, direction, normal)

        return items

    def _add_window_panels(
        self,
        wall: WallSegment,
        opening: Opening,
        direction: Vec2,
        normal: Vec2,
    ):
        (x1, z1) = wall.start
        length = ((wall.end[0] - wall.start[0]) ** 2 + (wall.end[1] - wall.start[1]) ** 2) ** 0.5
        if length < 1e-4:
            return

        center = max(0.0, min(opening.offset_along_wall, length))
        half = opening.width / 2.0
        start = max(0.0, center - half)
        end = min(length, center + half)
        if end - start < 1e-4:
            return

        seg_start = (x1 + direction[0] * start, z1 + direction[1] * start)
        seg_end = (x1 + direction[0] * end, z1 + direction[1] * end)

        panel_thickness = max(0.03, wall.thickness * 0.25)
        panel_offset = wall.thickness / 2.0 + panel_thickness / 2.0 + 0.01
        glass_color = (0.55, 0.8, 1.0, 0.35)

        for side in (-1.0, 1.0):
            mesh = self._make_wall_segment_box(
                seg_start,
                seg_end,
                opening.height,
                panel_thickness,
                glass_color,
                bottom=opening.sill_height,
                normal=normal,
                offset_normal=panel_offset * side,
                draw_edges=False,
            )
            if mesh is None:
                continue
            mesh.setGLOptions("translucent")
            mesh.setDepthValue(10)
            self.view.addItem(mesh)
            self.exterior_items.append(mesh)

    def _add_door_instance(
        self,
        wall: WallSegment,
        opening: Opening,
        direction: Vec2,
        normal: Vec2,
    ):
        (x1, z1) = wall.start
        length = ((wall.end[0] - wall.start[0]) ** 2 + (wall.end[1] - wall.start[1]) ** 2) ** 0.5
        if length < 1e-4:
            return

        center = max(0.0, min(opening.offset_along_wall, length))
        half = opening.width / 2.0
        hinge_offset = max(0.0, center - half)
        hinge_x = x1 + direction[0] * hinge_offset
        hinge_z = z1 + direction[1] * hinge_offset

        hinge_gl = np.array(self._to_gl_pos((hinge_x, opening.sill_height, hinge_z)), dtype=float)

        door_thickness = max(0.04, wall.thickness * 0.6)
        door_color = (0.55, 0.32, 0.2, 1.0)

        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=door_color,
            shader="shaded",
            drawEdges=False,
        )
        mesh.setDepthValue(5)
        self.view.addItem(mesh)
        self.door_items.append(mesh)

        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        open_angle = 90.0
        door_id = opening.id if getattr(opening, "id", None) else None

        door = DoorInstance(
            item=mesh,
            hinge=hinge_gl,
            width=opening.width,
            thickness=door_thickness,
            height=opening.height,
            base_angle=angle,
            open_angle=open_angle,
            wall_ref=wall,
            offset_along_wall=opening.offset_along_wall,
            door_id=door_id,
        )
        self._update_door_transform(door)
        if door.door_id and door.door_id in self._door_state_cache:
            self._set_door_state(door, self._door_state_cache[door.door_id], snap=True)
        self.doors.append(door)
    def _make_wall_windows(self, wall: WallSegment):
        """
        Very simple window quads slightly in front of the wall.
        v1: just 1 or 2 windows per segment if long enough.
        """
        (x1, z1), (x2, z2) = wall.start, wall.end
        length = ((x2 - x1) ** 2 + (z2 - z1) ** 2) ** 0.5
        if length < 1e-3:
            return None

        dx = (x2 - x1) / length
        dy = (z2 - z1) / length
        nx = -dy
        ny = dx

        window_items: List[gl.GLMeshItem] = []
        openings = [op for op in getattr(wall, "openings", []) if op.type == "window"]

        if not openings:
            # Fallback: simple evenly spaced windows
            num_windows = 2 if length > 6.0 else 1
            w_width = 1.0
            w_height = 1.2
            w_thickness = wall.thickness * 0.5
            sill_height = 0.9
            offset = wall.thickness * 0.6

            for i in range(num_windows):
                t = (i + 1) / (num_windows + 1)  # center fraction along wall
                wx = x1 + dx * length * t
                wy = z1 + dy * length * t
                wz = sill_height + (w_height / 2.0) + 0.01

                wx_front = wx + nx * offset
                wy_front = wy + ny * offset

                md = get_unit_cube_meshdata()
                mesh = gl.GLMeshItem(
                    meshdata=md,
                    smooth=False,
                    color=(0.6, 0.8, 1.0, 0.9),  # blue-ish "glass"
                    shader="shaded",
                    drawEdges=False,
                )
                mesh.scale(w_width, w_thickness, w_height)
                angle = np.degrees(np.arctan2(dy, dx))
                mesh.rotate(-angle, 0, 0, 1)
                mesh.translate(wx_front, wy_front, wz)
                window_items.append(mesh)
            return window_items

        # Plan-driven openings
        for op in openings:
            offset = max(0.0, min(op.offset_along_wall, length))
            wx = x1 + dx * offset
            wy = z1 + dy * offset
            wz = op.sill_height + (op.height / 2.0) + 0.01

            front_offset = wall.thickness / 2.0 + 0.01
            wx_front = wx + nx * front_offset
            wy_front = wy + ny * front_offset

            md = get_unit_cube_meshdata()
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                color=(0.6, 0.8, 1.0, 0.9),
                shader="shaded",
                drawEdges=False,
            )
            mesh.scale(op.width, wall.thickness * 0.5, op.height)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.translate(wx_front, wy_front, wz)
            window_items.append(mesh)

        return window_items

    def _make_wall_door(self, wall: WallSegment):
        """
        Simple door quad on the front wall.
        """
        (x1, z1), (x2, z2) = wall.start, wall.end
        length = ((x2 - x1) ** 2 + (z2 - z1) ** 2) ** 0.5
        if length < 1e-3:
            return None

        dx = (x2 - x1) / length
        dy = (z2 - z1) / length

        # Normal in x-y plane
        nx = -dy
        ny = dx

        openings = [op for op in getattr(wall, "openings", []) if op.type == "door"]
        if not openings:
            # Door centered around the segment midpoint
            mx = (x1 + x2) / 2.0
            my = (z1 + z2) / 2.0
            door_width = 0.9
            door_height = 2.0
            door_thickness = wall.thickness * 0.6
            offset = wall.thickness * 0.6

            dx_front = mx + nx * offset
            dy_front = my + ny * offset
            md = get_unit_cube_meshdata()
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                color=(0.5, 0.3, 0.2, 1.0),  # brown-ish
                shader="shaded",
                drawEdges=False,
            )
            mesh.scale(door_width, door_thickness, door_height)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.translate(dx_front, dy_front, door_height / 2.0)
            return mesh

        # Plan-driven doors (support multiple)
        door_items: List[gl.GLMeshItem] = []
        for op in openings:
            offset = max(0.0, min(op.offset_along_wall, length))
            px = x1 + dx * offset
            py = z1 + dy * offset
            pz = op.sill_height + (op.height / 2.0)

            front_offset = wall.thickness / 2.0 + 0.01
            px_front = px + nx * front_offset
            py_front = py + ny * front_offset

            md = get_unit_cube_meshdata()
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                color=(0.5, 0.3, 0.2, 1.0),
                shader="shaded",
                drawEdges=False,
            )
            mesh.scale(op.width, wall.thickness * 0.6, op.height)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.translate(px_front, py_front, pz)
            door_items.append(mesh)

        return door_items if door_items else None

    # ---- Interior rendering (debug / layout view) ----

    def _build_interior(self, house: House):
        for room in house.rooms:
            if room.name == "garden":
                # Skip "garden" room here; exterior already has a garden plane.
                continue
            mesh = self._make_room_box(room)
            self.view.addItem(mesh)
            self.interior_items.append(mesh)

    def _make_room_box(self, room: Room) -> gl.GLMeshItem:
        return self._make_box(
            center=room.center,
            size=room.size,
            color=room.color,
            draw_edges=True,
            normal=(0, 1, 0),
            room=room.name,
        )

    # ---- Sky ----

    def _build_sky(self):
        self.clear_items(self.sky_items)
        self._sky_bodies = [
            {"name": "sun", "color": (1.0, 0.95, 0.7, 1.0), "size": 3.0},
            {"name": "moon", "color": (0.75, 0.8, 0.95, 1.0), "size": 2.2},
        ]
        for body in self._sky_bodies:
            mesh = self._make_sky_body(body["color"], body["size"])
            body["mesh"] = mesh
        self._update_sky_positions(force=True)

    def _make_sky_body(self, color, size: float) -> gl.GLMeshItem:
        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=color,
            shader="shaded",
            drawEdges=False,
        )
        mesh.scale(size, size, size)
        self.view.addItem(mesh)
        self.sky_items.append(mesh)
        return mesh

    def _update_sky_positions(self, force: bool = False):
        now = time.time()
        if not force and now - self._sky_last_update < 1.0:
            return
        self._sky_last_update = now

        sun_dir = self._sun_direction()

        center = self._vector_to_np(self.view.opts.get("center", self.cam_center))
        sun_pos = center + sun_dir * self.sky_radius
        moon_pos = center - sun_dir * self.sky_radius

        sun_intensity = max(0.0, min(1.0, (sun_dir[2] + 0.2) / 1.2))
        night_color = np.array([10.0, 10.0, 20.0], dtype=float)
        day_color = np.array([120.0, 170.0, 230.0], dtype=float)
        bg = night_color + (day_color - night_color) * sun_intensity
        self.view.setBackgroundColor(tuple(int(c) for c in bg))
        self._update_lighting()

        for body in getattr(self, "_sky_bodies", []):
            mesh = body.get("mesh")
            if mesh is None:
                continue
            if body.get("name") == "sun":
                pos = sun_pos
            else:
                pos = moon_pos
            mesh.resetTransform()
            size = float(body.get("size", 2.0))
            mesh.scale(size, size, size)
            mesh.translate(pos[0], pos[1], pos[2])

    def _sun_direction(self) -> np.ndarray:
        sun_times = self._get_sunrise_sunset()
        if sun_times is None:
            return self._fallback_sun_direction()
        sunrise, sunset = sun_times
        now = datetime.now().astimezone()
        if sunrise <= now <= sunset:
            total = (sunset - sunrise).total_seconds()
            progress = (now - sunrise).total_seconds() / max(total, 1.0)
            angle = progress * math.pi
        else:
            if now < sunrise:
                night_start = sunset - timedelta(days=1)
                night_end = sunrise
            else:
                night_start = sunset
                night_end = sunrise + timedelta(days=1)
            total = (night_end - night_start).total_seconds()
            progress = (now - night_start).total_seconds() / max(total, 1.0)
            angle = math.pi + progress * math.pi
        return np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=float)

    def _fallback_sun_direction(self) -> np.ndarray:
        local = time.localtime()
        day_fraction = (local.tm_hour + local.tm_min / 60.0 + local.tm_sec / 3600.0) / 24.0
        angle = (day_fraction - 0.25) * (2.0 * math.pi)
        return np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=float)

    def _get_sunrise_sunset(self) -> Optional[Tuple[datetime, datetime]]:
        today = date.today()
        cache = self._sun_times_cache
        if (
            cache.get("date") == today
            and cache.get("sunrise")
            and cache.get("sunset")
            and cache.get("source") == "api"
        ):
            return cache["sunrise"], cache["sunset"]
        if time.time() - cache.get("fetched_at", 0.0) < 3600.0:
            if cache.get("sunrise") and cache.get("sunset"):
                return cache.get("sunrise"), cache.get("sunset")
            return None

        lat, lon = self.sun_location
        url = (
            "https://api.sunrise-sunset.org/json"
            f"?lat={lat}&lng={lon}&formatted=0"
        )
        try:
            with urllib.request.urlopen(url, timeout=4.0) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, ValueError, OSError):
            cache["fetched_at"] = time.time()
            return self._sunrise_sunset_fallback(today)

        if data.get("status") != "OK":
            cache["fetched_at"] = time.time()
            return self._sunrise_sunset_fallback(today)

        results = data.get("results", {})
        sunrise_str = results.get("sunrise")
        sunset_str = results.get("sunset")
        if not sunrise_str or not sunset_str:
            cache["fetched_at"] = time.time()
            return self._sunrise_sunset_fallback(today)

        sunrise = self._parse_iso_datetime(sunrise_str)
        sunset = self._parse_iso_datetime(sunset_str)
        if sunrise is None or sunset is None:
            cache["fetched_at"] = time.time()
            return self._sunrise_sunset_fallback(today)

        sunrise = sunrise.astimezone()
        sunset = sunset.astimezone()
        cache["date"] = today
        cache["sunrise"] = sunrise
        cache["sunset"] = sunset
        cache["fetched_at"] = time.time()
        cache["source"] = "api"
        return sunrise, sunset

    def _sunrise_sunset_fallback(self, for_date: date) -> Optional[Tuple[datetime, datetime]]:
        fallback = self._calculate_sunrise_sunset(for_date)
        cache = self._sun_times_cache
        cache["fetched_at"] = time.time()
        cache["source"] = "fallback"
        if fallback is None:
            cache["sunrise"] = None
            cache["sunset"] = None
            return None
        sunrise, sunset = fallback
        cache["date"] = for_date
        cache["sunrise"] = sunrise
        cache["sunset"] = sunset
        return sunrise, sunset

    def _calculate_sunrise_sunset(self, for_date: date) -> Optional[Tuple[datetime, datetime]]:
        """Approximate sunrise/sunset using a NOAA-style fallback calculation."""
        lat, lon = self.sun_location
        day_of_year = for_date.timetuple().tm_yday
        zenith = 90.833

        def _calc(is_sunrise: bool) -> Optional[datetime]:
            lng_hour = lon / 15.0
            if is_sunrise:
                t = day_of_year + ((6.0 - lng_hour) / 24.0)
            else:
                t = day_of_year + ((18.0 - lng_hour) / 24.0)

            mean_anomaly = (0.9856 * t) - 3.289
            true_long = (
                mean_anomaly
                + (1.916 * math.sin(math.radians(mean_anomaly)))
                + (0.020 * math.sin(math.radians(2 * mean_anomaly)))
                + 282.634
            ) % 360.0

            right_ascension = math.degrees(
                math.atan(0.91764 * math.tan(math.radians(true_long)))
            )
            right_ascension %= 360.0
            l_quadrant = math.floor(true_long / 90.0) * 90.0
            ra_quadrant = math.floor(right_ascension / 90.0) * 90.0
            right_ascension = (right_ascension + (l_quadrant - ra_quadrant)) / 15.0

            sin_dec = 0.39782 * math.sin(math.radians(true_long))
            cos_dec = math.cos(math.asin(sin_dec))
            cos_lat = math.cos(math.radians(lat))
            if abs(cos_lat) < 1e-6:
                return None
            cos_h = (
                math.cos(math.radians(zenith))
                - (sin_dec * math.sin(math.radians(lat)))
            ) / (cos_dec * cos_lat)

            if cos_h > 1 or cos_h < -1:
                return None

            if is_sunrise:
                hour_angle = 360.0 - math.degrees(math.acos(cos_h))
            else:
                hour_angle = math.degrees(math.acos(cos_h))
            hour_angle /= 15.0

            local_mean_time = hour_angle + right_ascension - (0.06571 * t) - 6.622
            utc_time = (local_mean_time - lng_hour) % 24.0
            utc_dt = datetime(
                for_date.year, for_date.month, for_date.day, tzinfo=timezone.utc
            ) + timedelta(hours=utc_time)
            return utc_dt.astimezone()

        sunrise = _calc(True)
        sunset = _calc(False)
        if sunrise is None or sunset is None:
            return None
        return sunrise, sunset

    def _parse_iso_datetime(self, value: str) -> Optional[datetime]:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    # ---- Player / first-person ----

    def set_first_person_enabled(self, enabled: bool):
        if enabled == self.first_person_enabled:
            return
        self.first_person_enabled = enabled
        self._last_mouse_pos = None

        if enabled:
            if self.architect_action.isChecked():
                self.architect_action.setChecked(False)
            if self.furnish_action.isChecked():
                self.furnish_action.setChecked(False)
            self._orbit_camera_state = {
                "center": self.view.opts.get("center", self.cam_center),
                "distance": self.view.opts.get("distance", self.cam_distance),
                "azimuth": self.view.opts.get("azimuth", self.cam_azimuth),
                "elevation": self.view.opts.get("elevation", self.cam_elevation),
            }
            self._ensure_player()
            self._sync_first_person_camera()
            self.view.setFocus()
            self._set_mouse_capture(True)
            if not self.mouse_capture_supported:
                self.statusBar().showMessage(
                    "Wayland detected: mouse capture disabled. Hold right mouse button to look.",
                    5000,
                )
        else:
            self.keys_down.clear()
            self._reset_interaction_hold()
            self._set_mouse_capture(False)
            if self._orbit_camera_state is not None:
                state = self._orbit_camera_state
                self.view.setCameraPosition(
                    pos=state["center"],
                    distance=state["distance"],
                    azimuth=state["azimuth"],
                    elevation=state["elevation"],
                )
                self.cam_center = state["center"]
                self.cam_distance = state["distance"]
                self.cam_azimuth = state["azimuth"]
                self.cam_elevation = state["elevation"]

        self._update_player_visibility()
        self._update_crosshair_visibility()
        self._update_grid_visibility()
        self._update_interaction_target()

    def _update_crosshair_position(self):
        if not hasattr(self, "crosshair"):
            return
        rect = self.view.rect()
        size = self.crosshair.size()
        x = max(0, (rect.width() - size.width()) // 2)
        y = max(0, (rect.height() - size.height()) // 2)
        self.crosshair.move(int(x), int(y))

    def _update_crosshair_visibility(self):
        if not hasattr(self, "crosshair"):
            return
        visible = self.first_person_enabled and not self._radial_menu_active
        self.crosshair.setVisible(visible)
        if visible:
            self._update_crosshair_position()

    def _update_grid_visibility(self):
        if not hasattr(self, "grid_item"):
            return
        self.grid_item.setVisible(not self.first_person_enabled)

    def set_player_avatar_enabled(self, enabled: bool):
        if enabled == self.player_avatar_enabled:
            return
        self.player_avatar_enabled = enabled
        self._rebuild_player_representation()

    def _rebuild_player_representation(self):
        self.clear_items(self.player_items)
        self.player_item = None
        self.player_avatar_parts = []
        if self.player_pos is None:
            self.player_pos = self._default_player_position()
        self._ensure_player()
        self._update_player_visibility()

    def _ensure_player(self):
        if self.player_pos is None:
            self.player_pos = self._default_player_position()
        if self.player_avatar_enabled:
            if self.player_item is not None:
                self.view.removeItem(self.player_item)
                if self.player_item in self.player_items:
                    self.player_items.remove(self.player_item)
                self.player_item = None
            if not self.player_avatar_parts:
                self._create_player_avatar_parts()
        else:
            if self.player_avatar_parts:
                for part in self.player_avatar_parts:
                    self.view.removeItem(part.mesh)
                    if part.mesh in self.player_items:
                        self.player_items.remove(part.mesh)
                self.player_avatar_parts = []
            if self.player_item is None:
                self.player_item = self._make_player_mesh()
                self.view.addItem(self.player_item)
                self.player_items.append(self.player_item)
        self._update_player_mesh()

    def _create_player_avatar_parts(self):
        self.player_avatar_parts = []
        md = get_unit_sphere_meshdata()
        for part_def in self._schema_part_defs(
            self.player_schema_path,
            target_height=self.player_height,
            color_map={
                "head": (0.95, 0.84, 0.78, 1.0),
                "hair": (0.2, 0.16, 0.12, 1.0),
                "left_eye": (0.96, 0.96, 0.98, 1.0),
                "right_eye": (0.96, 0.96, 0.98, 1.0),
                "left_ear": (0.94, 0.85, 0.8, 1.0),
                "right_ear": (0.94, 0.85, 0.8, 1.0),
                "mouth": (0.78, 0.48, 0.5, 1.0),
                "throat": (0.9, 0.78, 0.7, 1.0),
                "chest": (0.32, 0.34, 0.5, 1.0),
                "core": (0.28, 0.3, 0.46, 1.0),
                "left_arm": (0.35, 0.38, 0.55, 1.0),
                "right_arm": (0.35, 0.38, 0.55, 1.0),
                "left_leg": (0.16, 0.18, 0.26, 1.0),
                "right_leg": (0.16, 0.18, 0.26, 1.0),
            },
            default_color=(0.4, 0.4, 0.5, 1.0),
        ):
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                color=part_def.color,
                shader="shaded",
                drawEdges=False,
            )
            if part_def.gl_options is not None:
                mesh.setGLOptions(part_def.gl_options)
            self.view.addItem(mesh)
            self.player_items.append(mesh)
            self.player_avatar_parts.append(
                FurniturePart(
                    mesh=mesh,
                    size=part_def.size,
                    offset=part_def.offset,
                    base_size=part_def.size,
                    base_offset=part_def.offset,
                )
            )

    def _schema_part_defs_with_keys(
        self,
        schema_path: str,
        target_height: Optional[float] = None,
        color_map: Optional[dict] = None,
        default_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.7, 1.0),
    ) -> List[Tuple[str, FurniturePartDef]]:
        schema = self._load_body_schema(schema_path)
        anchors = schema.get("anchors", {})
        bounds = schema.get("body_bounds", {})
        extents = bounds.get("extents", [0.3, 0.3, 1.0])
        center = bounds.get("center", [0.0, 0.0, float(extents[2])])
        schema_height = max(float(extents[2]) * 2.0, 1e-3)
        scale = float(target_height) / schema_height if target_height else 1.0

        parts: List[Tuple[str, FurniturePartDef]] = []
        order = [
            "left_leg",
            "right_leg",
            "left_foot",
            "right_foot",
            "core",
            "chest",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "throat",
            "head",
            "hair",
            "left_ear",
            "right_ear",
            "left_eye",
            "right_eye",
            "mouth",
        ]
        for key in order:
            anchor = anchors.get(key)
            if not isinstance(anchor, dict):
                continue
            anchor_center = anchor.get("center", [0.0, 0.0, 0.0])
            radius = float(anchor.get("radius", 0.12))
            if len(anchor_center) < 3:
                continue
            offset = (
                (float(anchor_center[0]) - float(center[0])) * scale,
                (float(anchor_center[1]) - float(center[1])) * scale,
                (float(anchor_center[2]) - float(center[2])) * scale,
            )
            size = (radius * 2.0 * scale, radius * 2.0 * scale, radius * 2.0 * scale)
            color = default_color
            if color_map and key in color_map:
                color = color_map[key]
            parts.append((key, FurniturePartDef(size=size, offset=offset, color=color)))
        return parts

    def _schema_part_defs(
        self,
        schema_path: str,
        target_height: Optional[float] = None,
        color_map: Optional[dict] = None,
        default_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.7, 1.0),
    ) -> List[FurniturePartDef]:
        return [
            part_def
            for _key, part_def in self._schema_part_defs_with_keys(
                schema_path,
                target_height=target_height,
                color_map=color_map,
                default_color=default_color,
            )
        ]

    def _load_body_schema(self, path: str) -> dict:
        if path in self._schema_cache:
            return self._schema_cache[path]
        if not path or not os.path.exists(path):
            data = {}
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        self._schema_cache[path] = data
        return data

    def _ensure_ina_avatar(self):
        if self.architect_state.ina_spawn_point is None:
            return
        self.ina_pos = np.array(self._to_gl_pos(self.architect_state.ina_spawn_point), dtype=float)
        if not self.ina_avatar_parts:
            self._create_ina_avatar_parts()
        self._update_ina_avatar_mesh()

    def _create_ina_avatar_parts(self):
        self.ina_avatar_parts = []
        md = get_unit_sphere_meshdata()
        for part_def in self._schema_part_defs(
            self.ina_schema_path,
            target_height=None,
            color_map={
                "head": (0.95, 0.86, 0.82, 1.0),
                "hair": (0.25, 0.18, 0.12, 1.0),
                "left_eye": (0.96, 0.96, 0.98, 1.0),
                "right_eye": (0.96, 0.96, 0.98, 1.0),
                "left_ear": (0.94, 0.85, 0.8, 1.0),
                "right_ear": (0.94, 0.85, 0.8, 1.0),
                "mouth": (0.78, 0.48, 0.5, 1.0),
                "throat": (0.92, 0.82, 0.76, 1.0),
                "chest": (0.45, 0.36, 0.5, 1.0),
                "core": (0.4, 0.32, 0.48, 1.0),
                "left_arm": (0.48, 0.4, 0.55, 1.0),
                "right_arm": (0.48, 0.4, 0.55, 1.0),
                "left_hand": (0.52, 0.44, 0.6, 1.0),
                "right_hand": (0.52, 0.44, 0.6, 1.0),
                "left_leg": (0.22, 0.18, 0.28, 1.0),
                "right_leg": (0.22, 0.18, 0.28, 1.0),
                "left_foot": (0.24, 0.2, 0.3, 1.0),
                "right_foot": (0.24, 0.2, 0.3, 1.0),
            },
            default_color=(0.4, 0.35, 0.5, 1.0),
        ):
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                color=part_def.color,
                shader="shaded",
                drawEdges=False,
            )
            if part_def.gl_options is not None:
                mesh.setGLOptions(part_def.gl_options)
            self.view.addItem(mesh)
            self.ina_items.append(mesh)
            self.ina_avatar_parts.append(
                FurniturePart(
                    mesh=mesh,
                    size=part_def.size,
                    offset=part_def.offset,
                    base_size=part_def.size,
                    base_offset=part_def.offset,
                )
            )

    def _reset_player(self):
        self._clear_climb_surface()
        self.player_pos = self._default_player_position()
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self._update_player_mesh()

    def _default_player_position(self):
        ground_z = self._ground_z()
        if self.exterior_model is not None:
            spawn_point = getattr(self.exterior_model, "spawn_point", None)
            if spawn_point is not None:
                gl_spawn = self._to_gl_pos(spawn_point)
                return np.array(
                    [gl_spawn[0], gl_spawn[1], spawn_point[1] + self.player_eye_height],
                    dtype=float,
                )
            base_model = (
                self.exterior_model.garden_center[0],
                ground_z,
                self.exterior_model.garden_center[2],
            )
            gl_base = self._to_gl_pos(base_model)
            return np.array(
                [gl_base[0], gl_base[1], ground_z + self.player_eye_height],
                dtype=float,
            )
        center = self.view.opts.get("center", self.scene_center)
        if isinstance(center, pg.Vector):
            return np.array([center.x(), center.y(), ground_z + self.player_eye_height], dtype=float)
        try:
            cx, cy, _ = center
            return np.array([float(cx), float(cy), ground_z + self.player_eye_height], dtype=float)
        except Exception:
            return np.array([0.0, 0.0, ground_z + self.player_eye_height], dtype=float)

    def _make_player_mesh(self) -> gl.GLMeshItem:
        md = get_unit_cube_meshdata()
        return gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=(1.0, 0.4, 0.3, 0.85),
            shader="shaded",
            drawEdges=True,
        )

    def _update_player_mesh(self):
        if self.player_pos is None:
            return
        if self.player_avatar_enabled:
            self._update_player_avatar_mesh()
            return
        if self.player_item is None:
            return
        center = np.array(self.player_pos, dtype=float)
        body_offset = self.player_eye_height - (self.player_height / 2.0)
        center[2] -= body_offset

        self.player_item.resetTransform()
        self.player_item.scale(self.player_width, self.player_depth, self.player_height)
        self.player_item.translate(center[0], center[1], center[2])

    def _advance_anim_phase(
        self,
        *,
        phase: float,
        speed: float,
        now: float,
        last_ts: float,
        idle_speed: float = 0.2,
        rate: float = 6.0,
    ) -> Tuple[float, float]:
        dt = now - last_ts
        if dt <= 0.0:
            dt = 0.1
        dt = max(0.02, min(dt, 0.2))
        phase_speed = max(speed, idle_speed)
        phase = (phase + phase_speed * rate * dt) % (2.0 * math.pi)
        return phase, now

    def _player_animation_state(self) -> Tuple[float, float, float]:
        if self.player_pos is None:
            return self._player_anim_phase, 0.0, 0.4
        now = time.perf_counter()
        if self._player_anim_last_pos is None:
            self._player_anim_last_pos = np.array(self.player_pos, dtype=float)
            self._player_anim_last_ts = now
            return self._player_anim_phase, 0.0, 0.4
        dt = now - self._player_anim_last_ts
        if dt <= 0.0:
            dt = 0.1
        delta = np.array(self.player_pos, dtype=float) - self._player_anim_last_pos
        speed = float(np.linalg.norm(delta[:2]) / max(dt, 1e-3))
        self._player_anim_last_pos = np.array(self.player_pos, dtype=float)
        self._player_anim_phase, self._player_anim_last_ts = self._advance_anim_phase(
            phase=self._player_anim_phase,
            speed=speed,
            now=now,
            last_ts=self._player_anim_last_ts,
            idle_speed=0.15,
            rate=6.5,
        )
        max_speed = self.player_speed * self.player_sprint_multiplier
        intensity = min(speed / max(max_speed, 0.1), 1.0)
        if self.player_crouched:
            intensity *= 0.8
        idle = max(0.0, 0.4 - intensity)
        return self._player_anim_phase, intensity, idle

    def _ina_animation_state(self) -> Tuple[float, float, float]:
        speed = None
        intensity = None
        if self.ina_anim_use_inastate and get_inastate is not None:
            try:
                feedback = get_inastate("motor_feedback")
            except Exception:
                feedback = None
            if isinstance(feedback, dict):
                try:
                    speed = float(feedback.get("speed", 0.0))
                except Exception:
                    speed = 0.0
                try:
                    intensity = float(feedback.get("motion_intensity", 0.0))
                except Exception:
                    intensity = None
        if speed is None:
            speed = float(np.linalg.norm(getattr(self, "ina_velocity", np.zeros(3))[:2]))
        now = time.perf_counter()
        if self._ina_anim_last_pos is None:
            self._ina_anim_last_pos = (
                np.array(self.ina_pos, dtype=float) if self.ina_pos is not None else None
            )
            self._ina_anim_last_ts = now
        self._ina_anim_phase, self._ina_anim_last_ts = self._advance_anim_phase(
            phase=self._ina_anim_phase,
            speed=speed,
            now=now,
            last_ts=self._ina_anim_last_ts,
            idle_speed=0.12,
            rate=6.0,
        )
        if intensity is None:
            intensity = min(speed / 2.4, 1.0)
        idle = max(0.0, 0.5 - intensity)
        return self._ina_anim_phase, intensity, idle

    def _animation_offset(
        self,
        key: str,
        *,
        phase: float,
        intensity: float,
        idle: float,
        stride_scale: float = 1.0,
    ) -> Tuple[float, float, float]:
        stride = 0.12 * intensity * stride_scale
        lift = 0.07 * intensity * stride_scale
        sway = 0.04 * intensity * stride_scale
        bob = 0.02 * (intensity + 0.4 * idle)
        breath = 0.01 * idle

        def _step(phase_val: float) -> float:
            return math.sin(phase_val)

        dx = dy = dz = 0.0
        if key in ("left_leg", "left_foot"):
            step = _step(phase)
            dx = step * stride
            dz = max(0.0, step) * lift * (0.6 if key == "left_leg" else 1.0)
        elif key in ("right_leg", "right_foot"):
            step = _step(phase + math.pi)
            dx = step * stride
            dz = max(0.0, step) * lift * (0.6 if key == "right_leg" else 1.0)
        elif key in ("left_arm", "left_hand"):
            step = _step(phase + math.pi)
            dx = step * stride * 0.6
            dz = abs(step) * lift * 0.25
        elif key in ("right_arm", "right_hand"):
            step = _step(phase)
            dx = step * stride * 0.6
            dz = abs(step) * lift * 0.25
        elif key in ("core", "chest"):
            dy = _step(phase) * sway
            dz = bob
        elif key in ("head", "hair", "left_ear", "right_ear", "left_eye", "right_eye", "mouth", "throat"):
            dy = _step(phase) * sway * 0.3
            dz = bob * 1.5 + breath * _step(phase * 0.5)
        else:
            dz = bob * 0.5

        return dx, dy, dz

    def _update_player_avatar_mesh(self):
        if not self.player_avatar_parts:
            self._create_player_avatar_parts()
        part_defs = self._schema_part_defs_with_keys(
            self.player_schema_path,
            target_height=self.player_height,
            color_map=None,
        )
        if len(part_defs) != len(self.player_avatar_parts):
            self._create_player_avatar_parts()
            part_defs = self._schema_part_defs_with_keys(
                self.player_schema_path,
                target_height=self.player_height,
                color_map=None,
            )

        center = np.array(self.player_pos, dtype=float)
        body_offset = self.player_eye_height - (self.player_height / 2.0)
        center[2] -= body_offset

        phase, intensity, idle = self._player_animation_state()
        stride_scale = 0.7 if self.player_crouched else 1.0
        for part, (key, part_def) in zip(self.player_avatar_parts, part_defs):
            part.size = part_def.size
            dx, dy, dz = self._animation_offset(
                key,
                phase=phase,
                intensity=intensity,
                idle=idle,
                stride_scale=stride_scale,
            )
            part.offset = (
                part_def.offset[0] + dx,
                part_def.offset[1] + dy,
                part_def.offset[2] + dz,
            )
            part.mesh.resetTransform()
            part.mesh.scale(part.size[0], part.size[1], part.size[2])
            part_center = center + np.array(part.offset, dtype=float)
            part.mesh.translate(part_center[0], part_center[1], part_center[2])

    def _update_ina_avatar_mesh(self):
        if self.ina_pos is None:
            return
        if not self.ina_avatar_parts:
            self._create_ina_avatar_parts()
        part_defs = self._schema_part_defs_with_keys(
            self.ina_schema_path, target_height=None, color_map=None
        )
        if len(part_defs) != len(self.ina_avatar_parts):
            self._create_ina_avatar_parts()
            part_defs = self._schema_part_defs_with_keys(
                self.ina_schema_path, target_height=None, color_map=None
            )

        center = np.array(self.ina_pos, dtype=float)
        min_z = None
        for _key, part_def in part_defs:
            part_min = part_def.offset[2] - (part_def.size[2] / 2.0)
            if min_z is None or part_min < min_z:
                min_z = part_min
        if min_z is None:
            min_z = -max(self.player_height / 2.0, 0.1)
        center[2] -= min_z
        ground_z = 0.0
        if hasattr(self, "_ground_z"):
            try:
                ground_z = float(self._ground_z())
            except Exception:
                ground_z = 0.0
        if (center[2] + min_z) < ground_z:
            center[2] += ground_z - (center[2] + min_z)
        phase, intensity, idle = self._ina_animation_state()
        for part, (key, part_def) in zip(self.ina_avatar_parts, part_defs):
            part.size = part_def.size
            dx, dy, dz = self._animation_offset(
                key,
                phase=phase,
                intensity=intensity,
                idle=idle,
                stride_scale=0.95,
            )
            part.offset = (
                part_def.offset[0] + dx,
                part_def.offset[1] + dy,
                part_def.offset[2] + dz,
            )
            part.mesh.resetTransform()
            part.mesh.scale(part.size[0], part.size[1], part.size[2])
            part_center = center + np.array(part.offset, dtype=float)
            part.mesh.translate(part_center[0], part_center[1], part_center[2])

    def _update_player_visibility(self):
        for item in self.player_items:
            item.setVisible(not self.first_person_enabled)

    def _forward_vector(self, include_pitch: bool = True):
        yaw = np.radians(self.player_yaw)
        pitch = np.radians(self.player_pitch if include_pitch else 0.0)
        return np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            dtype=float,
        )

    def _ground_z(self) -> float:
        if getattr(self, "exterior_model", None) is None:
            return 0.0
        return float(
            self.exterior_model.garden_center[1]
            + (self.exterior_model.garden_size[1] / 2.0)
        )

    def _detect_mouse_capture_support(self) -> bool:
        platform_name = QtGui.QGuiApplication.platformName().lower()
        if "wayland" in platform_name:
            return False
        if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            return False
        return True

    def _load_player_schema(self):
        path = self.player_schema_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        bounds = data.get("body_bounds", {})
        extents = bounds.get("extents")
        if isinstance(extents, (list, tuple)) and len(extents) >= 3:
            try:
                width = float(extents[0]) * 2.0
                depth = float(extents[1]) * 2.0
                height = float(extents[2]) * 2.0
                if width > 1e-3 and depth > 1e-3 and height > 1e-3:
                    self.player_width = width
                    self.player_depth = depth
                    self.player_height = height
            except Exception:
                pass

        anchors = data.get("anchors", {})
        head = anchors.get("head", {}) if isinstance(anchors, dict) else {}
        eye_height = None
        if isinstance(head, dict):
            center = head.get("center")
            radius = head.get("radius", 0.0)
            if isinstance(center, (list, tuple)) and len(center) >= 3:
                try:
                    eye_height = float(center[2])
                    eye_height -= float(radius) * 0.2
                except Exception:
                    eye_height = None

        if eye_height is None and self.player_height > 1e-3:
            eye_height = self.player_height * 0.94
        if eye_height is not None:
            self.player_eye_height = max(0.1, float(eye_height))
        if self.player_avatar_enabled:
            self._rebuild_player_representation()
        elif self.player_item is not None:
            self._update_player_mesh()

    def _set_mouse_cursor_hidden(self, hidden: bool):
        if hidden == self._mouse_cursor_hidden:
            return
        if hidden:
            self.view.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))
        else:
            self.view.unsetCursor()
        self._mouse_cursor_hidden = hidden

    def _set_mouse_capture(self, enabled: bool):
        if not self.mouse_capture_supported:
            self._mouse_captured = False
            self._mouse_look_active = False
            self._set_mouse_cursor_hidden(False)
            return
        if enabled == self._mouse_captured:
            return

        if enabled:
            self.view.grabMouse()
            self._set_mouse_cursor_hidden(True)
            self._center_mouse_cursor()
        else:
            self.view.releaseMouse()
            self._set_mouse_cursor_hidden(False)
            self._last_mouse_pos = None
            self._mouse_look_active = False

        self._mouse_captured = enabled

    def _center_mouse_cursor(self):
        if not self.mouse_capture_supported:
            return
        if not self.view.isVisible():
            return
        center = self.view.rect().center()
        global_pos = self.view.mapToGlobal(center)
        self._recentering = True
        QtGui.QCursor.setPos(global_pos)
        self._last_mouse_pos = center

    def _sync_first_person_camera(self):
        if self.player_pos is None:
            return
        forward = self._forward_vector(include_pitch=True)
        center = self.player_pos + forward * self.player_look_distance
        azimuth = (self.player_yaw + 180.0) % 360.0
        elevation = -self.player_pitch

        self.view.setCameraPosition(
            pos=pg.Vector(center[0], center[1], center[2]),
            distance=self.player_look_distance,
            azimuth=azimuth,
            elevation=elevation,
        )

    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        self._update_sky_positions()
        self._update_door_animations(min(dt, 0.05))
        self._update_tv_stream()
        if not self.first_person_enabled:
            self._update_interaction_target()
            return
        self._apply_player_input(min(dt, 0.05))
        self._maybe_respawn_player()
        self._update_interaction_target()

    def _apply_player_input(self, dt: float):
        if self.player_pos is None:
            return

        yaw = np.radians(self.player_yaw)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)

        move = np.zeros(3, dtype=float)
        if QtCore.Qt.Key_W in self.keys_down:
            move += forward
        if QtCore.Qt.Key_S in self.keys_down:
            move -= forward
        if QtCore.Qt.Key_D in self.keys_down:
            move -= right
        if QtCore.Qt.Key_A in self.keys_down:
            move += right
        if self.seated:
            move[:] = 0.0
        self._set_crouched(
            QtCore.Qt.Key_Control in self.keys_down and not self.seated and self.ground_snap_enabled
        )
        if not self.ground_snap_enabled:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
            if QtCore.Qt.Key_Space in self.keys_down:
                move += up
            if QtCore.Qt.Key_C in self.keys_down:
                move -= up

        norm = np.linalg.norm(move)
        if norm > 1e-6:
            move /= norm

        speed = self.player_speed
        if QtCore.Qt.Key_Shift in self.keys_down:
            speed *= self.player_sprint_multiplier
        if self.player_crouched:
            speed *= self.player_crouch_speed_multiplier

        if norm > 1e-6:
            desired = self.player_pos + move * speed * dt
            self.player_pos = self._resolve_player_collision(self.player_pos, desired)

        if self.ground_snap_enabled and not self.seated:
            surface_z = self._update_climb_surface()
            ground_eye_z = self._ground_z() + self.player_eye_height
            if surface_z is not None:
                ground_eye_z = max(ground_eye_z, surface_z + self.player_eye_height)
            grounded = self.player_pos[2] <= ground_eye_z + 1e-3
            if grounded and self.player_vertical_velocity < 0.0:
                self.player_vertical_velocity = 0.0
            if grounded:
                self.player_pos[2] = ground_eye_z
                if QtCore.Qt.Key_Space in self.keys_down and not self._jump_consumed:
                    self.player_vertical_velocity = self.player_jump_speed
                    self._jump_consumed = True
                    grounded = False
            if not grounded:
                self.player_vertical_velocity -= self.player_gravity * dt
                self.player_pos[2] += self.player_vertical_velocity * dt
                if self.player_pos[2] <= ground_eye_z:
                    self.player_pos[2] = ground_eye_z
                    self.player_vertical_velocity = 0.0
        elif not self.seated and self.player_pos[2] < self.player_eye_height:
            self.player_pos[2] = self.player_eye_height
            self.player_vertical_velocity = 0.0
        elif not self.ground_snap_enabled:
            self.player_vertical_velocity = 0.0

        yaw_delta = 0.0
        pitch_delta = 0.0
        if QtCore.Qt.Key_Left in self.keys_down:
            yaw_delta += self.player_turn_speed * dt
        if QtCore.Qt.Key_Right in self.keys_down:
            yaw_delta -= self.player_turn_speed * dt
        if QtCore.Qt.Key_Up in self.keys_down:
            pitch_delta += self.player_turn_speed * dt
        if QtCore.Qt.Key_Down in self.keys_down:
            pitch_delta -= self.player_turn_speed * dt

        if yaw_delta or pitch_delta:
            self.player_yaw = (self.player_yaw + yaw_delta) % 360.0
            self.player_pitch = float(
                np.clip(self.player_pitch + pitch_delta, -85.0, 85.0)
            )

        self._sync_first_person_camera()
        self._update_player_mesh()

    def _set_crouched(self, crouched: bool) -> None:
        if crouched == self.player_crouched:
            return
        self.player_crouched = crouched
        self.player_eye_height = (
            self.player_eye_height_stand * self.player_crouch_factor
            if crouched
            else self.player_eye_height_stand
        )
        if self.player_pos is not None and self.ground_snap_enabled and not self.seated:
            ground_eye_z = self._ground_z() + self.player_eye_height
            self.player_pos[2] = ground_eye_z
            self.player_vertical_velocity = 0.0
        self._sync_first_person_camera()

    def _resolve_player_collision(self, current_pos: np.ndarray, desired_pos: np.ndarray) -> np.ndarray:
        if not self._collides_with_walls(desired_pos):
            return desired_pos
        resolved = np.array(current_pos, dtype=float)
        test = np.array(resolved, dtype=float)
        test[0] = desired_pos[0]
        if not self._collides_with_walls(test):
            resolved[0] = desired_pos[0]
        test = np.array(resolved, dtype=float)
        test[1] = desired_pos[1]
        if not self._collides_with_walls(test):
            resolved[1] = desired_pos[1]
        return resolved

    def _maybe_respawn_player(self):
        if self.player_pos is None:
            return
        if not np.isfinite(self.player_pos).all():
            self._respawn_player("Respawned: invalid position.")
            return
        ground_z = self._ground_z()
        if self.player_pos[2] < ground_z - self.player_respawn_depth:
            self._respawn_player("Respawned after fall.")

    def _respawn_player(self, message: str):
        self.seated = False
        self.seated_return_pos = None
        self.player_vertical_velocity = 0.0
        self._jump_consumed = False
        self._reset_player()
        self._sync_first_person_camera()
        self.statusBar().showMessage(message, 1600)

    def _collides_with_walls(self, pos: np.ndarray) -> bool:
        if self.exterior_model is None:
            return False
        radius = max(self.player_width, self.player_depth) * 0.5
        px, py = float(pos[0]), float(pos[1])
        for wall in self.exterior_model.walls:
            dist, offset = self._distance_to_segment((px, py), wall.start, wall.end)
            if dist <= (wall.thickness * 0.5 + radius):
                if self._door_gap_allows(wall, offset, radius):
                    continue
                return True
        if self._collides_with_fences(pos, radius):
            return True
        if self._collides_with_furniture(px, py, float(pos[2]), radius):
            return True
        return False

    def _collides_with_furniture(self, px: float, py: float, pz: float, radius: float) -> bool:
        if not self.furniture_instances:
            return False
        foot_z = float(pz) - float(self.player_eye_height)
        for instance in self.furniture_instances:
            bounds = self._furniture_vertical_bounds(instance, use_base=False)
            if bounds is not None:
                top_z = bounds[1]
                if foot_z >= top_z - self._climb_surface_margin:
                    continue
            dx = px - float(instance.position[0])
            dy = py - float(instance.position[1])
            limit = radius + float(instance.collision_radius)
            if (dx * dx + dy * dy) <= (limit * limit):
                return True
        return False

    def _collides_with_fences(self, pos: np.ndarray, radius: float) -> bool:
        if self.exterior_model is None:
            return False
        fences = []
        if self.architect_state.fences:
            fences = self.architect_state.fences
        elif getattr(self.exterior_model, "fences", None):
            fences = self.exterior_model.fences
        if not fences:
            return False
        px, py = float(pos[0]), float(pos[1])
        base_z = float(pos[2]) - float(self.player_eye_height)
        for fence in fences:
            if base_z >= float(fence.height) - 0.05:
                continue
            dist, _ = self._distance_to_segment((px, py), fence.start, fence.end)
            if dist <= (float(fence.thickness) * 0.5 + radius):
                return True
        return False

    def _door_gap_allows(self, wall: WallSegment, offset: float, radius: float) -> bool:
        for opening in wall.openings:
            if opening.type != "door":
                continue
            half_width = (opening.width * 0.5) + radius
            if abs(offset - opening.offset_along_wall) <= half_width:
                door = self._door_for_opening(wall, opening)
                if door is not None and abs(door.current_angle) < self.door_block_angle:
                    return False
                return True
        return False

    def _door_for_opening(self, wall: WallSegment, opening: Opening) -> Optional[DoorInstance]:
        for door in self.doors:
            if door.wall_ref is not wall:
                continue
            if abs(door.offset_along_wall - opening.offset_along_wall) <= 1e-3:
                return door
        return None

    # ---- Generic mesh helper ----

    def _handle_first_person_key_event(self, event, pressed: bool) -> bool:
        key = event.key()
        if pressed and not event.isAutoRepeat():
            if key == QtCore.Qt.Key_F:
                self.first_person_action.setChecked(not self.first_person_enabled)
                return True
            if key == QtCore.Qt.Key_E and self.first_person_enabled:
                self._begin_context_interaction()
                return True
            if key == QtCore.Qt.Key_Escape and self.first_person_enabled:
                self.first_person_action.setChecked(False)
                return True
            if key == QtCore.Qt.Key_R and self.first_person_enabled:
                self.reset_camera()
                return True

        if not self.first_person_enabled:
            return False

        if not pressed and key == QtCore.Qt.Key_E:
            if event.isAutoRepeat():
                return True
            self._finish_context_interaction()
            return True

        if key in self._player_control_keys:
            if event.isAutoRepeat():
                return True
            if pressed:
                self.keys_down.add(key)
            else:
                self.keys_down.discard(key)
                if key == QtCore.Qt.Key_Space:
                    self._jump_consumed = False
            return True

        return False

    def _handle_context_interaction(self):
        if self.player_pos is None:
            return
        if self._radial_menu_active:
            return
        if self.seated:
            self._stand_from_seat()
            return

        target = self._pick_interaction_target()
        if target is None:
            self.statusBar().showMessage("Nothing to interact with.", 1200)
            return
        self._handle_context_interaction_target(target)

    def _begin_context_interaction(self):
        if self.player_pos is None:
            return
        if self._radial_menu_active:
            return
        if self.seated:
            self._stand_from_seat()
            return
        target = self._pick_interaction_target()
        if target is None:
            self.statusBar().showMessage("Nothing to interact with.", 1200)
            return
        self._interact_press_target = target
        self._interact_press_active = True
        self._interact_hold_opened = False
        if self._interact_hold_timer.isActive():
            self._interact_hold_timer.stop()
        self._interact_hold_timer.start(max(1, int(self.interact_hold_threshold * 1000)))

    def _finish_context_interaction(self):
        if not self._interact_press_active:
            return
        if self._interact_hold_timer.isActive():
            self._interact_hold_timer.stop()
        target = self._interact_press_target
        menu_opened = self._interact_hold_opened
        self._interact_press_active = False
        self._interact_press_target = None
        self._interact_hold_opened = False
        if menu_opened:
            return
        if target is None:
            self.statusBar().showMessage("Nothing to interact with.", 1200)
            return
        self._handle_context_interaction_target(target)

    def _handle_context_interaction_target(self, target: dict):
        if self.player_pos is None:
            return
        if self._radial_menu_active:
            return
        if self.seated:
            self._stand_from_seat()
            return
        kind = target.get("kind")
        payload = target.get("payload")
        if kind == "door":
            self._toggle_door(payload)
        elif kind == "switch":
            self._toggle_switch(payload)
        elif kind == "bed":
            bed, bed_pos = payload
            self._lie_in_bed(bed, bed_pos, tuck=False)
        elif kind == "sofa":
            sofa, seat_pos = payload
            self._sit_on_sofa(sofa, seat_pos)
        elif kind == "desk":
            desk, seat_pos = payload
            self._use_desk(desk, seat_pos)
        elif kind == "chair":
            chair, seat_pos = payload
            self._sit_in_chair(chair, seat_pos)
        elif kind == "fridge":
            self._use_fridge(payload)
        elif kind == "microwave":
            self._use_microwave(payload)
        elif kind == "cooker":
            self._use_cooker(payload)
        elif kind == "hifi":
            self._use_hifi(payload)
        elif kind == "tv":
            self._use_tv(payload)
        elif kind == "bookshelf":
            self._use_bookshelf(payload)
        elif kind == "climb":
            self._climb_furniture(payload)

    def _open_context_menu(self, target: Optional[dict]) -> bool:
        if target is None:
            return False
        kind = target.get("kind")
        payload = target.get("payload")
        if kind == "switch":
            self._open_switch_menu(payload)
            return True
        if kind == "bed":
            bed, bed_pos = payload
            self._open_bed_menu(bed, bed_pos)
            return True
        if kind == "sofa":
            sofa, seat_pos = payload
            self._open_sofa_menu(sofa, seat_pos)
            return True
        if kind == "desk":
            desk, seat_pos = payload
            self._open_desk_menu(desk, seat_pos)
            return True
        if kind == "chair":
            chair, seat_pos = payload
            self._open_chair_menu(chair, seat_pos)
            return True
        if kind == "fridge":
            self._open_fridge_menu(payload)
            return True
        if kind == "microwave":
            self._open_microwave_menu(payload)
            return True
        if kind == "cooker":
            self._open_cooker_menu(payload)
            return True
        if kind == "hifi":
            self._open_hifi_menu(payload)
            return True
        if kind == "tv":
            self._open_tv_menu(payload)
            return True
        if kind == "bookshelf":
            self._open_bookshelf_menu(payload)
            return True
        return False

    def _on_interact_hold_timeout(self):
        if not self._interact_press_active or self._interact_hold_opened:
            return
        if not self.first_person_enabled or self._radial_menu_active:
            return
        if self._open_context_menu(self._interact_press_target):
            self._interact_hold_opened = True

    def _reset_interaction_hold(self):
        if self._interact_hold_timer.isActive():
            self._interact_hold_timer.stop()
        self._interact_press_active = False
        self._interact_press_target = None
        self._interact_hold_opened = False

    def _interaction_candidates(self) -> List[dict]:
        if self.player_pos is None:
            return []
        candidates: List[dict] = []
        used_instances: set[int] = set()
        door_info = self._nearest_door()
        if door_info is not None:
            door, _ = door_info
            pos = np.array(door.hinge, dtype=float)
            pos[2] += float(door.height) * 0.5
            candidates.append({"kind": "door", "payload": door, "pos": pos})

        switch_info = self._nearest_switch()
        if switch_info is not None:
            switch, _ = switch_info
            center = self._switch_world_center(
                (float(switch.position[0]), float(switch.position[1])),
                float(switch.height),
                switch.room,
            )
            pos = np.array(self._to_gl_pos(center), dtype=float)
            candidates.append({"kind": "switch", "payload": switch, "pos": pos})

        bed_info = self._nearest_bed()
        if bed_info is not None:
            bed, bed_pos, _ = bed_info
            used_instances.add(bed.instance_id)
            candidates.append({"kind": "bed", "payload": (bed, bed_pos), "pos": bed_pos})

        sofa_info = self._nearest_sofa()
        if sofa_info is not None:
            sofa, seat_pos, interact_pos, _ = sofa_info
            used_instances.add(sofa.instance_id)
            candidates.append(
                {"kind": "sofa", "payload": (sofa, seat_pos), "pos": interact_pos}
            )

        desk_info = self._nearest_desk()
        if desk_info is not None:
            desk, seat_pos, _ = desk_info
            used_instances.add(desk.instance_id)
            candidates.append({"kind": "desk", "payload": (desk, seat_pos), "pos": seat_pos})

        kitchen_info = self._nearest_kitchen_appliance()
        if kitchen_info is not None:
            kind, instance, interact_pos, _ = kitchen_info
            used_instances.add(instance.instance_id)
            candidates.append({"kind": kind, "payload": instance, "pos": interact_pos})

        hifi_info = self._nearest_hifi()
        if hifi_info is not None:
            hifi, interact_pos, _ = hifi_info
            used_instances.add(hifi.instance_id)
            candidates.append({"kind": "hifi", "payload": hifi, "pos": interact_pos})

        tv_info = self._nearest_tv()
        if tv_info is not None:
            tv, interact_pos, _ = tv_info
            used_instances.add(tv.instance_id)
            candidates.append({"kind": "tv", "payload": tv, "pos": interact_pos})

        bookshelf_info = self._nearest_bookshelf()
        if bookshelf_info is not None:
            bookshelf, interact_pos, _ = bookshelf_info
            used_instances.add(bookshelf.instance_id)
            candidates.append({"kind": "bookshelf", "payload": bookshelf, "pos": interact_pos})

        chair_info = self._nearest_chair()
        if chair_info is not None:
            chair, seat_pos, _ = chair_info
            used_instances.add(chair.instance_id)
            candidates.append({"kind": "chair", "payload": (chair, seat_pos), "pos": seat_pos})

        climb_info = self._nearest_climbable_surface(used_instances)
        if climb_info is not None:
            instance, surface_pos, _ = climb_info
            candidates.append({"kind": "climb", "payload": instance, "pos": surface_pos})

        return candidates

    def _pick_interaction_target(self) -> Optional[dict]:
        candidates = self._interaction_candidates()
        if not candidates or self.player_pos is None:
            return None

        forward = self._forward_vector(include_pitch=True)
        norm = np.linalg.norm(forward)
        if norm > 1e-6:
            forward = forward / norm

        aim_bias = 1.5
        scored = []
        for cand in candidates:
            to_target = np.array(cand["pos"], dtype=float) - np.array(self.player_pos, dtype=float)
            dist = float(np.linalg.norm(to_target))
            if dist < 1e-6:
                dist = 1e-6
            direction = to_target / dist
            dot = float(np.dot(direction, forward))
            score = dist * (1.0 + aim_bias * (1.0 - max(dot, 0.0)))
            cand["dist"] = dist
            cand["dot"] = dot
            cand["score"] = score
            scored.append(cand)

        aim_candidates = [cand for cand in scored if cand["dot"] >= 0.2]
        if aim_candidates:
            return min(aim_candidates, key=lambda item: item["score"])
        return min(scored, key=lambda item: item["dist"])

    def _ensure_interaction_highlight(self):
        if self._interaction_highlight is not None:
            return
        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=(1.0, 0.9, 0.4, 0.85),
            shader="shaded",
            drawEdges=True,
        )
        mesh.setGLOptions("translucent")
        mesh.setDepthValue(2)
        mesh.setVisible(False)
        self.view.addItem(mesh)
        self._interaction_highlight = mesh

    def _set_interaction_highlight(self, pos: np.ndarray, size: float, color: Tuple[float, float, float, float]):
        self._ensure_interaction_highlight()
        if self._interaction_highlight is None:
            return
        mesh = self._interaction_highlight
        mesh.resetTransform()
        mesh.scale(size, size, size)
        mesh.translate(float(pos[0]), float(pos[1]), float(pos[2]))
        if hasattr(mesh, "setColor"):
            mesh.setColor(color)
        mesh.setVisible(True)

    def _clear_interaction_highlight(self):
        if self._interaction_highlight is None:
            return
        self._interaction_highlight.setVisible(False)

    def _update_interaction_target(self):
        if not self.first_person_enabled or self.player_pos is None or self._radial_menu_active:
            self._interaction_target = None
            self._clear_interaction_highlight()
            return

        target = self._pick_interaction_target()
        self._interaction_target = target
        if target is None:
            self._clear_interaction_highlight()
            return

        kind = target["kind"]
        pos = np.array(target["pos"], dtype=float)
        if kind == "door":
            color = (1.0, 0.85, 0.35, 0.85)
            size = 0.2
        elif kind == "switch":
            color = (0.6, 0.95, 1.0, 0.9)
            size = 0.16
        elif kind == "bed":
            color = (0.85, 0.6, 1.0, 0.85)
            size = 0.28
            pos[2] += 0.08
        elif kind == "sofa":
            color = (0.9, 0.75, 0.55, 0.85)
            size = 0.26
            pos[2] += 0.08
        elif kind == "desk":
            color = (0.65, 0.85, 1.0, 0.85)
            size = 0.2
            pos[2] += 0.05
        elif kind == "fridge":
            color = (0.8, 0.9, 1.0, 0.85)
            size = 0.22
            pos[2] += 0.12
        elif kind == "microwave":
            color = (0.9, 0.85, 0.6, 0.85)
            size = 0.2
            pos[2] += 0.1
        elif kind == "cooker":
            color = (1.0, 0.7, 0.5, 0.85)
            size = 0.24
            pos[2] += 0.12
        elif kind == "hifi":
            color = (0.75, 0.9, 1.0, 0.85)
            size = 0.2
            pos[2] += 0.1
        elif kind == "tv":
            color = (0.65, 0.8, 1.0, 0.85)
            size = 0.26
            pos[2] += 0.12
        elif kind == "bookshelf":
            color = (0.9, 0.78, 0.5, 0.85)
            size = 0.24
            pos[2] += 0.1
        elif kind == "climb":
            color = (0.65, 1.0, 0.65, 0.85)
            size = 0.22
            pos[2] += 0.08
        else:
            color = (0.7, 1.0, 0.7, 0.85)
            size = 0.22
            pos[2] += 0.08
        self._set_interaction_highlight(pos, size, color)

    def _open_radial_menu(self, actions: List[Tuple[str, callable]]):
        if not actions:
            return
        if self.first_person_enabled and self.mouse_capture_supported:
            self._set_mouse_capture(False)
        self._radial_menu_active = True
        self._update_crosshair_visibility()
        center = self.view.mapToGlobal(self.view.rect().center())
        self.radial_menu.show_menu(center, actions)

    def _on_radial_menu_closed(self):
        self._radial_menu_active = False
        if self.first_person_enabled and self.mouse_capture_supported:
            self._set_mouse_capture(True)
        self._update_crosshair_visibility()

    def _nearest_door(self) -> Optional[Tuple[DoorInstance, float]]:
        if self.player_pos is None or not self.doors:
            return None
        player_xy = np.array([self.player_pos[0], self.player_pos[1]], dtype=float)
        best = None
        best_dist = None
        for door in self.doors:
            hinge_xy = np.array([door.hinge[0], door.hinge[1]], dtype=float)
            dist = float(np.linalg.norm(player_xy - hinge_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = door
        if best is None or best_dist is None or best_dist > self.door_interact_distance:
            return None
        return best, best_dist

    def _toggle_door(self, door: DoorInstance):
        if door is None:
            return
        open_state = abs(door.target_angle) < 1e-3
        self._set_door_state(door, open_state, snap=False)
        if door.door_id:
            self._door_state_cache[door.door_id] = open_state
        if self._door_state_callback is not None and door.door_id:
            try:
                self._door_state_callback(door.door_id, open_state)
            except Exception:
                pass

    def set_door_state_callback(self, callback) -> None:
        self._door_state_callback = callback

    def apply_door_states(self, door_states: Optional[dict], *, snap: bool = False) -> None:
        if not isinstance(door_states, dict):
            return
        for door_id, raw_state in door_states.items():
            parsed = self._parse_door_state(raw_state)
            if parsed is None:
                continue
            self._door_state_cache[str(door_id)] = parsed
        for door in self.doors:
            if not door.door_id:
                continue
            if door.door_id not in self._door_state_cache:
                continue
            self._set_door_state(door, self._door_state_cache[door.door_id], snap=snap)

    def _parse_door_state(self, raw_state: object) -> Optional[bool]:
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

    def _set_door_state(self, door: DoorInstance, open_state: bool, *, snap: bool = False) -> None:
        door.target_angle = door.open_angle if open_state else 0.0
        if snap:
            door.current_angle = door.target_angle
            self._update_door_transform(door)

    def _nearest_switch(self) -> Optional[Tuple[ArchitectLightSwitch, float]]:
        if self.player_pos is None or not self.architect_state.light_switches:
            return None
        player_xyz = np.array(self.player_pos, dtype=float)
        best = None
        best_dist = None
        for switch in self.architect_state.light_switches:
            center = self._switch_world_center(
                (float(switch.position[0]), float(switch.position[1])),
                float(switch.height),
                switch.room,
            )
            gl_center = np.array(self._to_gl_pos(center), dtype=float)
            dist = float(np.linalg.norm(player_xyz - gl_center))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = switch
        if best is None or best_dist is None or best_dist > 2.0:
            return None
        return best, best_dist

    def _toggle_switch(self, switch: ArchitectLightSwitch):
        room = switch.room or "global"
        state = self._get_room_light_state(room)
        state["enabled"] = not state["enabled"]
        self._refresh_light_items_from_state()
        self._update_lighting(force=True)
        state_label = "on" if state["enabled"] else "off"
        self.statusBar().showMessage(f"Lights {state_label}.", 1200)

    def _open_switch_menu(self, switch: ArchitectLightSwitch):
        room = switch.room or "global"
        bright_intensity = self.max_light_intensity
        actions = [
            ("On", lambda: self._set_room_light(room, enabled=True)),
            ("Off", lambda: self._set_room_light(room, enabled=False)),
            ("Low", lambda: self._set_room_light(room, intensity=0.35, enabled=True)),
            ("Med", lambda: self._set_room_light(room, intensity=0.65, enabled=True)),
            ("High", lambda: self._set_room_light(room, intensity=1.0, enabled=True)),
            ("Bright", lambda: self._set_room_light(room, intensity=bright_intensity, enabled=True)),
            ("Dimmer...", lambda: self._open_light_dimmer(room)),
            ("Warm", lambda: self._set_room_light(room, color=(1.0, 0.85, 0.6, 1.0), enabled=True)),
            ("Neutral", lambda: self._set_room_light(room, color=(1.0, 0.95, 0.8, 1.0), enabled=True)),
            ("Cool", lambda: self._set_room_light(room, color=(0.75, 0.85, 1.0, 1.0), enabled=True)),
        ]
        self._open_radial_menu(actions)

    def _open_light_dimmer(self, room: str):
        if self._light_dimmer_dialog is not None:
            self._light_dimmer_dialog.close()
            self._light_dimmer_dialog = None

        state = self._get_room_light_state(room)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Light Dimmer ({room})")
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        layout = QtWidgets.QVBoxLayout(dialog)

        value_label = QtWidgets.QLabel()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, int(self.max_light_intensity * 100))
        start_value = int(min(state["intensity"], self.max_light_intensity) * 100)
        slider.setValue(start_value)

        def _update_label(val: int):
            intensity = val / 100.0
            value_label.setText(f"Intensity: {intensity:.2f}x")

        def _apply_value(val: int):
            intensity = val / 100.0
            self._set_room_light(room, intensity=intensity, enabled=True)
            _update_label(val)

        slider.valueChanged.connect(_apply_value)
        _update_label(start_value)

        layout.addWidget(value_label)
        layout.addWidget(slider)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        self._light_dimmer_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_light_dimmer_dialog", None))
        dialog.show()

    def _set_room_light(
        self,
        room: str,
        enabled: Optional[bool] = None,
        intensity: Optional[float] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
    ):
        state = self._get_room_light_state(room)
        if enabled is not None:
            state["enabled"] = bool(enabled)
        if intensity is not None:
            clamped = max(0.0, min(self.max_light_intensity, float(intensity)))
            state["intensity"] = clamped
        if color is not None:
            state["color"] = color
        self._refresh_light_items_from_state()
        self._update_lighting(force=True)

    def _get_room_light_state(self, room: str) -> dict:
        state = self.room_light_state.get(room)
        if state is None:
            state = {
                "enabled": True,
                "intensity": 1.0,
                "color": (1.0, 1.0, 1.0, 1.0),
            }
            self.room_light_state[room] = state
        return state

    def _nearest_bed(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if instance.key != "bed":
                continue
            bed_pos = self._furniture_seat_position(instance)
            bed_xy = np.array(bed_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - bed_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, bed_pos)
        if best is None or best_dist is None:
            return None
        instance, bed_pos = best
        if best_dist > max(instance.interact_radius, 1.5):
            return None
        return instance, bed_pos, best_dist

    def _sofa_interaction_position(
        self,
        instance: FurnitureInstance,
        seat_pos: np.ndarray,
    ) -> np.ndarray:
        angle = np.radians(instance.rotation)
        front_dir = np.array([math.sin(angle), -math.cos(angle), 0.0], dtype=float)
        front_dist = 0.35
        return seat_pos + front_dir * front_dist

    def _nearest_sofa(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if not self._is_sofa_instance(instance):
                continue
            if not instance.seat_offset or instance.interact_radius <= 0.0:
                continue
            seat_pos = self._furniture_seat_position(instance)
            interact_pos = self._sofa_interaction_position(instance, seat_pos)
            interact_xy = np.array(interact_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - interact_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, seat_pos, interact_pos)
        if best is None or best_dist is None:
            return None
        instance, seat_pos, interact_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, seat_pos, interact_pos, best_dist

    def _nearest_desk(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if not self._is_desk_instance(instance):
                continue
            if not instance.seat_offset or instance.interact_radius <= 0.0:
                continue
            seat_pos = self._furniture_seat_position(instance)
            seat_xy = np.array(seat_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - seat_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, seat_pos)
        if best is None or best_dist is None:
            return None
        instance, seat_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, seat_pos, best_dist

    def _kitchen_interaction_position(self, instance: FurnitureInstance) -> np.ndarray:
        angle = np.radians(instance.rotation)
        front_dir = np.array([-math.sin(angle), math.cos(angle), 0.0], dtype=float)
        front_dist = max(instance.collision_radius * 0.6, 0.4)
        return instance.position + front_dir * front_dist

    def _hifi_interaction_position(self, instance: FurnitureInstance) -> np.ndarray:
        angle = np.radians(instance.rotation)
        front_dir = np.array([-math.sin(angle), math.cos(angle), 0.0], dtype=float)
        front_dist = max(instance.collision_radius * 0.7, 0.35)
        return instance.position + front_dir * front_dist

    def _tv_interaction_position(self, instance: FurnitureInstance) -> np.ndarray:
        angle = np.radians(instance.rotation)
        front_dir = np.array([-math.sin(angle), math.cos(angle), 0.0], dtype=float)
        front_dist = max(instance.collision_radius * 0.75, 0.5)
        return instance.position + front_dir * front_dist

    def _bookshelf_interaction_position(self, instance: FurnitureInstance) -> np.ndarray:
        pos = np.array(instance.position, dtype=float)
        bounds = self._furniture_vertical_bounds(instance, use_base=False)
        if bounds is not None:
            pos[2] = (bounds[0] + bounds[1]) * 0.5
        else:
            pos[2] += 0.8
        return pos

    def _nearest_hifi(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if not self._is_hifi_instance(instance):
                continue
            if instance.interact_radius <= 0.0:
                continue
            interact_pos = self._hifi_interaction_position(instance)
            interact_xy = np.array(interact_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - interact_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, interact_pos)
        if best is None or best_dist is None:
            return None
        instance, interact_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, interact_pos, best_dist

    def _nearest_bookshelf(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if not self._is_bookshelf_instance(instance):
                continue
            if instance.interact_radius <= 0.0:
                continue
            interact_pos = self._bookshelf_interaction_position(instance)
            interact_xy = np.array(interact_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - interact_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, interact_pos)
        if best is None or best_dist is None:
            return None
        instance, interact_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, interact_pos, best_dist

    def _nearest_tv(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if not self._is_tv_instance(instance):
                continue
            if instance.interact_radius <= 0.0:
                continue
            interact_pos = self._tv_interaction_position(instance)
            interact_xy = np.array(interact_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - interact_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, interact_pos)
        if best is None or best_dist is None:
            return None
        instance, interact_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, interact_pos, best_dist

    def _nearest_kitchen_appliance(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            kind = self._kitchen_appliance_kind(instance)
            if not kind:
                continue
            if instance.interact_radius <= 0.0:
                continue
            interact_pos = self._kitchen_interaction_position(instance)
            interact_xy = np.array(interact_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - interact_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (kind, instance, interact_pos)
        if best is None or best_dist is None:
            return None
        kind, instance, interact_pos = best
        if best_dist > instance.interact_radius:
            return None
        return kind, instance, interact_pos, best_dist

    def _append_climb_action(self, actions: List[Tuple[str, callable]], instance: FurnitureInstance) -> None:
        if not self._is_climbable_instance(instance):
            return
        actions.append(("Climb", lambda: self._climb_furniture(instance)))

    def _open_bed_menu(self, bed: FurnitureInstance, bed_pos: np.ndarray):
        actions = [
            ("Get In Bed", lambda: self._lie_in_bed(bed, bed_pos, tuck=False)),
            ("Tuck In", lambda: self._lie_in_bed(bed, bed_pos, tuck=True)),
            ("Make Bed", lambda: self._make_bed(bed)),
        ]
        self._append_climb_action(actions, bed)
        self._open_radial_menu(actions)

    def _open_sofa_menu(self, sofa: FurnitureInstance, seat_pos: np.ndarray):
        actions = [
            ("Sit", lambda: self._sit_on_sofa(sofa, seat_pos)),
            ("Lie Down", lambda: self._lie_on_sofa(sofa, seat_pos)),
        ]
        self._append_climb_action(actions, sofa)
        self._open_radial_menu(actions)

    def _open_desk_menu(self, desk: FurnitureInstance, seat_pos: np.ndarray):
        actions = [
            ("Use Computer", lambda: self._use_desk(desk, seat_pos)),
            ("Sit", lambda: self._sit_in_chair(desk, seat_pos)),
        ]
        self._append_climb_action(actions, desk)
        self._open_radial_menu(actions)

    def set_desk_use_callback(self, callback) -> None:
        self._desk_use_callback = callback

    def is_using_desk(self) -> bool:
        return bool(self._desk_in_use)

    def _open_chair_menu(self, chair: FurnitureInstance, seat_pos: np.ndarray):
        actions = [
            ("Sit", lambda: self._sit_in_chair(chair, seat_pos)),
        ]
        self._append_climb_action(actions, chair)
        self._open_radial_menu(actions)

    def _open_fridge_menu(self, fridge: FurnitureInstance):
        actions = [
            ("Grab Snack", lambda: self._request_kitchen_meal("snack", "fridge")),
        ]
        if self._kitchen_set_available(fridge):
            actions.append(("Make Large Meal", lambda: self._request_kitchen_meal("large_meal", "kitchen")))
        self._open_radial_menu(actions)

    def _open_microwave_menu(self, microwave: FurnitureInstance):
        actions = [
            ("Heat Small Meal", lambda: self._request_kitchen_meal("small_meal", "microwave")),
        ]
        if self._kitchen_set_available(microwave):
            actions.append(("Make Large Meal", lambda: self._request_kitchen_meal("large_meal", "kitchen")))
        self._open_radial_menu(actions)

    def _open_cooker_menu(self, cooker: FurnitureInstance):
        actions = [
            ("Cook Meal", lambda: self._request_kitchen_meal("meal", "cooker")),
        ]
        if self._kitchen_set_available(cooker):
            actions.append(("Make Large Meal", lambda: self._request_kitchen_meal("large_meal", "kitchen")))
        self._open_radial_menu(actions)

    def _open_hifi_menu(self, hifi: FurnitureInstance):
        actions = [
            ("Play / Pause", lambda: self._hifi_player_action("play-pause", "Hi-fi: play/pause.")),
            ("Next Track", lambda: self._hifi_player_action("next", "Hi-fi: next track.")),
            ("Previous Track", lambda: self._hifi_player_action("previous", "Hi-fi: previous track.")),
            ("Volume Up", lambda: self._hifi_volume_adjust(0.05, label="Hi-fi")),
            ("Volume Down", lambda: self._hifi_volume_adjust(-0.05, label="Hi-fi")),
        ]
        self._open_radial_menu(actions)

    def _open_tv_menu(self, tv: FurnitureInstance):
        actions = [
            ("Play Channel", lambda: self._launch_tv_channel()),
            ("Channel: Stremio", lambda: self._set_tv_channel("stremio")),
            ("Channel: YouTube", lambda: self._set_tv_channel("youtube")),
            ("Channel: Spotify", lambda: self._set_tv_channel("spotify")),
            ("Set Stremio Window", lambda: self._open_tv_window_picker("stremio")),
            ("Set YouTube Window", lambda: self._open_tv_window_picker("youtube", prefer_firefox=True)),
            ("Set Spotify Window", lambda: self._open_tv_window_picker("spotify")),
            ("Open Firefox", lambda: self._launch_firefox()),
            ("Open Spotify", lambda: self._launch_spotify()),
        ]
        if getattr(tv, "knocked", False):
            actions.insert(0, ("Right TV", lambda: self._toggle_tv_knock(tv)))
        else:
            actions.append(("Knock TV", lambda: self._toggle_tv_knock(tv)))
        self._open_radial_menu(actions)

    def _open_bookshelf_menu(self, bookshelf: FurnitureInstance):
        actions = [
            ("Read Books", lambda: self._use_bookshelf(bookshelf)),
        ]
        self._append_climb_action(actions, bookshelf)
        self._open_radial_menu(actions)

    def _kitchen_set_available(self, instance: FurnitureInstance) -> bool:
        room = self._room_for_point((float(instance.position[0]), float(instance.position[1])))
        needed = {"fridge", "microwave", "cooker"}
        for other in self.furniture_instances:
            kind = self._kitchen_appliance_kind(other)
            if not kind:
                continue
            if room is not None:
                other_room = self._room_for_point((float(other.position[0]), float(other.position[1])))
                if other_room != room:
                    continue
            needed.discard(kind)
            if not needed:
                return True
        return False

    def _use_fridge(self, fridge: FurnitureInstance):
        self._request_kitchen_meal("snack", "fridge")

    def _use_microwave(self, microwave: FurnitureInstance):
        self._request_kitchen_meal("small_meal", "microwave")

    def _use_cooker(self, cooker: FurnitureInstance):
        self._request_kitchen_meal("meal", "cooker")

    def _use_hifi(self, hifi: FurnitureInstance):
        self._hifi_player_action("play-pause", "Hi-fi: play/pause.")

    def _use_tv(self, tv: FurnitureInstance):
        if getattr(tv, "knocked", False):
            self.statusBar().showMessage("TV is knocked over.", 1600)
            return
        self._launch_tv_channel()

    def _use_bookshelf(self, bookshelf: FurnitureInstance):
        if self._trigger_self_read(source="books"):
            self.statusBar().showMessage("Bookshelf: reading queued.", 1600)
        else:
            self.statusBar().showMessage("Bookshelf: reading unavailable.", 1600)

    def _trigger_self_read(self, source: Optional[str] = None) -> bool:
        if safe_popen is None:
            return False
        if not Path("raw_file_manager.py").exists():
            return False
        env = dict(os.environ)
        if source:
            env["SELF_READ_SOURCE"] = source
        return safe_popen(["python", "raw_file_manager.py"], env=env) is not None

    def _toggle_tv_knock(self, tv: FurnitureInstance):
        tv.knocked = not getattr(tv, "knocked", False)
        if tv.knocked:
            bounds = self._furniture_vertical_bounds(tv, use_base=False)
            height = 0.0
            if bounds is not None:
                height = bounds[1] - bounds[0]
            tv.tilt_deg = 90.0
            tv.pose_offset = (0.0, 0.0, -height * 0.4)
            entry = self._tv_stream_items.get(tv.instance_id)
            if entry and entry.get("item") is not None:
                entry["item"].setVisible(False)
            self.statusBar().showMessage("TV knocked over.", 1600)
        else:
            tv.tilt_deg = 0.0
            tv.pose_offset = (0.0, 0.0, 0.0)
            entry = self._tv_stream_items.get(tv.instance_id)
            if entry and entry.get("item") is not None:
                entry["item"].setVisible(True)
            self.statusBar().showMessage("TV righted.", 1600)
        self._update_furniture_instance_transform(tv)

    def _request_kitchen_meal(self, meal_name: str, source: str):
        try:
            from model_manager import request_meal
        except Exception:
            self.statusBar().showMessage("Kitchen connection unavailable.", 1600)
            return
        if request_meal(meal_name, reason=f"kitchen:{source}"):
            label = meal_name.replace("_", " ")
            self.statusBar().showMessage(f"Kitchen: {label} queued.", 1600)
        else:
            self.statusBar().showMessage("Kitchen busy. Try again later.", 1600)

    def _hifi_player_action(self, action: str, message: str):
        if self._hifi_playerctl([action]):
            self.statusBar().showMessage(message, 1400)

    def _hifi_volume_adjust(self, delta: float, label: str = "Hi-fi"):
        step = f"{abs(delta):.2f}{'+' if delta >= 0.0 else '-'}"
        if self._hifi_playerctl(["volume", step]):
            direction = "up" if delta >= 0.0 else "down"
            self.statusBar().showMessage(f"{label}: volume {direction}.", 1400)

    def _launch_firefox(self, url: Optional[str] = None) -> bool:
        firefox_path = shutil.which("firefox")
        cmd = None
        if firefox_path:
            cmd = [firefox_path]
            if url:
                cmd.append(url)
        elif url:
            opener = shutil.which("xdg-open")
            if opener:
                cmd = [opener, url]
        if not cmd:
            self.statusBar().showMessage("TV: Firefox not available.", 2000)
            return False
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self.statusBar().showMessage("TV: failed to launch Firefox.", 2000)
            return False
        return True

    def _set_tv_channel(self, channel: str):
        normalized = channel.strip().lower()
        if normalized not in ("stremio", "youtube", "spotify"):
            self.statusBar().showMessage("TV: unknown channel.", 1800)
            return
        self.tv_channel = normalized
        has_window = self._apply_tv_channel_window(normalized)
        label = "Stremio" if normalized == "stremio" else "YouTube"
        if normalized == "spotify":
            label = "Spotify"
        suffix = "" if has_window else " (set a window)"
        self.statusBar().showMessage(f"TV channel set to {label}{suffix}.", 1600)

    def _launch_tv_channel(self) -> bool:
        channel = (self.tv_channel or "stremio").strip().lower()
        self._apply_tv_channel_window(channel)
        if channel == "youtube":
            return self._launch_firefox("https://www.youtube.com")
        if channel == "spotify":
            return self._launch_spotify()
        return self._launch_stremio()

    def _load_tv_channel(self) -> str:
        config = self._load_config_data()
        channel = (
            config.get("tv_settings", {})
            .get("active_channel", "stremio")
        )
        if not isinstance(channel, str):
            return "stremio"
        channel = channel.strip().lower()
        return channel if channel in ("stremio", "youtube", "spotify") else "stremio"

    def _apply_tv_channel_window(self, channel: str) -> bool:
        config = self._load_config_data()
        tv_settings = config.get("tv_settings", {})
        channels = tv_settings.get("channels", {})
        tv_settings["active_channel"] = channel
        config["tv_settings"] = tv_settings
        info = channels.get(channel)
        if not isinstance(info, dict):
            self._save_config_data(config)
            return False
        bounds = info.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
            self._save_config_data(config)
            return False
        self._save_config_data(config)
        return True

    def _launch_stremio(self) -> bool:
        stremio_paths = ["stremio", "stremio-desktop"]
        cmd = None
        for candidate in stremio_paths:
            path = shutil.which(candidate)
            if path:
                cmd = [path]
                break
        if cmd is None:
            flatpak = shutil.which("flatpak")
            if flatpak:
                cmd = [flatpak, "run", "com.stremio.Stremio"]
        if not cmd:
            self.statusBar().showMessage("TV: Stremio not available.", 2000)
            return False
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self.statusBar().showMessage("TV: failed to launch Stremio.", 2000)
            return False
        return True

    def _launch_spotify(self) -> bool:
        spotify_paths = ["spotify", "spotify-launcher"]
        cmd = None
        for candidate in spotify_paths:
            path = shutil.which(candidate)
            if path:
                cmd = [path]
                break
        if cmd is None:
            flatpak = shutil.which("flatpak")
            if flatpak:
                cmd = [flatpak, "run", "com.spotify.Client"]
        if not cmd:
            self.statusBar().showMessage("TV: Spotify not available.", 2000)
            return False
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self.statusBar().showMessage("TV: failed to launch Spotify.", 2000)
            return False
        return True

    def _open_tv_window_picker(self, channel: str, prefer_firefox: bool = False):
        windows, preferred_found = self._list_window_candidates(prefer_firefox=prefer_firefox)
        if not windows:
            if self._prompt_tv_window_bounds(channel):
                self.statusBar().showMessage(
                    f"TV: {channel.title()} window set (manual bounds).",
                    2000,
                )
                return
            self.statusBar().showMessage("TV: no windows found (wmctrl missing).", 2200)
            return
        items = []
        for win in windows:
            class_hint = f" [{win['wm_class']}]" if win.get("wm_class") else ""
            items.append(
                f"{win['title']}{class_hint} ({win['width']}x{win['height']}+{win['x']}+{win['y']})"
            )
        label = "Firefox Window" if prefer_firefox and preferred_found else "Window"
        prompt = f"Choose a window for {channel.title()}:"
        if prefer_firefox and not preferred_found:
            prompt = f"No Firefox windows found; choose a window for {channel.title()}:"
        selection, ok = QtWidgets.QInputDialog.getItem(
            self,
            f"Select {label}",
            prompt,
            items,
            0,
            False,
        )
        if not ok:
            return
        index = items.index(selection)
        chosen = windows[index]
        self._store_tv_channel_window(channel, chosen)
        self.statusBar().showMessage(
            f"TV: {channel.title()} window set to '{chosen['title']}'.",
            2000,
        )

    def _prompt_tv_window_bounds(self, channel: str) -> bool:
        placeholder = "x,y,width,height"
        value, ok = QtWidgets.QInputDialog.getText(
            self,
            "Manual Window Bounds",
            f"Enter {channel.title()} window bounds ({placeholder}):",
            text="",
        )
        if not ok:
            return False
        raw = value.strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 4:
            self.statusBar().showMessage("TV: expected x,y,width,height.", 2000)
            return False
        try:
            x, y, w, h = (int(p) for p in parts)
        except ValueError:
            self.statusBar().showMessage("TV: bounds must be integers.", 2000)
            return False
        if w <= 1 or h <= 1:
            self.statusBar().showMessage("TV: width/height must be > 1.", 2000)
            return False
        title, _ = QtWidgets.QInputDialog.getText(
            self,
            "Window Title (Optional)",
            f"Label for {channel.title()} window:",
            text=channel.title(),
        )
        window_info = {
            "window_id": None,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "title": (title or channel.title()).strip() or channel.title(),
        }
        self._store_tv_channel_window(channel, window_info)
        return True

    def _list_window_candidates(self, prefer_firefox: bool = False) -> Tuple[List[dict], bool]:
        wmctrl_path = shutil.which("wmctrl")
        if not wmctrl_path:
            return [], False
        try:
            result = subprocess.run(
                [wmctrl_path, "-lxG"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return [], False
        windows = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 8)
            if len(parts) < 9:
                continue
            win_id, _desktop, x, y, w, h, _host, wm_class, title = parts
            try:
                info = {
                    "window_id": win_id,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "title": title.strip(),
                    "wm_class": wm_class.strip(),
                }
            except ValueError:
                continue
            if info["width"] <= 0 or info["height"] <= 0:
                continue
            title_lower = info["title"].lower()
            class_lower = info["wm_class"].lower()
            info["preferred"] = (
                "firefox" in title_lower
                or "mozilla" in title_lower
                or "firefox" in class_lower
                or "mozilla" in class_lower
            )
            windows.append(info)
        if prefer_firefox:
            firefox = [win for win in windows if win.get("preferred")]
            if firefox:
                return firefox, True
            return windows, False
        return windows, False

    def _store_tv_channel_window(self, channel: str, window_info: dict):
        config = self._load_config_data()
        tv_settings = config.get("tv_settings", {})
        channels = tv_settings.get("channels", {})
        channels[channel] = {
            "window_id": window_info.get("window_id"),
            "title": window_info.get("title"),
            "wm_class": window_info.get("wm_class"),
            "bounds": [
                window_info.get("x"),
                window_info.get("y"),
                window_info.get("width"),
                window_info.get("height"),
            ],
        }
        tv_settings["channels"] = channels
        tv_settings["active_channel"] = self.tv_channel
        config["tv_settings"] = tv_settings
        self._save_config_data(config)

    def _load_config_data(self) -> dict:
        path = Path("config.json")
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_config_data(self, data: dict) -> bool:
        path = Path("config.json")
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
        except Exception:
            self.statusBar().showMessage("TV: failed to save config.json.", 2200)
            return False
        return True

    def _refresh_tv_stream_config(self) -> None:
        path = Path("config.json")
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._tv_stream_bounds = None
            self._tv_stream_config_mtime = None
            return
        if self._tv_stream_config_mtime is not None and stat.st_mtime == self._tv_stream_config_mtime:
            return
        self._tv_stream_config_mtime = stat.st_mtime
        try:
            with path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            self._tv_stream_bounds = None
            return
        tv_settings = cfg.get("tv_settings", {})
        channel = tv_settings.get("active_channel", self.tv_channel)
        if not isinstance(channel, str):
            self._tv_stream_bounds = None
            return
        channel = channel.strip().lower()
        channels = tv_settings.get("channels", {})
        if not isinstance(channels, dict):
            self._tv_stream_bounds = None
            return
        info = channels.get(channel)
        if not isinstance(info, dict):
            self._tv_stream_bounds = None
            return
        bounds = info.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
            self._tv_stream_bounds = None
            return
        try:
            left = int(bounds[0])
            top = int(bounds[1])
            width = int(bounds[2])
            height = int(bounds[3])
        except (TypeError, ValueError):
            self._tv_stream_bounds = None
            return
        if width <= 1 or height <= 1:
            self._tv_stream_bounds = None
            return
        self._tv_stream_bounds = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

    def _update_tv_stream(self):
        if not self.furniture_instances:
            return
        self._refresh_tv_stream_config()
        if not self._tv_stream_bounds:
            return
        now = time.perf_counter()
        if (now - self._tv_stream_last_update) < self._tv_stream_interval:
            return
        self._tv_stream_last_update = now
        frame = None
        try:
            import mss  # type: ignore

            with mss.mss() as grabber:
                raw = grabber.grab(self._tv_stream_bounds)
            if raw is not None:
                frame = np.array(raw)
                if frame.shape[-1] == 4:
                    frame = frame[:, :, [2, 1, 0, 3]]
        except Exception:
            frame = None
        if frame is None:
            try:
                import pyautogui  # type: ignore

                bounds = self._tv_stream_bounds
                image = pyautogui.screenshot(
                    region=(
                        bounds["left"],
                        bounds["top"],
                        bounds["width"],
                        bounds["height"],
                    )
                )
                frame = np.array(image)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    alpha = np.full(frame.shape[:2] + (1,), 255, dtype=frame.dtype)
                    frame = np.concatenate([frame, alpha], axis=2)
            except Exception:
                return
        if frame is None or frame.ndim < 3 or frame.shape[2] < 3:
            return
        target_w, target_h = self._tv_stream_size
        try:
            import cv2  # type: ignore

            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception:
            step_y = max(int(frame.shape[0] / target_h), 1)
            step_x = max(int(frame.shape[1] / target_w), 1)
            frame = frame[::step_y, ::step_x]
            frame = frame[:target_h, :target_w]
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            return
        if frame.shape[2] == 3:
            alpha = np.full(frame.shape[:2] + (1,), 255, dtype=frame.dtype)
            frame = np.concatenate([frame, alpha], axis=2)
        frame = frame.astype(np.uint8, copy=False)
        data = np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
        for instance in self.furniture_instances:
            if not self._is_tv_instance(instance):
                continue
            if getattr(instance, "knocked", False):
                continue
            entry = self._ensure_tv_stream_item(instance, target_w, target_h)
            if entry is None:
                continue
            item = entry["item"]
            item.setData(data)
            self._update_tv_stream_item_transform(instance, item, target_w, target_h)

    def _ensure_tv_stream_item(self, instance: FurnitureInstance, width_px: int, height_px: int):
        entry = self._tv_stream_items.get(instance.instance_id)
        if entry and entry.get("size") == (width_px, height_px):
            return entry
        if entry and entry.get("item") is not None:
            self.view.removeItem(entry["item"])
        blank = np.zeros((width_px, height_px, 4), dtype=np.ubyte)
        item = gl.GLImageItem(blank, smooth=True, glOptions="opaque")
        self.view.addItem(item)
        entry = {"item": item, "size": (width_px, height_px)}
        self._tv_stream_items[instance.instance_id] = entry
        return entry

    def _update_tv_stream_item_transform(
        self,
        instance: FurnitureInstance,
        item: gl.GLImageItem,
        width_px: int,
        height_px: int,
    ):
        if not instance.parts:
            return
        screen_part = max(
            instance.parts,
            key=lambda part: float(part.size[0]) * float(part.size[2]),
        )
        angle = np.radians(instance.rotation)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rot = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        rotated_offset = rot @ np.array(screen_part.offset, dtype=float)
        center = instance.position + rotated_offset
        width = float(screen_part.size[0])
        height = float(screen_part.size[2])
        depth = float(screen_part.size[1])
        right_dir = np.array([cos_a, sin_a, 0.0], dtype=float)
        up_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        front_dir = np.array([-sin_a, cos_a, 0.0], dtype=float)
        front_offset = depth * 0.5 + 0.01
        origin = (
            center
            + front_dir * front_offset
            - right_dir * (width / 2.0)
            - up_dir * (height / 2.0)
        )
        if width_px <= 0 or height_px <= 0:
            return
        scale_x = width / float(width_px)
        scale_y = height / float(height_px)
        m = np.eye(4, dtype=float)
        m[0:3, 0] = right_dir * scale_x
        m[0:3, 1] = up_dir * scale_y
        m[0:3, 2] = front_dir
        m[0:3, 3] = origin
        item.setTransform(pg.Transform3D(m))

    def _hifi_playerctl(self, args: List[str]) -> bool:
        if sys.platform != "linux":
            self.statusBar().showMessage("Media control requires Linux MPRIS.", 2000)
            return False
        playerctl_path = shutil.which("playerctl")
        if not playerctl_path:
            self.statusBar().showMessage("Media: install playerctl to control Spotify.", 2200)
            return False
        cmd = [playerctl_path]
        player = self._resolve_hifi_player()
        if player:
            cmd.extend(["-p", player])
        cmd.extend(args)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.statusBar().showMessage("Media: playerctl missing.", 2000)
            return False
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or "").strip()
            if "No players found" in msg or "Player is not running" in msg:
                self._hifi_player_cache = None
                self._hifi_player_cache_ts = 0.0
                self.statusBar().showMessage("Media: open Spotify and try again.", 2000)
            elif msg:
                self.statusBar().showMessage(f"Media: {msg}", 2000)
            else:
                self.statusBar().showMessage("Media: command failed.", 2000)
            return False
        return True

    def _resolve_hifi_player(self) -> Optional[str]:
        now = time.time()
        if (
            self._hifi_player_cache
            and (now - self._hifi_player_cache_ts) < 4.0
        ):
            return self._hifi_player_cache
        env_override = os.environ.get("INAZUMA_HIFI_PLAYER")
        if env_override:
            self._hifi_player_cache = env_override
            self._hifi_player_cache_ts = now
            return env_override
        playerctl_path = shutil.which("playerctl")
        if not playerctl_path:
            return None
        result = subprocess.run(
            [playerctl_path, "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        players = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not players:
            return None
        selected = None
        for name in players:
            if "spotify" in name.lower():
                selected = name
                break
        if selected is None:
            selected = players[0]
        self._hifi_player_cache = selected
        self._hifi_player_cache_ts = now
        return selected

    def _sit_on_sofa(self, sofa: FurnitureInstance, seat_pos: np.ndarray):
        self._clear_climb_surface()
        self._sit_in_chair(sofa, seat_pos)

    def _lie_on_sofa(self, sofa: FurnitureInstance, seat_pos: np.ndarray):
        self._clear_climb_surface()
        self.seated = True
        self.seated_return_pos = np.array(self.player_pos, dtype=float)
        self.seated_return_yaw = self.player_yaw
        eye_height = self.player_eye_height * 0.38
        self.player_pos = np.array(
            [seat_pos[0], seat_pos[1], seat_pos[2] + eye_height],
            dtype=float,
        )
        self._sync_first_person_camera()

    def _use_desk(self, desk: FurnitureInstance, seat_pos: np.ndarray):
        self._clear_climb_surface()
        self._sit_in_chair(desk, seat_pos)
        self._desk_in_use = True
        if self._desk_use_callback is not None:
            try:
                self._desk_use_callback(desk)
            except Exception:
                pass
        self.statusBar().showMessage("Using desk.", 1200)

    def _climb_furniture(self, instance: FurnitureInstance):
        if self.player_pos is None:
            return
        surface_z = self._furniture_surface_z(instance)
        if surface_z is None:
            return
        self.seated = False
        self.seated_return_pos = None
        self._clear_climb_surface()
        self._climb_surface_instance = instance
        self._climb_surface_radius = max(instance.collision_radius * 1.05, 0.6)
        center_xy = np.array(instance.position[:2], dtype=float)
        target_xy = np.array(self.player_pos[:2], dtype=float)
        offset = target_xy - center_xy
        dist = float(np.linalg.norm(offset))
        if dist > self._climb_surface_radius:
            if dist > 1e-6:
                target_xy = center_xy + offset / dist * (self._climb_surface_radius * 0.8)
            else:
                target_xy = center_xy
        self.player_vertical_velocity = 0.0
        self._jump_consumed = False
        self.player_pos = np.array(
            [target_xy[0], target_xy[1], surface_z + self.player_eye_height],
            dtype=float,
        )
        self._sync_first_person_camera()
        self.statusBar().showMessage("Climbed up.", 1200)

    def _lie_in_bed(self, bed: FurnitureInstance, bed_pos: np.ndarray, tuck: bool):
        self._clear_climb_surface()
        self.seated = True
        self.seated_return_pos = np.array(self.player_pos, dtype=float)
        self.seated_return_yaw = self.player_yaw
        factor = 0.32 if tuck else 0.4
        eye_height = self.player_eye_height * factor
        self.player_pos = np.array(
            [bed_pos[0], bed_pos[1], bed_pos[2] + eye_height],
            dtype=float,
        )
        self._sync_first_person_camera()

    def _make_bed(self, bed: FurnitureInstance):
        if bed.key != "bed":
            return
        if getattr(bed, "_made", False):
            for part in bed.parts:
                part.size = part.base_size
                part.offset = part.base_offset
            bed._made = False
            self._update_furniture_instance_transform(bed)
            self.statusBar().showMessage("Bed unmade.", 1200)
            return

        for idx, part in enumerate(bed.parts):
            if idx >= len(bed.parts) - 2:
                part.size = (
                    part.base_size[0],
                    part.base_size[1],
                    part.base_size[2] * 0.45,
                )
                part.offset = (
                    part.base_offset[0],
                    part.base_offset[1],
                    part.base_offset[2] - (part.base_size[2] * 0.25),
                )
        bed._made = True
        self._update_furniture_instance_transform(bed)
        self.statusBar().showMessage("Bed made.", 1200)

    def _nearest_chair(self):
        if self.player_pos is None:
            return None
        player_xy = np.array(self.player_pos[:2], dtype=float)
        best = None
        best_dist = None
        for instance in self.furniture_instances:
            if instance.key == "bed" or self._is_sofa_instance(instance):
                continue
            if self._is_desk_instance(instance):
                continue
            if not instance.seat_offset or instance.interact_radius <= 0.0:
                continue
            seat_pos = self._furniture_seat_position(instance)
            seat_xy = np.array(seat_pos[:2], dtype=float)
            dist = float(np.linalg.norm(player_xy - seat_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (instance, seat_pos)
        if best is None or best_dist is None:
            return None
        instance, seat_pos = best
        if best_dist > instance.interact_radius:
            return None
        return instance, seat_pos, best_dist

    def _furniture_seat_position(self, instance: FurnitureInstance) -> np.ndarray:
        seat_offset = instance.seat_offset or (0.0, 0.0, 0.0)
        angle = np.radians(instance.rotation)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rot = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        offset = rot @ np.array(seat_offset, dtype=float)
        return instance.position + offset

    def _sit_in_chair(self, chair: FurnitureInstance, seat_pos: np.ndarray):
        self._clear_climb_surface()
        self.seated = True
        self.seated_return_pos = np.array(self.player_pos, dtype=float)
        self.seated_return_yaw = self.player_yaw
        self._desk_in_use = False
        eye_height = self.player_eye_height * self.seated_eye_height_factor
        self.player_pos = np.array(
            [seat_pos[0], seat_pos[1], seat_pos[2] + eye_height],
            dtype=float,
        )
        self._sync_first_person_camera()

    def _stand_from_seat(self):
        self._clear_climb_surface()
        if self.seated_return_pos is not None:
            self.player_pos = np.array(self.seated_return_pos, dtype=float)
            self.player_yaw = self.seated_return_yaw
        self.seated = False
        self.seated_return_pos = None
        self._desk_in_use = False
        self._sync_first_person_camera()

    def _update_door_animations(self, dt: float):
        if not self.doors:
            return
        speed = self.door_open_speed
        for door in self.doors:
            if abs(door.current_angle - door.target_angle) < 0.1:
                door.current_angle = door.target_angle
                continue
            if door.current_angle < door.target_angle:
                door.current_angle = min(door.current_angle + speed * dt, door.target_angle)
            else:
                door.current_angle = max(door.current_angle - speed * dt, door.target_angle)
            self._update_door_transform(door)

    def _update_door_transform(self, door: DoorInstance):
        sx, sy, sz = door.width, door.thickness, door.height
        m = self._translation_matrix(door.hinge[0], door.hinge[1], door.hinge[2])
        m = m @ self._rotation_z_matrix(door.base_angle)
        m = m @ self._rotation_z_matrix(door.current_angle)
        m = m @ self._translation_matrix(door.width / 2.0, 0.0, door.height / 2.0)
        m = m @ self._scale_matrix(sx, sy, sz)
        door.item.setTransform(pg.Transform3D(m))

    @staticmethod
    def _translation_matrix(x: float, y: float, z: float) -> np.ndarray:
        m = np.eye(4, dtype=float)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    @staticmethod
    def _scale_matrix(x: float, y: float, z: float) -> np.ndarray:
        m = np.eye(4, dtype=float)
        m[0, 0] = x
        m[1, 1] = y
        m[2, 2] = z
        return m

    @staticmethod
    def _rotation_z_matrix(angle_deg: float) -> np.ndarray:
        angle = np.radians(angle_deg)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        m = np.eye(4, dtype=float)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return m

    def _handle_camera_keys(self, event) -> bool:
        if self.first_person_enabled:
            return False
        key = event.key()
        az_step = 5.0
        dist_step = 1.0
        elev_step = 2.0

        if key in (QtCore.Qt.Key_PageUp, QtCore.Qt.Key_Up):
            self._update_camera_position(delta_elevation=elev_step)
            return True
        if key in (QtCore.Qt.Key_PageDown, QtCore.Qt.Key_Down):
            self._update_camera_position(delta_elevation=-elev_step)
            return True
        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_A):
            self._update_camera_position(delta_azimuth=-az_step)
            return True
        if key in (QtCore.Qt.Key_E, QtCore.Qt.Key_D):
            self._update_camera_position(delta_azimuth=az_step)
            return True
        if key == QtCore.Qt.Key_W:
            self._update_camera_position(delta_distance=-dist_step)
            return True
        if key == QtCore.Qt.Key_S:
            self._update_camera_position(delta_distance=dist_step)
            return True
        if key == QtCore.Qt.Key_R:
            self.reset_camera()
            return True
        return False

    def _handle_mouse_look(self, event) -> bool:
        if not getattr(self, "first_person_enabled", False):
            return False

        if event.type() in (
            QtCore.QEvent.MouseButtonPress,
            QtCore.QEvent.MouseButtonRelease,
        ):
            if self.mouse_capture_supported:
                self._last_mouse_pos = event.pos()
                return True
            if event.button() == self.mouse_look_button:
                self._mouse_look_active = event.type() == QtCore.QEvent.MouseButtonPress
                self._last_mouse_pos = event.pos()
                self._set_mouse_cursor_hidden(self._mouse_look_active)
                return True
            return True

        if event.type() == QtCore.QEvent.MouseMove:
            if self.mouse_look_requires_button and not self._mouse_look_active:
                return True
            pos = event.pos()
            if self._recentering:
                self._recentering = False
                self._last_mouse_pos = pos
                return True
            if self._last_mouse_pos is None:
                self._last_mouse_pos = pos
                return True

            delta = pos - self._last_mouse_pos
            self._last_mouse_pos = pos

            self.player_yaw = (self.player_yaw - delta.x() * self.mouse_sensitivity) % 360.0
            self.player_pitch = float(
                np.clip(
                    self.player_pitch - delta.y() * self.mouse_sensitivity,
                    -85.0,
                    85.0,
                )
            )
            self._sync_first_person_camera()
            if self._mouse_captured:
                self._center_mouse_cursor()
            return True

        if event.type() == QtCore.QEvent.Wheel:
            return True

        return False

    def keyPressEvent(self, event):
        if self._handle_first_person_key_event(event, pressed=True):
            return
        if self._handle_camera_keys(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._handle_first_person_key_event(event, pressed=False):
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.view:
            if self.furnish_mode_enabled and not self.first_person_enabled:
                self._handle_furnish_event(event)
            if event.type() == QtCore.QEvent.KeyPress:
                if self._handle_first_person_key_event(event, pressed=True):
                    return True
                if self._handle_camera_keys(event):
                    return True
            if event.type() == QtCore.QEvent.KeyRelease:
                if self._handle_first_person_key_event(event, pressed=False):
                    return True
            if event.type() == QtCore.QEvent.FocusOut:
                if self.first_person_enabled and self.mouse_capture_supported:
                    self._set_mouse_capture(False)
            if event.type() == QtCore.QEvent.FocusIn:
                if self.first_person_enabled and self.mouse_capture_supported:
                    self._set_mouse_capture(True)
            if event.type() == QtCore.QEvent.Resize:
                self._update_crosshair_position()
            if self._handle_mouse_look(event):
                return True
        return super().eventFilter(obj, event)

    def _update_camera_position(
        self,
        delta_azimuth=0.0,
        delta_distance=0.0,
        delta_elevation=0.0,
        recenter=False,
    ):
        """
        Apply camera changes relative to current view or recenter to defaults.
        Uses GLViewWidget's native center/distance/elevation/azimuth to avoid drift.
        """
        # Pull current state (captures mouse interactions)
        current_center = self.view.opts.get("center", self.cam_center)
        current_distance = self.view.opts.get("distance", self.cam_distance)
        current_azimuth = self.view.opts.get("azimuth", self.cam_azimuth)
        current_elevation = self.view.opts.get("elevation", self.cam_elevation)

        if recenter:
            center = self.scene_center
            distance = self.scene_distance
            az = self.default_azimuth
            elev = self.default_elevation
        else:
            center = current_center
            distance = current_distance
            az = current_azimuth
            elev = current_elevation

        distance = max(2.0, distance + delta_distance)
        az += delta_azimuth
        elev = np.clip(elev + delta_elevation, -89.0, 89.0)

        self.cam_center = center
        self.cam_distance = distance
        self.cam_azimuth = az
        self.cam_elevation = elev

        # Update opts first so setCameraPosition uses the current center
        self.view.opts["center"] = center
        self.view.opts["distance"] = distance
        self.view.opts["azimuth"] = az
        self.view.opts["elevation"] = elev

        self.view.setCameraPosition(
            distance=distance,
            elevation=elev,
            azimuth=az,
        )
        self.view.update()

    def _make_box(
        self,
        center,
        size,
        color,
        draw_edges: bool = False,
        normal=None,
        room: Optional[str] = None,
        lit: bool = True,
    ) -> gl.GLMeshItem:
        w, h, d = self._to_gl_size(size)
        cx, cy, cz = self._to_gl_pos(center)
        n = self._to_gl_normal(normal) if normal is not None else None

        md = get_unit_cube_meshdata()
        mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=False,
            color=color,
            shader="shaded",
            drawEdges=draw_edges,
        )
        mesh.scale(w, h, d)
        if n is not None:
            target_normal = np.array(n, dtype=float)
            norm = np.linalg.norm(target_normal)
            if norm > 1e-6:
                target_normal /= norm
                up = np.array([0.0, 0.0, 1.0])
                dot = float(np.clip(np.dot(up, target_normal), -1.0, 1.0))
                angle = np.degrees(np.arccos(dot))
                if angle > 1e-4:
                    axis = np.cross(up, target_normal)
                    axis_len = np.linalg.norm(axis)
                    if axis_len < 1e-6:
                        axis = np.array([1.0, 0.0, 0.0])
                    else:
                        axis /= axis_len
                    mesh.rotate(angle, *axis)
        mesh.translate(cx, cy, cz)
        if lit:
            self._register_lit_item(mesh, color, room=room)
        return mesh

    def _register_lit_item(
        self,
        mesh: gl.GLMeshItem,
        color,
        room: Optional[str] = None,
        emissive: float = 0.0,
    ):
        base = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3]) if len(color) > 3 else 1.0,
        )
        mesh._base_color = base
        mesh._room_key = room
        mesh._emissive = max(0.0, min(1.0, float(emissive)))
        self._lit_items.append(mesh)
        self._apply_lighting_to_item(mesh)

    def _apply_lighting_to_item(self, mesh: gl.GLMeshItem):
        base = getattr(mesh, "_base_color", None)
        if base is None:
            return
        room = getattr(mesh, "_room_key", None)
        ambient = self._current_ambient_intensity()
        ambient_color = self._current_ambient_color()
        room_state = self._get_room_light_state(room or "global")
        room_intensity = room_state["intensity"] if room_state["enabled"] else 0.0
        room_color = room_state["color"]
        room_boost = 0.85 * room_intensity
        emissive = getattr(mesh, "_emissive", 0.0)

        r = base[0] * (ambient * ambient_color[0]) + base[0] * room_boost * room_color[0]
        g = base[1] * (ambient * ambient_color[1]) + base[1] * room_boost * room_color[1]
        b = base[2] * (ambient * ambient_color[2]) + base[2] * room_boost * room_color[2]
        if emissive > 0.0:
            r += base[0] * emissive
            g += base[1] * emissive
            b += base[2] * emissive
        max_value = self.max_light_value
        r = max(0.0, min(max_value, r))
        g = max(0.0, min(max_value, g))
        b = max(0.0, min(max_value, b))
        if hasattr(mesh, "setColor"):
            mesh.setColor((r, g, b, base[3]))

    def _update_lighting(self, force: bool = False):
        now = time.time()
        if not force and now - self._lighting_last_update < 0.5:
            return
        self._lighting_last_update = now
        for mesh in list(self._lit_items):
            self._apply_lighting_to_item(mesh)

    def _current_ambient_intensity(self) -> float:
        sun_dir = self._sun_direction()
        sun_intensity = max(0.0, min(1.0, (sun_dir[2] + 0.15) / 1.15))
        moon_dir = -sun_dir
        moon_intensity = max(0.0, min(1.0, (moon_dir[2] + 0.05) / 1.05))
        ambient = 0.2 + (0.7 * sun_intensity) + (0.15 * moon_intensity)
        return max(0.05, min(1.0, ambient))

    def _current_ambient_color(self) -> Tuple[float, float, float]:
        sun_dir = self._sun_direction()
        sun_intensity = max(0.0, min(1.0, (sun_dir[2] + 0.2) / 1.2))
        moon_dir = -sun_dir
        moon_intensity = max(0.0, min(1.0, (moon_dir[2] + 0.05) / 1.05))
        day = np.array([1.0, 0.96, 0.9], dtype=float)
        night = np.array([0.35, 0.42, 0.55], dtype=float)
        moon = np.array([0.6, 0.68, 0.8], dtype=float)
        color = night + (day - night) * sun_intensity
        color = color + (moon - color) * (moon_intensity * 0.25)
        return (
            float(np.clip(color[0], 0.0, 1.0)),
            float(np.clip(color[1], 0.0, 1.0)),
            float(np.clip(color[2], 0.0, 1.0)),
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = HouseViewer()
    viewer.resize(1280, 720)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
