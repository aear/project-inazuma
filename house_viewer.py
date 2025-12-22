# house_viewer.py

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from geometry_utils import get_unit_cube_meshdata
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
    current_angle: float = 0.0
    target_angle: float = 0.0


@dataclass
class FurniturePartDef:
    size: Vec3
    offset: Vec3
    color: Tuple[float, float, float, float]
    gl_options: Optional[str] = None


@dataclass
class FurniturePrototype:
    key: str
    label: str
    parts: List[FurniturePartDef]


@dataclass
class FurniturePart:
    mesh: gl.GLMeshItem
    size: Vec3
    offset: Vec3


@dataclass
class FurnitureInstance:
    instance_id: int
    key: str
    label: str
    position: np.ndarray
    rotation: float
    parts: List[FurniturePart]


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
            spawn_x = state.spawn_point[0]
            spawn_y = state.spawn_point[2]
            radius = max(0.15, self.viewer.architect_settings.grid_size * 0.4)
            pen = QtGui.QPen(QtGui.QColor(255, 180, 60, 220))
            pen.setWidthF(0.05)
            circle = QtWidgets.QGraphicsEllipseItem(
                spawn_x - radius,
                spawn_y - radius,
                radius * 2.0,
                radius * 2.0,
            )
            circle.setPen(pen)
            circle.setBrush(QtGui.QBrush(QtGui.QColor(255, 180, 60, 60)))
            self._scene.addItem(circle)
            self._scene.addLine(
                QtCore.QLineF(spawn_x - radius * 1.5, spawn_y, spawn_x + radius * 1.5, spawn_y),
                pen,
            )
            self._scene.addLine(
                QtCore.QLineF(spawn_x, spawn_y - radius * 1.5, spawn_x, spawn_y + radius * 1.5),
                pen,
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


class HouseViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ina House Viewer (Prototype)")

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((10, 10, 20))
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.architect_state = ArchitectState()
        self.architect_settings = ArchitectSettings()
        self.architect_dirty = False
        self.architect_loaded = False
        self.architect_plan_cache = None
        self.architect_room_counter = 1
        self.doors: List[DoorInstance] = []
        self.door_items: List[gl.GLGraphicsItem] = []
        self.door_open_speed = 120.0
        self.door_interact_distance = 2.0
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
        self.player_pos = None  # GL coords: x, y, z
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self.player_speed = 4.0
        self.player_sprint_multiplier = 2.0
        self.player_turn_speed = 90.0
        self.player_look_distance = 0.6
        self.mouse_sensitivity = 0.15
        self.ground_snap_enabled = True
        self.player_items: List[gl.GLGraphicsItem] = []
        self.player_item = None
        self.player_avatar_enabled = False
        self.player_avatar_parts: List[FurniturePart] = []
        self.player_schema_path = "player_schema.json"
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
        self.exterior_model = None
        self._player_control_keys = {
            QtCore.Qt.Key_W,
            QtCore.Qt.Key_A,
            QtCore.Qt.Key_S,
            QtCore.Qt.Key_D,
            QtCore.Qt.Key_Space,
            QtCore.Qt.Key_C,
            QtCore.Qt.Key_Shift,
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
        }

        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)

        # Reference grid
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.view.addItem(grid)

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

        avatar_act = QtWidgets.QAction("Ina Avatar", self)
        avatar_act.setCheckable(True)
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
        items.clear()

    def reload_scene(self):
        self.clear_items(self.exterior_items)
        self.clear_items(self.interior_items)
        self.clear_items(self.player_items)
        self.clear_items(self.door_items)
        self.clear_items(self.furniture_items)
        self.clear_items(getattr(self, "light_items", []))
        self._clear_furnish_preview()
        self.doors.clear()
        self.player_item = None
        self.player_avatar_parts = []
        self.player_pos = None
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
        self._build_exterior(exterior)

        # Interior boxes (optional debug)
        if self.show_interior:
            self._build_interior(house)

        self._load_furniture_from_plan()
        self._load_lighting_from_plan()

        self._load_player_schema()
        self._ensure_player()
        self._update_player_visibility()
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
            ("spawn", "Spawn"),
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
        spawn_btn = QtWidgets.QPushButton("Spawn At Bed")
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
            ),
            FurniturePartDef(
                size=(pillow_width, pillow_depth, pillow_height),
                offset=(
                    -bed_width * 0.25,
                    bed_depth / 2.0 - headboard_thickness - pillow_depth / 2.0 - 0.05,
                    frame_height + mattress_height + pillow_height / 2.0,
                ),
                color=(0.95, 0.95, 0.96, 1.0),
            ),
        ]
        catalog["bed"] = FurniturePrototype(key="bed", label="Bed", parts=bed_parts)

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
                    )
                )
        catalog["table"] = FurniturePrototype(key="table", label="Table", parts=table_parts)

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
                    )
                )
        catalog["chair"] = FurniturePrototype(key="chair", label="Chair", parts=chair_parts)

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
            ),
            FurniturePartDef(
                size=(lamp_stem_width, lamp_stem_width, lamp_stem_height),
                offset=(0.0, 0.0, lamp_base_height + lamp_stem_height / 2.0),
                color=(0.35, 0.35, 0.38, 1.0),
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
            ),
        ]
        catalog["lamp"] = FurniturePrototype(key="lamp", label="Floor Lamp", parts=lamp_parts)

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
            message = "Spawn: click to set the player spawn point."
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
        self.furniture_preview.position = np.array(gl_pos, dtype=float)
        self._update_furniture_instance_transform(self.furniture_preview)

    def _update_furnish_preview_rotation(self):
        if self.furniture_preview is None:
            return
        self.furniture_preview.rotation = self.furniture_rotation
        self._update_furniture_instance_transform(self.furniture_preview)

    def _create_furniture_instance(
        self,
        proto: FurniturePrototype,
        position: np.ndarray,
        rotation: float,
        preview: bool,
        instance_id: int,
    ) -> FurnitureInstance:
        md = get_unit_cube_meshdata()
        parts: List[FurniturePart] = []
        for part_def in proto.parts:
            color = part_def.color
            if preview:
                alpha = min(0.35, color[3] * 0.35)
                color = (color[0], color[1], color[2], alpha)
            gl_options = part_def.gl_options
            if gl_options is None and color[3] < 0.99:
                gl_options = "translucent"
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
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
            parts.append(FurniturePart(mesh=mesh, size=part_def.size, offset=part_def.offset))

        instance = FurnitureInstance(
            instance_id=instance_id,
            key=proto.key,
            label=proto.label,
            position=np.array(position, dtype=float),
            rotation=rotation,
            parts=parts,
        )
        self._update_furniture_instance_transform(instance)
        return instance

    def _update_furniture_instance_transform(self, instance: FurnitureInstance):
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
        for part in instance.parts:
            offset = np.array(part.offset, dtype=float)
            rotated_offset = rot @ offset
            center = instance.position + rotated_offset
            part.mesh.resetTransform()
            part.mesh.scale(part.size[0], part.size[1], part.size[2])
            if abs(instance.rotation) > 1e-3:
                part.mesh.rotate(instance.rotation, 0, 0, 1)
            part.mesh.translate(center[0], center[1], center[2])

    def _place_furniture_at(self, gl_pos: np.ndarray):
        proto = self.furniture_catalog.get(self.furniture_active_key)
        if proto is None:
            return
        instance_id = self._furniture_instance_counter
        self._furniture_instance_counter += 1
        instance = self._create_furniture_instance(
            proto,
            gl_pos,
            self.furniture_rotation,
            preview=False,
            instance_id=instance_id,
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
            center = (float(pos[0]), ceiling_y, float(pos[1]))
            mesh = self._make_box(center=center, size=ceiling_size, color=color, draw_edges=True)
            mesh.setGLOptions("translucent")
            self.view.addItem(mesh)
            self.light_items.append(mesh)

        switch_size = (0.12, 0.18, 0.05)
        for switch in lighting.get("switches", []):
            pos = switch.get("position", [0.0, 0.0])
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            height = float(switch.get("height", self.architect_settings.switch_height))
            color = self._normalize_color(
                switch.get("color", self.architect_settings.switch_color),
                self.architect_settings.switch_color,
            )
            center = (float(pos[0]), height, float(pos[1]))
            mesh = self._make_box(
                center=center,
                size=switch_size,
                color=color,
                draw_edges=True,
            )
            self.view.addItem(mesh)
            self.light_items.append(mesh)

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
        storey["spawn"] = {"position": [float(v) for v in spawn]}
        if self._save_plan_data(data, plan_path, status_callback=self._set_furnish_status):
            self._set_furnish_status("Spawn set to bed.")

        if self.exterior_model is not None:
            self.exterior_model.spawn_point = spawn
        self.architect_state.spawn_point = spawn
        self.architect_settings.spawn_height = spawn[1]
        self.player_pos = None
        self._ensure_player()
        if self.first_person_enabled:
            self._sync_first_person_camera()

    def _remove_furniture_instance(self, instance: FurnitureInstance):
        for part in instance.parts:
            self.view.removeItem(part.mesh)
            if part.mesh in self.furniture_items:
                self.furniture_items.remove(part.mesh)
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
        if opening_type == "door":
            opening = Opening(
                type="door",
                width=self.architect_settings.door_width,
                height=self.architect_settings.door_height,
                sill_height=self.architect_settings.door_sill,
                offset_along_wall=best_offset,
            )
        else:
            opening = Opening(
                type="window",
                width=self.architect_settings.window_width,
                height=self.architect_settings.window_height,
                sill_height=self.architect_settings.window_sill,
                offset_along_wall=best_offset,
            )
        best_wall.openings.append(opening)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def _room_for_point(self, pos: Vec2) -> Optional[str]:
        x, z = float(pos[0]), float(pos[1])
        for room in self.architect_state.rooms:
            center_x, _, center_z = room.center
            width, _, depth = room.size
            if abs(x - center_x) <= width / 2.0 and abs(z - center_z) <= depth / 2.0:
                return room.name
        return None

    def architect_set_spawn(self, pos: Vec2):
        spawn = (float(pos[0]), float(self.architect_settings.spawn_height), float(pos[1]))
        self.architect_state.spawn_point = spawn
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_ceiling_light(self, pos: Vec2):
        light = ArchitectCeilingLight(
            position=(float(pos[0]), float(pos[1])),
            color=self.architect_settings.ceiling_light_color,
            room=self._room_for_point(pos),
        )
        self.architect_state.ceiling_lights.append(light)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_add_light_switch(self, pos: Vec2):
        switch = ArchitectLightSwitch(
            position=(float(pos[0]), float(pos[1])),
            height=float(self.architect_settings.switch_height),
            room=self._room_for_point(pos),
        )
        self.architect_state.light_switches.append(switch)
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

    def architect_clear(self):
        self.architect_state = ArchitectState()
        self.architect_room_counter = 1
        self._set_architect_dirty()
        self.architect_canvas.refresh_scene()

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
                self._set_architect_dirty()
                self.architect_canvas.refresh_scene()
                self._set_architect_status("Spawn point deleted.")
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
            spawn_point=exterior.spawn_point,
        )
        self.architect_room_counter = len(self.architect_state.rooms) + 1
        self.architect_dirty = False
        self.architect_loaded = True
        if exterior.spawn_point is not None:
            self.architect_settings.spawn_height = float(exterior.spawn_point[1])
        self._sync_architect_controls()
        self.architect_canvas.refresh_scene()
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
                openings.append(
                    {
                        "type": op.type,
                        "id": f"{op.type}_{idx}_{op_idx}",
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

        if self.architect_state.spawn_point is not None:
            storey["spawn"] = {"position": [float(v) for v in self.architect_state.spawn_point]}
        else:
            storey.pop("spawn", None)

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
            }
            if switch.room:
                payload["room"] = switch.room
            switches.append(payload)
        lighting["switches"] = switches

    def _update_plan_with_furniture(self, storey):
        furniture_entries = []
        for instance in self.furniture_instances:
            model_pos = self._to_model_pos(instance.position)
            furniture_entries.append(
                {
                    "id": f"furniture_{instance.instance_id}",
                    "key": instance.key,
                    "position": [float(model_pos[0]), float(model_pos[1]), float(model_pos[2])],
                    "rotation": float(instance.rotation),
                }
            )
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

        points = self._expand_polygon_radial(outline, roof_overhang)
        triangles = self._triangulate_polygon(points)
        if not triangles:
            return None

        top_y = roof_height
        bottom_y = roof_height - roof_thickness

        verts = []
        for x, z in points:
            verts.append(self._to_gl_pos((x, top_y, z)))
        for x, z in points:
            verts.append(self._to_gl_pos((x, bottom_y, z)))

        n = len(points)
        faces = []
        for a, b, c in triangles:
            faces.append([a, b, c])
            faces.append([a + n, c + n, b + n])

        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j + n])
            faces.append([i, j + n, i + n])

        mesh_data = gl.MeshData(vertexes=np.array(verts, dtype=float), faces=np.array(faces, dtype=int))
        mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            color=color,
            shader="shaded",
            drawEdges=False,
        )
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

        door = DoorInstance(
            item=mesh,
            hinge=hinge_gl,
            width=opening.width,
            thickness=door_thickness,
            height=opening.height,
            base_angle=angle,
            open_angle=open_angle,
        )
        self._update_door_transform(door)
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
        )

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
        md = get_unit_cube_meshdata()
        for part_def in self._player_avatar_part_defs():
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                color=part_def.color,
                shader="shaded",
                drawEdges=True,
            )
            if part_def.gl_options is not None:
                mesh.setGLOptions(part_def.gl_options)
            self.view.addItem(mesh)
            self.player_items.append(mesh)
            self.player_avatar_parts.append(
                FurniturePart(mesh=mesh, size=part_def.size, offset=part_def.offset)
            )

    def _player_avatar_part_defs(self) -> List[FurniturePartDef]:
        width = max(self.player_width, 0.1)
        depth = max(self.player_depth, 0.1)
        height = max(self.player_height, 0.1)

        leg_height = height * 0.45
        torso_height = height * 0.35
        head_height = height * 0.2
        total = leg_height + torso_height + head_height
        if total > height and total > 1e-6:
            scale = height / total
            leg_height *= scale
            torso_height *= scale
            head_height *= scale

        def offset_from_bottom(bottom: float, part_height: float) -> float:
            return bottom + part_height / 2.0 - height / 2.0

        parts: List[FurniturePartDef] = []
        bottom = 0.0
        parts.append(
            FurniturePartDef(
                size=(width * 0.5, depth * 0.5, leg_height),
                offset=(0.0, 0.0, offset_from_bottom(bottom, leg_height)),
                color=(0.18, 0.18, 0.22, 1.0),
            )
        )
        bottom += leg_height
        parts.append(
            FurniturePartDef(
                size=(width * 0.7, depth * 0.6, torso_height),
                offset=(0.0, 0.0, offset_from_bottom(bottom, torso_height)),
                color=(0.35, 0.32, 0.4, 1.0),
            )
        )
        bottom += torso_height
        parts.append(
            FurniturePartDef(
                size=(width * 0.5, depth * 0.55, head_height),
                offset=(0.0, 0.0, offset_from_bottom(bottom, head_height)),
                color=(0.95, 0.84, 0.78, 1.0),
            )
        )
        hair_height = head_height * 0.6
        hair_bottom = bottom + head_height * 0.4
        parts.append(
            FurniturePartDef(
                size=(width * 0.58, depth * 0.6, hair_height),
                offset=(0.0, 0.0, offset_from_bottom(hair_bottom, hair_height)),
                color=(0.12, 0.08, 0.16, 1.0),
            )
        )
        return parts

    def _reset_player(self):
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

    def _update_player_avatar_mesh(self):
        if not self.player_avatar_parts:
            self._create_player_avatar_parts()
        part_defs = self._player_avatar_part_defs()
        if len(part_defs) != len(self.player_avatar_parts):
            self._create_player_avatar_parts()
            part_defs = self._player_avatar_part_defs()

        center = np.array(self.player_pos, dtype=float)
        body_offset = self.player_eye_height - (self.player_height / 2.0)
        center[2] -= body_offset

        for part, part_def in zip(self.player_avatar_parts, part_defs):
            part.size = part_def.size
            part.offset = part_def.offset
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
        if self.exterior_model is None:
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
        self._update_door_animations(min(dt, 0.05))
        if not self.first_person_enabled:
            return
        self._apply_player_input(min(dt, 0.05))

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

        if norm > 1e-6:
            self.player_pos += move * speed * dt

        if self.ground_snap_enabled:
            ground_eye_z = self._ground_z() + self.player_eye_height
            self.player_pos[2] = ground_eye_z
        elif self.player_pos[2] < self.player_eye_height:
            self.player_pos[2] = self.player_eye_height

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

    # ---- Generic mesh helper ----

    def _handle_first_person_key_event(self, event, pressed: bool) -> bool:
        key = event.key()
        if pressed and not event.isAutoRepeat():
            if key == QtCore.Qt.Key_F:
                self.first_person_action.setChecked(not self.first_person_enabled)
                return True
            if key == QtCore.Qt.Key_E and self.first_person_enabled:
                self._toggle_nearest_door()
                return True
            if key == QtCore.Qt.Key_Escape and self.first_person_enabled:
                self.first_person_action.setChecked(False)
                return True
            if key == QtCore.Qt.Key_R and self.first_person_enabled:
                self.reset_camera()
                return True

        if not self.first_person_enabled:
            return False

        if key in self._player_control_keys:
            if event.isAutoRepeat():
                return True
            if pressed:
                self.keys_down.add(key)
            else:
                self.keys_down.discard(key)
            return True

        return False

    def _toggle_nearest_door(self):
        if self.player_pos is None or not self.doors:
            return
        player_xy = np.array([self.player_pos[0], self.player_pos[1]], dtype=float)
        best = None
        best_dist = None
        for door in self.doors:
            hinge_xy = np.array([door.hinge[0], door.hinge[1]], dtype=float)
            dist = float(np.linalg.norm(player_xy - hinge_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = door
        if best is None or best_dist is None:
            return
        if best_dist > self.door_interact_distance:
            self.statusBar().showMessage("No door nearby.", 1500)
            return
        best.target_angle = best.open_angle if abs(best.target_angle) < 1e-3 else 0.0

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
        return mesh


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = HouseViewer()
    viewer.resize(1280, 720)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
