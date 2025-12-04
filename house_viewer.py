# house_viewer.py

import sys
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from geometry_utils import get_unit_cube_meshdata
from house_model import (
    create_prototype_house,
    create_prototype_exterior,
    load_house_from_plan,
    Room,
    House,
    ExteriorModel,
    WallSegment,
)


class HouseViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ina House Viewer (Prototype)")

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((10, 10, 20))
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view.installEventFilter(self)
        self.setCentralWidget(self.view)

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

        # Store GL items so we can remove them on reload
        self.exterior_items: List[gl.GLGraphicsItem] = []
        self.interior_items: List[gl.GLGraphicsItem] = []

        # For now: allow toggling interior debug rendering if you want it
        self.show_interior = True

        self.reload_scene()

    # ---- Coordinate helpers ----

    @staticmethod
    def _to_gl_pos(pos):
        """Map model coords (x, y-up, z-depth) to GL coords (x, y-depth, z-up)."""
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
        self._update_camera_position(recenter=True)

    # ---- Scene management ----

    def clear_items(self, items: List[gl.GLGraphicsItem]):
        for item in items:
            self.view.removeItem(item)
        items.clear()

    def reload_scene(self):
        self.clear_items(self.exterior_items)
        self.clear_items(self.interior_items)

        # Exterior
        try:
            house, exterior = load_house_from_plan()
        except FileNotFoundError:
            house = create_prototype_house()
            exterior = create_prototype_exterior()
        except Exception:
            house = create_prototype_house()
            exterior = create_prototype_exterior()
        self._build_exterior(exterior)

        # Interior boxes (optional debug)
        if self.show_interior:
            self._build_interior(house)

    # ---- Exterior rendering ----

    def _build_exterior(self, exterior: ExteriorModel):
        """
        Build garden, walls, and a simple flat roof.
        """
        xs = [p[0] for p in exterior.footprint.outline]
        zs = [p[1] for p in exterior.footprint.outline]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

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
                fence_mesh = self._make_fence_box(fence)
                self.view.addItem(fence_mesh)
                self.exterior_items.append(fence_mesh)
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

        # Walls
        for wall in exterior.walls:
            wall_mesh = self._make_wall_box(wall)
            self.view.addItem(wall_mesh)
            self.exterior_items.append(wall_mesh)

            # v1: windows/door are just different-colour quads
            if wall.has_window:
                window_mesh = self._make_wall_windows(wall)
                if window_mesh is not None:
                    for item in window_mesh:
                        self.view.addItem(item)
                        self.exterior_items.append(item)
            if wall.has_door:
                door_mesh = self._make_wall_door(wall)
                if door_mesh is not None:
                    if isinstance(door_mesh, list):
                        for item in door_mesh:
                            self.view.addItem(item)
                            self.exterior_items.append(item)
                    else:
                        self.view.addItem(door_mesh)
                        self.exterior_items.append(door_mesh)

        # Roof: single flat box slightly above wall height
        # Use footprint extents to size it.
        roof_margin = 0.2
        roof_center = (
            (min_x + max_x) / 2.0,
            exterior.roof_height,
            (min_z + max_z) / 2.0,
        )
        roof_size = (
            (max_x - min_x) + exterior.roof_overhang * 2.0,
            exterior.roof_thickness,
            (max_z - min_z) + exterior.roof_overhang * 2.0,
        )

        roof_mesh = self._make_box(
            center=roof_center,
            size=roof_size,
            color=exterior.roof_color,
            draw_edges=False,
        )
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
        mesh.translate(mx, my, height / 2.0)

        # Rotate cube so its long axis matches the segment direction in the x-y plane.
        angle = np.degrees(np.arctan2(dy, dx))
        mesh.rotate(-angle, 0, 0, 1)

        # Scale after orientation so height is along GL z.
        mesh.scale(length, thickness, height)

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
        mesh.translate(mx, my, height / 2.0)
        angle = np.degrees(np.arctan2(dy, dx))
        mesh.rotate(-angle, 0, 0, 1)
        mesh.scale(length, thickness, height)
        return mesh

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
                mesh.translate(wx_front, wy_front, wz)
                angle = np.degrees(np.arctan2(dy, dx))
                mesh.rotate(-angle, 0, 0, 1)
                mesh.scale(w_width, w_thickness, w_height)
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
            mesh.translate(wx_front, wy_front, wz)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.scale(op.width, wall.thickness * 0.5, op.height)
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
            mesh.translate(dx_front, dy_front, door_height / 2.0)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.scale(door_width, door_thickness, door_height)
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
            mesh.translate(px_front, py_front, pz)
            angle = np.degrees(np.arctan2(dy, dx))
            mesh.rotate(-angle, 0, 0, 1)
            mesh.scale(op.width, wall.thickness * 0.6, op.height)
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

    # ---- Generic mesh helper ----

    def _handle_camera_keys(self, event) -> bool:
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

    def keyPressEvent(self, event):
        if self._handle_camera_keys(event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.view and event.type() == QtCore.QEvent.KeyPress:
            if self._handle_camera_keys(event):
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
        mesh.translate(cx, cy, cz)
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
        mesh.scale(w, h, d)
        return mesh


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = HouseViewer()
    viewer.resize(1280, 720)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
