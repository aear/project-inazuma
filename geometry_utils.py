# geometry_utils.py

from __future__ import annotations

from typing import Optional

import math
import numpy as np
import pyqtgraph.opengl as gl


# Cached mesh primitives so we don't recreate them every time.
_CUBE_MESHDATA: Optional[gl.MeshData] = None
_SPHERE_MESHDATA: Optional[gl.MeshData] = None
_CYLINDER_MESHDATA: Optional[gl.MeshData] = None


def get_unit_cube_meshdata() -> gl.MeshData:
    """
    Return a MeshData representing a unit cube in [0, 1]^3.

    We define the vertices and faces ourselves so we're not relying on
    any version-specific helpers from pyqtgraph. This gives us a stable,
    debuggable primitive we can reuse everywhere.
    """
    global _CUBE_MESHDATA
    if _CUBE_MESHDATA is not None:
        return _CUBE_MESHDATA

    # 8 vertices of a cube centered at origin in [-0.5, 0.5]^3
    verts = np.array(
        [
            [-0.5, -0.5, -0.5],  # 0
            [ 0.5, -0.5, -0.5],  # 1
            [ 0.5,  0.5, -0.5],  # 2
            [-0.5,  0.5, -0.5],  # 3
            [-0.5, -0.5,  0.5],  # 4
            [ 0.5, -0.5,  0.5],  # 5
            [ 0.5,  0.5,  0.5],  # 6
            [-0.5,  0.5,  0.5],  # 7
        ],
        dtype=float,
    )

    # 12 triangles (2 per face) – consistent winding order
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [1, 2, 6], [1, 6, 5],  # right
            [2, 3, 7], [2, 7, 6],  # back
            [3, 0, 4], [3, 4, 7],  # left
        ],
        dtype=int,
    )

    _CUBE_MESHDATA = gl.MeshData(vertexes=verts, faces=faces)
    return _CUBE_MESHDATA


def get_unit_sphere_meshdata(segments: int = 16, rings: int = 12) -> gl.MeshData:
    """
    Return a MeshData representing a unit sphere centered at origin with radius 0.5.

    The sphere is built as a UV sphere so we can keep the mesh lightweight
    while looking smoother than a cube for avatar parts.
    """
    global _SPHERE_MESHDATA
    if _SPHERE_MESHDATA is not None:
        return _SPHERE_MESHDATA

    segments = max(3, int(segments))
    rings = max(3, int(rings))
    radius = 0.5

    verts = []
    faces = []

    # Top pole.
    verts.append([0.0, 0.0, radius])

    # Rings between poles.
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            x = radius * sin_phi * math.cos(theta)
            y = radius * sin_phi * math.sin(theta)
            z = radius * cos_phi
            verts.append([x, y, z])

    # Bottom pole.
    verts.append([0.0, 0.0, -radius])

    top_index = 0
    bottom_index = len(verts) - 1
    first_ring = 1
    last_ring = 1 + (rings - 2) * segments

    # Top cap.
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = first_ring + seg
        b = first_ring + next_seg
        faces.append([top_index, a, b])

    # Middle quads.
    for ring in range(1, rings - 1):
        ring_start = 1 + (ring - 1) * segments
        next_ring_start = ring_start + segments
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            a = ring_start + seg
            b = next_ring_start + seg
            c = next_ring_start + next_seg
            d = ring_start + next_seg
            faces.append([a, b, c])
            faces.append([a, c, d])

    # Bottom cap (reverse winding).
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = last_ring + seg
        b = last_ring + next_seg
        faces.append([bottom_index, b, a])

    _SPHERE_MESHDATA = gl.MeshData(
        vertexes=np.array(verts, dtype=float),
        faces=np.array(faces, dtype=int),
    )
    return _SPHERE_MESHDATA


def get_unit_cylinder_meshdata(segments: int = 16) -> gl.MeshData:
    """
    Return a MeshData representing a unit cylinder centered at origin.

    Cylinder radius is 0.5 and height is 1.0 along the Z axis.
    """
    global _CYLINDER_MESHDATA
    if _CYLINDER_MESHDATA is not None:
        return _CYLINDER_MESHDATA

    segments = max(3, int(segments))
    radius = 0.5
    z_top = 0.5
    z_bottom = -0.5

    verts = []
    faces = []

    # Centers for caps.
    verts.append([0.0, 0.0, z_top])
    verts.append([0.0, 0.0, z_bottom])

    # Ring vertices.
    for seg in range(segments):
        theta = 2.0 * math.pi * seg / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append([x, y, z_top])
    for seg in range(segments):
        theta = 2.0 * math.pi * seg / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append([x, y, z_bottom])

    top_center = 0
    bottom_center = 1
    top_start = 2
    bottom_start = 2 + segments

    # Top cap.
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = top_start + seg
        b = top_start + next_seg
        faces.append([top_center, a, b])

    # Bottom cap (reverse winding).
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = bottom_start + seg
        b = bottom_start + next_seg
        faces.append([bottom_center, b, a])

    # Side faces.
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = top_start + seg
        b = top_start + next_seg
        c = bottom_start + next_seg
        d = bottom_start + seg
        faces.append([a, d, c])
        faces.append([a, c, b])

    _CYLINDER_MESHDATA = gl.MeshData(
        vertexes=np.array(verts, dtype=float),
        faces=np.array(faces, dtype=int),
    )
    return _CYLINDER_MESHDATA
