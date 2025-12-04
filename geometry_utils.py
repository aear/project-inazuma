# geometry_utils.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph.opengl as gl


# Cached unit cube mesh so we don't recreate it every time.
_CUBE_MESHDATA: Optional[gl.MeshData] = None


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
