# house_model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

# Basic vector aliases
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Color = Tuple[float, float, float, float]  # RGBA 0–1


@dataclass
class Opening:
    type: str  # "window" or "door"
    width: float
    height: float
    sill_height: float
    offset_along_wall: float  # distance from wall start along segment


# ---------- INTERIOR MODEL ----------

@dataclass
class Room:
    name: str
    center: Vec3      # x, y, z
    size: Vec3        # width, height, depth
    color: Color      # RGBA 0–1


@dataclass
class House:
    rooms: List[Room]


def create_prototype_house() -> House:
    """
    Simple fixed layout:
    - Four indoor rooms arranged in a 2x2 grid
    - Garden as a ground plane around the house
    """
    rooms: List[Room] = []

    # Ground level y=0, walls up to y=2.4 (interior height)
    height = 2.4
    depth = 4.0
    width = 4.0
    gap = 0.5

    # Arrange rooms on a simple 2x2 grid so the house is not overly elongated.
    # Centers are spaced by (room_size + gap) in both X (width) and Z (depth).
    half_span = (width + gap) / 2.0  # 2.25 with current numbers
    x_positions = [-half_span, half_span]
    z_positions = [-half_span, half_span]
    y_center = height / 2.0

    # Front row (z negative): bedroom + bathroom
    rooms.append(Room(
        name="bedroom",
        center=(x_positions[0], y_center, z_positions[0]),
        size=(width, height, depth),
        color=(0.8, 0.7, 0.9, 0.4)
    ))
    rooms.append(Room(
        name="bathroom",
        center=(x_positions[1], y_center, z_positions[0]),
        size=(width, height, depth),
        color=(0.7, 0.8, 0.9, 0.4)
    ))

    # Back row (z positive): kitchen + living
    rooms.append(Room(
        name="kitchen",
        center=(x_positions[0], y_center, z_positions[1]),
        size=(width, height, depth),
        color=(0.9, 0.8, 0.7, 0.4)
    ))
    rooms.append(Room(
        name="living",
        center=(x_positions[1], y_center, z_positions[1]),
        size=(width, height, depth),
        color=(0.8, 0.8, 0.8, 0.4)
    ))

    # Garden as a ground plane (we render this separately in the exterior)
    # Keeping a "room" entry is still useful for debugging / interior-only view.
    rooms.append(Room(
        name="garden",
        center=(0.0, -0.025, 0.0),
        size=(12.0, 0.05, 12.0),
        color=(0.7, 0.9, 0.7, 1.0)
    ))

    return House(rooms=rooms)


# ---------- EXTERIOR MODEL ----------

@dataclass
class Footprint:
    """2D outline on the ground plane (x, z), walls extrude upward."""
    outline: List[Vec2]   # ordered loop of points
    wall_height: float    # height of walls (y)


@dataclass
class WallSegment:
    start: Vec2
    end: Vec2
    height: float
    thickness: float
    color: Color
    openings: List[Opening]
    has_window: bool = False
    has_door: bool = False
    is_front: bool = False


@dataclass
class ExteriorModel:
    footprint: Footprint
    walls: List[WallSegment]
    roof_height: float
    roof_color: Color
    roof_overhang: float
    roof_thickness: float
    garden_center: Vec3
    garden_size: Vec3
    garden_color: Color
    garden_normal: Vec3
    fences: List["FenceSegment"]
    spawn_point: Optional[Vec3] = None


@dataclass
class FenceSegment:
    start: Vec2
    end: Vec2
    height: float
    thickness: float
    color: Color


def _distance_2d(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def load_house_from_plan(path: str = "ina_house_plan.json") -> Tuple[House, ExteriorModel]:
    """
    Load house + exterior data from a JSON plan file.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    site = data.get("site", {})
    ground = site.get("ground", {})
    ground_size = ground.get("size", [20.0, 20.0])
    ground_color = tuple(ground.get("color", (0.7, 0.9, 0.7, 1.0)))

    building = data["building"]
    storey = building["storeys"][0]  # simple: first storey

    # Footprint
    points = storey["footprint"]["points"]
    footprint = Footprint(outline=[(p[0], p[1]) for p in points], wall_height=storey["height"])

    # Walls with openings
    walls: List[WallSegment] = []
    wall_defaults = storey["exterior_walls"]
    default_thickness = wall_defaults.get("default_thickness", 0.2)
    default_color = tuple(wall_defaults.get("default_color", (0.85, 0.85, 0.9, 1.0)))

    for seg in wall_defaults.get("segments", []):
        openings: List[Opening] = []
        for op in seg.get("openings", []):
            openings.append(Opening(
                type=op["type"],
                width=op["width"],
                height=op["height"],
                sill_height=op.get("sill_height", 0.0),
                offset_along_wall=op.get("offset_along_wall", 0.0),
            ))

        wall_color = tuple(seg.get("color", default_color))
        wall_height = seg.get("height", storey["height"])
        thickness = seg.get("thickness", default_thickness)
        start = tuple(seg["from"])
        end = tuple(seg["to"])

        walls.append(WallSegment(
            start=start,
            end=end,
            height=wall_height,
            thickness=thickness,
            color=wall_color,
            openings=openings,
            has_window=any(op.type == "window" for op in openings),
            has_door=any(op.type == "door" for op in openings),
            is_front=False,
        ))

    # Roof
    roof = storey.get("roof", {})
    roof_overhang = roof.get("overhang", 0.2)
    roof_thickness = roof.get("thickness", 0.1)
    roof_height = storey["height"] + roof_thickness
    roof_color = tuple(roof.get("color", (0.75, 0.75, 0.78, 1.0)))

    # Garden / ground plane
    garden_height = 0.05
    garden_center: Vec3 = (0.0, -garden_height / 2.0, 0.0)
    garden_size: Vec3 = (ground_size[0], garden_height, ground_size[1])
    garden_color: Color = tuple(ground_color)
    garden_normal: Vec3 = (0.0, 1.0, 0.0)

    # Fences
    fences: List[FenceSegment] = []
    for fence in storey.get("fences", []):
        fences.append(FenceSegment(
            start=tuple(fence["from"]),
            end=tuple(fence["to"]),
            height=fence.get("height", 1.0),
            thickness=fence.get("thickness", 0.05),
            color=tuple(fence.get("color", (1.0, 1.0, 1.0, 1.0))),
        ))

    # Spawn point
    spawn_point = None
    spawn = storey.get("spawn", {})
    position = spawn.get("position")
    if position and len(position) >= 3:
        spawn_point = (float(position[0]), float(position[1]), float(position[2]))

    exterior = ExteriorModel(
        footprint=footprint,
        walls=walls,
        roof_height=roof_height,
        roof_color=roof_color,
        roof_overhang=roof_overhang,
        roof_thickness=roof_thickness,
        garden_center=garden_center,
        garden_size=garden_size,
        garden_color=garden_color,
        garden_normal=garden_normal,
        fences=fences,
        spawn_point=spawn_point,
    )

    # Interior rooms
    rooms: List[Room] = []
    for room in storey.get("interior_rooms", []):
        rooms.append(Room(
            name=room.get("name", room.get("id", "room")),
            center=tuple(room["center"]),
            size=tuple(room["size"]),
            color=tuple(room.get("color", (0.8, 0.8, 0.8, 0.4))),
        ))

    return House(rooms=rooms), exterior

def create_prototype_exterior() -> ExteriorModel:
    """
    Creates a simple rectangular house shell with:
    - Four walls
    - One front door + a few windows
    - Flat roof
    - Garden plane around the house
    """
    wall_height = 2.7

    # Footprint: simple rectangle around the interior rooms.
    # Sized to comfortably contain the 2x2 interior grid (rooms at +/-2.25).
    half_w = 6.0   # house half width (x)
    half_d = 6.0   # house half depth (z)

    outline: List[Vec2] = [
        (-half_w, -half_d),  # front-left  (looking from +z towards -z)
        ( half_w, -half_d),  # front-right
        ( half_w,  half_d),  # back-right
        (-half_w,  half_d),  # back-left
    ]
    footprint = Footprint(outline=outline, wall_height=wall_height)

    # Walk outline edges → walls
    walls: List[WallSegment] = []
    wall_color: Color = (0.85, 0.85, 0.9, 1.0)

    for i in range(len(outline)):
        start = outline[i]
        end = outline[(i + 1) % len(outline)]

        # Front wall = first edge (y = -half_d)
        is_front = (start[1] == -half_d and end[1] == -half_d)

        seg = WallSegment(
            start=start,
            end=end,
            height=wall_height,
            thickness=0.2,
            color=wall_color,
            openings=[],
            has_window=False,
            has_door=False,
            is_front=is_front,
        )
        walls.append(seg)

    # Mark front wall features:
    # We'll assume walls[0] is front, given our outline ordering.
    if walls:
        front_wall = walls[0]
        front_wall.openings.append(Opening(
            type="door",
            width=0.9,
            height=2.0,
            sill_height=0.0,
            offset_along_wall=_distance_2d(front_wall.start, front_wall.end) * 0.4,
        ))
        front_wall.openings.append(Opening(
            type="window",
            width=1.0,
            height=1.2,
            sill_height=0.9,
            offset_along_wall=_distance_2d(front_wall.start, front_wall.end) * 0.65,
        ))
        front_wall.has_door = True
        front_wall.has_window = True  # one window near the door, visually

    # Add windows to side walls as a simple start
    if len(walls) >= 4:
        # Right wall
        walls[1].openings.append(Opening(
            type="window",
            width=1.0,
            height=1.2,
            sill_height=0.9,
            offset_along_wall=_distance_2d(walls[1].start, walls[1].end) * 0.5,
        ))
        walls[1].has_window = True
        # Left wall
        walls[3].openings.append(Opening(
            type="window",
            width=1.0,
            height=1.2,
            sill_height=0.9,
            offset_along_wall=_distance_2d(walls[3].start, walls[3].end) * 0.5,
        ))
        walls[3].has_window = True

    # Flat roof just above wall height
    roof_overhang = 0.2
    roof_thickness = 0.1
    roof_height = wall_height + roof_thickness
    roof_color: Color = (0.75, 0.75, 0.78, 1.0)

    # Garden: a plane larger than footprint
    garden_margin = 5.0
    garden_height = 0.05
    garden_center: Vec3 = (0.0, -garden_height / 2.0, 0.0)
    garden_size: Vec3 = (
        (half_w + garden_margin) * 2.0,
        garden_height,
        (half_d + garden_margin) * 2.0,
    )
    garden_color: Color = (0.70, 0.90, 0.70, 1.0)
    garden_normal: Vec3 = (0.0, 1.0, 0.0)
    fences: List[FenceSegment] = []

    return ExteriorModel(
        footprint=footprint,
        walls=walls,
        roof_height=roof_height,
        roof_color=roof_color,
        roof_overhang=roof_overhang,
        roof_thickness=roof_thickness,
        garden_center=garden_center,
        garden_size=garden_size,
        garden_color=garden_color,
        garden_normal=garden_normal,
        fences=fences,
    )
