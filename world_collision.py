"""Shared, dependency-free collision helpers for the house world."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from house_model import ExteriorModel, WallSegment, load_house_from_plan

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def distance_to_segment(point: Vec2, start: Vec2, end: Vec2) -> Tuple[float, float]:
    """Return distance to a segment and distance of the projection from its start."""
    px, py = point
    sx, sy = start
    vx = end[0] - sx
    vy = end[1] - sy
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-8:
        return math.hypot(px - sx, py - sy), 0.0
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / length_sq))
    return math.hypot(px - (sx + t * vx), py - (sy + t * vy)), math.sqrt(length_sq) * t


def door_gap_allows(
    wall: WallSegment,
    offset: float,
    radius: float,
    door_states: Mapping[str, bool],
) -> bool:
    """Whether a circular body fits wholly through an open door aperture."""
    for opening in wall.openings:
        if opening.type != "door":
            continue
        # Shrink the usable aperture by the body's radius on both sides.  The old
        # viewer code enlarged it, permitting bodies to clip through door jambs.
        half_clearance = max(0.0, (float(opening.width) * 0.5) - radius)
        if abs(offset - float(opening.offset_along_wall)) <= half_clearance:
            return bool(opening.id and door_states.get(str(opening.id), False))
    return False


@dataclass(frozen=True)
class HouseCollisionMap:
    exterior: ExteriorModel

    @classmethod
    def from_plan(cls, path: str) -> "HouseCollisionMap":
        _house, exterior = load_house_from_plan(path)
        return cls(exterior=exterior)

    def collides(
        self,
        position: Sequence[float],
        *,
        radius: float,
        door_states: Optional[Mapping[str, bool]] = None,
        foot_z: Optional[float] = None,
    ) -> bool:
        px, py = float(position[0]), float(position[1])
        base_z = float(position[2] if foot_z is None else foot_z)
        states = door_states or {}
        for wall in self.exterior.walls:
            if base_z >= float(wall.height) - 0.05:
                continue
            distance, offset = distance_to_segment((px, py), wall.start, wall.end)
            if distance <= (float(wall.thickness) * 0.5 + radius):
                if door_gap_allows(wall, offset, radius, states):
                    continue
                return True
        for fence in self.exterior.fences:
            if base_z >= float(fence.height) - 0.05:
                continue
            distance, _offset = distance_to_segment((px, py), fence.start, fence.end)
            if distance <= (float(fence.thickness) * 0.5 + radius):
                return True
        return False

    def resolve_motion(
        self,
        current: Sequence[float],
        desired: Sequence[float],
        *,
        radius: float,
        door_states: Optional[Mapping[str, bool]] = None,
        foot_z: Optional[float] = None,
        max_step: float = 0.1,
    ) -> Vec3:
        """Sweep a move in small steps, sliding along blocked axes."""
        result = [float(current[0]), float(current[1]), float(current[2])]
        delta = [float(desired[i]) - result[i] for i in range(3)]
        horizontal = math.hypot(delta[0], delta[1])
        steps = max(1, int(math.ceil(horizontal / max(max_step, 1e-3))))
        increment = [component / steps for component in delta]
        for _ in range(steps):
            target = [result[i] + increment[i] for i in range(3)]
            target_foot_z = target[2] if foot_z is None else foot_z + (target[2] - float(current[2]))
            if not self.collides(target, radius=radius, door_states=door_states, foot_z=target_foot_z):
                result = target
                continue
            x_only = [target[0], result[1], target[2]]
            if not self.collides(x_only, radius=radius, door_states=door_states, foot_z=target_foot_z):
                result = x_only
            y_only = [result[0], target[1], target[2]]
            if not self.collides(y_only, radius=radius, door_states=door_states, foot_z=target_foot_z):
                result = y_only
        return (result[0], result[1], result[2])
