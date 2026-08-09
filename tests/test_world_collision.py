from house_model import ExteriorModel, FenceSegment, Footprint, Opening, WallSegment
from world_collision import HouseCollisionMap, door_gap_allows


def _collision_map() -> HouseCollisionMap:
    wall = WallSegment(
        start=(-2.0, 0.0),
        end=(2.0, 0.0),
        height=2.7,
        thickness=0.2,
        color=(1.0, 1.0, 1.0, 1.0),
        openings=[
            Opening(
                type="door",
                width=1.0,
                height=2.0,
                sill_height=0.0,
                offset_along_wall=2.0,
                id="front_door",
            )
        ],
    )
    exterior = ExteriorModel(
        footprint=Footprint(outline=[], wall_height=2.7),
        walls=[wall],
        roof_height=2.8,
        roof_color=(1.0, 1.0, 1.0, 1.0),
        roof_overhang=0.0,
        roof_thickness=0.1,
        garden_center=(0.0, 0.0, 0.0),
        garden_size=(10.0, 0.05, 10.0),
        garden_color=(1.0, 1.0, 1.0, 1.0),
        garden_normal=(0.0, 1.0, 0.0),
        fences=[
            FenceSegment(
                start=(-2.0, 2.0),
                end=(2.0, 2.0),
                height=1.0,
                thickness=0.1,
                color=(1.0, 1.0, 1.0, 1.0),
            )
        ],
    )
    return HouseCollisionMap(exterior)


def test_closed_door_blocks_and_open_door_allows_clear_center() -> None:
    collision = _collision_map()
    position = (0.0, 0.0, 0.0)

    assert collision.collides(position, radius=0.2, door_states={})
    assert not collision.collides(
        position, radius=0.2, door_states={"front_door": True}
    )


def test_body_radius_shrinks_door_clearance_instead_of_widening_it() -> None:
    wall = _collision_map().exterior.walls[0]

    assert door_gap_allows(wall, 2.29, 0.2, {"front_door": True})
    assert not door_gap_allows(wall, 2.31, 0.2, {"front_door": True})


def test_swept_motion_cannot_tunnel_through_closed_wall() -> None:
    collision = _collision_map()

    resolved = collision.resolve_motion(
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        radius=0.2,
        door_states={},
    )

    assert resolved[1] < 0.0
    assert not collision.collides(resolved, radius=0.2, door_states={})


def test_low_fence_does_not_block_a_body_above_it() -> None:
    collision = _collision_map()

    assert collision.collides((1.0, 2.0, 0.0), radius=0.2)
    assert not collision.collides((1.0, 2.0, 1.1), radius=0.2)
