from tests.test_world_collision import _collision_map
from world_server import MotionLimits, WorldState


def _walk_forward(state: WorldState, entity_id: str, steps: int = 40) -> list[float]:
    state.update_input(
        entity_id,
        forward=1.0,
        strafe=0.0,
        up=0.0,
        turn=0.0,
        run=False,
        now=1.0,
    )
    for _ in range(steps):
        state.step(0.05, MotionLimits())
    return state.snapshot()["entities"][entity_id]["position"]


def test_world_state_blocks_entity_at_closed_door() -> None:
    state = WorldState(
        bounds=(-5.0, 5.0, -5.0, 5.0),
        tv_channel="spotify",
        collision_map=_collision_map(),
    )
    state.ensure_entity("ina", "ina", "Ina", position=(0.0, -1.0, 0.0))

    position = _walk_forward(state, "ina")

    assert position[1] < 0.0


def test_world_state_allows_entity_through_open_door() -> None:
    state = WorldState(
        bounds=(-5.0, 5.0, -5.0, 5.0),
        tv_channel="spotify",
        collision_map=_collision_map(),
        door_states={"front_door": True},
    )
    state.ensure_entity("ina", "ina", "Ina", position=(0.0, -1.0, 0.0))

    position = _walk_forward(state, "ina")

    assert position[1] > 0.0
