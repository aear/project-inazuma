import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motor_controls import MotorFeedback, motor_feedback_to_touch


def _feedback(**overrides):
    payload = {
        "speed": 0.0,
        "vertical_speed": 0.0,
        "accel_mag": 0.0,
        "turn_rate": 0.0,
        "balance": 1.0,
        "gravity_load": 1.0,
        "grounded": True,
        "motion_intensity": 0.0,
        "timestamp": "2026-02-09T00:00:00+00:00",
    }
    payload.update(overrides)
    return MotorFeedback(**payload)


def test_touch_reports_solid_ground_and_standing_still():
    touch = motor_feedback_to_touch(
        _feedback(
            speed=0.0,
            vertical_speed=0.0,
            turn_rate=0.0,
            gravity_load=0.95,
            grounded=True,
        )
    )
    assert touch["grounded"] is True
    assert touch["support_surface"] == "solid"
    assert touch["stance"] == "standing_still"
    assert touch["surface_solidity"] > 0.9
    assert len(touch["contacts"]) == 2


def test_touch_reports_airborne_without_support():
    touch = motor_feedback_to_touch(
        _feedback(
            speed=0.8,
            vertical_speed=0.6,
            turn_rate=18.0,
            gravity_load=0.45,
            grounded=False,
            motion_intensity=0.5,
        )
    )
    assert touch["grounded"] is False
    assert touch["support_surface"] == "none"
    assert touch["stance"] == "airborne"
    assert touch["surface_solidity"] == 0.0
    assert touch["contacts"] == []


def test_touch_reports_grounded_motion_stance():
    touch = motor_feedback_to_touch(
        _feedback(
            speed=1.2,
            vertical_speed=0.0,
            turn_rate=12.0,
            gravity_load=1.0,
            grounded=True,
            motion_intensity=0.4,
        )
    )
    assert touch["grounded"] is True
    assert touch["support_surface"] == "solid"
    assert touch["stance"] == "moving_grounded"
