"""Motor control + proprioceptive feedback for Ina.

This module is meant to sit between cognition and movement.
It applies movement limits, simulates balance/gravity cues, and
feeds those sensations into Ina's body schema and inastate.
"""

from __future__ import annotations

import copy
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from world_protocol import clamp
from body_schema import get_default_body_schema, snapshot_default_body

try:
    from model_manager import get_inastate, update_inastate
except Exception:  # pragma: no cover - optional dependency
    get_inastate = None
    update_inastate = None


@dataclass
class MotorLimits:
    walk_speed: float = 1.4
    run_speed: float = 2.4
    accel: float = 2.8
    decel: float = 3.2
    turn_rate_deg: float = 110.0
    vertical_speed: float = 0.6
    gravity: float = 9.8
    jump_speed: float = 4.8
    max_z: float = 2.0

    @classmethod
    def from_house_viewer(cls) -> "MotorLimits":
        return cls(
            walk_speed=4.0,
            run_speed=8.0,
            accel=6.5,
            decel=7.2,
            turn_rate_deg=90.0,
            vertical_speed=1.2,
            gravity=9.8,
            jump_speed=4.8,
            max_z=6.0,
        )


@dataclass
class MotorInput:
    forward: float = 0.0
    strafe: float = 0.0
    up: float = 0.0
    turn: float = 0.0
    run: bool = False
    jump: bool = False


@dataclass
class MotorState:
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    grounded: bool = True
    last_ts: Optional[float] = None
    last_velocity: Optional[Tuple[float, float, float]] = None
    last_yaw: Optional[float] = None


@dataclass
class MotorFeedback:
    speed: float
    vertical_speed: float
    accel_mag: float
    turn_rate: float
    balance: float
    gravity_load: float
    grounded: bool
    motion_intensity: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed": round(self.speed, 4),
            "vertical_speed": round(self.vertical_speed, 4),
            "accel_mag": round(self.accel_mag, 4),
            "turn_rate": round(self.turn_rate, 3),
            "balance": round(self.balance, 4),
            "gravity_load": round(self.gravity_load, 4),
            "grounded": self.grounded,
            "motion_intensity": round(self.motion_intensity, 4),
            "timestamp": self.timestamp,
        }


class MotorController:
    def __init__(
        self,
        *,
        limits: Optional[MotorLimits] = None,
        bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
        ground_z: float = 0.0,
        publish_interval: float = 0.25,
        allow_fly: bool = False,
    ) -> None:
        self.limits = limits or MotorLimits()
        self.bounds = bounds
        self.ground_z = float(ground_z)
        self.allow_fly = bool(allow_fly)
        self.publish_interval = float(publish_interval)

        self.state = MotorState()
        self._last_publish_ts: float = 0.0

    def step(self, motor_input: MotorInput, dt: Optional[float] = None) -> MotorFeedback:
        now = time.monotonic()
        if dt is None:
            if self.state.last_ts is None:
                dt = 0.1
            else:
                dt = now - self.state.last_ts
        dt = max(0.02, min(0.2, dt))

        self.state.last_ts = now
        prev_velocity = tuple(self.state.velocity)
        prev_yaw = self.state.yaw_deg

        forward = clamp(motor_input.forward, -1.0, 1.0)
        strafe = clamp(motor_input.strafe, -1.0, 1.0)
        up = clamp(motor_input.up, -1.0, 1.0)
        turn = clamp(motor_input.turn, -1.0, 1.0)

        speed_target = self.limits.run_speed if motor_input.run else self.limits.walk_speed
        input_mag = (forward * forward + strafe * strafe) ** 0.5
        if input_mag > 1.0:
            forward /= input_mag
            strafe /= input_mag
            input_mag = 1.0

        yaw_delta = turn * self.limits.turn_rate_deg * dt
        self.state.yaw_deg = (self.state.yaw_deg + yaw_delta) % 360.0

        yaw_rad = math.radians(self.state.yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        target_vx = (strafe * speed_target) * cos_y - (forward * speed_target) * sin_y
        target_vy = (strafe * speed_target) * sin_y + (forward * speed_target) * cos_y

        accel = self.limits.accel if input_mag > 0.001 else self.limits.decel
        max_delta = accel * dt

        delta_x = target_vx - self.state.velocity[0]
        delta_y = target_vy - self.state.velocity[1]
        delta_mag = (delta_x * delta_x + delta_y * delta_y) ** 0.5
        if delta_mag > max_delta and delta_mag > 0.0:
            scale = max_delta / delta_mag
            delta_x *= scale
            delta_y *= scale

        self.state.velocity[0] += delta_x
        self.state.velocity[1] += delta_y

        if self.allow_fly:
            self.state.velocity[2] = up * self.limits.vertical_speed
            self.state.grounded = False
        else:
            if motor_input.jump and self.state.grounded:
                self.state.velocity[2] = self.limits.jump_speed
                self.state.grounded = False
            if not self.state.grounded:
                self.state.velocity[2] -= self.limits.gravity * dt

        self.state.position[0] += self.state.velocity[0] * dt
        self.state.position[1] += self.state.velocity[1] * dt
        self.state.position[2] += self.state.velocity[2] * dt

        min_x, max_x, min_y, max_y = self.bounds
        self.state.position[0] = clamp(self.state.position[0], min_x, max_x)
        self.state.position[1] = clamp(self.state.position[1], min_y, max_y)
        self.state.position[2] = clamp(self.state.position[2], self.ground_z, self.limits.max_z)

        if self.state.position[2] <= self.ground_z + 1e-3:
            self.state.position[2] = self.ground_z
            if self.state.velocity[2] < 0.0:
                self.state.velocity[2] = 0.0
            self.state.grounded = True

        feedback = self._compute_feedback(prev_velocity, prev_yaw, dt)
        self._maybe_publish(feedback)
        return feedback

    def observe_state(
        self,
        *,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        yaw_deg: Optional[float] = None,
        now: Optional[float] = None,
    ) -> MotorFeedback:
        if now is None:
            now = time.monotonic()
        if self.state.last_ts is None:
            dt = 0.1
        else:
            dt = max(0.02, min(0.2, now - self.state.last_ts))
        self.state.last_ts = now

        prev_velocity = tuple(self.state.velocity)
        prev_yaw = self.state.yaw_deg

        self.state.position = [float(position[0]), float(position[1]), float(position[2])]
        self.state.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        if yaw_deg is not None:
            self.state.yaw_deg = float(yaw_deg) % 360.0

        self.state.grounded = self.state.position[2] <= self.ground_z + 1e-3

        feedback = self._compute_feedback(prev_velocity, prev_yaw, dt)
        self._maybe_publish(feedback)
        return feedback

    def _compute_feedback(
        self, prev_velocity: Tuple[float, float, float], prev_yaw: float, dt: float
    ) -> MotorFeedback:
        vx, vy, vz = self.state.velocity
        speed = (vx * vx + vy * vy) ** 0.5
        vertical_speed = float(vz)

        dvx = vx - prev_velocity[0]
        dvy = vy - prev_velocity[1]
        dvz = vz - prev_velocity[2]
        accel_mag = (dvx * dvx + dvy * dvy + dvz * dvz) ** 0.5 / max(dt, 1e-3)

        yaw_delta = (self.state.yaw_deg - prev_yaw) % 360.0
        if yaw_delta > 180.0:
            yaw_delta -= 360.0
        turn_rate = yaw_delta / max(dt, 1e-3)

        turn_load = min(abs(turn_rate) / max(self.limits.turn_rate_deg, 1.0), 1.0)
        accel_load = min(accel_mag / max(self.limits.accel, 1.0), 1.0)
        balance = clamp(1.0 - (0.6 * turn_load + 0.4 * accel_load), 0.0, 1.0)

        vertical_accel = dvz / max(dt, 1e-3)
        gravity_load = clamp(1.0 + (vertical_accel / max(self.limits.gravity, 1.0)), 0.2, 1.6)
        if not self.state.grounded:
            gravity_load = min(gravity_load, 0.9)

        speed_ratio = speed / max(self.limits.run_speed, 1.0)
        motion_intensity = clamp((speed_ratio + accel_load + turn_load) / 3.0, 0.0, 1.0)

        return MotorFeedback(
            speed=speed,
            vertical_speed=vertical_speed,
            accel_mag=accel_mag,
            turn_rate=turn_rate,
            balance=balance,
            gravity_load=gravity_load,
            grounded=self.state.grounded,
            motion_intensity=motion_intensity,
            timestamp=_now_iso(),
        )

    def _maybe_publish(self, feedback: MotorFeedback) -> None:
        now = time.monotonic()
        if now - self._last_publish_ts < self.publish_interval:
            return
        self._last_publish_ts = now
        publish_motor_feedback(feedback)


def publish_motor_feedback(
    feedback: MotorFeedback,
    *,
    schema_path: Optional[str] = None,
    strength: float = 0.6,
) -> Optional[Dict[str, Dict[str, float]]]:
    if update_inastate is None:
        return None

    adjustments = motor_feedback_to_body_adjustment(feedback)
    touch_feedback = motor_feedback_to_touch(feedback)
    base_state = _load_body_state(schema_path=schema_path)
    merged_state = merge_body_state(base_state, adjustments, strength=strength)

    update_inastate("motor_feedback", feedback.to_dict())
    update_inastate("motor_body_adjustment", adjustments)
    update_inastate("touch_feedback", touch_feedback)
    update_inastate("body_state", merged_state)
    update_inastate("last_motor_update", feedback.timestamp)
    update_inastate("last_touch_update", touch_feedback.get("timestamp", feedback.timestamp))

    return merged_state


def motor_feedback_to_body_adjustment(feedback: MotorFeedback) -> Dict[str, Dict[str, float]]:
    adjustments: Dict[str, Dict[str, float]] = {}

    imbalance = clamp(1.0 - feedback.balance, 0.0, 1.0)
    effort = clamp(feedback.motion_intensity, 0.0, 1.0)
    weight_axis = clamp((feedback.gravity_load - 1.0) * 0.8, -1.0, 1.0)
    vertical_factor = clamp(abs(feedback.vertical_speed) / 6.0, 0.0, 1.0)
    turn_factor = clamp(abs(feedback.turn_rate) / 120.0, 0.0, 1.0)

    def bump(region: str, tension: float = 0.0, openness: float = 0.0, weight: float = 0.0, energy: float = 0.0):
        if region not in adjustments:
            adjustments[region] = {"tension": 0.0, "openness": 0.0, "weight": 0.0, "energy": 0.0}
        adjustments[region]["tension"] += tension
        adjustments[region]["openness"] += openness
        adjustments[region]["weight"] += weight
        adjustments[region]["energy"] += energy

    bump("left_leg", tension=0.35 * effort + 0.25 * imbalance, weight=0.6 * weight_axis, energy=0.25 * effort)
    bump("right_leg", tension=0.35 * effort + 0.25 * imbalance, weight=0.6 * weight_axis, energy=0.25 * effort)
    bump("left_foot", tension=0.3 * effort + 0.2 * imbalance, weight=0.55 * weight_axis, energy=0.2 * effort)
    bump("right_foot", tension=0.3 * effort + 0.2 * imbalance, weight=0.55 * weight_axis, energy=0.2 * effort)
    bump("core", tension=0.4 * imbalance + 0.2 * effort, weight=0.4 * weight_axis, energy=0.15 * effort)
    bump("chest", tension=0.15 * imbalance, openness=-0.2 * imbalance)
    bump("throat", tension=0.1 * imbalance)
    bump("head", tension=0.1 * imbalance + 0.05 * turn_factor, weight=0.25 * weight_axis + 0.2 * vertical_factor)
    bump("left_arm", tension=0.1 * effort + 0.1 * turn_factor, energy=0.1 * effort)
    bump("right_arm", tension=0.1 * effort + 0.1 * turn_factor, energy=0.1 * effort)
    bump("left_hand", tension=0.08 * effort + 0.08 * turn_factor)
    bump("right_hand", tension=0.08 * effort + 0.08 * turn_factor)

    return adjustments


def motor_feedback_to_touch(feedback: MotorFeedback) -> Dict[str, Any]:
    grounded = bool(feedback.grounded)
    pressure = clamp(feedback.gravity_load if grounded else 0.0, 0.0, 1.0)
    movement = clamp(feedback.speed / 3.0, 0.0, 1.0)
    impact = 0.0
    if grounded and feedback.vertical_speed < -0.01:
        impact = clamp(abs(feedback.vertical_speed) / 3.0, 0.0, 1.0)
    sway = clamp(1.0 - feedback.balance, 0.0, 1.0)
    contacts = []
    if grounded:
        contacts = [
            {"region": "left_foot", "pressure": round(pressure, 4)},
            {"region": "right_foot", "pressure": round(pressure, 4)},
        ]
    standing_still = (
        grounded
        and movement < 0.03
        and abs(feedback.vertical_speed) < 0.03
        and abs(feedback.turn_rate) < 6.0
    )
    stance = "standing_still" if standing_still else ("moving_grounded" if grounded else "airborne")
    surface_solidity = clamp((0.75 * pressure) + (0.25 * (1.0 - impact)), 0.0, 1.0) if grounded else 0.0
    return {
        "grounded": grounded,
        "foot_pressure": round(pressure, 4),
        "support_surface": "solid" if grounded else "none",
        "surface_solidity": round(surface_solidity, 4),
        "stance": stance,
        "movement": round(movement, 4),
        "impact": round(impact, 4),
        "sway": round(sway, 4),
        "contacts": contacts,
        "timestamp": feedback.timestamp,
    }


def merge_body_state(
    base_state: Dict[str, Dict[str, float]],
    adjustments: Dict[str, Dict[str, float]],
    *,
    strength: float = 0.6,
) -> Dict[str, Dict[str, float]]:
    merged = copy.deepcopy(base_state)
    for region_id, axes in adjustments.items():
        if region_id not in merged:
            merged[region_id] = {"tension": 0.0, "openness": 0.0, "weight": 0.0, "energy": 0.0}
        for axis, delta in axes.items():
            current = float(merged[region_id].get(axis, 0.0))
            merged[region_id][axis] = clamp(current + (delta * strength), -1.0, 1.0)
    return merged


def _load_body_state(schema_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    base_state: Optional[Dict[str, Dict[str, float]]] = None
    if get_inastate is not None:
        try:
            state = get_inastate("body_state")
            if isinstance(state, dict):
                base_state = state
        except Exception:
            base_state = None

    if base_state is None:
        fallback = snapshot_default_body(schema_path)
        if fallback is not None:
            base_state = fallback

    if base_state is None:
        schema = get_default_body_schema(schema_path)
        if schema:
            base_state = schema.snapshot()

    return copy.deepcopy(base_state or {})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
