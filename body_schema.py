# body_schema.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
import copy
import math

DEFAULT_ANCHORS = {
    "head": {"center": [0.0, 0.0, 9.5], "radius": 1.4},
    "throat": {"center": [0.0, 0.0, 7.8], "radius": 0.9},
    "chest": {"center": [0.0, 0.0, 6.2], "radius": 1.6},
    "core": {"center": [0.0, 0.0, 4.4], "radius": 1.4},
    "left_arm": {"center": [-2.4, 0.0, 5.8], "radius": 1.2},
    "right_arm": {"center": [2.4, 0.0, 5.8], "radius": 1.2},
    "left_leg": {"center": [-1.0, 0.0, 1.6], "radius": 1.3},
    "right_leg": {"center": [1.0, 0.0, 1.6], "radius": 1.3},
}


@dataclass
class BodyRegionState:
  """State for a single body region."""
  tension: float = 0.0
  openness: float = 0.0
  weight: float = 0.0
  energy: float = 0.0

  def clamp(self) -> None:
      self.tension = max(-1.0, min(1.0, self.tension))
      self.openness = max(-1.0, min(1.0, self.openness))
      self.weight = max(-1.0, min(1.0, self.weight))
      self.energy = max(-1.0, min(1.0, self.energy))


class BodySchema:
  """
  Ina's internal body schema.

  Responsibilities:
  - Load static schema (regions, axes, named postures) from JSON.
  - Maintain a current per-region state.
  - Update body state from emotion vectors (via a simple mapping).
  - Apply / register named postures.
  - Provide compact snapshots for memory fragments and for avatar bridges.
  """

  def __init__(self, schema_data: Dict[str, Any]):
      self._schema_data = schema_data
      self.regions = [r["id"] for r in schema_data.get("body_regions", [])]

      # load default state
      self._state: Dict[str, BodyRegionState] = {}
      default_state = schema_data.get("default_state", {})
      for region_id in self.regions:
          region_defaults = default_state.get(region_id, {})
          self._state[region_id] = BodyRegionState(
              tension=region_defaults.get("tension", 0.0),
              openness=region_defaults.get("openness", 0.0),
              weight=region_defaults.get("weight", 0.0),
              energy=region_defaults.get("energy", 0.0),
          )

      self.named_postures: Dict[str, Dict[str, Dict[str, float]]] = \
          copy.deepcopy(schema_data.get("named_postures", {}))
      self._anchors = self._load_anchors(schema_data)

  # ---- construction helpers ----

  @classmethod
  def from_file(cls, path: str) -> "BodySchema":
      with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
      return cls(data)

  # ---- core state API ----

  def reset(self, posture: str = "neutral") -> None:
      """Reset body to default or a named posture."""
      if posture in self.named_postures:
          self.apply_posture(posture)
          return

      # fallback: reset to default_state
      default_state = self._schema_data.get("default_state", {})
      for region_id in self.regions:
          region_defaults = default_state.get(region_id, {})
          self._state[region_id] = BodyRegionState(
              tension=region_defaults.get("tension", 0.0),
              openness=region_defaults.get("openness", 0.0),
              weight=region_defaults.get("weight", 0.0),
              energy=region_defaults.get("energy", 0.0),
          )

  def get_region_state(self, region_id: str) -> Optional[BodyRegionState]:
      return self._state.get(region_id)

  def snapshot(self) -> Dict[str, Dict[str, float]]:
      """
      Return a serialisable snapshot of current body state, suitable for:
      - storing on fragments
      - writing to inastate.json
      - driving an avatar bridge
      """
      out: Dict[str, Dict[str, float]] = {}
      for region_id, state in self._state.items():
          out[region_id] = {
              "tension": state.tension,
              "openness": state.openness,
              "weight": state.weight,
              "energy": state.energy,
          }
      return out

  # ---- posture API ----

  def apply_posture(self, name: str, blend: float = 1.0) -> None:
      """
      Move current body state toward a named posture.
      blend=1.0 -> full overwrite
      blend=0.5 -> halfway between current and target
      """
      if name not in self.named_postures:
          return

      target = self.named_postures[name]
      blend = max(0.0, min(1.0, blend))

      for region_id, target_axes in target.items():
          if region_id not in self._state:
              continue
          region_state = self._state[region_id]
          for axis, target_value in target_axes.items():
              current_value = getattr(region_state, axis, 0.0)
              new_value = current_value + (target_value - current_value) * blend
              setattr(region_state, axis, new_value)
          region_state.clamp()

  def register_posture(self, name: str,
                       posture_state: Dict[str, Dict[str, float]]) -> None:
      """
      Register a new named posture (e.g. from training data).
      posture_state has same structure as snapshot().
      """
      self.named_postures[name] = copy.deepcopy(posture_state)

  # ---- emotion mapping API ----

  def update_from_emotion(self,
                          emotion_values: Dict[str, float],
                          strength: float = 0.3) -> None:
      """
      Update body state from Ina's current emotional sliders.

      emotion_values: e.g. {
          "intensity": 0.5,
          "positivity": 0.8,
          "negativity": -0.2,
          "stress": -0.4,
          "novelty": 0.9,
          "familiarity": -0.3,
          ...
      }

      strength: 0..1 scaling factor for how strongly the body reacts.
      """
      strength = max(0.0, min(1.0, strength))

      intensity  = emotion_values.get("intensity", 0.0)
      stress     = emotion_values.get("stress", 0.0)
      positivity = emotion_values.get("positivity", 0.0)
      novelty    = emotion_values.get("novelty", 0.0)
      energy_axis = emotion_values.get("_core_energy", 0.0)

      # Example simple rule-set; you can tune this later or learn it.
      for region_id, region_state in self._state.items():
          # baseline deltas
          d_tension = (intensity + stress) * 0.5 * strength
          d_openness = (positivity - stress) * 0.4 * strength
          d_weight = (-energy_axis - stress) * 0.3 * strength
          d_energy = (intensity + novelty) * 0.5 * strength

          # head maybe reacts a bit more to novelty
          if region_id == "head":
              d_energy += novelty * 0.2 * strength

          # core reacts more to stress
          if region_id == "core":
              d_tension += stress * 0.2 * strength
              d_weight += stress * 0.1 * strength

          region_state.tension += d_tension
          region_state.openness += d_openness
          region_state.weight += d_weight
          region_state.energy += d_energy

          region_state.clamp()

  # ---- spatial helpers ----

  def _load_anchors(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
      anchors: Dict[str, Dict[str, float]] = {}
      raw = data.get("anchors", {})

      for region_id in self.regions:
          fallback = DEFAULT_ANCHORS.get(region_id, {"center": [0.0, 0.0, 0.0], "radius": 1.0})
          raw_anchor = raw.get(region_id, {}) if isinstance(raw, dict) else {}
          center = raw_anchor.get("center", fallback["center"])
          radius = raw_anchor.get("radius", fallback["radius"])
          try:
              center_vals = [float(center[i]) for i in range(3)]
          except Exception:
              center_vals = fallback["center"]
          try:
              radius_val = float(radius)
          except Exception:
              radius_val = fallback["radius"]
          anchors[region_id] = {"center": center_vals, "radius": radius_val}

      return anchors

  def region_anchor(self, region_id: str) -> Dict[str, float]:
      """
      Return spatial anchor (center + radius) for a region, falling back
      to a neutral anchor if missing.
      """
      if region_id in self._anchors:
          return self._anchors[region_id]
      return {"center": [0.0, 0.0, 0.0], "radius": 1.0}


# ---- module-level convenience helpers --------------------------------------

_DEFAULT_SCHEMA_PATH = Path("body_schema.json")
_DEFAULT_BODY_SCHEMA: Optional[BodySchema] = None


def get_default_body_schema(schema_path: Optional[str] = None) -> Optional[BodySchema]:
    """
    Lazily load and cache the default body schema instance.
    """
    global _DEFAULT_BODY_SCHEMA, _DEFAULT_SCHEMA_PATH
    path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH

    # Reload if path changes or cache is empty
    if _DEFAULT_BODY_SCHEMA is None or path != _DEFAULT_SCHEMA_PATH:
        _DEFAULT_SCHEMA_PATH = path
        try:
            _DEFAULT_BODY_SCHEMA = BodySchema.from_file(str(path))
        except Exception:
            _DEFAULT_BODY_SCHEMA = None

    return _DEFAULT_BODY_SCHEMA


def snapshot_default_body(schema_path: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Return a copy of the current default body snapshot if available.
    """
    schema = get_default_body_schema(schema_path)
    if not schema:
        return None
    return copy.deepcopy(schema.snapshot())


def get_region_anchors(schema_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Expose the anchor map (center + radius) for each known body region.
    Useful for spatially projecting neural nodes into body space.
    """
    schema = get_default_body_schema(schema_path)
    if not schema:
        return {}
    return copy.deepcopy(schema._anchors)
