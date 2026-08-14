"""Audience-specific, lightweight social reaction modelling."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gui_hook import log_to_statusbox
from model_manager import load_config
from origin_record import make_origin

try:  # pragma: no cover - optional dependency
    from meaning_map import get_symbol_neighbors
except Exception:  # pragma: no cover
    def get_symbol_neighbors(symbol_id: str | None = None, tags: Optional[Iterable[str]] = None, k: int = 5) -> list:
        return []

try:  # pragma: no cover - optional dependency
    from vision_digest import run_text_recognition
except Exception:  # pragma: no cover
    def run_text_recognition(image: Any, child: str | None = None) -> list:
        return []


class HeuristicMirrorTransformer:
    """Predict reactions with a separate bounded model for each audience."""

    VERSION = "V2"

    def __init__(self, child: Optional[str] = None, root_path: Path | str = "AI_Children") -> None:
        config = load_config()
        self.child = child or config.get("current_child", "default_child")
        self.root = Path(root_path)
        self.mirror_path = self.root / self.child / "mirror"
        self.log_path = self.mirror_path / "mirror_log.jsonl"
        self.model_path = self.mirror_path / "audience_models.json"
        self.mirror_path.mkdir(parents=True, exist_ok=True)
        self.models = self._load_models()

    @staticmethod
    def _audience_key(audience: Any) -> str:
        try:
            raw = json.dumps(audience, sort_keys=True, default=str)
        except Exception:
            raw = str(audience)
        label = str(audience or "generic")[:48]
        return f"{label}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"

    def _load_models(self) -> Dict[str, Any]:
        if not self.model_path.is_file():
            return {}
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_models(self) -> None:
        self.model_path.write_text(json.dumps(self.models, indent=2, sort_keys=True), encoding="utf-8")

    def _log(self, action: str, detail: Optional[Dict[str, Any]] = None) -> None:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "action": action, "detail": detail or {}}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def observe_reaction(
        self, perceived_audience: Any, presented_emotions: Dict[str, float],
        observed_emotions: Dict[str, float], *, event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update one audience transform from an actual interaction outcome."""
        key = self._audience_key(perceived_audience)
        model = self.models.setdefault(key, {"audience": perceived_audience, "count": 0, "emotions": {}})
        count = int(model.get("count", 0)) + 1
        rate = max(0.05, min(0.5, 1.0 / count))
        for emotion in sorted(set(presented_emotions) | set(observed_emotions)):
            source = float(presented_emotions.get(emotion, 0.0) or 0.0)
            actual = float(observed_emotions.get(emotion, 0.0) or 0.0)
            params = model.setdefault("emotions", {}).setdefault(emotion, {"scale": 0.8, "bias": 0.0})
            predicted = float(params["scale"]) * source + float(params["bias"] or 0.0)
            error = actual - predicted
            params["scale"] = round(max(-2.0, min(2.0, float(params["scale"]) + rate * error * source)), 6)
            params["bias"] = round(max(-1.0, min(1.0, float(params["bias"]) + rate * error)), 6)
        model["count"] = count
        model["last_event_id"] = event_id
        model["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_models()
        return dict(model)

    def mirror(
        self, symbolic_state: Dict[str, Any], emotional_vector: Optional[Dict[str, float]] = None,
        perceived_audience: Optional[Any] = None,
    ) -> Dict[str, Any]:
        emotional_vector = emotional_vector or {}
        tags = list(symbolic_state.get("tags", []))
        mirrored_symbols = []
        for tag in tags:
            try:
                neighbors = get_symbol_neighbors(tags=[tag], k=1)
                mirrored_symbols.append(neighbors[0] if neighbors else tag)
            except Exception:
                mirrored_symbols.append(tag)
        vision_tags = []
        if symbolic_state.get("image") is not None:
            try:
                vision_tags = run_text_recognition(symbolic_state["image"], child=self.child)
            except Exception:
                vision_tags = []

        key = self._audience_key(perceived_audience)
        model = self.models.get(key, {})
        parameters = model.get("emotions", {}) if isinstance(model, dict) else {}
        predicted_emotions = {}
        for emotion, value in emotional_vector.items():
            params = parameters.get(emotion, {})
            scale = float(params.get("scale", 0.8) or 0.0)
            bias = float(params.get("bias", 0.0) or 0.0)
            predicted_emotions[emotion] = round(max(-1.0, min(1.0, scale * float(value) + bias)), 4)
        misalignment = {key: round(float(value) - predicted_emotions.get(key, 0.0), 4) for key, value in emotional_vector.items()}
        empathy_vector = {key: round((float(value) + predicted_emotions.get(key, 0.0)) / 2, 4) for key, value in emotional_vector.items()}
        origin = make_origin(
            self.__class__.__name__, self.VERSION, inputs={"tags": tags, "emotions": emotional_vector},
            trigger="perspective_request", metadata={"audience_key": key, "observations": model.get("count", 0) if isinstance(model, dict) else 0},
        )
        result = {
            "mirrored_symbols": mirrored_symbols, "vision_tags": vision_tags,
            "predicted_emotions": predicted_emotions, "misalignment": misalignment,
            "empathy_vector": empathy_vector, "audience": perceived_audience,
            "audience_model_observations": int(model.get("count", 0) or 0) if isinstance(model, dict) else 0,
            "origins": [origin],
        }
        self._log("mirror", {"audience": perceived_audience, "result": result})
        log_to_statusbox("[Mirror] Generated audience-specific perspective.")
        return result
