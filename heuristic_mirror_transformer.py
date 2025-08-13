"""heuristic_mirror_transformer.py

Heuristic Mirror Transformer simulates how Ina believes an audience would
interpret her internal state.  The transformer receives a symbolic
representation of the current state, an emotional vector, and an optional
``perceived_audience`` descriptor.  It then produces a mirrored view of the
symbols and emotions, highlighting potential misalignments while preserving
Ina's own perspective.

The implementation is intentionally lightweight.  Many integrations are
optional and fall back to benign defaults if the required modules are not
available.  This keeps the component easy to test while still offering hooks
for vision, language and emotion subsystems when they exist.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gui_hook import log_to_statusbox
from model_manager import load_config

# ------------------------------------------------------------------ optional
try:  # pragma: no cover - optional dependency
    from meaning_map import get_symbol_neighbors
except Exception:  # pragma: no cover - fallback used in tests
    def get_symbol_neighbors(symbol_id: str | None = None,
                             tags: Optional[Iterable[str]] = None,
                             k: int = 5) -> list:
        return []

try:  # pragma: no cover - optional dependency
    from vision_digest import run_text_recognition
except Exception:  # pragma: no cover - fallback used in tests
    def run_text_recognition(image: Any, child: str | None = None) -> list:
        return []


class HeuristicMirrorTransformer:
    """Generate an externally viewed representation of Ina's state.

    Parameters
    ----------
    child:
        Override the ``current_child`` from the config.  Useful for testing.
    root_path:
        Base directory where child specific data is stored.  Defaults to
        ``AI_Children`` in the repository root.
    """

    def __init__(self, child: Optional[str] = None, root_path: Path | str = "AI_Children"):
        config = load_config()
        self.child = child or config.get("current_child", "default_child")
        self.root = Path(root_path)
        self.mirror_path = self.root / self.child / "mirror"
        self.log_path = self.mirror_path / "mirror_log.jsonl"
        self.mirror_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def _log(self, action: str, detail: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ----------------------------------------------------------------- public
    def mirror(self,
               symbolic_state: Dict[str, Any],
               emotional_vector: Optional[Dict[str, float]] = None,
               perceived_audience: Optional[Any] = None) -> Dict[str, Any]:
        """Return how an audience might interpret the provided state.

        Parameters
        ----------
        symbolic_state:
            Dictionary describing the current symbolic representation.
            Expected keys include ``tags`` and optional vision features.
        emotional_vector:
            Mapping of emotion slider names to values.
        perceived_audience:
            Any descriptor representing the expected viewer.  Only logged.
        """

        emotional_vector = emotional_vector or {}
        tags = list(symbolic_state.get("tags", []))

        # Translate symbols through meaning map to simulate audience view
        mirrored_symbols = []
        for tag in tags:
            try:
                neighbors = get_symbol_neighbors(tags=[tag], k=1)
                mirrored_symbols.append(neighbors[0] if neighbors else tag)
            except Exception:  # pragma: no cover - should rarely occur
                mirrored_symbols.append(tag)

        # Attempt to pull visual hints if image data is present
        vision_tags: list[str] = []
        if symbolic_state.get("image") is not None:
            try:
                vision_tags = run_text_recognition(symbolic_state["image"], child=self.child)
            except Exception:  # pragma: no cover
                vision_tags = []

        # Predict how the audience might feel and measure misalignment
        predicted_emotions = {k: round(v * 0.8, 4) for k, v in emotional_vector.items()}
        misalignment = {k: round(v - predicted_emotions.get(k, 0.0), 4)
                        for k, v in emotional_vector.items()}
        empathy_vector = {k: round((v + predicted_emotions.get(k, 0.0)) / 2, 4)
                          for k, v in emotional_vector.items()}

        result = {
            "mirrored_symbols": mirrored_symbols,
            "vision_tags": vision_tags,
            "predicted_emotions": predicted_emotions,
            "misalignment": misalignment,
            "empathy_vector": empathy_vector,
            "audience": perceived_audience,
        }

        self._log("mirror", {"audience": perceived_audience, "result": result})
        log_to_statusbox("[Mirror] Generated mirrored perspective.")
        return result
