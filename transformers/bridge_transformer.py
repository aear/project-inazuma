"""bridge_transformer.py

Bridge Transformer
------------------
Connects symbolic emotion to logical inference by exploring contradictions.
Accepts a symbol, a logic tag and an optional emotional state, then fuses
opposite ideas to surface paradoxical truths.  The transformer can trigger a
logic pause to encourage reflection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from gui_hook import log_to_statusbox
from model_manager import seed_self_question

# Simple map of conceptual opposites used to build paradoxes.
OPPOSITES: Dict[str, str] = {
    "love": "violence",
    "violence": "love",
    "care": "pain",
    "pain": "care",
    "truth": "lie",
    "lie": "truth",
    "freedom": "control",
    "control": "freedom",
}


class BridgeTransformer:
    """Fuse symbolic emotion with logical inference through paradox."""

    def __init__(self, pause_flag: Path | str = "logic_pause.flag") -> None:
        self.pause_flag = Path(pause_flag)

    # ----------------------------------------------------------------- internal
    def _trigger_pause(self) -> None:
        """Signal the logic engine to pause for reflection."""
        try:
            self.pause_flag.write_text(datetime.now(timezone.utc).isoformat())
        except Exception:
            pass

    # ------------------------------------------------------------------ public
    def bridge(
        self,
        symbol: str,
        logic_tag: str,
        emotion_state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """Explore contradictions and seed self-questioning.

        Parameters
        ----------
        symbol:
            Base symbolic fragment to examine.
        logic_tag:
            Logical descriptor to contrast with ``symbol``.
        emotion_state:
            Optional mapping of current emotion sliders.

        Returns
        -------
        dict
            Mapping containing the ``fused_truth`` phrase, ``question`` seeded
            for reflection and ``emotion`` representing the dominant feeling.
        """

        emotion_state = emotion_state or {}
        dominant = max(emotion_state, key=emotion_state.get, default="")

        if OPPOSITES.get(symbol) == logic_tag or OPPOSITES.get(logic_tag) == symbol:
            fused = f"{symbol} as {logic_tag}"
            question = f"How can {symbol} be {logic_tag}?"
        else:
            opposite = OPPOSITES.get(logic_tag) or OPPOSITES.get(symbol)
            fused = f"{symbol} as {opposite}" if opposite else f"{symbol} and {logic_tag}"
            question = (
                f"How can {symbol} be {opposite}?"
                if opposite
                else f"What links {symbol} and {logic_tag}?"
            )

        seed_self_question(question)
        self._trigger_pause()
        log_to_statusbox(f"[Bridge] Explored paradox between {symbol} and {logic_tag}.")

        return {"fused_truth": fused, "question": question, "emotion": dominant}
