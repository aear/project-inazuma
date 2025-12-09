"""Desktop optic nerve capture for Ina.

This module watches Fedora's Desktop 1 (primary workspace) like an optic nerve,
logging experiential events via ExperienceLogger and optionally storing compact
vision fragments for compatibility with the existing memory system.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Preferred screen grabber for a single desktop/workspace.
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    mss = None

try:  # Fallback capture path.
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pyautogui = None

from live_experience_bridge import LiveExperienceBridge


def _load_child() -> str:
    path = Path("config.json")
    if not path.exists():
        return "Inazuma_Yagami"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("current_child", "Inazuma_Yagami")
    except Exception:
        return "Inazuma_Yagami"


class DesktopOpticNerve:
    """Capture Desktop 1 frames and funnel them into experiential memory."""

    def __init__(
        self,
        *,
        child: Optional[str] = None,
        monitor_index: int = 1,
        capture_interval: float = 5.0,
        delta_threshold: float = 50.0,
        tags: Optional[Iterable[str]] = None,
        narrative: Optional[str] = None,
        write_fragments: bool = True,
    ) -> None:
        self.child = child or _load_child()
        self.monitor_index = monitor_index
        self.capture_interval = capture_interval
        self.delta_threshold = delta_threshold
        self.tags = list(tags) if tags else ["vision", "desktop1", "optic_nerve"]
        self.narrative = narrative or "Observed Desktop 1 (primary workspace)."
        self.write_fragments = write_fragments

        self.bridge = LiveExperienceBridge(child=self.child)
        self.logger = self.bridge.logger
        self.bridge.configure_screen_capture(tags=self.tags, narrative=self.narrative)

        self._capture = None
        self._episode_started = False

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------
    def ensure_episode(self) -> None:
        if self._episode_started:
            return
        self.logger.start_episode(
            situation_tags=self.tags,
            preconditions={
                "workspace": "Desktop 1",
                "monitor_index": self.monitor_index,
            },
            intent="Passive optic nerve feed for Desktop 1",
        )
        self._episode_started = True

    def close_episode(self, result: Optional[Dict[str, Any]] = None) -> None:
        if self._episode_started:
            self.logger.finish_and_narrate_episode(
                result=result
                or {"status": "stream_active", "workspace": "Desktop 1"},
                feedback_hooks=[{"event": "vision_stream"}],
            )
        self._episode_started = False
        if self._capture:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None

    # ------------------------------------------------------------------
    # Capture + logging
    # ------------------------------------------------------------------
    def capture_frame(self) -> Optional[np.ndarray]:
        """Grab the current Desktop 1 frame."""
        if mss:
            if self._capture is None:
                self._capture = mss.mss()
            try:
                monitor = (
                    self._capture.monitors[self.monitor_index]
                    if self.monitor_index in range(1, len(self._capture.monitors))
                    else self._capture.monitors[1]
                )
            except Exception:
                monitor = self._capture.monitors[1]
            try:
                raw = self._capture.grab(monitor)
            except Exception:
                raw = None
            if raw is not None:
                frame = np.array(raw)
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                return frame

        if pyautogui:
            try:
                image = pyautogui.screenshot()
                frame = np.array(image)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = frame[:, :, ::-1]
                return frame
            except Exception:
                return None
        return None

    @staticmethod
    def compute_delta(prev: np.ndarray, current: np.ndarray) -> float:
        if prev is None or current is None:
            return 0.0
        if prev.shape != current.shape:
            return 0.0
        delta = np.abs(prev.astype(np.int32) - current.astype(np.int32))
        return float(np.sum(delta) / delta.size)

    def log_snapshot(
        self, frame: np.ndarray, delta_score: float, timestamp: Optional[str] = None
    ) -> str:
        self.ensure_episode()
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        metadata = {
            "delta_score": round(float(delta_score), 4),
            "workspace": "Desktop 1",
            "monitor_index": self.monitor_index,
        }
        event_id = self.bridge.log_screen_snapshot(
            frame, tags=self.tags, narrative=self.narrative, metadata=metadata
        )
        if self.write_fragments:
            self._write_fragment(frame, ts, event_id, metadata)
        return event_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write_fragment(
        self,
        frame: np.ndarray,
        timestamp: str,
        event_id: str,
        metadata: Dict[str, Any],
    ) -> str:
        flat = self._embed_frame(frame)
        frag_id = f"frag_optic_{int(time.time())}"
        fragment = {
            "id": frag_id,
            "summary": "Desktop 1 optic nerve glimpse",
            "tags": self.tags + ["fragment"],
            "timestamp": timestamp,
            "source": "desktop_optic_nerve",
            "event_ref": event_id,
            "clarity": 0.55,
            "modality": "screen",
            "image_features": flat,
            "metadata": metadata,
        }
        path = Path("AI_Children") / self.child / "memory" / "fragments"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{frag_id}.json", "w", encoding="utf-8") as fh:
            json.dump(fragment, fh, indent=4)
        return frag_id

    @staticmethod
    def _embed_frame(frame: Any) -> List[float]:
        """
        Compress a frame into a stable, low-dimensional embedding:
        - Patch means per channel (8x8 grid)
        - Global stats for robustness
        """
        try:
            import numpy as np  # type: ignore

            arr = np.array(frame, dtype=float)
            if arr.ndim == 2:  # grayscale
                arr = arr[:, :, None]

            h, w, c = arr.shape
            grid = 8
            step_y = max(h // grid, 1)
            step_x = max(w // grid, 1)

            patches: List[float] = []
            for y in range(0, h, step_y):
                for x in range(0, w, step_x):
                    patch = arr[y : min(y + step_y, h), x : min(x + step_x, w)]
                    patches.extend(patch.mean(axis=(0, 1)).tolist())
                    if len(patches) >= 3 * grid * grid:
                        break
                if len(patches) >= 3 * grid * grid:
                    break

            global_stats = [
                float(arr.mean()),
                float(arr.std()),
                float(arr.min()),
                float(arr.max()),
            ]

            embed = patches + global_stats
            # Normalize to 0-1 range for consistency
            scale = 255.0 if arr.max() > 1.0 else 1.0
            return [round(v / scale, 6) for v in embed[:512]]
        except Exception:
            flattened: List[float] = []

            def _walk(node: Any) -> None:
                if isinstance(node, (list, tuple)):
                    for item in node:
                        _walk(item)
                else:
                    try:
                        flattened.append(float(node))
                    except Exception:
                        flattened.append(0.0)

            _walk(frame)
            return flattened[:512]

    # ------------------------------------------------------------------
    # Standalone loop (optional)
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        self.ensure_episode()
        prev = self.capture_frame()
        try:
            while True:
                time.sleep(self.capture_interval)
                curr = self.capture_frame()
                if curr is None or prev is None:
                    prev = curr
                    continue
                delta = self.compute_delta(prev, curr)
                if delta >= self.delta_threshold:
                    self.log_snapshot(curr, delta)
                prev = curr
        finally:
            self.close_episode(result={"status": "stopped", "workspace": "Desktop 1"})


def run_optic_nerve() -> None:
    nerve = DesktopOpticNerve()
    nerve.run_forever()


if __name__ == "__main__":
    run_optic_nerve()
