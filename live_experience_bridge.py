"""Bridge live sensory streams into grounded experience logs.

This module coordinates high-frequency inputs such as screen captures and
spoken dialogue so that they are reflected inside the experiential memory
structure introduced by :mod:`experience_logger`.  The primary goal is to
ensure that Ina can relate on-going interactions to concrete episodes rather
than detached data fragments.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore

from experience_logger import ExperienceLogger

try:  # Optional dependency that offers pixel capture utilities.
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    mss = None

try:  # Pillow is preferred for persisting screen snapshots.
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None

try:  # OpenCV provides reliable fallbacks for image persistence.
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ScreenCaptureConfig:
    """Configuration parameters for live screen sampling."""

    interval_seconds: float = 5.0
    region: Optional[Dict[str, int]] = None
    tags: Iterable[str] = field(default_factory=lambda: ["screen", "live"])
    narrative: str = "Observed the operator's screen."


@dataclass
class AudioCaptureConfig:
    """Configuration parameters for spoken dialogue ingestion."""

    speaker: str = "operator"
    tags: Iterable[str] = field(default_factory=lambda: ["conversation", "live"])
    narrative_prefix: str = "Conversation with the operator"


class LiveExperienceBridge:
    """Translate live sensory signals into grounded experiential events."""

    def __init__(
        self,
        *,
        child: str = "Inazuma_Yagami",
        base_path: Optional[Path] = None,
        logger: Optional[ExperienceLogger] = None,
    ) -> None:
        self.child = child
        self._base_path = Path(base_path) if base_path else Path("AI_Children")
        self.logger = logger or ExperienceLogger(child=child, base_path=self._base_path)
        experiences_root = (
            self._base_path / child / "memory" / "experiences" / "live_media"
        )
        self._media_dir = _ensure_dir(experiences_root)
        self._screen_config = ScreenCaptureConfig()
        self._audio_config = AudioCaptureConfig()
        self._screen_thread: Optional[threading.Thread] = None
        self._screen_stop = threading.Event()

    # ------------------------------------------------------------------
    # Screen capture integration
    # ------------------------------------------------------------------
    def configure_screen_capture(self, **kwargs: Any) -> None:
        """Update the parameters used for live screen sampling."""

        for key, value in kwargs.items():
            if hasattr(self._screen_config, key):
                setattr(self._screen_config, key, value)

    def start_screen_capture(self) -> None:
        """Launch a background loop that periodically records the screen."""

        if mss is None:
            raise RuntimeError(
                "mss is required for live screen capture but is not available."
            )
        if self._screen_thread and self._screen_thread.is_alive():
            return

        self._screen_stop.clear()
        self._screen_thread = threading.Thread(
            target=self._screen_loop, name="LiveScreenCapture", daemon=True
        )
        self._screen_thread.start()

    def stop_screen_capture(self) -> None:
        """Signal the capture loop to terminate."""

        self._screen_stop.set()
        if self._screen_thread and self._screen_thread.is_alive():
            self._screen_thread.join(timeout=2.0)

    def _screen_loop(self) -> None:
        assert mss is not None  # Guarded by start_screen_capture
        capture = mss.mss()
        try:
            while not self._screen_stop.is_set():
                frame = self._grab_screen_frame(capture)
                if frame is not None:
                    self.log_screen_snapshot(frame)
                time.sleep(max(0.1, self._screen_config.interval_seconds))
        finally:
            capture.close()

    def _grab_screen_frame(self, capture: "mss.mss") -> Optional[np.ndarray]:
        try:
            monitor = (
                self._screen_config.region
                if self._screen_config.region
                else capture.monitors[1]
            )
            raw = capture.grab(monitor)
            frame = np.array(raw) if np is not None else raw
            if np is not None and frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            return frame
        except Exception:
            return None

    def log_screen_snapshot(
        self,
        frame: np.ndarray,
        *,
        tags: Optional[Iterable[str]] = None,
        narrative: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a single screen observation as an experiential event."""

        if np is None:
            raise RuntimeError("numpy is required to process screen snapshots.")

        situation_tags = list(tags or self._screen_config.tags)
        narrative_text = narrative or self._screen_config.narrative
        recognized_text, feature_sample = self._describe_frame(frame)
        perceived_entities = []
        if recognized_text:
            perceived_entities.append(
                {
                    "type": "screen_text",
                    "content": " ".join(recognized_text),
                    "tokens": recognized_text,
                }
            )
        if metadata:
            perceived_entities.append({"type": "metadata", **metadata})

        event_id = self.logger.log_event(
            situation_tags=situation_tags,
            perceived_entities=perceived_entities,
            narrative=narrative_text,
            internal_state={"feature_sample": feature_sample},
        )
        self._persist_frame(event_id, frame, recognized_text, metadata)
        if recognized_text:
            utterance = " ".join(recognized_text)
            self.logger.attach_word_usage(
                event_id,
                speaker="screen_text",
                utterance=utterance,
                words=recognized_text,
                entity_links=[{"type": "screen_text", "event_id": event_id}],
            )
        return event_id

    def _describe_frame(self, frame: np.ndarray) -> Tuple[List[str], List[float]]:
        sample = frame.reshape(-1).astype(float)[:512]
        feature_sample = np.round(sample / 255.0, 4).tolist()
        recognized: List[str] = []
        try:
            from vision_digest import run_text_recognition

            recognized = run_text_recognition(frame, self.child)
        except Exception:
            recognized = []
        return recognized, feature_sample

    def _persist_frame(
        self,
        event_id: str,
        frame: np.ndarray,
        recognized_text: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        media_meta: Dict[str, Any] = {
            "event_id": event_id,
            "shape": list(frame.shape),
            "recognized_text": recognized_text,
            "metadata": metadata or {},
        }
        saved = False
        image_path = self._media_dir / f"{event_id}_screen.png"
        try:
            if Image is not None:
                img = Image.fromarray(frame.astype(np.uint8))
                img.save(image_path)
                saved = True
            elif cv2 is not None:
                cv2.imwrite(str(image_path), frame)
                saved = True
        except Exception:
            saved = False

        if not saved:
            fallback = self._media_dir / f"{event_id}_screen.npy"
            try:
                np.save(fallback, frame)
                media_meta["raw_file"] = fallback.name
            except Exception:
                media_meta["raw_file"] = None
        else:
            media_meta["image_file"] = image_path.name

        meta_path = self._media_dir / f"{event_id}_screen.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(media_meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Dialogue integration
    # ------------------------------------------------------------------
    def log_conversation_turn(
        self,
        utterance: str,
        *,
        speaker: Optional[str] = None,
        event_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        entity_links: Optional[Iterable[Dict[str, Any]]] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Capture a spoken utterance and bind it to an experiential event."""

        speaker_name = speaker or self._audio_config.speaker
        situation_tags = list(tags or self._audio_config.tags)

        if event_id is None:
            narrative = f"{self._audio_config.narrative_prefix}: {speaker_name} said '{utterance}'"
            event_id = self.logger.log_event(
                situation_tags=situation_tags,
                narrative=narrative,
                internal_state={"speaker": speaker_name},
                timestamp=timestamp,
            )
        annotation = self.logger.attach_word_usage(
            event_id,
            speaker=speaker_name,
            utterance=utterance,
            entity_links=entity_links,
            timestamp=timestamp,
        )
        meta_path = self._media_dir / f"{event_id}_dialogue.json"
        try:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            else:
                payload = {"event_id": event_id, "turns": []}
        except Exception:
            payload = {"event_id": event_id, "turns": []}
        payload["turns"].append(annotation)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return event_id


__all__ = ["LiveExperienceBridge", "ScreenCaptureConfig", "AudioCaptureConfig"]

