"""Low-latency export of rebuildable world-view frames."""
from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from storage_layout import fast_runtime_path


def resolve_world_frame_path(filename: str, config_path: Path = Path("config.json")) -> Optional[Path]:
    """Route rebuildable viewer frames to fast runtime storage when available."""

    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    child = str(config.get("current_child") or "default_child")
    fallback = Path("AI_Children") / child / "memory" / "vision_session" / filename
    return fast_runtime_path(
        child,
        filename,
        fallback,
        subdir="vision_session",
        config=config,
    )


class AsyncFrameExporter:
    """Keep PNG compression and filesystem writes off the Qt input thread."""

    def __init__(self, *, interval: float = 1.0, thread_name: str = "world_frame_export") -> None:
        self.interval = max(0.1, float(interval))
        self._last_export = 0.0
        self._future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)

    def export(self, view, path: Optional[Path], now: float) -> None:
        if path is None:
            return

        future = self._future
        if future is not None:
            if not future.done():
                return
            self._future = None
            try:
                future.result()
            except Exception:
                pass

        if (now - self._last_export) < self.interval:
            return
        self._last_export = now

        try:
            pixmap = view.grab()
        except Exception:
            return
        if pixmap.isNull():
            return

        # QImage is reentrant and the copy detaches it from the GUI-owned pixmap.
        image = pixmap.toImage().copy()
        self._future = self._executor.submit(self._save, image, Path(path))

    @staticmethod
    def _save(image, path: Path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(image.save(str(path), "PNG"))
