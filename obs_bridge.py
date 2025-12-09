"""Optional OBS WebSocket bridge to give Ina a unified vision feed.

This helper is intentionally lightweight and defensive:
- Uses ``simpleobsws`` if available, otherwise cleanly disables itself.
- Grabs a composited screenshot from the current program scene.
- Can (optionally) ask OBS to save the replay buffer for richer context.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

try:  # OBS WebSocket v5+ client (async)
    import simpleobsws  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    simpleobsws = None


class OBSWebSocketBridge:
    """Synchronous-friendly wrapper around OBS WebSocket screenshots."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
        source: Optional[str] = None,
        enabled: bool = True,
        use_replay_buffer: bool = False,
        replay_min_interval: float = 120.0,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.password = password or ""
        self.source = source
        self.enabled = bool(enabled)
        self.use_replay_buffer = bool(use_replay_buffer)
        self.replay_min_interval = float(replay_min_interval)
        self.logger = logger

        self._last_error: Optional[str] = None
        self._last_error_ts: float = 0.0
        self._last_replay_save: float = 0.0

        if self.enabled and simpleobsws is None:
            self._log_once("OBS WebSocket bridge disabled: install simpleobsws to enable vision via OBS.")

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls, cfg: Optional[Dict[str, Any]], logger: Optional[Callable[[str], None]] = None
    ):
        if not cfg or cfg.get("enabled") is False:
            return None
        return cls(
            host=cfg.get("host", "localhost"),
            port=int(cfg.get("port", 4455)),
            password=cfg.get("password", "") or "",
            source=cfg.get("source"),
            enabled=cfg.get("enabled", True),
            use_replay_buffer=cfg.get("use_replay_buffer", False),
            replay_min_interval=float(cfg.get("replay_min_interval", 120)),
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        return self.enabled and simpleobsws is not None

    @property
    def can_save_replay(self) -> bool:
        return self.is_available and self.use_replay_buffer

    def capture_frame(self) -> Optional[np.ndarray]:
        """Fetch a composited frame from the current OBS program scene."""
        if not self.is_available:
            return None
        try:
            return self._run_async(self._capture_once())
        except Exception as exc:  # pragma: no cover - runtime guard
            self._log_once(f"OBS capture failed: {exc}")
            return None

    def save_replay_buffer(self) -> bool:
        """Ask OBS to save the replay buffer (throttled)."""
        if not self.can_save_replay:
            return False

        now = time.time()
        if now - self._last_replay_save < self.replay_min_interval:
            return False

        try:
            ok = bool(self._run_async(self._save_replay()))
            if ok:
                self._last_replay_save = now
                self._log("OBS replay buffer saved.")
            return ok
        except Exception as exc:  # pragma: no cover - runtime guard
            self._log_once(f"OBS replay buffer request failed: {exc}")
            return False

    def set_record_directory(self, directory: str) -> bool:
        """Ask OBS to route recordings to a specific directory."""
        if not self.is_available or not directory:
            return False
        try:
            ok = bool(self._run_async(self._set_record_directory(directory)))
            if ok:
                self._log(f"Recording directory set to: {directory}")
            return ok
        except Exception as exc:  # pragma: no cover - runtime guard
            self._log_once(f"OBS record directory request failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _make_client(self):
        if simpleobsws is None:
            return None
        params = simpleobsws.IdentificationParameters(ignoreNonFatalRequestChecks=True)
        url = f"ws://{self.host}:{self.port}"
        return simpleobsws.WebSocketClient(
            url=url, password=self.password, identification_parameters=params
        )

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    async def _capture_once(self) -> Optional[np.ndarray]:
        client = self._make_client()
        if client is None:
            return None

        try:
            await client.connect()
            await client.wait_until_identified()

            source_name = self.source
            if not source_name:
                scene_resp = await client.call(simpleobsws.Request("GetCurrentProgramScene"))
                if scene_resp and getattr(scene_resp, "responseData", None):
                    source_name = scene_resp.responseData.get("currentProgramSceneName")

            if not source_name:
                return None

            req = simpleobsws.Request(
                "GetSourceScreenshot",
                {"sourceName": source_name, "imageFormat": "png"},
            )
            resp = await client.call(req)
            data_url = getattr(resp, "responseData", {}).get("imageData")
            return self._decode_image(data_url)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

    async def _save_replay(self) -> bool:
        client = self._make_client()
        if client is None:
            return False

        try:
            await client.connect()
            await client.wait_until_identified()
            await client.call(simpleobsws.Request("SaveReplayBuffer"))
            return True
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

    async def _set_record_directory(self, directory: str) -> bool:
        client = self._make_client()
        if client is None:
            return False
        try:
            await client.connect()
            await client.wait_until_identified()
            await client.call(
                simpleobsws.Request(
                    "SetRecordDirectory", {"recordDirectory": directory}
                )
            )
            return True
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

    @staticmethod
    def _decode_image(data_url: Optional[str]) -> Optional[np.ndarray]:
        if not data_url or "," not in data_url:
            return None
        try:
            encoded = data_url.split(",", 1)[1]
            raw = base64.b64decode(encoded)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Logging helpers                                                    #
    # ------------------------------------------------------------------ #
    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(f"[OBS] {message}")

    def _log_once(self, message: str) -> None:
        now = time.time()
        if message == self._last_error and (now - self._last_error_ts) < 30:
            return
        self._last_error = message
        self._last_error_ts = now
        self._log(message)
