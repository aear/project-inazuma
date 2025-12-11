from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import json
import logging
import os
from pathlib import Path
import threading
import uuid

try:
    from text_memory import record_text_observation
except Exception:  # pragma: no cover - optional dependency
    record_text_observation = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SenderInfo:
    """Represents who a message is from, independent of backend."""
    internal_id: str         # stable ID inside Ina (e.g. "sakura", "ina", "system")
    backend_id: str          # raw ID from backend (e.g. Discord user ID)
    display_name: str        # human readable name
    is_self: bool = False    # True if this is Ina herself
    backend: str = "unknown" # e.g. "discord", "gui", "mercury"


@dataclass
class ChannelInfo:
    """Represents where a message belongs (channel / DM / thread)."""
    internal_id: str         # stable channel ID inside Ina
    backend_id: str          # raw ID from backend
    name: str                # human readable name
    is_private: bool = False
    backend: str = "unknown"


@dataclass
class CommsMessage:
    """Canonical representation of a message flowing through comms_core."""
    id: str
    direction: str           # "inbound" or "outbound"
    backend: str             # source/target backend, e.g. "discord", "gui"
    sender: SenderInfo
    channel: ChannelInfo
    text: str
    created_at: str          # ISO 8601 UTC timestamp
    reply_to_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommsResponse:
    """
    Standard result from Ina's internal processing.

    text: main reply text to send outward.
    metadata: structured data that backends or logs might care about:
        - emotion_vector
        - energy_state
        - precision_state
        - debug_notes
        - etc.
    """
    text: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class CommsCore:
    """
    Central hub for all communication in and out of Ina.

    - Normalises messages from different backends into a single schema.
    - Hands them to the internal processing pipeline (model_manager / logic_engine / etc.).
    - Dispatches resulting replies back to registered backends.
    - Logs everything for later analysis.

    This module is intentionally agnostic about *how* Ina thinks.
    You plug in your own `process_inbound` callback that takes a CommsMessage
    and returns a CommsResponse.
    """

    def __init__(
        self,
        *,
        instance_name: str = "ina",
        log_dir: Optional[Path] = None,
        process_inbound: Optional[Callable[[CommsMessage], CommsResponse]] = None,
        raw_fallback: Optional[Callable[[CommsMessage], None]] = None,
    ) -> None:
        self.instance_name = instance_name
        self._backends: Dict[str, Callable[[CommsMessage], None]] = {}
        self._process_inbound = process_inbound or self._default_process_inbound
        self._raw_fallback = raw_fallback

        # basic log file setup
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "comms_core.jsonl"

        # threading lock for file writes
        self._log_lock = threading.Lock()

        logger.info("CommsCore initialised: instance=%s log_dir=%s", instance_name, self.log_dir)

    # ------------------------------------------------------------------
    # Backend registration
    # ------------------------------------------------------------------
    def register_backend(self, name: str, send_callable: Callable[[CommsMessage], None]) -> None:
        """
        Register a backend that can send messages out.

        `send_callable` will be called with a CommsMessage for each outbound
        message targeting this backend.
        """
        if name in self._backends:
            logger.warning("Comms backend %s is being overwritten", name)
        self._backends[name] = send_callable
        logger.info("Comms backend registered: %s", name)

    # ------------------------------------------------------------------
    # Inbound path (called by backends)
    # ------------------------------------------------------------------
    def receive_inbound(
        self,
        *,
        backend: str,
        backend_message_id: str,
        sender: SenderInfo,
        channel: ChannelInfo,
        text: str,
        reply_to_backend_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CommsMessage]:
        """
        Entry point for backends to deliver an inbound message into Ina.

        Returns the outbound CommsMessage that was sent as a direct reply,
        or None if there was no reply.
        """
        metadata = metadata or {}

        msg = CommsMessage(
            id=self._new_message_id(),
            direction="inbound",
            backend=backend,
            sender=sender,
            channel=channel,
            text=text,
            created_at=self._now_iso(),
            reply_to_id=reply_to_backend_id,
            metadata={
                "backend_message_id": backend_message_id,
                **metadata,
            },
        )

        if record_text_observation and text:
            try:
                tags = [backend, "inbound"]
                if metadata.get("is_dm"):
                    tags.append("dm")
                if metadata.get("is_owner_friend"):
                    tags.append("owner_friend")
                record_text_observation(
                    text=text,
                    source=f"{backend}:{channel.name}",
                    tags=tags,
                )
            except Exception:
                logger.exception("Failed to record text fragment for inbound message %s", backend_message_id)

        self._log_message(msg)

        # Hand message to Ina's internal pipeline
        response = self._process_inbound(msg)

        if response is None or response.text is None:
            logger.debug("No response generated for message %s", msg.id)
            return None

        # Create outbound message
        outbound = CommsMessage(
            id=self._new_message_id(),
            direction="outbound",
            backend=backend,  # reply via same backend by default
            sender=SenderInfo(
                internal_id=self.instance_name,
                backend_id=self.instance_name,
                display_name=self.instance_name,
                is_self=True,
                backend=backend,
            ),
            channel=channel,
            text=response.text,
            created_at=self._now_iso(),
            reply_to_id=msg.id,
            metadata=response.metadata,
        )

        self._log_message(outbound)
        self._dispatch_outbound(outbound)
        return outbound

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------
    def send_outbound(
        self,
        *,
        backend: str,
        channel: ChannelInfo,
        text: str,
        reply_to_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CommsMessage:
        """
        Send an outbound message initiated internally (not directly replying
        to a specific inbound message).
        """
        metadata = metadata or {}

        outbound = CommsMessage(
            id=self._new_message_id(),
            direction="outbound",
            backend=backend,
            sender=SenderInfo(
                internal_id=self.instance_name,
                backend_id=self.instance_name,
                display_name=self.instance_name,
                is_self=True,
                backend=backend,
            ),
            channel=channel,
            text=text,
            created_at=self._now_iso(),
            reply_to_id=reply_to_id,
            metadata=metadata,
        )

        self._log_message(outbound)
        self._dispatch_outbound(outbound)
        return outbound

    def _dispatch_outbound(self, msg: CommsMessage) -> None:
        """Send an outbound message to its backend."""
        backend = msg.backend
        send_fn = self._backends.get(backend)
        if not send_fn:
            self._fallback_outbound(msg, reason="no backend registered")
            return
        try:
            send_fn(msg)
        except Exception:
            logger.exception("Error while dispatching message %s via backend %s", msg.id, backend)
            self._fallback_outbound(msg, reason="send failed")

    def _fallback_outbound(self, msg: CommsMessage, *, reason: str) -> None:
        """Optional fallback path when no backend is available or send fails."""
        if self._raw_fallback:
            try:
                self._raw_fallback(msg)
                return
            except Exception:
                logger.exception("Raw fallback failed for message %s (reason=%s)", msg.id, reason)
        logger.error("Dropping outbound message %s (reason=%s, text=%s)", msg.id, reason, msg.text)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_message(self, msg: CommsMessage) -> None:
        """Append message as JSONL to the comms log."""
        record = dataclasses.asdict(msg)
        try:
            with self._log_lock:
                with self._log_path.open("a", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")
        except Exception:
            logger.exception("Failed to write comms log record for %s", msg.id)

    # ------------------------------------------------------------------
    # Defaults & utilities
    # ------------------------------------------------------------------
    def _default_process_inbound(self, msg: CommsMessage) -> CommsResponse:
        """
        Fallback processing pipeline.

        Replace this by passing `process_inbound` when you create CommsCore.
        For now it just echoes the text with a small prefix so you can verify
        the Discord bridge end-to-end before wiring in model_manager.py.
        """
        logger.debug("Default process_inbound used for message %s", msg.id)
        return CommsResponse(
            text=f"[echo:{self.instance_name}] {msg.text}",
            metadata={
                "debug": True,
                "note": "Default echo pipeline in comms_core._default_process_inbound",
            },
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_message_id() -> str:
        return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Optional helper: secrets loader for backends (e.g. Discord)
# ---------------------------------------------------------------------------

def load_secret(
    key: str,
    *,
    env_name: Optional[str] = None,
    secrets_file: Optional[Path] = None,
) -> Optional[str]:
    """
    Load a secret (e.g. DISCORD_BOT_TOKEN) with a simple, local-only strategy.

    Priority:
    1. Environment variable
    2. JSON file (default: ./secrets.json) with shape: {"KEY": "value"}

    This is mostly for backends like backend_discord.py – comms_core itself
    does not use the secret, it just provides a helper.
    """
    env_var = env_name or key
    value = os.getenv(env_var)
    if value:
        return value

    secrets_path = secrets_file or Path("secrets.json")
    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text(encoding="utf-8"))
            value = data.get(key)
            if value:
                return value
        except Exception:
            logger.exception("Failed to read secrets file at %s", secrets_path)

    return None
