# discord_bridge.py

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio

import discord

from comms_core import CommsCore, CommsResponse, load_secret
from backend_discord import (
    make_sender_info_from_discord,
    make_channel_info_from_discord,
    register_discord_backend,
)
from social_map import (
    get_owner_user_id,
    is_owner_friend,
    is_high_trust,
    get_high_trust_contacts,
    record_dm_attempt,
    update_social_entry,
)
from language_processing import generate_symbolic_reply_from_text
from live_experience_bridge import LiveExperienceBridge
from model_manager import update_inastate
try:
    from lm_studio_adapter import LMStudioAdapter
except Exception:
    LMStudioAdapter = None  # type: ignore

# ---------------------------------------------------------------------------
# Basic logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bridge")
CONFIG_PATH = Path("config.json")
_CHAT_ADAPTER = None


def _load_discord_sinks():
    """
    Attempt to import Discord voice sinks, falling back to the legacy
    discord.ext.voice_recv extension if present. Emits a targeted warning with
    install guidance if neither is available.
    """
    version = getattr(discord, "__version__", "unknown")
    try:
        from discord import sinks as discord_sinks  # type: ignore
        return discord_sinks
    except Exception as first_exc:
        try:
            from discord.ext import voice_recv as voice_sinks  # type: ignore
        except Exception as exc:
            logger.warning(
                "discord voice receive modules not available (discord.py %s); voice capture disabled. "
                "Install py-cord[voice] (or discord-ext-voice-recv) to enable discord.sinks. import errors: %s / %s",
                version,
                first_exc,
                exc,
            )
            return None
        logger.info(
            "Loaded discord voice sinks from discord.ext.voice_recv extension (discord.py %s).",
            version,
        )
        return voice_sinks


sinks = _load_discord_sinks()


def log_discord_voice_capabilities():
    """Emit a one-time info log about discord voice support to aid debugging."""
    version = getattr(discord, "__version__", "unknown")
    sink_path = getattr(sinks, "__file__", None) if sinks else None
    has_start_recording = hasattr(getattr(discord, "VoiceClient", None), "start_recording")
    logger.info(
        "Discord voice capabilities: version=%s sinks=%s start_recording=%s sink_path=%s",
        version,
        bool(sinks),
        has_start_recording,
        sink_path,
    )


def _install_voice_debug_hooks():
    """
    Add lightweight logging on voice state/server updates to track session/token details.
    """
    try:
        vc_cls = discord.VoiceClient
    except Exception:
        return
    if getattr(vc_cls, "_ina_voice_hooks", False):
        return

    vc_cls._ina_voice_hooks = True

    orig_vs = vc_cls.on_voice_state_update
    orig_vserv = vc_cls.on_voice_server_update

    async def wrapped_vs(self, data, *args, **kwargs):
        logger.info(
            "Voice state update: session_id=%s channel_id=%s handshaking=%s reconnecting=%s",
            data.get("session_id"),
            data.get("channel_id"),
            getattr(self, "_handshaking", None),
            getattr(self, "_potentially_reconnecting", None),
        )
        return await orig_vs(self, data, *args, **kwargs)

    async def wrapped_vserv(self, data, *args, **kwargs):
        logger.info(
            "Voice server update: token_present=%s endpoint=%s guild_id=%s",
            bool(data.get("token")),
            data.get("endpoint"),
            data.get("guild_id"),
        )
        return await orig_vserv(self, data, *args, **kwargs)

    vc_cls.on_voice_state_update = wrapped_vs  # type: ignore
    vc_cls.on_voice_server_update = wrapped_vserv  # type: ignore


def load_root_config() -> dict:
    """Lightweight loader for config.json without pulling in the full stack."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read config at %s", CONFIG_PATH)
        return {}


def get_discord_config() -> dict:
    """Return the discord block from config.json, or an empty dict."""
    cfg = load_root_config()
    section = cfg.get("discord") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def get_voice_io_config() -> dict:
    """
    Return voice input settings with sensible defaults.
    """
    cfg = load_root_config()
    child = cfg.get("current_child", "Inazuma_Yagami") if isinstance(cfg, dict) else "Inazuma_Yagami"
    discord_cfg = cfg.get("discord") if isinstance(cfg, dict) else None
    voice_cfg = discord_cfg if isinstance(discord_cfg, dict) else {}
    return {
        "voice_label": voice_cfg.get("voice_label", "discord_voice"),
        "voice_pipe_path": voice_cfg.get("voice_pipe_path"),
        "voice_buffer_dir": voice_cfg.get("voice_buffer_dir")
        or str(Path("AI_Children") / child / "memory" / "discord_voice"),
        "voice_chunk_seconds": max(5, int(voice_cfg.get("voice_chunk_seconds", 15) or 15)),
    }


def get_outbox_policy() -> dict:
    cfg = get_discord_config()
    policy = cfg.get("outbox_policy") if isinstance(cfg, dict) else None
    defaults = {
        "max_burst": 5,
        "max_age_minutes": 5.0,
        "archive_path": None,
    }
    if not isinstance(policy, dict):
        return defaults
    result = defaults.copy()
    if policy.get("max_burst") is not None:
        try:
            result["max_burst"] = max(0, int(policy["max_burst"]))
        except Exception:
            logger.warning("Invalid discord.outbox_policy.max_burst value; using default %s", defaults["max_burst"])
    if policy.get("max_age_minutes") is not None:
        try:
            result["max_age_minutes"] = max(0.0, float(policy["max_age_minutes"]))
        except Exception:
            logger.warning(
                "Invalid discord.outbox_policy.max_age_minutes; using default %s", defaults["max_age_minutes"]
            )
    archive_path = policy.get("archive_path")
    if archive_path:
        result["archive_path"] = str(archive_path)
    return result


def get_current_child() -> str:
    cfg = load_root_config()
    return cfg.get("current_child", "Inazuma_Yagami") if isinstance(cfg, dict) else "Inazuma_Yagami"


def get_chat_adapter():
    """
    Lazy-load a simple text responder. Uses LMStudioAdapter if available,
    otherwise falls back to echo.
    """
    global _CHAT_ADAPTER
    if _CHAT_ADAPTER is not None:
        return _CHAT_ADAPTER
    if LMStudioAdapter is None:
        return None
    try:
        _CHAT_ADAPTER = LMStudioAdapter(child=get_current_child())
    except Exception:
        logger.exception("Failed to initialise LMStudioAdapter; falling back to echo.")
        _CHAT_ADAPTER = None
    return _CHAT_ADAPTER


def _extract_tokens(text: str):
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text or "")]


def _log_raw_outbound(msg):
    """
    Raw fallback: log outbound text when Discord send is unavailable.
    Keeps visibility into Ina's replies even if Discord is down.
    """
    chan = getattr(msg, "channel", None)
    chan_name = getattr(chan, "name", "unknown") if chan else "unknown"
    logger.info("[RAW OUTBOUND] channel=%s text=%s", chan_name, msg.text)


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_channel_by_name(client: discord.Client, name: str, channel_type) -> discord.abc.GuildChannel | None:
    """
    Search across all guilds the bot can see to find a channel by exact name and type.
    """
    target = name.lower().strip()
    for guild in client.guilds:
        for channel in guild.channels:
            if isinstance(channel, channel_type) and channel.name.lower() == target:
                return channel
    return None


def resolve_configured_channels(client: discord.Client):
    """
    Resolve text/voice channel targets using IDs when present, otherwise by name.
    Returns a tuple (text_channel, voice_channel) which may contain None if not found.
    """
    cfg = get_discord_config()
    text_id = _coerce_int(cfg.get("text_channel_id"))
    voice_id = _coerce_int(cfg.get("voice_channel_id"))
    text_name = (cfg.get("text_channel_name") or "").strip()
    voice_name = (cfg.get("voice_channel_name") or "").strip()

    text_chan = client.get_channel(text_id) if text_id else None
    if not text_chan and text_name:
        text_chan = _find_channel_by_name(client, text_name, discord.TextChannel)

    voice_chan = client.get_channel(voice_id) if voice_id else None
    if not voice_chan and voice_name:
        voice_chan = _find_channel_by_name(client, voice_name, discord.VoiceChannel)

    if text_chan:
        logger.info("Resolved text channel: %s (id=%s, guild=%s)", text_chan.name, text_chan.id, text_chan.guild)
    else:
        logger.warning("Text channel not resolved. Set discord.text_channel_id or discord.text_channel_name in config.json.")

    if voice_chan:
        logger.info("Resolved voice channel: %s (id=%s, guild=%s)", voice_chan.name, voice_chan.id, voice_chan.guild)
    else:
        logger.warning("Voice channel not resolved. Set discord.voice_channel_id or discord.voice_channel_name in config.json.")

    return text_chan, voice_chan


# ---------------------------------------------------------------------------
# Config – change these for your setup
# ---------------------------------------------------------------------------

DEFAULT_OWNER_ID = 123456789012345678  # <-- replace via config.json -> discord.owner_user_id
VOICE_JOIN_COMMANDS = {"/ina join", "/ina voice", "/ina voice join", "/ina join voice"}


def _resolve_primary_user_id() -> int:
    cfg = get_discord_config()
    raw_id = cfg.get("owner_user_id") if cfg else None
    if raw_id is not None:
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            logger.warning("discord.owner_user_id is not an integer; checking social map next.")

    owner_from_social = get_owner_user_id(cfg)
    if owner_from_social:
        logger.info("Using owner user id from social map: %s", owner_from_social)
        return owner_from_social

    logger.warning(
        "discord.owner_user_id not set; using default placeholder (%s). "
        "Set config.json->discord.owner_user_id or tag an entry as 'owner' in social_map.json.",
        DEFAULT_OWNER_ID,
    )
    return DEFAULT_OWNER_ID


SAKURA_USER_ID = _resolve_primary_user_id()

# Optional: instance name for Ina, used in comms_core
INA_INSTANCE_NAME = "ina"

# Name of the backend as registered in CommsCore
BACKEND_NAME = "discord"


# ---------------------------------------------------------------------------
# Optional: custom processing pipeline hook
# ---------------------------------------------------------------------------

def process_inbound_message(msg) -> CommsResponse:
    """
    This is where you'd plug Ina's real brain in.

    For now, this tries a lightweight grounded-language adapter; if that is
    unavailable, it falls back to an echo so the bridge remains testable.
    """
    cfg = get_discord_config()
    if isinstance(cfg, dict) and cfg.get("allow_replies") is False:
        return CommsResponse(
            text=None,
            metadata={
                "adapter": "disabled",
                "reason": "discord.allow_replies=false",
            },
        )

    child = get_current_child()

    # Give Ina the option to stay silent based on her urge to type/communicate.
    root_cfg: dict = {}
    try:
        root_cfg = load_root_config()
        min_urge = float(root_cfg.get("min_urge_to_type", 0.35))
    except Exception:
        min_urge = 0.35
    try:
        inastate_path = Path("AI_Children") / child / "memory" / "inastate.json"
        state = json.loads(inastate_path.read_text(encoding="utf-8")) if inastate_path.exists() else {}
    except Exception:
        state = {}
    urge_state = state.get("urge_to_type") or state.get("urge_to_communicate") or {}
    try:
        urge_level = float(urge_state.get("level", 0.0))
    except Exception:
        urge_level = 0.0
    ignore_urge = bool(root_cfg.get("ignore_urge_for_typing", False)) if isinstance(root_cfg, dict) else False
    if not ignore_urge and urge_level < min_urge:
        return CommsResponse(
            text=None,
            metadata={
                "adapter": "urge_gate",
                "reason": "low_urge_to_reply",
                "urge_level": urge_level,
                "threshold": min_urge,
            },
        )

    reply_text = None
    adapter = get_chat_adapter()
    metadata = {"source": "discord_bridge.process_inbound_message", "adapter": "echo"}

    tokens = _extract_tokens(msg.text)
    symbolic = generate_symbolic_reply_from_text(
        msg.text,
        child=child,
        base_path=Path("AI_Children"),
        max_symbols=4,
    )
    symbolic_unknown: list[str] = symbolic.get("unknown") if symbolic else []
    symbolic_text = symbolic.get("text") if symbolic else None
    if symbolic:
        metadata.update(
            {
                "adapter": "language_processing",
                "symbols": symbolic.get("symbols"),
                "unknown_words": symbolic.get("unknown"),
            }
        )
        if not symbolic_unknown:
            return CommsResponse(text=symbolic_text, metadata=metadata)

    if adapter:
        try:
            entity_links = [
                {
                    "type": "discord_message",
                    "author_id": msg.sender.backend_id,
                    "author_name": msg.sender.display_name,
                    "channel_id": msg.channel.backend_id,
                    "channel_name": msg.channel.name,
                    "guild_id": msg.metadata.get("discord_guild_id") if msg.metadata else None,
                    "is_dm": msg.metadata.get("is_dm") if msg.metadata else None,
                }
            ]
            # If Ina still has unknown words, ask LM adapter for explanations/examples.
            explain_targets = symbolic_unknown or tokens
            if symbolic_unknown or (symbolic is None and explain_targets):
                prompt = (
                    "Please explain the following word(s) for Ina using simple meanings "
                    "and one short example each: "
                    + ", ".join(sorted(set(explain_targets))[:8])
                )
                reply_text = adapter.handle_prompt(
                    prompt,
                    speaker=msg.sender.display_name or msg.sender.internal_id,
                    tags=["discord", "text", "lexicon_explain"],
                    entity_links=entity_links,
                    response_tags=["discord", "ina", "lexicon_explain"],
                )
                metadata["adapter"] = "lm_explain"
                metadata["unknown_words"] = explain_targets
                if symbolic_text:
                    metadata["symbolic_hint"] = symbolic_text
            else:
                reply_text = adapter.handle_prompt(
                    msg.text,
                    speaker=msg.sender.display_name or msg.sender.internal_id,
                    tags=["discord", "text"],
                    entity_links=entity_links,
                    response_tags=["discord", "ina"],
                )
                metadata["adapter"] = "lmstudio"
        except Exception:
            logger.exception("LMStudioAdapter failed; falling back to echo.")

    if not reply_text:
        reply_text = f"{INA_INSTANCE_NAME}: {msg.text}"

    return CommsResponse(
        text=reply_text,
        metadata={
            **metadata,
            "debug": adapter is None,
        },
    )


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

class InaDiscordClient(discord.Client):
    """
    Discord client that connects DMs and a configured text channel to Ina via CommsCore.
    """

    def __init__(self, comms: CommsCore, *args, **kwargs) -> None:
        intents = kwargs.pop("intents", None)
        if intents is None:
            intents = discord.Intents.default()
            intents.guilds = True
            intents.messages = True
            intents.message_content = True  # REQUIRED to read message content
            intents.dm_messages = True
            intents.voice_states = True

        super().__init__(intents=intents, *args, **kwargs)
        self.comms = comms
        self.text_channel = None
        self.voice_channel = None
        self.voice_client = None
        self.child = get_current_child()
        voice_cfg = get_voice_io_config()
        self.voice_label = voice_cfg["voice_label"]
        self.voice_pipe_path = Path(voice_cfg["voice_pipe_path"]) if voice_cfg.get("voice_pipe_path") else None
        self.voice_buffer_dir = Path(voice_cfg["voice_buffer_dir"])
        self.voice_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.voice_chunk_seconds = voice_cfg["voice_chunk_seconds"]
        self._recording_active = False
        self._active_sink = None
        self.history_bridge = LiveExperienceBridge(child=self.child)
        child_memory = Path("AI_Children") / self.child / "memory"
        self._typed_outbox_path = child_memory / "typed_outbox.jsonl"
        self._typed_outbox_history_path = child_memory / "typed_outbox_history.jsonl"
        self._outbox_policy = get_outbox_policy()
        archive_override = self._outbox_policy.get("archive_path")
        self._typed_archive_path = Path(archive_override) if archive_override else child_memory / "typed_outbox_archive.jsonl"
        self._typed_outbox_seen = set()
        self._load_outbox_history()
        self._typed_outbox_task = None

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user and self.user.id)
        logger.info("Discord bridge is active. DMs from owner (%s) + configured text channel will be routed.", SAKURA_USER_ID)
        self.text_channel, self.voice_channel = resolve_configured_channels(self)
        if self._typed_outbox_task is None:
            self._typed_outbox_task = asyncio.create_task(self._watch_typed_outbox())

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages from ourselves or other bots
        if message.author.bot:
            return

        content = (message.content or "").strip()
        lower = content.lower()

        # DMs stay owner-only
        if message.guild is None:
            is_owner = message.author.id == SAKURA_USER_ID
            owner_friend = is_owner_friend(message.author.id)
            high_trust = is_high_trust(message.author.id)
            added = False
            if not is_owner:
                try:
                    added = record_dm_attempt(
                        user_id=message.author.id,
                        display_name=message.author.display_name or str(message.author),
                    )
                except Exception:
                    added = False
                    logger.exception(
                        "Failed to record DM attempt in social map for user %s (%s)",
                        message.author,
                        message.author.id,
                    )
            if not (is_owner or owner_friend or high_trust):
                logger.info(
                    "Ignoring DM from untrusted user %s (%s)%s",
                    message.author,
                    message.author.id,
                    " [logged to social_map]" if added else "",
                )
                return
            logger.info(
                "Inbound DM from %s: %s (channel %s)",
                "owner"
                if is_owner
                else "trusted friend"
                if owner_friend
                else "high-trust contact",
                message.content,
                message.channel.id,
            )
            self._record_social_contact(message)
            self._remember_last_dm_contact(message)
            if lower in VOICE_JOIN_COMMANDS:
                await self._handle_voice_join(message)
                return
            self._route_to_comms(message, is_dm=True, owner_friend=owner_friend, high_trust=high_trust)
            return

        # Guild messages: only handle those in the configured text channel
        if self.text_channel is None or message.channel.id != self.text_channel.id:
            return

        self._record_social_contact(message)

        if lower in VOICE_JOIN_COMMANDS:
            await self._handle_voice_join(message)
            return
        if lower in {"/ina learn history", "/ina history learn"} and message.author.id == SAKURA_USER_ID:
            await message.channel.send("Scanning recent history for language training...")
            await self._ingest_message_history()
            await message.channel.send("History scan complete.")
            return
        if lower in {"/ina status", "/ina ping"}:
            await message.channel.send("Ina is listening here.")
            return

        logger.info("Inbound guild message in text channel %s: %s", message.channel.id, content)
        self._route_to_comms(message, is_dm=False)

    async def _handle_voice_join(self, message: discord.Message) -> None:
        target_channel = self.voice_channel
        author_voice = getattr(message.author, "voice", None)
        if target_channel is None and author_voice and author_voice.channel:
            target_channel = author_voice.channel

        if target_channel is None:
            await message.channel.send("No voice channel configured or detected to join.")
            return

        async def _attempt_join(reason: str | None = None) -> tuple[bool, bool]:
            try:
                await self.ensure_voice_connected(target_channel)
                suffix = f" ({reason})" if reason else ""
                await message.channel.send(f"Joined voice channel: {target_channel.name}{suffix}")
                return True, True
            except discord.errors.ConnectionClosed as exc:
                logger.warning(
                    "Voice gateway closed while joining %s (code=%s, attempt=%s)",
                    target_channel,
                    exc.code,
                    reason or "initial",
                )
                if exc.code == 4006:
                    await message.channel.send(
                        "Discord reported an invalid voice session (4006). Resetting the voice client and retrying..."
                    )
                    await self._reset_voice_client()
                    return False, False
                await message.channel.send(
                    f"Voice gateway closed with code {exc.code}. "
                    "Make sure only py-cord[voice] is installed (no discord.py mix), then restart Ina."
                )
                return True, True
            except discord.errors.ClientException as exc:
                if "Already connected" in str(exc):
                    logger.info("Already connected to %s. Resetting voice client and retrying.", target_channel)
                    await message.channel.send(
                        "Discord thinks I'm still tied to an older voice session. Resetting and trying again..."
                    )
                    await self._reset_voice_client()
                    return False, False
                logger.exception("Voice client exception while joining %s: %s", target_channel, exc)
                await message.channel.send(f"Voice client error: {exc}")
                return True, True
            except Exception:
                logger.exception("Failed to join voice channel %s", target_channel)
                await message.channel.send(f"Failed to join voice channel: {target_channel.name}")
                return True, True

        completed, terminal = await _attempt_join()
        if not completed and not terminal:
            await asyncio.sleep(1.0)
            await _attempt_join("after reset")

    def _route_to_comms(
        self,
        message: discord.Message,
        *,
        is_dm: bool,
        owner_friend: bool = False,
        high_trust: bool = False,
    ) -> None:
        sender = make_sender_info_from_discord(message, backend_name=BACKEND_NAME)
        channel = make_channel_info_from_discord(message, backend_name=BACKEND_NAME)
        metadata = {
            "discord_author_id": str(message.author.id),
            "discord_channel_id": str(message.channel.id),
            "is_dm": is_dm,
            "is_owner_friend": owner_friend,
            "is_high_trust": high_trust,
        }
        if message.guild:
            metadata["discord_guild_id"] = str(message.guild.id)

        # Hand this into Ina via CommsCore
        # This will synchronously run the processing pipeline and,
        # if a response is generated, CommsCore will trigger the outbound
        # path which sends a message back using the registered backend.
        self.comms.receive_inbound(
            backend=BACKEND_NAME,
            backend_message_id=str(message.id),
            sender=sender,
            channel=channel,
            text=message.content or "",
            reply_to_backend_id=str(message.id),
            metadata=metadata,
        )

    def _record_social_contact(self, message: discord.Message) -> None:
        """
        Touch the social map entry so trust and recency stay fresh.
        """
        display_name = (
            getattr(message.author, "display_name", None)
            or getattr(message.author, "global_name", None)
            or getattr(message.author, "name", None)
            or str(message.author)
        )
        try:
            update_social_entry(
                message.author.id,
                display_name=display_name,
                last_interaction=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            logger.exception("Failed to update social map after contact from %s", message.author)

    def _remember_last_dm_contact(self, message: discord.Message) -> None:
        """
        Keep a lightweight hint in Ina's state about who last reached out via DM.
        """
        try:
            channel = message.channel
            payload = {
                "user_id": str(message.author.id),
                "display_name": getattr(message.author, "display_name", None)
                or getattr(message.author, "global_name", None)
                or getattr(message.author, "name", None)
                or str(message.author),
                "channel_id": str(getattr(channel, "id", "")),
                "channel_name": getattr(channel, "name", None) or "dm",
                "is_dm": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            update_inastate("last_heard_contact", payload)
        except Exception:
            logger.exception("Failed to record last DM contact in inastate.")

    def _read_typed_outbox(self):
        if not self._typed_outbox_path.exists():
            return []
        entries = []
        max_batch = int(self._outbox_policy.get("max_burst") or 0)
        max_age_minutes = float(self._outbox_policy.get("max_age_minutes") or 0.0)
        expiry_cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes) if max_age_minutes > 0 else None
        )
        try:
            with self._typed_outbox_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        logger.exception("Failed to parse typed outbox line: %s", line[:120])
                        continue
                    entry_id = str(
                        entry.get("id") or entry.get("uuid") or entry.get("created_at") or len(self._typed_outbox_seen)
                    )
                    if entry_id in self._typed_outbox_seen:
                        continue
                    entry["id"] = entry_id
                    if expiry_cutoff and self._entry_is_stale(entry, expiry_cutoff):
                        self._archive_outbox_entry(entry, "stale_buffer")
                        continue
                    self._typed_outbox_seen.add(entry_id)
                    entries.append(entry)
                    if max_batch and len(entries) >= max_batch:
                        break
        except Exception:
            logger.exception("Failed to read typed outbox at %s", self._typed_outbox_path)

        if len(self._typed_outbox_seen) > 5000:
            # Avoid unbounded growth if the file grows large.
            self._typed_outbox_seen = set(list(self._typed_outbox_seen)[-2000:])
        return entries

    def _load_outbox_history(self) -> None:
        if not self._typed_outbox_history_path.exists():
            return
        try:
            with self._typed_outbox_history_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    entry_id = str(entry.get("id") or entry.get("entry_id") or "")
                    if entry_id:
                        self._typed_outbox_seen.add(entry_id)
        except Exception:
            logger.exception("Failed to load typed outbox history from %s", self._typed_outbox_history_path)

    def _log_outbox_history(self, entry_id: str, status: str, *, reason: Optional[str] = None) -> None:
        if not entry_id:
            return
        payload = {
            "id": str(entry_id),
            "status": status,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._typed_outbox_history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._typed_outbox_history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append typed outbox history for entry %s", entry_id)
        self._typed_outbox_seen.add(entry_id)

    def _entry_timestamp(self, entry: dict) -> Optional[datetime]:
        created_at = entry.get("created_at")
        if not created_at:
            return None
        try:
            stamp = datetime.fromisoformat(created_at)
        except Exception:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp

    def _entry_is_stale(self, entry: dict, cutoff: datetime) -> bool:
        stamp = self._entry_timestamp(entry)
        if not stamp:
            return False
        return stamp < cutoff

    def _archive_outbox_entry(self, entry: dict, reason: str) -> None:
        entry_id = str(entry.get("id") or entry.get("uuid") or entry.get("created_at") or "")
        archived = {
            **entry,
            "archive_reason": reason,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._typed_archive_path.parent.mkdir(parents=True, exist_ok=True)
            with self._typed_archive_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(archived, ensure_ascii=False) + "\n")
            logger.info("Archived typed outbox entry %s (%s)", entry_id or "<unknown>", reason)
        except Exception:
            logger.exception("Failed to archive typed outbox entry %s", entry_id or "<unknown>")
        if entry.get("text"):
            try:
                self.history_bridge.log_conversation_turn(
                    entry.get("text", ""),
                    speaker=INA_INSTANCE_NAME,
                    tags=["typed_outbox", "archive", reason],
                    entity_links=[
                        {
                            "type": "typed_outbox_entry",
                            "id": entry_id or entry.get("id") or "",
                            "target": entry.get("target"),
                            "status": "archived",
                            "reason": reason,
                        }
                    ],
                    timestamp=entry.get("created_at") or archived["archived_at"],
                )
            except Exception:
                logger.exception("Failed to log archived outbox entry %s to history bridge", entry_id or "<unknown>")
        if entry_id:
            self._log_outbox_history(entry_id, "archived", reason=reason)

    async def _deliver_typed_outbox_entry(self, entry: dict) -> None:
        text = entry.get("text")
        allow_empty = bool(entry.get("allow_empty"))
        attachment_path = entry.get("attachment_path")

        def _build_file():
            if not attachment_path:
                return None
            try:
                path = Path(attachment_path)
                if not path.exists() or not path.is_file():
                    logger.debug("Attachment path missing for entry %s: %s", entry.get("id"), attachment_path)
                    return None
                return discord.File(str(path), filename=path.name)
            except Exception:
                logger.exception("Failed to prepare attachment for entry %s", entry.get("id"))
                return None

        if text is None:
            if not allow_empty and not attachment_path:
                return
            text = ""
        text_str = str(text)
        if not text_str.strip() and not allow_empty and not attachment_path:
            logger.debug("Skipping empty typed outbox entry %s", entry.get("id"))
            return

        target = entry.get("target") or "owner_dm"
        channel_id = entry.get("channel_id")
        target_user_id = entry.get("user_id")
        sent = False

        async def _send_dm(user_id: int) -> bool:
            try:
                user = self.get_user(user_id) or await self.fetch_user(user_id)
                if not user:
                    return False
                file = _build_file()
                await user.send(text_str, file=file)
                return True
            except Exception:
                logger.exception("Failed to DM user %s for typed outbox entry %s", user_id, entry.get("id"))
                return False

        # Explicit user target first (owner or high-trust only)
        if target_user_id:
            try:
                uid = int(target_user_id)
                if uid == SAKURA_USER_ID or is_high_trust(uid):
                    sent = await _send_dm(uid)
                else:
                    logger.info(
                        "Typed outbox entry %s targets user %s without high trust; skipping.",
                        entry.get("id"),
                        target_user_id,
                    )
            except Exception:
                logger.exception("Invalid user_id on typed outbox entry %s: %s", entry.get("id"), target_user_id)

        # Owner DM fallback
        if not sent and target == "owner_dm":
            sent = await _send_dm(SAKURA_USER_ID)

        # High-trust DM selection
        if not sent and target in {"trusted_dm", "high_trust_dm"}:
            contacts = get_high_trust_contacts(limit=1)
            if contacts:
                try:
                    uid = int(contacts[0].get("user_id"))
                    sent = await _send_dm(uid)
                except Exception:
                    logger.exception("Failed to DM high-trust contact for entry %s", entry.get("id"))

        # Direct channel id
        if not sent and channel_id:
            try:
                channel = self.get_channel(int(channel_id)) or await self.fetch_channel(int(channel_id))
                if channel:
                    file = _build_file()
                    await channel.send(text_str, file=file)
                    sent = True
            except Exception:
                logger.exception(
                    "Failed to send typed outbox entry %s to channel %s", entry.get("id"), channel_id
                )

        if not sent and target == "text_channel" and self.text_channel:
            try:
                file = _build_file()
                await self.text_channel.send(text_str, file=file)
                sent = True
            except Exception:
                logger.exception(
                    "Failed to send typed outbox entry %s to configured text channel", entry.get("id")
                )

        if sent:
            logger.info(
                "Delivered typed outbox entry %s (target=%s, meta=%s)",
                entry.get("id"),
                target,
                entry.get("metadata"),
            )
            entry_id = entry.get("id")
            if entry_id:
                self._log_outbox_history(str(entry_id), "sent")
        else:
            logger.warning("Unable to deliver typed outbox entry %s; no usable target.", entry.get("id"))

    async def _watch_typed_outbox(self):
        while not self.is_closed():
            try:
                pending = self._read_typed_outbox()
                for entry in pending:
                    await self._deliver_typed_outbox_entry(entry)
            except Exception:
                logger.exception("Typed outbox dispatch loop failed.")
            await asyncio.sleep(3)

    async def _ingest_message_history(self, limit: int = 50) -> None:
        """
        Backfill recent Discord text + owner DMs into Ina's experience log for language training.
        """
        targets = []
        if self.text_channel:
            targets.append(("guild_text", self.text_channel))

        try:
            owner = self.get_user(SAKURA_USER_ID) or await self.fetch_user(SAKURA_USER_ID)
            if owner:
                dm = owner.dm_channel or await owner.create_dm()
                targets.append(("owner_dm", dm))
        except Exception:
            logger.exception("Failed to resolve owner DM channel for history ingest.")

        for label, channel in targets:
            try:
                async for msg in channel.history(limit=limit, oldest_first=True):
                    if msg.author.bot:
                        continue
                    content = (msg.content or "").strip()
                    if not content:
                        continue
                    is_dm = label == "owner_dm"
                    tags = ["discord", "history"]
                    if is_dm:
                        tags.append("dm")
                    self.history_bridge.log_conversation_turn(
                        content,
                        speaker=msg.author.display_name or str(msg.author),
                        tags=tags,
                        entity_links=[
                            {
                                "type": "discord_message",
                                "author_id": str(msg.author.id),
                                "channel_id": str(channel.id),
                                "is_dm": is_dm,
                            }
                        ],
                        timestamp=msg.created_at.replace(tzinfo=timezone.utc).isoformat(),
                    )
                logger.info("Ingested %s messages from %s", limit, label)
            except Exception:
                logger.exception("Failed to ingest history for %s", label)

    def _guild_voice_client(self, guild: Optional[discord.Guild]) -> Optional[discord.VoiceClient]:
        if guild is None:
            return None
        for vc in getattr(self, "voice_clients", []):
            try:
                if vc.guild and vc.guild.id == guild.id:
                    return vc
            except Exception:
                continue
        return None

    async def ensure_voice_connected(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """
        Connect or move Ina to the desired voice channel.
        """
        existing = self._guild_voice_client(channel.guild)
        if existing and existing is not self.voice_client:
            self.voice_client = existing

        if self.voice_client and self.voice_client.is_connected():
            if self.voice_client.channel and self.voice_client.channel.id == channel.id:
                await self._ensure_voice_capture()
                return self.voice_client
            await self.voice_client.move_to(channel)
        else:
            try:
                self.voice_client = await channel.connect(reconnect=True)
            except discord.errors.ClientException as exc:
                if "Already connected" in str(exc):
                    logger.info("Discord claims an existing voice session; forcing disconnect before retry.")
                    await self._reset_voice_client()
                    self.voice_client = await channel.connect(reconnect=True)
                else:
                    raise
        self.voice_channel = channel
        await self._ensure_voice_capture()
        return self.voice_client

    async def _ensure_voice_capture(self) -> None:
        """
        Start continuous chunked recording into a pipe/buffer directory if sinks are available.
        """
        if sinks is None:
            logger.warning("discord.sinks not available; voice capture disabled.")
            return
        if not self.voice_client or not self.voice_client.is_connected():
            return
        if not hasattr(self.voice_client, "start_recording"):
            logger.warning(
                "Discord client missing start_recording; install py-cord[voice] to enable voice capture support."
            )
            return
        if self._recording_active:
            return
        self._start_recording_segment()

    def _start_recording_segment(self):
        if sinks is None or not self.voice_client:
            return
        try:
            sink = getattr(sinks, "RawDataSink", None)
            if sink is None:
                sink = getattr(sinks, "RawSink", None)
            sink = sink() if sink else sinks.WaveSink()
        except Exception:
            logger.exception("Failed to create voice sink; voice capture disabled.")
            return
        self._active_sink = sink
        self._recording_active = True
        try:
            self.voice_client.start_recording(sink, self._on_record_complete)
        except Exception:
            self._recording_active = False
            logger.exception("Failed to start voice recording sink.")
            return
        loop = self.loop
        loop.call_later(self.voice_chunk_seconds, self._stop_recording_segment)

    def _stop_recording_segment(self):
        if not self.voice_client or not self._recording_active:
            return
        try:
            self.voice_client.stop_recording()
        except Exception:
            logger.exception("Failed to stop recording sink.")
            self._recording_active = False

    async def _reset_voice_client(self):
        """
        Forcefully disconnect the current voice client and reset recording state.
        """
        targets = set()
        if self.voice_client:
            targets.add(self.voice_client)
        for vc in getattr(self, "voice_clients", []):
            if vc:
                targets.add(vc)

        for vc in targets:
            try:
                await vc.disconnect(force=True)
            except Exception:
                logger.exception("Failed to disconnect voice client during reset.")

        self.voice_client = None
        self.voice_channel = None
        self._recording_active = False
        self._active_sink = None

    def _on_record_complete(self, sink, *args):
        """
        Called by discord.py when a recording segment completes.
        Writes PCM to pipe if configured and WAV chunks to buffer dir.
        """
        self.loop.create_task(self._after_record_complete(sink))

    async def _after_record_complete(self, sink):
        self._recording_active = False
        self._active_sink = None
        if not sink.audio_data:
            logger.warning("Voice sink produced no audio data for this segment.")
        else:
            await self._persist_audio_segment(sink)

        # Schedule next segment if still connected
        if self.voice_client and self.voice_client.is_connected():
            self._start_recording_segment()

    async def _persist_audio_segment(self, sink):
        """
        Persist the first user's audio to FIFO/buffer dir.
        """
        # pick first user entry
        audio_entry = next(iter(sink.audio_data.values()))
        try:
            audio_entry.file.seek(0)
        except Exception:
            pass
        pcm_bytes = audio_entry.file.read()

        # Write to FIFO/pipe if configured
        if self.voice_pipe_path and self.voice_pipe_path.exists():
            try:
                mode = self.voice_pipe_path.stat().st_mode
                is_fifo = (mode & 0o170000) == 0o010000  # stat.S_IFIFO
                if not is_fifo:
                    logger.warning("Configured voice_pipe_path is not a FIFO: %s", self.voice_pipe_path)
                fd = os.open(self.voice_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    os.write(fd, pcm_bytes)
                finally:
                    os.close(fd)
            except Exception:
                logger.exception("Failed to write Discord voice segment to pipe %s", self.voice_pipe_path)

        # Always write to buffer directory as WAV-like raw bytes (named .pcm)
        ts = datetime.now(timezone.utc).isoformat().replace(":", "_")
        ext = ".wav" if sinks and isinstance(sink, getattr(sinks, "WaveSink", ())) else ".pcm"
        out_path = self.voice_buffer_dir / f"{self.voice_label}_{ts}{ext}"
        try:
            out_path.write_bytes(pcm_bytes)
            logger.info("Discord voice segment saved to %s (%d bytes)", out_path, len(pcm_bytes))
        except Exception:
            logger.exception("Failed to persist Discord voice segment to %s", out_path)


# ---------------------------------------------------------------------------
# Bot startup
# ---------------------------------------------------------------------------

def get_discord_token() -> str:
    """
    Load the Discord bot token from either:
    - config.json -> discord.bot_token
    - environment variable DISCORD_BOT_TOKEN
    - secrets.json file in the working directory: {"DISCORD_BOT_TOKEN": "..."}
    """
    cfg = get_discord_config()
    token = cfg.get("bot_token") if cfg else None
    if not token:
        token = load_secret("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Discord token not found. Set discord.bot_token in config.json, set "
            "DISCORD_BOT_TOKEN in the environment, or create a secrets.json file "
            "with {'DISCORD_BOT_TOKEN': '...'}"
        )
    return token


def main() -> None:
    # Python 3.12+ does not create a default event loop; set one explicitly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    _install_voice_debug_hooks()
    # Create CommsCore with our custom process_inbound hook
    comms = CommsCore(
        instance_name=INA_INSTANCE_NAME,
        process_inbound=process_inbound_message,
        raw_fallback=_log_raw_outbound,
    )

    # Create Discord client
    client = InaDiscordClient(comms=comms, loop=loop)

    # Register Discord backend with CommsCore so outbound messages work
    register_discord_backend(comms, client, backend_name=BACKEND_NAME)

    log_discord_voice_capabilities()

    token = get_discord_token()

    # Run the Discord client
    client.run(token)


if __name__ == "__main__":
    main()
