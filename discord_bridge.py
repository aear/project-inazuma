# discord_bridge.py

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import discord

from comms_core import CommsCore, CommsResponse, load_secret
from backend_discord import (
    make_sender_info_from_discord,
    make_channel_info_from_discord,
    register_discord_backend,
)
from social_map import get_owner_user_id, is_owner_friend, record_dm_attempt
from language_processing import generate_symbolic_reply_from_text
from live_experience_bridge import LiveExperienceBridge
try:
    from lm_studio_adapter import LMStudioAdapter
except Exception:
    LMStudioAdapter = None  # type: ignore

try:
    from discord import sinks
except Exception:
    sinks = None

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
    child = get_current_child()
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
        voice_cfg = get_voice_io_config()
        self.voice_label = voice_cfg["voice_label"]
        self.voice_pipe_path = Path(voice_cfg["voice_pipe_path"]) if voice_cfg.get("voice_pipe_path") else None
        self.voice_buffer_dir = Path(voice_cfg["voice_buffer_dir"])
        self.voice_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.voice_chunk_seconds = voice_cfg["voice_chunk_seconds"]
        self._recording_active = False
        self._active_sink = None
        self.history_bridge = LiveExperienceBridge(child=get_current_child())

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user and self.user.id)
        logger.info("Discord bridge is active. DMs from owner (%s) + configured text channel will be routed.", SAKURA_USER_ID)
        self.text_channel, self.voice_channel = resolve_configured_channels(self)

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages from ourselves or other bots
        if message.author.bot:
            return

        # DMs stay owner-only
        if message.guild is None:
            is_owner = message.author.id == SAKURA_USER_ID
            owner_friend = is_owner_friend(message.author.id)
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
            if not (is_owner or owner_friend):
                logger.info(
                    "Ignoring DM from untrusted user %s (%s)%s",
                    message.author,
                    message.author.id,
                    " [logged to social_map]" if added else "",
                )
                return
            logger.info(
                "Inbound DM from %s: %s (channel %s)",
                "owner" if is_owner else "trusted friend",
                message.content,
                message.channel.id,
            )
            self._route_to_comms(message, is_dm=True, owner_friend=owner_friend)
            return

        # Guild messages: only handle those in the configured text channel
        if self.text_channel is None or message.channel.id != self.text_channel.id:
            return

        content = (message.content or "").strip()
        lower = content.lower()
        if lower in {"/ina join", "/ina voice", "/ina voice join", "/ina join voice"}:
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
        if target_channel is None and message.author.voice and message.author.voice.channel:
            target_channel = message.author.voice.channel

        if target_channel is None:
            await message.channel.send("No voice channel configured or detected to join.")
            return

        try:
            await self.ensure_voice_connected(target_channel)
            await message.channel.send(f"Joined voice channel: {target_channel.name}")
        except Exception:
            logger.exception("Failed to join voice channel %s", target_channel)
            await message.channel.send(f"Failed to join voice channel: {target_channel.name}")

    def _route_to_comms(self, message: discord.Message, *, is_dm: bool, owner_friend: bool = False) -> None:
        sender = make_sender_info_from_discord(message, backend_name=BACKEND_NAME)
        channel = make_channel_info_from_discord(message, backend_name=BACKEND_NAME)
        metadata = {
            "discord_author_id": str(message.author.id),
            "discord_channel_id": str(message.channel.id),
            "is_dm": is_dm,
            "is_owner_friend": owner_friend,
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

    async def ensure_voice_connected(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """
        Connect or move Ina to the desired voice channel.
        """
        if self.voice_client and self.voice_client.is_connected():
            if self.voice_client.channel and self.voice_client.channel.id == channel.id:
                await self._ensure_voice_capture()
                return self.voice_client
            await self.voice_client.move_to(channel)
        else:
            self.voice_client = await channel.connect(reconnect=True)
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
        if self._recording_active or not self.voice_client or not self.voice_client.is_connected():
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
    # Create CommsCore with our custom process_inbound hook
    comms = CommsCore(
        instance_name=INA_INSTANCE_NAME,
        process_inbound=process_inbound_message,
        raw_fallback=_log_raw_outbound,
    )

    # Create Discord client
    client = InaDiscordClient(comms=comms)

    # Register Discord backend with CommsCore so outbound messages work
    register_discord_backend(comms, client, backend_name=BACKEND_NAME)

    token = get_discord_token()

    # Run the Discord client
    client.run(token)


if __name__ == "__main__":
    main()
