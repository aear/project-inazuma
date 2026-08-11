# backend_discord.py

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import discord

from comms_core import CommsCore, SenderInfo, ChannelInfo, CommsMessage

logger = logging.getLogger(__name__)


def make_sender_info_from_discord(
    message: discord.Message,
    *,
    backend_name: str = "discord",
    self_user_id: object = None,
) -> SenderInfo:
    """Map a discord.Message author into a SenderInfo."""
    author = message.author

    return SenderInfo(
        internal_id=str(author.id),               # you can remap this later if you want
        backend_id=str(author.id),
        display_name=author.display_name,
        # A proxy or another bot is still an external speaker.  Only Ina's
        # actual Discord account is self-authored.
        is_self=(
            self_user_id is not None
            and str(author.id) == str(self_user_id)
        ),
        backend=backend_name,
    )


def make_channel_info_from_discord(
    message: discord.Message,
    *,
    backend_name: str = "discord",
) -> ChannelInfo:
    """Map a discord.Message channel (DM in our case) into a ChannelInfo."""

    channel = message.channel
    is_private = isinstance(channel, (discord.DMChannel, discord.GroupChannel))

    # For DMs this will usually not have a meaningful .name, so we fall back.
    name = getattr(channel, "name", None) or "dm"

    return ChannelInfo(
        internal_id=str(channel.id),
        backend_id=str(channel.id),
        name=name,
        is_private=is_private,
        backend=backend_name,
    )


def register_discord_backend(
    comms: CommsCore,
    client: discord.Client,
    *,
    backend_name: str = "discord",
    reply_to_trigger_message: bool = True,
) -> None:
    """
    Register a Discord backend with the given CommsCore instance.

    This function creates a `send_fn` that CommsCore can call whenever Ina
    wants to send an outbound message via Discord. The function schedules
    the actual Discord API call on the client's event loop.
    """

    def _get_loop():
        # discord.py 2.x discourages accessing .loop directly; use internal connection or running loop.
        conn = getattr(client, "_connection", None)
        loop = getattr(conn, "loop", None)
        if loop:
            return loop
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def send_fn(msg: CommsMessage) -> None:
        """Send an outbound CommsMessage using discord.py."""

        async def _send_async() -> None:
            channel = client.get_channel(int(msg.channel.backend_id))
            if channel is None:
                try:
                    # For DMs, get_channel can be None if we haven't seen it yet;
                    # fetch_channel may succeed if bot has access.
                    channel = await client.fetch_channel(int(msg.channel.backend_id))  # type: ignore
                except Exception:
                    logger.exception(
                        "Failed to fetch Discord channel %s for message %s",
                        msg.channel.backend_id,
                        msg.id,
                    )
                    return

            try:
                sender = getattr(client, "send_discord_message", None)
                metadata = msg.metadata or {}
                attachment_path = metadata.get("attachment_path")
                reply_to_message_id = (
                    metadata.get("reply_to_backend_id")
                    if reply_to_trigger_message and metadata.get("discord_native_reply", True) is not False
                    else None
                )

                def _build_file():
                    if not attachment_path:
                        return None
                    path = Path(str(attachment_path))
                    if not path.is_file():
                        logger.warning(
                            "Attachment missing for Discord message %s: %s",
                            msg.id,
                            attachment_path,
                        )
                        return None
                    return discord.File(str(path), filename=path.name)

                if sender:
                    await sender(
                        channel,
                        msg.text,
                        file_factory=_build_file if attachment_path else None,
                        reason=f"comms:{msg.id}",
                        reply_to_message_id=reply_to_message_id,
                    )
                else:
                    send_kwargs = {"file": _build_file()} if attachment_path else {}
                    get_partial_message = getattr(channel, "get_partial_message", None)
                    if reply_to_message_id and callable(get_partial_message):
                        try:
                            send_kwargs.update(
                                reference=get_partial_message(int(reply_to_message_id)),
                                mention_author=False,
                            )
                        except (TypeError, ValueError):
                            logger.warning(
                                "Invalid Discord reply target for message %s: %r",
                                msg.id,
                                reply_to_message_id,
                            )
                    await channel.send(msg.text, **send_kwargs)
            except Exception:
                logger.exception(
                    "Failed to send message %s to Discord channel %s",
                    msg.id,
                    msg.channel.backend_id,
                )

        # Schedule the send on the Discord event loop
        loop = _get_loop()
        if not loop:
            logger.error("No Discord event loop available; dropping outbound message %s", msg.id)
            return
        asyncio.run_coroutine_threadsafe(_send_async(), loop)

    comms.register_backend(backend_name, send_fn)
    logger.info("Discord backend registered with CommsCore as '%s'", backend_name)
