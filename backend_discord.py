# backend_discord.py

from __future__ import annotations

import logging
from typing import Callable

import discord

from comms_core import CommsCore, SenderInfo, ChannelInfo, CommsMessage

logger = logging.getLogger(__name__)


def make_sender_info_from_discord(
    message: discord.Message,
    *,
    backend_name: str = "discord",
) -> SenderInfo:
    """Map a discord.Message author into a SenderInfo."""
    author = message.author

    return SenderInfo(
        internal_id=str(author.id),               # you can remap this later if you want
        backend_id=str(author.id),
        display_name=author.display_name,
        is_self=author.bot,                      # True for bots (including Ina if she ever has a Discord user)
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
) -> None:
    """
    Register a Discord backend with the given CommsCore instance.

    This function creates a `send_fn` that CommsCore can call whenever Ina
    wants to send an outbound message via Discord. The function schedules
    the actual Discord API call on the client's event loop.
    """

    loop = client.loop

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
                await channel.send(msg.text)
            except Exception:
                logger.exception(
                    "Failed to send message %s to Discord channel %s",
                    msg.id,
                    msg.channel.backend_id,
                )

        # Schedule the send on the Discord event loop
        loop.create_task(_send_async())

    comms.register_backend(backend_name, send_fn)
    logger.info("Discord backend registered with CommsCore as '%s'", backend_name)
