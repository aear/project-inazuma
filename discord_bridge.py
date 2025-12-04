# discord_bridge.py

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import discord

from comms_core import CommsCore, CommsResponse, load_secret
from backend_discord import (
    make_sender_info_from_discord,
    make_channel_info_from_discord,
    register_discord_backend,
)

# ---------------------------------------------------------------------------
# Basic logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bridge")


# ---------------------------------------------------------------------------
# Config – change these for your setup
# ---------------------------------------------------------------------------

# TODO: replace this with your actual Discord user ID (integer).
# You can get it by enabling Developer Mode in Discord and right-clicking yourself → Copy ID.
SAKURA_USER_ID = 123456789012345678  # <-- PUT YOUR DISCORD USER ID HERE

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

    For now, this is just a thin wrapper around the default comms_core behaviour
    (or you can implement your own here).

    Example later:
        return model_manager.handle_inbound_comms(msg)

    Right now, we'll just use a very simple "echo but tagged" behaviour.
    """
    # Simple example: tag DMs from Sakura differently if you want.
    base_text = msg.text

    # You can add fancy routing here later:
    # - send to model_manager
    # - check for /commands
    # - introspection commands (/status, /energy, etc.)
    reply_text = f"{INA_INSTANCE_NAME}: {base_text}"

    return CommsResponse(
        text=reply_text,
        metadata={
            "source": "discord_bridge.process_inbound_message",
            "debug": True,
        },
    )


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

class InaDiscordClient(discord.Client):
    """
    Discord client that connects DMs from Sakura to Ina via CommsCore.

    DM-only, single-user:
    - Ignores all messages in guilds (servers)
    - Ignores all messages not from SAKURA_USER_ID
    """

    def __init__(self, comms: CommsCore, *args, **kwargs) -> None:
        intents = kwargs.pop("intents", None)
        if intents is None:
            intents = discord.Intents.default()
            intents.messages = True
            intents.message_content = True  # REQUIRED to read message content
            intents.dm_messages = True

        super().__init__(intents=intents, *args, **kwargs)
        self.comms = comms

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user and self.user.id)
        logger.info("DM-only bridge is active. Awaiting messages from Sakura (%s).", SAKURA_USER_ID)

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages from ourselves or other bots
        if message.author.bot:
            return

        # We only care about DMs, not guild channels
        if message.guild is not None:
            return

        # Single-user sandbox: only accept messages from Sakura
        if message.author.id != SAKURA_USER_ID:
            logger.info(
                "Ignoring DM from non-Sakura user %s (%s)",
                message.author,
                message.author.id,
            )
            return

        logger.info(
            "Inbound DM from Sakura: %s (channel %s)",
            message.content,
            message.channel.id,
        )

        sender = make_sender_info_from_discord(message, backend_name=BACKEND_NAME)
        channel = make_channel_info_from_discord(message, backend_name=BACKEND_NAME)

        # Hand this into Ina via CommsCore
        # This will synchronously run the processing pipeline and,
        # if a response is generated, CommsCore will trigger the outbound
        # path which sends a DM back using the registered backend.
        self.comms.receive_inbound(
            backend=BACKEND_NAME,
            backend_message_id=str(message.id),
            sender=sender,
            channel=channel,
            text=message.content,
            reply_to_backend_id=str(message.id),
            metadata={
                "discord_author_id": str(message.author.id),
                "discord_channel_id": str(message.channel.id),
                "is_dm": True,
            },
        )


# ---------------------------------------------------------------------------
# Bot startup
# ---------------------------------------------------------------------------

def get_discord_token() -> str:
    """
    Load the Discord bot token from either:
    - environment variable DISCORD_BOT_TOKEN, or
    - secrets.json file in the working directory: {"DISCORD_BOT_TOKEN": "..."}
    """
    token = load_secret("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Discord token not found. Set DISCORD_BOT_TOKEN in the environment "
            "or create a secrets.json file with {'DISCORD_BOT_TOKEN': '...'}"
        )
    return token


def main() -> None:
    # Create CommsCore with our custom process_inbound hook
    comms = CommsCore(
        instance_name=INA_INSTANCE_NAME,
        process_inbound=process_inbound_message,
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
