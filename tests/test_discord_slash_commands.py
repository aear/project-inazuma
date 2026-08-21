import discord_bridge as bridge
from types import SimpleNamespace


class _FakeClient:
    def __init__(self):
        self.commands = []

    def add_application_command(self, command):
        self.commands.append(command)


def test_ina_slash_commands_mirror_existing_text_commands(monkeypatch):
    monkeypatch.setattr(bridge, "get_discord_config", lambda: {})
    client = _FakeClient()
    root = bridge.register_ina_application_commands(client)

    assert client.commands == [root]
    assert root.name == "ina"
    root_names = {command.name for command in root.subcommands}
    assert root_names == {"status", "join", "leave", "learn"}
    learn = next(command for command in root.subcommands if command.name == "learn")
    assert [command.name for command in learn.subcommands] == ["history"]


def test_ina_slash_commands_accept_optional_fast_guild_registration(monkeypatch):
    monkeypatch.setattr(
        bridge, "get_discord_config", lambda: {"slash_command_guild_ids": ["123", "bad"]}
    )
    root = bridge.register_ina_application_commands(_FakeClient())
    assert root.guild_ids == [123]


def test_discord_space_identity_distinguishes_duplicate_channel_names():
    troubled = SimpleNamespace(
        guild=SimpleNamespace(id=1, name="The Troubled Family"),
        channel=SimpleNamespace(id=11, name="ina-text"),
    )
    umani = SimpleNamespace(
        guild=SimpleNamespace(id=2, name="Umani RP"),
        channel=SimpleNamespace(id=22, name="ina-text"),
    )
    first = bridge.discord_space_identity(troubled)
    second = bridge.discord_space_identity(umani, roleplay_mode="respond")
    assert first["identity"] == "discord:1:11"
    assert second["identity"] == "discord:2:22"
    assert first["label"] == "The Troubled Family / #ina-text"
    assert second["label"] == "Umani RP / #ina-text"
    assert second["conversation_mode"] == "roleplay"


def test_autonomous_voice_entry_benchmark_v1_speech_threshold_vs_v2_social_gate():
    """V2 requires a distinct high Discord urge plus identity and presence gates."""
    cfg = {
        "voice_channel_id": "44",
        "autonomous_voice_join": {
            "enabled": True,
            "min_urge": 0.8,
            "require_trusted_presence": True,
            "cooldown_seconds": 900,
        },
    }
    v1_local_speech_threshold = 0.25
    assert 0.7 >= v1_local_speech_threshold
    assert bridge.autonomous_voice_join_decision(
        cfg, urge_level=0.7, channel_id="44", trusted_member_present=True, now=1000
    )["reason"] == "urge_below_threshold"
    assert bridge.autonomous_voice_join_decision(
        cfg, urge_level=0.9, channel_id="wrong", trusted_member_present=True, now=1000
    )["reason"] == "channel_not_allowlisted"
    assert bridge.autonomous_voice_join_decision(
        cfg, urge_level=0.9, channel_id="44", trusted_member_present=False, now=1000
    )["reason"] == "no_trusted_person_present"
    assert bridge.autonomous_voice_join_decision(
        cfg, urge_level=0.9, channel_id="44", trusted_member_present=True, now=1000
    )["allowed"] is True
    assert bridge.autonomous_voice_join_decision(
        cfg, urge_level=0.9, channel_id="44", trusted_member_present=True,
        now=1100, last_join_at=1000,
    )["reason"] == "cooldown"
