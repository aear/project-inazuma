import discord_bridge as bridge


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
