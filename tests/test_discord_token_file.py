import os

from discord_runtime import resolve_discord_token


def test_discord_token_resolution_prefers_environment_then_file(tmp_path, monkeypatch):
    token_file = tmp_path / "discord.token"
    token_file.write_text("from-file\n", encoding="utf-8")
    cfg = {"discord": {"token_env": "INA_TEST_DISCORD_TOKEN", "token_file": str(token_file), "bot_token": "legacy"}}
    monkeypatch.delenv("INA_TEST_DISCORD_TOKEN", raising=False)
    assert resolve_discord_token(cfg) == "from-file"
    monkeypatch.setenv("INA_TEST_DISCORD_TOKEN", "from-env")
    assert resolve_discord_token(cfg) == "from-env"


def test_discord_token_file_is_documented_and_real_file_is_ignored():
    assert os.path.exists(".secrets/discord_bot_token.example")
    readme = open(".secrets/README.md", encoding="utf-8").read()
    assert '"token_file": ".secrets/discord_bot_token"' in readme
