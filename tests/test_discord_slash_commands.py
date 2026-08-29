import discord_bridge as bridge
import asyncio
from types import SimpleNamespace


class _FakeClient:
    def __init__(self):
        self.commands = []

    def add_application_command(self, command):
        self.commands.append(command)


def test_discord_speaking_indicator_benchmark_v1_implicit_vs_v2_explicit(monkeypatch):
    """V2 explicitly brackets playback instead of relying on player internals."""
    states = []

    class Websocket:
        async def speak(self, state):
            states.append(state)

    client = SimpleNamespace(
        child="Inazuma_Yagami",
        voice_client=SimpleNamespace(ws=Websocket()),
        voice_channel=SimpleNamespace(id=44),
    )
    observed = []
    monkeypatch.setattr(bridge, "update_inastate", lambda key, value: observed.append((key, value)))
    monkeypatch.setattr(bridge, "record_voice_cognition", lambda child, event, payload: True)

    assert asyncio.run(bridge.InaDiscordClient._set_discord_speaking(
        client, True, reason="benchmark_playback",
    )) is True
    assert asyncio.run(bridge.InaDiscordClient._set_discord_speaking(
        client, False, reason="benchmark_finished",
    )) is True

    assert states == [bridge.discord.SpeakingState.voice, bridge.discord.SpeakingState.none]
    assert [value["active"] for key, value in observed if key == "discord_voice_speaking"] == [True, False]


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


def test_autonomous_voice_entry_benchmark_v3_empty_room_invitation():
    """V3 permits an empty allowlisted room, but not an untrusted occupied room."""
    cfg = {
        "voice_channel_id": "44",
        "autonomous_voice_join": {
            "enabled": True,
            "min_urge": 0.8,
            "require_trusted_presence": True,
            "allow_empty_channel": True,
            "cooldown_seconds": 900,
        },
    }
    empty = bridge.autonomous_voice_join_decision(
        cfg,
        urge_level=0.9,
        channel_id="44",
        trusted_member_present=False,
        human_member_present=False,
        now=1000,
    )
    assert empty["allowed"] is True
    assert empty["reason"] == "high_urge_empty_room_invitation"

    occupied_by_untrusted_person = bridge.autonomous_voice_join_decision(
        cfg,
        urge_level=0.9,
        channel_id="44",
        trusted_member_present=False,
        human_member_present=True,
        now=1000,
    )
    assert occupied_by_untrusted_person["reason"] == "no_trusted_person_present"


def test_voice_connection_disables_pycord_internal_invalid_session_retry():
    calls = []

    class FakeVoiceClient:
        def is_connected(self):
            return True

    class FakeChannel:
        id = 44
        guild = SimpleNamespace(id=4)

        async def connect(self, **kwargs):
            calls.append(kwargs)
            return FakeVoiceClient()

    async def exercise():
        client = SimpleNamespace(voice_client=None, voice_clients=[], voice_channel=None)

        async def ensure_capture():
            calls.append("capture")

        client._ensure_voice_capture = ensure_capture
        client._guild_voice_client = lambda guild: None
        result = await bridge.InaDiscordClient.ensure_voice_connected(client, FakeChannel())
        assert result.is_connected()

    asyncio.run(exercise())
    assert calls == [{"reconnect": False}, "capture"]


def test_voice_capture_adapter_retains_sink_across_pycord_28_callback(monkeypatch):
    events = []

    class FakeSink:
        audio_data = {}

    class FakeVoiceClient:
        def start_recording(self, sink, callback):
            events.append(("started", sink))
            callback(None)

    class FakeLoop:
        def call_soon_threadsafe(self, callback, *args):
            callback(*args)

        def call_later(self, delay, callback, *args):
            events.append(("timer", delay, callback, args))

    fake_sinks = SimpleNamespace(RawDataSink=FakeSink)
    monkeypatch.setattr(bridge, "sinks", fake_sinks)
    client = SimpleNamespace(
        voice_client=FakeVoiceClient(),
        _active_sink=None,
        _recording_active=False,
        _capture_generation=0,
        _capture_restart_handle=None,
        loop=FakeLoop(),
        voice_chunk_seconds=15,
        _stop_recording_segment=lambda: None,
        _on_record_complete=lambda sink, generation, error: events.append(("complete", sink)),
        _schedule_capture_restart=lambda delay: None,
    )

    bridge.InaDiscordClient._start_recording_segment(client)

    started_sink = events[0][1]
    assert events[0] == ("started", started_sink)
    assert events[1] == ("complete", started_sink)
    assert events[2][:2] == ("timer", 15)


def test_voice_capture_sink_adapts_pycord_28_router_contract():
    voice_client = object()
    sink = bridge._create_voice_capture_sink(bridge.sinks, voice_client)

    assert sink.client is voice_client
    assert sink.__sink_listeners__ == ()
    assert tuple(sink.walk_children()) == ()
    assert isinstance(sink.audio_data, dict)
    assert sink.is_opus() is False


def test_voice_capture_sink_unwraps_pycord_28_voice_data_to_pcm():
    voice_client = object()
    sink = bridge._create_voice_capture_sink(bridge.sinks, voice_client)
    speaker = SimpleNamespace(id=71)

    sink.write(SimpleNamespace(pcm=b"\x01\x00\x02\x00"), speaker)
    sink.write(SimpleNamespace(pcm=b"\x03\x00"), speaker)
    sink.audio_data[71].file.seek(0)

    assert sink.audio_data[71].file.read() == b"\x01\x00\x02\x00\x03\x00"


def test_corrupt_opus_packet_benchmark_v1_router_exit_vs_v2_packet_skip():
    class CorruptPacket(Exception):
        pass

    class Decoder:
        def pop_data(self):
            raise CorruptPacket("corrupted stream")

    assert bridge._install_pycord_opus_resilience(Decoder, CorruptPacket) is True
    assert Decoder().pop_data() is None
    assert Decoder._ina_corrupt_packet_count == 1


def test_invalid_argument_opus_packet_is_local_but_decoder_state_error_is_fatal():
    class OpusFailure(Exception):
        def __init__(self, code, message):
            self.code = code
            super().__init__(message)

    class BadPacketDecoder:
        def pop_data(self):
            raise OpusFailure(-1, "invalid argument")

    class BadStateDecoder:
        def pop_data(self):
            raise OpusFailure(-6, "invalid state")

    bridge._install_pycord_opus_resilience(BadPacketDecoder, OpusFailure)
    bridge._install_pycord_opus_resilience(BadStateDecoder, OpusFailure)

    assert BadPacketDecoder().pop_data() is None
    assert BadPacketDecoder._ina_dropped_packet_reasons == {"invalid_argument": 1}
    try:
        BadStateDecoder().pop_data()
    except OpusFailure as exc:
        assert exc.code == -6
    else:
        raise AssertionError("decoder state failures must remain visible")


def test_stale_capture_callback_cannot_start_competing_segment():
    tasks = []
    current_sink = object()
    stale_sink = object()
    client = SimpleNamespace(
        _capture_generation=4,
        _active_sink=current_sink,
        loop=SimpleNamespace(create_task=lambda task: tasks.append(task)),
    )

    bridge.InaDiscordClient._on_record_complete(
        client, stale_sink, generation=3, error=RuntimeError("old reader"),
    )

    assert tasks == []
