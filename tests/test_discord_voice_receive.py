import struct
from types import SimpleNamespace

import discord_voice_receive as receive


class _ExtendedPacket:
    extended = True

    def __init__(self, words):
        self.header = (b"\x00" * 12) + struct.pack(">2sH", b"\xbe\xde", words)
        self.extended_header = None

    def update_extended_header(self, data):
        self.extended_header = data


def test_rtp_extension_benchmark_v1_fixed_eight_vs_v2_declared_length():
    """V2 retains Opus when RFC extension data is not Pycord's fixed 8 bytes."""
    packet = _ExtendedPacket(words=3)
    extension = b"abcdefghijkl"
    opus = b"opus-frame"
    plaintext = extension + opus

    v1_fixed_eight = plaintext[8:]
    v2_declared_length = receive.strip_rtp_extension_payload(packet, plaintext)

    assert v1_fixed_eight != opus
    assert v2_declared_length == opus
    assert packet.extended_header == extension


def test_rtp_extension_rejects_truncated_declared_data():
    packet = _ExtendedPacket(words=3)
    try:
        receive.strip_rtp_extension_payload(packet, b"too-short")
    except ValueError as exc:
        assert str(exc) == "truncated_rtp_extension"
    else:
        raise AssertionError("truncated extension should fail closed")


def test_sequence_window_reorders_and_accounts_for_bounded_gap():
    window = receive.SequenceWindow(max_pending=3)
    assert window.push(10, "a") == (["a"], 0, False)
    assert window.push(12, "c") == ([], 0, False)
    assert window.push(11, "b") == (["b", "c"], 0, False)
    assert window.push(11, "late") == ([], 0, True)

    gap = receive.SequenceWindow(max_pending=2)
    assert gap.push(20, "a") == (["a"], 0, False)
    assert gap.push(22, "c") == ([], 0, False)
    assert gap.push(23, "d") == (["c", "d"], 1, False)


class _Connection:
    def __init__(self):
        self.listener = None
        self.ssrc_user_map = {7: 42}
        self.dave_session = None

    def add_socket_listener(self, callback):
        self.listener = callback

    def remove_socket_listener(self, callback):
        assert self.listener == callback
        self.listener = None


def test_receiver_lifecycle_is_bounded_and_inspectable(monkeypatch):
    class FakeDecoder:
        def decode(self, opus, fec=False):
            assert opus == b"opus"
            assert fec is False
            return b"pcm"

    connection = _Connection()
    voice_client = SimpleNamespace(
        _connection=connection,
        mode="aead_xchacha20_poly1305_rtpsize",
        secret_key=b"k" * 32,
    )
    monkeypatch.setattr(receive, "Decoder", FakeDecoder)
    receiver = receive.InaDiscordVoiceReceiver(voice_client, max_pcm_bytes_per_speaker=192_000)
    receiver.start()
    assert connection.listener is not None

    receiver._decoders[7] = FakeDecoder()
    receiver._sequences[7] = receive.SequenceWindow()
    receiver._decode_ready(7, 42, b"opus")
    snapshot = receiver.snapshot()

    assert snapshot.audio_data[42].file.read() == b"pcm"
    assert snapshot.metrics["backend"] == "ina_v1"
    assert snapshot.metrics["running"] is True
    assert snapshot.metrics["pcm_frames"] == 1
    receiver.stop()
    assert connection.listener is None


def test_receiver_fails_closed_for_unnegotiated_transport_mode():
    receiver = receive.InaDiscordVoiceReceiver(SimpleNamespace(
        _connection=_Connection(), mode="unknown", secret_key=b"k" * 32,
    ))
    try:
        receiver.start()
    except RuntimeError as exc:
        assert str(exc) == "unsupported_voice_transport_mode:unknown"
    else:
        raise AssertionError("unknown Discord transport mode must not be guessed")
