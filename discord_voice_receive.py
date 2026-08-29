"""Ina-owned Discord RTP/DAVE/Opus receive backend.

Pycord continues to own gateway and voice connection negotiation. This module
owns bounded media reception after the connected UDP socket is available.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import io
import logging
import struct
import threading
import time
from types import SimpleNamespace
from typing import Any

import davey
from nacl.exceptions import CryptoError
from nacl.secret import Aead

from discord.opus import Decoder, OpusError
from discord.voice.packets.rtp import RTPPacket, decode


logger = logging.getLogger(__name__)
OPUS_SILENCE = b"\xf8\xff\xfe"


def is_rtcp_packet(data: bytes) -> bool:
    return len(data) >= 2 and 200 <= data[1] <= 204


def strip_rtp_extension_payload(packet: RTPPacket, plaintext: bytes) -> bytes:
    """Remove the variable RTP extension values, never a fixed byte count."""
    if not packet.extended:
        return plaintext
    if len(packet.header) < 16:
        raise ValueError("missing_rtp_extension_header")
    _profile, words = struct.unpack(">2sH", packet.header[-4:])
    extension_bytes = int(words) * 4
    if extension_bytes > len(plaintext):
        raise ValueError("truncated_rtp_extension")
    # Populate Pycord's inspectable extension fields while keeping media bytes
    # owned by this backend.
    # rtpsize packets already keep the four-byte extension preamble in
    # packet.header; Pycord's parser prepends it internally.
    packet.update_extended_header(plaintext[:extension_bytes])
    return plaintext[extension_bytes:]


class SequenceWindow:
    """Small per-speaker reorder window with explicit gap accounting."""

    def __init__(self, max_pending: int = 8) -> None:
        self.max_pending = max(2, min(32, int(max_pending)))
        self.expected: int | None = None
        self.pending: dict[int, Any] = {}

    @staticmethod
    def _ahead(sequence: int, expected: int) -> int:
        return (int(sequence) - int(expected)) & 0xFFFF

    def push(self, sequence: int, payload: Any) -> tuple[list[Any], int, bool]:
        sequence = int(sequence) & 0xFFFF
        if self.expected is None:
            self.expected = sequence
        distance = self._ahead(sequence, self.expected)
        if distance >= 0x8000:
            return [], 0, True
        self.pending.setdefault(sequence, payload)
        gap = 0
        if len(self.pending) >= self.max_pending or distance >= self.max_pending:
            nearest = min(self.pending, key=lambda item: self._ahead(item, self.expected))
            gap = self._ahead(nearest, self.expected)
            self.expected = nearest
        ready = []
        while self.expected in self.pending:
            ready.append(self.pending.pop(self.expected))
            self.expected = (self.expected + 1) & 0xFFFF
        return ready, gap, False


@dataclass
class ReceiveSnapshot:
    audio_data: dict[Any, Any]
    metrics: dict[str, Any]


class InaDiscordVoiceReceiver:
    """Continuous bounded receiver attached to an existing Pycord voice socket."""

    def __init__(
        self,
        voice_client: Any,
        *,
        max_speakers: int = 16,
        max_pcm_bytes_per_speaker: int = 8 * 1024 * 1024,
        reorder_packets: int = 8,
    ) -> None:
        self.voice_client = voice_client
        self.connection = voice_client._connection
        self.max_speakers = max(1, min(32, int(max_speakers)))
        self.max_pcm_bytes_per_speaker = max(192_000, int(max_pcm_bytes_per_speaker))
        self.reorder_packets = max(2, min(32, int(reorder_packets)))
        self._lock = threading.RLock()
        self._pcm: dict[int, bytearray] = {}
        self._decoders: dict[int, Decoder] = {}
        self._sequences: dict[int, SequenceWindow] = {}
        self._running = False
        self._box: Aead | None = None
        self._mode = str(getattr(voice_client, "mode", "") or "")
        self._started_at = None
        self._metrics = defaultdict(int)
        self._last_error: str | None = None

    def start(self) -> None:
        if self._running:
            return
        if self._mode != "aead_xchacha20_poly1305_rtpsize":
            raise RuntimeError(f"unsupported_voice_transport_mode:{self._mode or 'missing'}")
        secret = bytes(getattr(self.voice_client, "secret_key", ()) or ())
        if len(secret) != Aead.KEY_SIZE:
            raise RuntimeError("invalid_voice_transport_key")
        self._box = Aead(secret)
        self.connection.add_socket_listener(self._on_packet)
        self._running = True
        self._started_at = time.time()

    def stop(self) -> None:
        if not self._running:
            return
        try:
            self.connection.remove_socket_listener(self._on_packet)
        finally:
            self._running = False

    def _transport_decrypt(self, packet: RTPPacket) -> bytes:
        if self._box is None:
            raise RuntimeError("receiver_not_started")
        packet.adjust_rtpsize()
        nonce = packet.nonce + (b"\x00" * 20)
        plaintext = self._box.decrypt(bytes(packet.data), bytes(packet.header), nonce)
        return strip_rtp_extension_payload(packet, plaintext)

    def _media_decrypt(self, user_id: int, frame: bytes) -> bytes:
        session = getattr(self.connection, "dave_session", None)
        if session is None or not getattr(session, "ready", False):
            return frame
        return bytes(session.decrypt(int(user_id), davey.MediaType.audio, frame))

    def _decoder(self, ssrc: int) -> Decoder | None:
        decoder = self._decoders.get(ssrc)
        if decoder is not None:
            return decoder
        if len(self._decoders) >= self.max_speakers:
            self._metrics["speaker_limit_drops"] += 1
            return None
        decoder = self._decoders[ssrc] = Decoder()
        self._sequences[ssrc] = SequenceWindow(self.reorder_packets)
        return decoder

    def _decode_ready(self, ssrc: int, user_id: int, frame: bytes) -> None:
        decoder = self._decoder(ssrc)
        if decoder is None:
            return
        try:
            opus = self._media_decrypt(user_id, frame)
        except Exception as exc:
            self._metrics["dave_decrypt_drops"] += 1
            self._last_error = f"dave:{type(exc).__name__}"
            return
        try:
            pcm = decoder.decode(opus, fec=False)
        except OpusError as exc:
            self._metrics["opus_decode_drops"] += 1
            self._last_error = f"opus:{getattr(exc, 'code', 'unknown')}"
            return
        if not pcm:
            return
        with self._lock:
            bucket = self._pcm.setdefault(int(user_id), bytearray())
            available = self.max_pcm_bytes_per_speaker - len(bucket)
            if available <= 0:
                self._metrics["pcm_bound_drops"] += len(pcm)
                return
            bucket.extend(pcm[:available])
            if len(pcm) > available:
                self._metrics["pcm_bound_drops"] += len(pcm) - available
            self._metrics["pcm_frames"] += 1
            self._metrics["pcm_bytes"] += min(len(pcm), available)

    def _on_packet(self, packet_data: bytes) -> None:
        self._metrics["udp_packets"] += 1
        if is_rtcp_packet(packet_data):
            self._metrics["rtcp_packets"] += 1
            return
        try:
            packet = decode(packet_data)
            if not isinstance(packet, RTPPacket):
                self._metrics["non_rtp_packets"] += 1
                return
            frame = self._transport_decrypt(packet)
        except (CryptoError, ValueError, struct.error) as exc:
            self._metrics["transport_decrypt_drops"] += 1
            self._last_error = f"transport:{type(exc).__name__}"
            return
        except Exception as exc:
            self._metrics["packet_parse_drops"] += 1
            self._last_error = f"packet:{type(exc).__name__}"
            return
        user_id = getattr(self.connection, "ssrc_user_map", {}).get(packet.ssrc)
        if not user_id:
            self._metrics["unknown_ssrc_drops"] += 1
            return
        decoder = self._decoder(packet.ssrc)
        if decoder is None:
            return
        window = self._sequences[packet.ssrc]
        ready, gap, duplicate = window.push(packet.sequence, (int(user_id), frame))
        if gap:
            self._metrics["sequence_gap_packets"] += gap
        if duplicate:
            self._metrics["late_or_duplicate_packets"] += 1
        for ready_user_id, ready_frame in ready:
            self._decode_ready(packet.ssrc, ready_user_id, ready_frame)

    def metrics(self) -> dict[str, Any]:
        session = getattr(self.connection, "dave_session", None)
        payload = dict(self._metrics)
        payload.update({
            "backend": "ina_v1",
            "running": self._running,
            "transport_mode": self._mode,
            "active_speakers": len(self._decoders),
            "started_at": self._started_at,
            "last_error": self._last_error,
            "dave_ready": bool(session and getattr(session, "ready", False)),
            "dave_epoch": getattr(session, "epoch", None) if session else None,
        })
        if session is not None:
            try:
                stats = session.get_decryption_stats()
                payload["dave_library"] = {
                    name: getattr(stats, name, None)
                    for name in ("attempts", "successes", "failures", "passthroughs", "duration")
                }
            except Exception:
                payload["dave_library"] = {"status": "unavailable"}
        return payload

    def snapshot(self) -> ReceiveSnapshot:
        with self._lock:
            captured = self._pcm
            self._pcm = {}
        audio_data = {}
        for user_id, pcm in captured.items():
            if pcm:
                audio_data[user_id] = SimpleNamespace(file=io.BytesIO(bytes(pcm)))
        return ReceiveSnapshot(audio_data=audio_data, metrics=self.metrics())


__all__ = [
    "InaDiscordVoiceReceiver", "ReceiveSnapshot", "SequenceWindow",
    "is_rtcp_packet", "strip_rtp_extension_payload",
]
