"""Optional local TTS for Ina-chosen text. No playback or network access.

This is an external formant voice, not Ina's native voice or a learned vocoder.
The existing symbolic speech path stays available.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import wave

MAX_TEXT_BYTES = 512
MAX_WAV_BYTES = 8 * 1024 * 1024
VOICES = frozenset({"en-gb", "en-us", "de", "fr", "es"})


def render_private_speech(fs, *, text: str, voice: str = "en-gb") -> dict:
    """One finite synthesis into the existing capability-scoped personal drive."""
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("choose 1..512 UTF-8 bytes of text")
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        raise ValueError("speech text contains control characters")
    if voice not in VOICES:
        raise ValueError("choose a supported voice")
    executable = shutil.which("espeak-ng")
    receipt = {"schema": "ina.speech_bridge/V1", "backend": "espeak-ng", "voice": voice,
               "private": True, "played": False, "published": False, "automatic_learning": False,
               "native_voice": False, "intelligibility": "not_assessed",
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if executable is None:
        return {**receipt, "status": "unavailable", "reason": "espeak-ng is not installed"}
    with tempfile.TemporaryDirectory(prefix="ina_speech_") as directory:
        target = Path(directory) / "speech.wav"
        try:
            completed = subprocess.run(
                [executable, "--stdin", "-b", "1", "-D", "-v", voice, "-s", "175", "-w", str(target)],
                input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=12, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**receipt, "status": "unavailable", "reason": type(exc).__name__}
        if completed.returncode or not target.is_file():
            return {**receipt, "status": "unavailable", "reason": "synthesis failed"}
        if not 44 < target.stat().st_size <= MAX_WAV_BYTES:
            raise ValueError("synthesis exceeded WAV size bounds or produced no audio")
        payload = target.read_bytes()
    with wave.open(io.BytesIO(payload), "rb") as audio:
        rate, channels, width, frames = audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()
        if not (8000 <= rate <= 48000 and channels == 1 and width == 2 and 0 < frames <= rate * 45):
            raise ValueError("synthesis produced unsupported or over-budget audio")
        if len(audio.readframes(frames)) != frames * channels * width:
            raise ValueError("synthesis produced truncated audio")
    relative = "Speech/" + uuid.uuid4().hex + ".wav"
    path = fs.write("ina_hdd", relative, payload)
    return {**receipt, "status": "rendered", "path": str(path), "relative_path": relative,
            "sample_rate": rate, "duration_seconds": round(frames / rate, 6),
            "audio_sha256": hashlib.sha256(payload).hexdigest()}
