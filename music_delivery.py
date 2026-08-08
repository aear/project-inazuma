"""Persistent, shareable copies of locally rendered music."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time


DEFAULT_OPUS_CONVERSION_TIMEOUT_SECONDS = 180


def ensure_opus_sidecar(
    wav_path: Path | str,
    *,
    timeout_seconds: int = DEFAULT_OPUS_CONVERSION_TIMEOUT_SECONDS,
) -> dict:
    """Create or refresh ``<stem>.opus`` without modifying the source WAV."""
    source = Path(wav_path)
    opus = source.with_suffix(".opus")
    if source.suffix.lower() != ".wav" or not source.is_file():
        return {"status": "rejected", "reason": "source_wav_unavailable"}
    if source.is_symlink() or opus.is_symlink():
        return {"status": "rejected", "reason": "symlink_rejected"}
    try:
        if opus.is_file() and opus.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return {
                "status": "reused", "wav_path": str(source),
                "opus_path": str(opus), "bytes": opus.stat().st_size,
            }
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {"status": "failed", "reason": "ffmpeg_unavailable"}
        # A unique same-folder staging file preserves any usable older copy
        # if conversion is interrupted; os.replace publishes it atomically.
        staged = opus.with_name(
            f".{opus.stem}.{os.getpid()}.{time.time_ns()}.part.opus"
        )
        try:
            completed = subprocess.run(
                [
                    ffmpeg, "-nostdin", "-y", "-loglevel", "error",
                    "-i", str(source), "-vn", "-c:a", "libopus",
                    "-b:a", "128k", "-vbr", "on", str(staged),
                ],
                capture_output=True, text=True,
                timeout=max(1, int(timeout_seconds)), check=False,
            )
            if completed.returncode != 0 or not staged.is_file():
                return {
                    "status": "failed", "reason": "ffmpeg_failed",
                    "returncode": completed.returncode,
                    "stderr": (completed.stderr or "")[-500:],
                }
            os.replace(staged, opus)
        finally:
            staged.unlink(missing_ok=True)
        return {
            "status": "converted", "wav_path": str(source),
            "opus_path": str(opus), "bytes": opus.stat().st_size,
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "ffmpeg_timeout"}
    except (OSError, ValueError) as exc:
        return {
            "status": "failed", "reason": type(exc).__name__,
            "detail": str(exc)[:300],
        }
