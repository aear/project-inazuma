
import subprocess
import time
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from gui_hook import log_to_statusbox
from model_manager import load_config, get_sweet_spots
from audio_digest import analyze_audio_clip, generate_fragment
from fragmentation_engine import fragment_device_log
from transformers.fractal_multidimensional_transformers import FractalTransformer


LABELS_DEFAULT = ["mic_headset", "mic_webcam", "output_headset", "output_TV"]
FLUSH_INTERVAL = 60  # seconds
FRAGMENT_INTERVAL = 3600  # seconds

transformer = FractalTransformer()
config = load_config()
child = config.get("current_child", "default_child")
save_path = Path("AI_Children") / child / "memory" / "audio_session"
save_path.mkdir(parents=True, exist_ok=True)

# Allow the user to slim down or reroute capture labels (e.g., to an OBS mix).
LABELS = config.get("audio_labels") or LABELS_DEFAULT
device_overrides = config.get("audio_device_overrides", {})
stereo_labels = set(config.get("stereo_audio_labels", []))

# === Global buffer tracking ===
audio_buffers = {label: [] for label in LABELS}
last_flush = {label: time.time() for label in LABELS}
last_fragmentation = time.time()

# Preferred index keys per label (includes old misspellings for outputs)
INDEX_KEYS = {
    "mic_headset": ["mic_headset_index"],
    "mic_webcam": ["mic_webcam_index"],
    "output_headset": ["output_headset_index", "ouput_headset_index"],
    "output_TV": ["output_TV_index", "ouput_TV_index"],
}


def _safe_unlink(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _transcode_wav_to_opus(wav_path: Path, opus_path: Path, channels: int) -> Path:
    if not wav_path.exists():
        return wav_path

    transcode_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-c:a",
        "libopus",
        "-b:a",
        "96k",
        "-ar",
        "48000",
        "-ac",
        str(channels),
        str(opus_path),
    ]

    try:
        subprocess.run(transcode_cmd, check=True)
        _safe_unlink(wav_path)
        return opus_path
    except subprocess.CalledProcessError as exc:
        log_to_statusbox(f"[Audio] Opus transcode failed for {wav_path.name}: {exc}")
    except FileNotFoundError as exc:
        log_to_statusbox(f"[Audio] ffmpeg missing during transcode: {exc}")
    return wav_path


def _index_for_label(label):
    for key in INDEX_KEYS.get(label, []):
        raw = config.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None

def resolve_plughw_device(label_name, fallback="default"):
    """
    Resolve device via aplay when available; otherwise use provided fallback.
    """
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        output = result.stdout
        for line in output.splitlines():
            if label_name.lower() in line.lower():
                card_match = re.search(r"card (\d+):", line)
                dev_match = re.search(r"device (\d+):", line)
                if card_match and dev_match:
                    card = card_match.group(1)
                    device = dev_match.group(1)
                    resolved = f"plughw:{card},{device}"
                    log_to_statusbox(f"[Audio] Resolved {label_name} to {resolved}")
                    return resolved
    except FileNotFoundError:
        log_to_statusbox("[Audio] aplay not found; using configured indices.")
    except Exception as e:
        log_to_statusbox(f"[Audio] Failed to resolve {label_name}: {e}")
    return fallback

def record_clip(label, device_string):
    # Prefer plughw for automatic format conversion to avoid channel errors
    if device_string.startswith("hw:"):
        device_string = "plughw:" + device_string.split(":", 1)[1]

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
    base_name = f"{label}_{timestamp}"
    wav_path = save_path / f"{base_name}.wav"
    opus_path = save_path / f"{base_name}.opus"
    channels = 2 if ("output" in label or label in stereo_labels) else 1
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-fflags", "+genpts",
        "-f", "alsa",
        "-use_wallclock_as_timestamps", "1",
        "-i", device_string,
        "-t", str(FLUSH_INTERVAL),
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", str(channels),
        str(wav_path)
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        clip_path = _transcode_wav_to_opus(wav_path, opus_path, channels)
        log_to_statusbox(f"[Audio] {label} saved {clip_path.name}")
        return clip_path
    except subprocess.CalledProcessError as e:
        log_to_statusbox(f"[Audio] {label} failed on {device_string}: {e}")
        _safe_unlink(wav_path)
        # Retry once with default if we were using plughw
        if device_string != "default":
            fallback_cmd = list(ffmpeg_cmd)
            fallback_cmd[fallback_cmd.index(device_string)] = "default"
            try:
                subprocess.run(fallback_cmd, check=True)
                clip_path = _transcode_wav_to_opus(wav_path, opus_path, channels)
                log_to_statusbox(
                    f"[Audio] {label} saved {clip_path.name} via default device fallback."
                )
                return clip_path
            except subprocess.CalledProcessError as e2:
                log_to_statusbox(f"[Audio] {label} fallback failed: {e2}")
                _safe_unlink(wav_path)
        return None


def append_session_log(label, clip_path, duration):
    """Append metadata about the captured clip to the device log."""
    log_file = save_path / f"{label}_log.json"
    entry = {
        "path": str(clip_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration": duration,
    }
    try:
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        log.append(entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        log_to_statusbox(f"[Audio] Failed to append session log for {label}: {e}")


def flush_and_digest(label, device_string):
    clip_path = record_clip(label, device_string)
    if not clip_path or not clip_path.exists():
        return
    try:
        analysis = analyze_audio_clip(clip_path, transformer, child=child, label=label)
        if analysis:
            generate_fragment(clip_path, analysis, child=child, label=label)
            append_session_log(label, clip_path, analysis.get("duration", FLUSH_INTERVAL))
    except Exception as e:
        log_to_statusbox(f"[Audio] Digest failed on {label}: {e}")


def run_audio_loop():
    devices = {}
    for label in LABELS:
        if label in device_overrides:
            devices[label] = device_overrides[label]
            log_to_statusbox(f"[Audio] {label} using override device '{devices[label]}'")
            continue
        name_hint = config.get(f"{label}_name", label)
        idx = _index_for_label(label)
        fallback = f"plughw:{idx},0" if idx is not None else "default"
        devices[label] = resolve_plughw_device(name_hint, fallback=fallback)

    while True:
        now = time.time()
        for label in LABELS:
            if now - last_flush[label] >= FLUSH_INTERVAL:
                flush_and_digest(label, devices[label])
                last_flush[label] = now

        global last_fragmentation
        if now - last_fragmentation >= FRAGMENT_INTERVAL:
            for label in LABELS:
                try:
                    fragment_device_log(label, child=child)
                except Exception as e:
                    log_to_statusbox(f"[Audio] Fragmentation failed for {label}: {e}")
            log_to_statusbox("[Audio] Hourly audio fragmentation complete")
            last_fragmentation = now

        time.sleep(1)


if __name__ == "__main__":
    run_audio_loop()
