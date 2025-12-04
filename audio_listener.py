
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

LABELS = ["mic_headset", "mic_webcam", "output_headset", "output_TV"]
FLUSH_INTERVAL = 60  # seconds
FRAGMENT_INTERVAL = 3600  # seconds

# === Global buffer tracking ===
audio_buffers = {label: [] for label in LABELS}
last_flush = {label: time.time() for label in LABELS}
last_fragmentation = time.time()

transformer = FractalTransformer()
config = load_config()
child = config.get("current_child", "default_child")
save_path = Path("AI_Children") / child / "memory" / "audio_session"
save_path.mkdir(parents=True, exist_ok=True)

def resolve_plughw_device(label_name, fallback="hw:0,0"):
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
    except Exception as e:
        log_to_statusbox(f"[Audio] Failed to resolve {label_name}: {e}")
    return fallback

def record_clip(label, device_string):
    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
    filename = f"{label}_{timestamp}.mp3"
    output_path = save_path / filename
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "alsa", "-i", device_string,
        "-t", str(FLUSH_INTERVAL),
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "2" if "output" in label else "1",
        str(output_path)
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        log_to_statusbox(f"[Audio] {label} saved {filename}")
        return output_path
    except subprocess.CalledProcessError as e:
        log_to_statusbox(f"[Audio] {label} failed: {e}")
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
    devices = {
        label: resolve_plughw_device(config.get(f"{label}_name", label))
        for label in LABELS
    }

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
