
import subprocess
import time
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from gui_hook import log_to_statusbox
from model_manager import load_config, get_sweet_spots
from audio_digest import analyze_audio_clip, generate_fragment
from fractal_multidimensional_transformers import FractalTransformer

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


def flush_and_digest(label, device_string):
    clip_path = record_clip(label, device_string)
    if not clip_path or not clip_path.exists():
        return
    try:
        clarity, pitch, tags = analyze_audio_clip(str(clip_path))
        generate_fragment(
            file_path=str(clip_path),
            clarity_input=clarity,
            pitch_info=pitch,
            tags=tags + [label],
            label=label,
            child=child,
            transformer=transformer
        )
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

        # Hourly fragmentation logic (stub)
        global last_fragmentation
        if now - last_fragmentation >= FRAGMENT_INTERVAL:
            log_to_statusbox("[Audio] Hourly audio fragmentation (placeholder)")
            # (Insert symbolic training logic here if needed)
            last_fragmentation = now

        time.sleep(1)


if __name__ == "__main__":
    run_audio_loop()
