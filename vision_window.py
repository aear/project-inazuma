
# === vision_window.py (Full Rewrite) ===
# Captures frames from both webcam and screen
# Compares successive frames to detect motion/gesture changes
# Stores symbolic fragments based on visual delta events

import cv2
import numpy as np
import time
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from model_manager import load_config
from gui_hook import log_to_statusbox
from optic_nerve import DesktopOpticNerve
from obs_bridge import OBSWebSocketBridge

try:  # Optional fallback if mss is unavailable
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None


FRAME_INTERVAL = 5  # seconds between frame captures
DELTA_THRESHOLD = 50  # pixel intensity difference threshold
webcam_buffer = []
last_video_flush = time.time()
last_digest_time = time.time()
VIDEO_DURATION = 60  # seconds
optic_nerve = DesktopOpticNerve()
obs_bridge = None


def capture_webcam_frame(device_index=0):
    cap = cv2.VideoCapture(device_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def capture_display_frame():
    if obs_bridge:
        frame = obs_bridge.capture_frame()
        if frame is not None:
            return frame

    frame = optic_nerve.capture_frame() if optic_nerve else None
    if frame is not None:
        return frame
    if pyautogui is None:
        return None
    try:
        image = pyautogui.screenshot()
        frame = np.array(image)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        return frame
    except Exception:
        return None

def compute_delta(prev, current):
    delta = cv2.absdiff(prev, current)
    score = np.sum(delta) / delta.size
    return score

def save_frame(frame, source, label, child):
    timestamp = datetime.now(timezone.utc).isoformat()
    path = Path("AI_Children") / child / "memory" / "vision_session"
    path.mkdir(parents=True, exist_ok=True)
    fname = f"{label}_{timestamp.replace(':', '_')}.jpg"
    cv2.imwrite(str(path / fname), frame)
    return fname, timestamp

def video_flush(child):
    global webcam_buffer, last_video_flush, last_digest_time
    if not webcam_buffer:
        return

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
    video_dir = Path("AI_Children") / child / "memory" / "vision_session"
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"webcam_buffer_{timestamp}.mp4"

    height, width = webcam_buffer[0].shape[:2]
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame in webcam_buffer:
        out.write(frame)
    out.release()

    print(f"[Vision] Flushed {len(webcam_buffer)} webcam frames to: {out_path.name}")
    log_to_statusbox(f"[Vision] Video saved: {out_path.name}")

    # Symbolic fragment from video
    flat = webcam_buffer[-1].flatten().tolist()
    frag = {
        "id": f"frag_video_{int(time.time())}",
        "summary": "webcam motion sequence",
        "tags": ["vision", "symbolic", "video"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "vision_window",
        "modality": "video",
        "clarity": 0.45,
        "emotions": {"novelty": 0.4, "focus": 0.35},
        "video_features": flat[:512]
    }
    frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag['id']}.json"
    with open(frag_path, "w") as f:
        json.dump(frag, f, indent=4)

    webcam_buffer.clear()
    last_video_flush = time.time()
    # Flush vision digest once an hour
    if time.time() - last_digest_time >= 3600:
        log_to_statusbox("[Vision] Running vision_digest cleanup...")
        subprocess.call(["python", "vision_digest.py"])
        last_digest_time = time.time()



def log_symbolic_vision(child, summary, tags, timestamp, clarity=0.5, image=None):
    frag_id = f"frag_vision_{int(time.time())}"
    frag = {
        "id": frag_id,
        "summary": summary,
        "tags": tags,
        "timestamp": timestamp,
        "source": "vision_window",
        "clarity": clarity,
        "emotions": {"novelty": 0.5, "focus": 0.3}
    }

    if image is not None:
        flat = image.flatten().tolist()
        frag["image_features"] = flat[:512]
        frag["modality"] = "image"

    path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"
    with open(path, "w") as f:
        json.dump(frag, f, indent=4)
    print(f"[Vision] Symbolic fragment saved: {frag_id}")


def vision_loop():
    global last_digest_time, obs_bridge
    config = load_config()
    child = config.get("current_child", "default_child")

    obs_bridge = OBSWebSocketBridge.from_config(
        config.get("obs_websocket"), logger=log_to_statusbox
    )
    if obs_bridge and obs_bridge.is_available:
        obs_record_dir = (config.get("obs_websocket") or {}).get("record_directory")
        if obs_record_dir:
            if obs_bridge.set_record_directory(obs_record_dir):
                log_to_statusbox(f"[Vision] OBS record dir set to {obs_record_dir}")
            else:
                log_to_statusbox(f"[Vision] OBS record dir not set (check path/permissions).")
        log_to_statusbox(
            f"[Vision] OBS WebSocket ready on {obs_bridge.host}:{obs_bridge.port}"
            f" (source={obs_bridge.source or 'program scene'})"
        )

    prev_webcam = capture_webcam_frame()
    prev_screen = capture_display_frame()

    if optic_nerve:
        optic_nerve.ensure_episode()

    try:
        while True:
            time.sleep(FRAME_INTERVAL)

            curr_webcam = capture_webcam_frame()
            if curr_webcam is not None:
                webcam_buffer.append(curr_webcam)

            curr_screen = capture_display_frame()

            if curr_webcam is not None and prev_webcam is not None:
                delta_web = compute_delta(prev_webcam, curr_webcam)
                log_to_statusbox(f"[Vision] Webcam delta: {delta_web:.2f}")

                if delta_web > DELTA_THRESHOLD:
                    fname, ts = save_frame(curr_webcam, curr_webcam, "webcam", child)
                    log_to_statusbox(f"[Vision] Saved frame: {fname}")
                    log_symbolic_vision(child, "gesture detected via webcam", ["symbolic", "vision", "gesture"], ts, image=curr_webcam)

            if curr_screen is not None and prev_screen is not None:
                delta_scr = compute_delta(prev_screen, curr_screen)
                log_to_statusbox(f"[Vision] Screen delta: {delta_scr:.2f}")

                if delta_scr > DELTA_THRESHOLD:
                    fname, ts = save_frame(curr_screen, curr_screen, "screen", child)
                    log_to_statusbox(f"[Vision] Saved frame: {fname}")
                    log_symbolic_vision(child, "transition detected via screen", ["symbolic", "vision", "transition"], ts, image=curr_screen)
                    if optic_nerve:
                        optic_nerve.log_snapshot(curr_screen, delta_scr, timestamp=ts)
                    if obs_bridge and obs_bridge.can_save_replay:
                        obs_bridge.save_replay_buffer()

            # 🧠 Periodic MP4 flush from buffered webcam frames
            if time.time() - last_video_flush >= VIDEO_DURATION:
                video_flush(child)

            # Flush vision digest once an hour
            if time.time() - last_digest_time >= 3600:
                log_to_statusbox("[Vision] Running vision_digest cleanup...")
                subprocess.call(["python", "vision_digest.py"])
                last_digest_time = time.time()

            prev_webcam = curr_webcam
            prev_screen = curr_screen
    finally:
        if optic_nerve:
            optic_nerve.close_episode(result={"status": "stopped", "workspace": "Desktop 1"})


if __name__ == "__main__":
    vision_loop()
