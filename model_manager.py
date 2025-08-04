# === model_manager.py (Final Rewrite + Module Awareness) ===

import os
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from gui_hook import log_to_statusbox

def load_config():
    path = Path("config.json")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

config = load_config()
CHILD = config.get("current_child", "Inazuma_Yagami")
MEMORY_PATH = Path("AI_Children") / CHILD / "memory"

def get_sweet_spots():
    path = MEMORY_PATH / "sweet_spots.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "stress_range": {"max": 0.7},
        "intensity_range": {"max": 0.8},
        "energy_drain_threshold": 0.6,
        "map_rebuild_fuzz": 0.7,
        "map_rebuild_drift": 0.5
    }

def get_inastate(key, default=None):
    path = MEMORY_PATH / "inastate.json"
    if not path.exists():
        return default
    try:
        with open(path, "r") as f:
            return json.load(f).get(key, default)
    except:
        return default


def update_inastate(key, value):
    path = MEMORY_PATH / "inastate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except:
            pass
    state[key] = value
    with open(path, "w") as f:
        json.dump(state, f, indent=4)

def seed_self_question(question):
    path = MEMORY_PATH / "self_questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            data = []
    else:
        data = []
    data.append({
        "question": question,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    with open(path, "w") as f:
        json.dump(data[-100:], f, indent=4)
    log_to_statusbox(f"[Manager] Self-question seeded: {question}")

def launch_background_loops():
    subprocess.Popen(["python", "audio_listener.py"])
    subprocess.Popen(["python", "vision_window.py"])
    log_to_statusbox("[Manager] Background loops launched.")

def monitor_energy():
    emo = get_inastate("emotion_snapshot") or {}
    dreaming = get_inastate("dreaming")
    meditating = get_inastate("meditating")

    stress = emo.get("stress", 0.0)
    intensity = emo.get("intensity", 0.0)

    energy = get_inastate("current_energy") or 0.5

    if dreaming:
        recovery = 0.02 if intensity > 0.5 else 0.04
        energy = min(1.0, energy + recovery)
    elif meditating:
        energy = min(1.0, energy + 0.01)
    else:
        drain = (stress + intensity) / 2
        energy = max(0.0, energy - drain * 0.01)

    update_inastate("current_energy", round(energy, 4))
    log_to_statusbox(f"[Manager] Energy updated: {energy:.4f}")

def feedback_inhibition():
    stress = get_inastate("emotion_stress") or 0.0
    last_stress = get_inastate("previous_stress") or 0.0
    update_inastate("previous_stress", stress)
    if stress > 0.6 and (stress - last_stress) > 0.2:
        log_to_statusbox("[Manager] Logic inhibited due to stress spike.")
        return True
    return False

def boredom_check():
    boredom = get_inastate("emotion_boredom") or 0.0
    if boredom > 0.4:
        subprocess.call(["python", "boredom_state.py"])
        log_to_statusbox("[Manager] Boredom triggered curiosity loop.")

def rebuild_maps_if_needed():
    emo = get_inastate("emotion_snapshot") or {}
    fuzz = emo.get("fuzz_level", 0.0)
    drift = emo.get("symbolic_drift", 0.0)
    if fuzz > 0.7 or drift > 0.5:
        log_to_statusbox("[Manager] Rebuilding maps due to emotional drift.")
        subprocess.call(["python", "memory_graph.py"])
        subprocess.call(["python", "meaning_map.py"])
        subprocess.call(["python", "logic_map_builder.py"])
        subprocess.call(["python", "emotion_map.py"])
        update_inastate("last_map_rebuild", datetime.now(timezone.utc).isoformat())

def run_internal_loop():
    monitor_energy()

    def check_audio_index_change():
        config = load_config()
        state = get_inastate("audio_device_cache") or {}

        current = {
            "mic_headset_index": config.get("mic_headset_index"),
            "mic_webcam_index": config.get("mic_webcam_index"),
            "output_headset_index": config.get("output_headset_index"),
            "output_TV_index": config.get("output_TV_index")
        }

        if current != state:
            update_inastate("audio_device_cache", current)
            log_to_statusbox("[Manager] Detected change in audio config — restarting audio listener.")
            return True

        return False

    if check_audio_index_change():
        subprocess.call(["pkill", "-f", "audio_listener.py"])
        time.sleep(2)  # Let config settle and avoid early InputStream calls
        subprocess.Popen(["python", "audio_listener.py"])



    if get_inastate("emotion_snapshot", {}).get("focus", 0.0) > 0.5:
        subprocess.Popen(["python", "meditation_state.py"])

    if get_inastate("emotion_snapshot", {}).get("fuzz_level", 0.0) > 0.7:
        subprocess.Popen(["python", "dreamstate.py"])

    subprocess.run(["python", "emotion_engine.py"], check=False)
    subprocess.run(["python", "instinct_engine.py"], check=False)
    subprocess.Popen(["python", "early_comm.py"])

    if not feedback_inhibition():
        subprocess.Popen(["python", "predictive_layer.py"])
        subprocess.Popen(["python", "logic_engine.py"])

    boredom_check()
    rebuild_maps_if_needed()

def schedule_runtime():
    log_to_statusbox("[Manager] Starting main runtime loop...")
    while True:
        try:
            run_internal_loop()
            time.sleep(10)
        except Exception as e:
            log_to_statusbox(f"[Manager ERROR] Runtime loop crashed: {e}")
            time.sleep(5)

if __name__ == "__main__":  
    launch_background_loops()
    schedule_runtime()
