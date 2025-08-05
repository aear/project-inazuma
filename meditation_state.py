
# === meditation_state.py (Full Rewrite) ===
# Reflection mode — builds maps, performs symbolic introspection, exits on emotional drift

import subprocess
from safe_popen import safe_popen
import time
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from model_manager import (
    mark_module_running, clear_module_running, update_inastate, get_inastate,
    get_sweet_spots, seed_self_question, load_config
)
from gui_hook import log_to_statusbox


def save_log(entry):
    child = load_config().get("current_child", "default_child")
    log_path = Path("AI_Children") / child / "memory" / "meditation_log.json"
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log[-100:], f, indent=2)

def enter_meditation():
    child = load_config().get("current_child", "default_child")
    update_inastate("meditating", True)
    update_inastate("last_meditation_time", datetime.now(timezone.utc).isoformat())
    log_to_statusbox("[Meditation] Entering meditation mode.")
    state_path = Path("AI_Children") / child / "memory" / "session_state.json"
    state = {}
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except:
            pass
    state["meditation_started_at"] = datetime.now(timezone.utc).isoformat()
    with open(state_path, "w") as f:
        json.dump(state, f, indent=4)

def exit_meditation(reason="natural"):
    update_inastate("meditating", False)
    clear_module_running("meditation_state")
    log_to_statusbox(f"[Meditation] Exiting meditation: {reason}")


def meditate_loop():
    enter_meditation()
    loop_count = 0
    max_loops = 10

    while loop_count < max_loops:
        loop_count += 1
        log_to_statusbox(f"[Meditation] Reflective cycle {loop_count}/{max_loops}")
        try:
            subprocess.run(["python", "emotion_engine.py"], check=False)
            subprocess.run(["python", "who_am_i.py"], check=False)
            subprocess.run(["python", "memory_graph.py"], check=False)
            subprocess.run(["python", "meaning_map.py"], check=False)
            subprocess.run(["python", "logic_map_builder.py"], check=False)
            subprocess.run(["python", "emotion_map.py"], check=False)
            subprocess.run(["python", "predictive_layer.py"], check=False)
            log_to_statusbox("[Meditation] Ran core reflection modules.")
        except Exception as e:
            log_to_statusbox(f"[Meditation] Subprocess error: {e}")

        emo = get_inastate("current_emotions") or {}
        log_to_statusbox(f"[Meditation] Emotional snapshot: {json.dumps(emo, indent=2)}")
        fuzz = emo.get("fuzz_level", 0.0)
        stress = emo.get("stress", 0.0)
        negativity = emo.get("negativity", 0.0)
        intensity = emo.get("intensity", 0.0)

        spots = get_sweet_spots()
        if intensity > spots.get("speech_rate", {}).get("max", 1.2):
            seed_self_question("Why does my thinking feel intense?")
            log_to_statusbox("[Meditation] Seeded question: thinking intensity.")

        if stress > spots.get("cpu_temperature", {}).get("max", 0.7):
            seed_self_question("Why am I stressed during meditation?")
            log_to_statusbox("[Meditation] Seeded question: thinking intensity.")
        if negativity > 0.5:
            exit_meditation("negativity spike")
            return
        if fuzz > 0.7 and stress > 0.5:
            exit_meditation("transition to dream")
            safe_popen(["python", "dreamstate.py"])
            return

        save_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loop": loop_count,
            "emotions": emo
        })
        log_to_statusbox(f"[Meditation] Logged cycle {loop_count} to meditation_log.json")


        time.sleep(30)

    exit_meditation("completed")

if __name__ == "__main__":
    meditate_loop()
