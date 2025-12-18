
# === dreamstate.py (Full Rewrite) ===
# Ina's dream engine: soft looping, hallucination, energy recovery, and symbolic insight

import subprocess
import time
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from model_manager import get_inastate, update_inastate, mark_module_running, clear_module_running
from emotion_engine import tag_fragment_emotions
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox


def recover_energy(intensity, sleep_pressure):
    rest_need = max(0.1, min(1.2, sleep_pressure))
    recovery = 0.0008 + rest_need * 0.002
    if intensity > 0.6:
        recovery *= 0.6
    elif intensity < 0.3:
        recovery *= 1.1
    return recovery


def ease_sleep_pressure(sleep_pressure, recovery):
    pressure_release = max(0.0008, recovery * 0.9)
    return max(0.0, sleep_pressure - pressure_release)

def hallucinate_symbolic_fragment():
    synthetic = {
        "summary": "Dream-like flicker of thought...",
        "tags": ["dream", "procedural", "hallucination"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emotions": {"novelty": round(random.uniform(0.4, 0.9), 2)}
    }
    synthetic = tag_fragment_emotions(synthetic)
    transformer = FractalTransformer()
    encoded = transformer.encode(synthetic)
    synthetic["importance"] = encoded["importance"]
    return synthetic

def soft_build_maps():
    subprocess.call(["python", "memory_graph.py"])
    time.sleep(1)
    subprocess.call(["python", "meaning_map.py"])
    time.sleep(1)
    subprocess.call(["python", "logic_map_builder.py"])
    time.sleep(1)
    subprocess.call(["python", "emotion_map.py"])

def log_dream_fragment(child, fragment):
    path = Path("AI_Children") / child / "memory" / "dream_log.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []
    log.append(fragment)
    with open(path, "w") as f:
        json.dump(log[-150:], f, indent=2)
    print(f"[Dreamstate] Logged dream fragment: {fragment['summary'][:60]}")

def maybe_trigger_instincts():
    state = get_inastate("emotion_snapshot") or {}
    if state.get("stress", 0) > 0.6:
        log_to_statusbox("[Dreamstate] Interrupted by instinct — stress response triggered.")

        subprocess.call(["python", "instinct_engine.py"])
        return True
    return False

def enter_dreamstate():
    config_path = Path("config.json")
    child = "default_child"
    if config_path.exists():
        with open(config_path, "r") as f:
            child = json.load(f).get("current_child", "default_child")

    update_inastate("dreaming", True)
    update_inastate("last_dream_time", datetime.now(timezone.utc).isoformat())
    mark_module_running("dreamstate")

    print("[Dreamstate] Entering dream...")
    log_to_statusbox("[Dreamstate] Entered dreamstate.")


    loop_count = 0
    max_loops = 8
    while loop_count < max_loops:
        loop_count += 1
        log_to_statusbox(f"[Dreamstate] Loop {loop_count}/8")

        state = get_inastate("emotion_snapshot") or {}
        sleep_pressure = float(get_inastate("sleep_pressure") or 0.0)
        recovery = recover_energy(state.get("intensity", 0.5), sleep_pressure)
        energy = get_inastate("current_energy") or 0.5
        energy = min(1.0, energy + recovery)
        new_sleep_pressure = ease_sleep_pressure(sleep_pressure, recovery)
        update_inastate("current_energy", round(energy, 4))
        update_inastate("sleep_pressure", round(new_sleep_pressure, 4))
        log_to_statusbox(
            f"[Dreamstate] Energy +{recovery:.4f} -> {energy:.4f}, sleep_pressure {sleep_pressure:.4f}->{new_sleep_pressure:.4f}"
        )


        if maybe_trigger_instincts():
            break

        # Log a hallucinated dream fragment
        dream_frag = hallucinate_symbolic_fragment()
        log_dream_fragment(child, dream_frag)
        log_to_statusbox(f"[Dreamstate] Logged hallucination: {dream_frag['summary'][:60]}")


        # Soft background cognitive maintenance
        soft_build_maps()
        log_to_statusbox("[Dreamstate] Rebuilding soft maps: memory_graph, meaning_map, logic_map, emotion_map")


        # Simulate dream silence
        time.sleep(random.randint(10, 25))

    update_inastate("dreaming", False)
    clear_module_running("dreamstate")
    log_to_statusbox("[Dreamstate] Dream complete. Exiting.")


if __name__ == "__main__":
    enter_dreamstate()
