
# === PATCH: boredom_state.py ===
# Adds full curiosity loop with symbol generation, image rendering, logic exploration, and raw file reading

import json
import subprocess
import time
import random
from pathlib import Path
from model_manager import get_inastate, update_inastate, load_config
from gui_hook import log_to_statusbox


def _call_local(script):
    path = Path(script)
    if not path.exists():
        log_to_statusbox(f"[Boredom] Skipping missing module: {script}")
        return
    subprocess.call(["python", str(path)])


def boredom_still_active():
    return get_inastate("emotion_boredom") and get_inastate("emotion_boredom") > 0.3

def generate_symbols():
    _call_local("symbol_generator.py")
    print("[Boredom] Symbols generated.")

def render_symbol_images():
    _call_local("symbol_visualiser.py")
    print("[Boredom] Symbol images rendered.")

def read_raw_files():
    _call_local("raw_file_manager.py")
    print("[Boredom] Raw project files explored.")

def evolve_logic():
    _call_local("logic_engine.py")
    print("[Boredom] Logic patterns explored.")

def do_visual_and_audio_exploration():
    _call_local("audio_listener.py")
    _call_local("vision_window.py")

def open_paint_window():
    _call_local("paint_window.py")
    print("[Boredom] Paint window opened.")

def load_impulse_preferences():
    config = load_config()
    child = config.get("current_child", "default_child")
    path = Path("AI_Children") / child / "memory" / "impulse_preferences.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_to_statusbox(f"[Boredom] Failed to load impulse preferences: {e}")
        return {}

def emotion_similarity(emotions, signature):
    if not signature:
        return 0.5
    total = 0.0
    count = 0
    for key, value in signature.items():
        if key in emotions:
            total += abs(emotions[key] - value)
            count += 1
    if count == 0:
        return 0.5
    distance = min(1.0, total / count)
    return max(0.0, 1.0 - distance)

def emotion_shift(emotions, signature):
    if not signature:
        return 0.0
    total = 0.0
    count = 0
    for key, value in signature.items():
        if key in emotions:
            total += abs(emotions[key] - value)
            count += 1
    if count == 0:
        return 0.0
    return min(1.0, total / count)

def choose_exploration(emotions, preferences):
    weights = {"symbols": 1.0, "render": 1.0, "read": 1.0, "logic": 1.0, "sense": 1.0, "paint": 1.0}
    rationales = []

    boredom_level = get_inastate("emotion_boredom") or 0.0
    curiosity = emotions.get("curiosity", 0.0)
    focus = emotions.get("focus", 0.0)
    intensity = emotions.get("intensity", 0.0)
    joy = emotions.get("joy", 0.0)
    sadness = emotions.get("sadness", 0.0)

    reading_pref = preferences.get("reading", {})
    music_pref = preferences.get("music", {})

    reading_score = reading_pref.get("preference_score", 0.5)
    reading_similarity = emotion_similarity(emotions, reading_pref.get("recent_emotion_signature", {}))
    reading_clarity = reading_pref.get("average_clarity", 0.5)
    reading_enjoyment = reading_pref.get("average_enjoyment", 0.5)
    weights["read"] += (
        curiosity * 0.9 +
        focus * 0.6 +
        reading_score * 1.2 +
        reading_similarity * 0.8 +
        reading_clarity * 0.4 +
        reading_enjoyment * 0.4
    )
    rationales.append(
        f"read→curiosity {curiosity:.2f}, focus {focus:.2f}, pref {reading_score:.2f}, sim {reading_similarity:.2f}"
    )

    music_score = music_pref.get("preference_score", 0.5)
    music_similarity = emotion_similarity(emotions, music_pref.get("recent_emotion_signature", {}))
    music_shift = emotion_shift(emotions, music_pref.get("recent_emotion_signature", {}))
    music_enjoyment = music_pref.get("average_enjoyment", 0.5)
    music_clarity = music_pref.get("average_clarity", 0.5)
    affect_delta = max(abs(joy - sadness), music_shift)
    weights["sense"] += (
        intensity * 0.8 +
        music_score * 1.1 +
        music_enjoyment * 0.8 +
        music_clarity * 0.3 +
        affect_delta * 0.9 +
        music_similarity * 0.5
    )
    rationales.append(
        f"sense→intensity {intensity:.2f}, affect Δ {affect_delta:.2f}, pref {music_score:.2f}, sim {music_similarity:.2f}"
    )

    weights["logic"] += max(focus - boredom_level, 0.0) * 0.6
    weights["symbols"] += boredom_level * 0.5 + curiosity * 0.4
    weights["render"] += boredom_level * 0.3 + (1.0 - intensity) * 0.2
    weights["paint"] += boredom_level * 0.5 + curiosity * 0.3 + joy * 0.2

    options = list(weights.keys())
    weight_values = [max(w, 0.01) for w in weights.values()]
    choice = random.choices(options, weights=weight_values, k=1)[0]
    rationale = "; ".join(rationales)
    log_to_statusbox(f"[Boredom] Weighted choice → {choice} ({rationale})")
    return choice

def boredom_exploration_loop():
    print("[Boredom] Entering curiosity loop...")
    loop_counter = 0
    while boredom_still_active() and loop_counter < 10:
        emotions = get_inastate("emotion_snapshot") or {}
        preferences = load_impulse_preferences()
        choice = choose_exploration(emotions, preferences)
        if choice == "symbols":
            generate_symbols()
        elif choice == "render":
            render_symbol_images()
        elif choice == "read":
            read_raw_files()
        elif choice == "logic":
            evolve_logic()
        elif choice == "sense":
            do_visual_and_audio_exploration()
        elif choice == "paint":
            open_paint_window()
        time.sleep(2)
        loop_counter += 1

    update_inastate("last_boredom_exploration", time.time())
    print("[Boredom] Curiosity loop ended.")

if __name__ == "__main__":
    boredom_exploration_loop()
