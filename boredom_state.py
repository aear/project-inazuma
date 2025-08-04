
# === PATCH: boredom_state.py ===
# Adds full curiosity loop with symbol generation, image rendering, logic exploration, and raw file reading

import subprocess
import time
import random
from model_manager import get_inastate, update_inastate

def boredom_still_active():
    return get_inastate("emotion_boredom") and get_inastate("emotion_boredom") > 0.3

def generate_symbols():
    subprocess.call(["python", "symbol_generator.py"])
    print("[Boredom] Symbols generated.")

def render_symbol_images():
    subprocess.call(["python", "symbol_visualiser.py"])
    print("[Boredom] Symbol images rendered.")

def read_raw_files():
    subprocess.call(["python", "raw_file_manager.py"])
    print("[Boredom] Raw project files explored.")

def evolve_logic():
    subprocess.call(["python", "logic_engine.py"])
    print("[Boredom] Logic patterns explored.")

def do_visual_and_audio_exploration():
    subprocess.call(["python", "audio_listener.py"])
    subprocess.call(["python", "vision_window.py"])

def boredom_exploration_loop():
    print("[Boredom] Entering curiosity loop...")
    loop_counter = 0
    while boredom_still_active() and loop_counter < 10:
        choice = random.choice(["symbols", "render", "read", "logic", "sense"])
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
        time.sleep(2)
        loop_counter += 1

    update_inastate("last_boredom_exploration", time.time())
    print("[Boredom] Curiosity loop ended.")

if __name__ == "__main__":
    boredom_exploration_loop()
