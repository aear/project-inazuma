import json
import os
from pathlib import Path

CONFIG_FILE = "config.json"
FUZZ_SOURCES = ["stress", "saturation", "intensity", "complexity", "negativity"]

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def load_fragment(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_fragment(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def calculate_fuzz(emotions):
    fuzz = 0.0
    count = 0
    for key in FUZZ_SOURCES:
        value = emotions.get(key)
        if isinstance(value, (int, float)):
            fuzz += abs(value)
            count += 1
    if count == 0:
        return 1.0  # From the void: fully fuzzy, fully clear
    fuzz = fuzz / count
    return round(min(fuzz, 1.0), 3)

def run_fuzzification():
    config = load_config()
    child_name = config.get("current_child")
    if not child_name:
        print("[Fuzzify] No current_child set in config.json")
        return

    frag_dir = Path("AI_Children") / child_name / "memory" / "fragments"
    if not frag_dir.exists():
        print(f"[Fuzzify] Fragment directory not found: {frag_dir}")
        return

    for frag_file in frag_dir.glob("frag_*.json"):
        frag = load_fragment(frag_file)
        emotions = frag.get("emotions", {})
        fuzz = calculate_fuzz(emotions)
        if "identity" in frag.get("tags", []):
            print(f"[Fuzzify] Anchoring symbol '{frag.get('id')}' — adjusting clarity.")
            fuzz = round(fuzz * 0.5, 3)  # Dampened fuzz, still some uncertainty
            clarity = round(0.7 + 0.3 * (1.0 - fuzz), 3)
        else:
            clarity = 1.0  # Full undefined potential for everything else

        frag["fuzz_level"] = fuzz
        frag["clarity"] = clarity

        save_fragment(frag_file, frag)
        print(f"[Fuzzify] {frag_file.name}: fuzz={fuzz}, clarity={frag['clarity']}")

if __name__ == "__main__":
    run_fuzzification()