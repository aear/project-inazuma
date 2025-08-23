# === emotion_engine.py (Slider-Based Emotion Engine) ===

import json
from datetime import datetime, timezone
from pathlib import Path
from model_manager import (
    load_config,
    update_inastate,
    get_inastate,
)
from gui_hook import log_to_statusbox

SLIDERS = [
    "intensity", "attention", "trust", "care", "curiosity", "novelty", "familiarity", "stress", "risk",
    "negativity", "positivity", "simplicity", "complexity", "interest", "clarity", "fuzziness", "alignment",
    "safety", "threat", "presence", "isolation", "connection", "ownership", "externality"
]

def load_fragments(path):
    fragments = []
    for f in sorted(Path(path).glob("frag_*.json")):
        if f.is_file() and f.suffix == ".json":
            try:
                with open(f, "r") as fh:
                    fragments.append(json.load(fh))
            except (OSError, json.JSONDecodeError) as e:
                log_to_statusbox(f"[Emotion] Failed to load {f.name}: {e}")
            with open(f, "r") as handle:
                fragments.append(json.load(handle))
    return fragments

def calculate_emotion_state(fragments):
    totals = {k: 0.0 for k in SLIDERS}
    counts = {k: 0 for k in SLIDERS}

    for frag in fragments:
        emotions = frag.get("emotions", {})
        for k in SLIDERS:
            if k in emotions:
                totals[k] += emotions[k]
                counts[k] += 1

    averages = {
        k: round(totals[k] / counts[k], 4) if counts[k] > 0 else 0.0
        for k in SLIDERS
    }
    return averages

def log_emotion_snapshot(child, snapshot):
    log_path = Path("AI_Children") / child / "memory" / "emotion_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot
    }

    try:
        if log_path.exists():
            with open(log_path, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(entry)
        history = history[-200:]

        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"[Emotion] Logged snapshot with {len(snapshot)} sliders.")
    except Exception as e:
        log_to_statusbox(f"[Emotion] Failed to write emotion log: {e}")

def tag_fragment(fragment, state):
    tagged = {}
    for k in SLIDERS:
        if abs(state.get(k, 0.0)) > 0.2:
            tagged[k] = round(state[k], 3)
    return tagged

def run_emotion_engine():
    config = load_config()
    child = config.get("current_child", "default_child")
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    if not frag_dir.exists():
        return

    log_to_statusbox("[Emotion] Tagging fragments...")
    fragments = load_fragments(frag_dir)
    snapshot = calculate_emotion_state(fragments)

    current_pred = get_inastate("current_prediction") or {}
    pred_clarity = (
        current_pred.get("predicted_vector", {}).get("clarity")
        if isinstance(current_pred, dict)
        else None
    )
    if pred_clarity is not None:
        snapshot["prediction_clarity"] = pred_clarity

    for fpath in frag_dir.glob("frag_*.json"):
        try:
            with open(fpath, "r") as f:
                frag = json.load(f)
            frag["emotions"] = tag_fragment(frag, snapshot)
            with open(fpath, "w") as f:
                json.dump(frag, f, indent=4)
        except Exception as e:
            log_to_statusbox(f"[Emotion] Failed to tag {fpath.name}: {e}")
            continue

    update_inastate("emotion_snapshot", snapshot)
    log_emotion_snapshot(child, snapshot)
    log_to_statusbox("[Emotion] Emotion snapshot stored.")

if __name__ == "__main__":
    run_emotion_engine()
