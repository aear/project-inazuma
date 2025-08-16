# precision_evolution.py

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from model_manager import get_inastate, get_sweet_spots
from transformers.fractal_multidimensional_transformers import load_precision_profile, FractalTransformer
from gui_hook import log_to_statusbox

CONFIG_FILE = "config.json"
CHILD = "Inazuma_Yagami"
BASE = Path("AI_Children") / CHILD / "models"
BASE.mkdir(parents=True, exist_ok=True)
HINT_PATH = BASE.parent / "precision_hint.json"
CONFIG_PATH = BASE / "precision_config.json"
LOG_JSON = BASE / "precision_learning.json"

def load_self_reflection(child):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)
    
def load_precision_config():
    path = Path("precision_config.json")
    if not path.exists():
        return {"max_precision": 64}
    with open(path, "r") as f:
        return json.load(f)
    
def load_precision_hint():
    path = Path("precision_hint.json")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def evaluate_efficiency():
    config = load_config()
    child = config.get("current_child", "default_child")
    transformer = FractalTransformer()

    # === Load emotional snapshot
    emo = get_inastate("emotion_snapshot") or {"curiosity": 0.6, "trust": 0.4}
    spot = get_sweet_spots()
    min_prec = spot.get("prediction_precision", {}).get("min", 32)
    max_prec = spot.get("prediction_precision", {}).get("max", 48)

    test_fragment = {
        "summary": "symbol awareness probe",
        "tags": ["symbolic"],
        "emotions": emo
    }

    # === Evaluate
    results = []
    for p in [1, 2, 4, 8, 16, 32, 48, 64]:
        if p > max_prec:
            continue
        transformer.precision = p
        encoded = transformer.encode_fragment(test_fragment)
        avg = round(sum(encoded["vector"]) / len(encoded["vector"]), 4)
        results.append((p, avg))

    results.sort(key=lambda x: x[1], reverse=True)
    selected = results[0]
    log_path = Path("AI_Children") / child / "memory" / "precision_learning.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbolic_precision_test": results,
        "selected_precision": selected[0],
        "reason": "symbolic pattern response"
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
    except Exception as e:
        log_to_statusbox(f"[Precision] Failed to write log: {e}")
        return

    # === Save config
    cfg_path = Path("precision_config.json")
    try:
        with open(cfg_path, "w") as f:
            json.dump({"max_precision": selected[0]}, f, indent=2)
    except Exception as e:
        log_to_statusbox(f"[Precision] Failed to save config: {e}")
        return

    print(f"[Precision] Efficiency test picked: {selected[0]} with score {selected[1]}")

def run_precision_test():
    config = load_config()
    child = config.get("current_child", "default_child")
    transformer = FractalTransformer()

    # === Load settings
    reflection = load_self_reflection(child)
    preferred = reflection.get("preferred_sweet_spots", {}).get("prediction_precision", {})
    comfort_max = preferred.get("max", 48)

    hard_limit = load_precision_config().get("max_precision", 64)
    override = load_precision_hint().get("override_precision")

    if override and 1 <= override <= hard_limit:
        print(f"[Precision] OVERRIDE: Instinct or logic requested precision {override}")
        selected_precision = override
        reason = "manual override"
    else:
        # === Test precision range
        precision_range = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
        precision_range = [p for p in precision_range if p <= comfort_max]
        results = []

        test_fragment = {
            "summary": "symbol awareness probe",
            "tags": ["symbolic"],
            "emotions": get_inastate("emotion_snapshot") or {"trust": 0.4, "curiosity": 0.6}
        }

        for p in precision_range:
            transformer.precision = p
            result = transformer.encode_fragment(test_fragment)
            score = round(sum(result["vector"]) / len(result["vector"]), 4)
            results.append((p, score))

        results.sort(key=lambda x: x[1], reverse=True)
        selected_precision = results[0][0]
        reason = "comfort_max adaptive selection"

    # === Log trail
    trail = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "selected_precision": selected_precision,
        "symbol_word_id": get_inastate("last_symbol_word_id"),
        "emotion_stress": get_inastate("emotion_stress"),
        "symbol_overload": get_inastate("symbol_overload")
    }

    log_path = Path("AI_Children") / child / "memory" / "precision_learning.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with open(log_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(trail)
    history = history[-100:]
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"[Precision] Selected: {selected_precision} — Reason: {reason}")
    return selected_precision


def apply_precision_strategy():
    current = load_precision_profile()
    current_precision = current.get("max_precision", 64)
    emotion = get_inastate("current_emotions") or {}
    spots = get_sweet_spots()

    min_prec = spots.get("prediction_precision", {}).get("min", 32)
    max_prec = spots.get("prediction_precision", {}).get("max", 48)
    updated = False
    reason = "stable baseline"

    # === Apply hint
    if HINT_PATH.exists():
        try:
            with open(HINT_PATH, "r") as f:
                hint = json.load(f)
            suggested = hint.get("suggested_max_precision")
            if suggested and 16 <= suggested <= 64:
                current["max_precision"] = suggested
                reason = hint.get("reason", "instinct hint")
                updated = True
            os.remove(HINT_PATH)
        except Exception as e:
            print(f"[Precision] Failed to apply hint: {e}")

    # === Emotion fallback
    if not updated:
        intensity = emotion.get("intensity", 0.0)
        stress = emotion.get("stress", 0.0)
        avg_overload = (intensity + stress) / 2
        if avg_overload > 0.6 and current_precision > min_prec:
            current["max_precision"] = max(min_prec, current_precision - 8)
            reason = "emotional overload fallback"
            updated = True

    # === Sweet spot check
    if not updated:
        if current_precision > max_prec:
            current["max_precision"] = max_prec
            reason = "above sweet spot"
            updated = True
        elif current_precision < min_prec:
            current["max_precision"] = min_prec
            reason = "below sweet spot"
            updated = True

    # === Save updated config and log
    if updated:
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(current, f, indent=4)
            print(f"[Precision] Updated config: precision={current['max_precision']} | Reason: {reason}")
        except Exception as e:
            print(f"[Precision] Failed to write config: {e}")

        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "precision": current["max_precision"],
                "reason": reason
            }
            if LOG_JSON.exists():
                with open(LOG_JSON, "r") as f:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
            else:
                logs = []

            logs.append(entry)
            logs = logs[-200:]  # Keep last 200 entries

            with open(LOG_JSON, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[Precision] Failed to update JSON log: {e}")
    else:
        print(f"[Precision] No update needed. Current precision: {current_precision}")


if __name__ == "__main__":
    apply_precision_strategy()
