# precision_evolution.py

import json
import os
import random
import precision_memory_map
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


def _coerce_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _emotion_values(raw):
    if isinstance(raw, dict) and isinstance(raw.get("values"), dict):
        raw = raw.get("values")
    if not isinstance(raw, dict):
        return {}
    values = {}
    for key, value in raw.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _precision_memory_context(child):
    scheduler = get_inastate("process_scheduler") or {}
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    planner = scheduler.get("planner") if isinstance(scheduler.get("planner"), dict) else {}
    slot_summary = scheduler.get("slot_summary") if isinstance(scheduler.get("slot_summary"), dict) else {}
    queue = scheduler.get("queue") if isinstance(scheduler.get("queue"), list) else []
    queue_depth = _coerce_float(planner.get("queue_depth"), float(len(queue)))
    max_queue = max(1.0, _coerce_float(slot_summary.get("max_queue_slots"), max(queue_depth, 1.0)))

    active_modules = get_inastate("running_modules") or []
    if isinstance(active_modules, str):
        active_modules = [active_modules]
    if not isinstance(active_modules, list):
        active_modules = []
    for entry in planner.get("running") or []:
        if not isinstance(entry, dict):
            continue
        module = entry.get("module") or entry.get("task_key") or entry.get("label")
        if module:
            active_modules.append(str(module))

    return {
        "active_modules": sorted(set(str(name) for name in active_modules if str(name))),
        "queue_pressure": max(0.0, min(1.0, queue_depth / max_queue)),
        "emotion_state": _emotion_values(get_inastate("emotion_snapshot") or get_inastate("current_emotions") or {}),
        "energy": _coerce_float(get_inastate("current_energy"), 0.5),
    }


def _precision_hint_candidates(child):
    return [
        Path("precision_hint.json"),
        Path("AI_Children") / child / "precision_hint.json",
        Path("AI_Children") / child / "memory" / "precision_hint.json",
    ]


def _load_precision_hint_for_child(child):
    for path in _precision_hint_candidates(child):
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload, path
        except Exception:
            continue
    return {}, None


def _write_precision_memory_suggestion(child, context, suggestion):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "precision_memory_map",
        "advisory": True,
        "context": context,
        "suggestion": suggestion,
    }
    path = Path("AI_Children") / child / "memory" / "precision_memory_suggestion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

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
    config = load_config()
    child = config.get("current_child", CHILD)
    current = load_precision_profile(child)
    current_precision = current.get("max_precision", 64)
    updated = False
    suggestion_recorded = False
    suggestion = {}
    context = {}
    reason = "stable baseline"

    # === Extreme-state hints remain explicit overrides.
    hint, hint_path = _load_precision_hint_for_child(child)
    if hint:
        try:
            suggested = hint.get("suggested_max_precision", hint.get("override_precision"))
            if suggested is not None:
                current["max_precision"] = float(suggested)
                reason = hint.get("reason", "instinct hint")
                updated = True
            if hint_path is not None:
                os.remove(hint_path)
        except Exception as e:
            print(f"[Precision] Failed to apply hint: {e}")

    # === Memory-map path: advisory only, no direct precision override.
    if not updated:
        try:
            context = _precision_memory_context(child)
            suggestion = precision_memory_map.suggest_precision(context, child=child)
            if suggestion.get("global") is not None:
                _write_precision_memory_suggestion(child, context, suggestion)
                reason = "precision memory map suggestion"
                suggestion_recorded = True
        except Exception as e:
            print(f"[Precision] Failed to read precision memory map: {e}")

    try:
        if LOG_JSON.exists():
            with open(LOG_JSON, "r") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
        else:
            logs = []
    except Exception:
        logs = []

    if updated:
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(current, f, indent=4)
            print(f"[Precision] Updated config: precision={current['max_precision']} | Reason: {reason}")
        except Exception as e:
            print(f"[Precision] Failed to write config: {e}")

        logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "precision": current["max_precision"],
            "reason": reason,
            "source": "precision_hint",
        })
    elif suggestion_recorded:
        logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "precision_suggestion": suggestion,
            "context": context,
            "reason": reason,
            "advisory": True,
        })
        print(
            f"[Precision] Memory suggestion recorded: precision={suggestion.get('global')} "
            f"confidence={suggestion.get('confidence')} | advisory only"
        )
    else:
        print(f"[Precision] No memory suggestion available. Current precision: {current_precision}")

    try:
        LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
        logs = logs[-200:]
        with open(LOG_JSON, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[Precision] Failed to update JSON log: {e}")


if __name__ == "__main__":
    apply_precision_strategy()
