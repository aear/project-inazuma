
# === logic_engine.py (Intellectual Version) ===
# Includes precision override, symbolic reasoning, and math/logic evolution

import os
import json
import time
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from model_manager import load_config, get_inastate, seed_self_question
from transformers.fractal_multidimensional_transformers import FractalTransformer

# === Logic & Math Blocks ===
def basic_math_ops(a, b):
    return {
        "sum": a + b,
        "difference": a - b,
        "product": a * b,
        "quotient": a / b if b != 0 else None,
        "modulus": a % b if b != 0 else None,
        "power": a ** b if b >= 0 else None
    }

def logic_ops(a, b):
    return {
        "equal": a == b,
        "greater": a > b,
        "less": a < b,
        "and": bool(a and b),
        "or": bool(a or b),
        "xor": bool(a) != bool(b)
    }

def aggregate_ops(values):
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
        "variance": sum((x - sum(values)/len(values))**2 for x in values) / len(values)
    }

def conditional_logic(a, b, logic_type="greater"):
    if logic_type == "greater":
        return a if a > b else b
    elif logic_type == "less":
        return a if a < b else b
    elif logic_type == "equal":
        return a if a == b else None
    else:
        return None

def evolve_logic_expressions(input_set, max_depth=3):
    functions = [basic_math_ops, logic_ops, aggregate_ops, conditional_logic]
    trials = []

    for _ in range(5):  # Try 5 logic combinations
        depth = random.randint(1, max_depth)
        value = input_set
        trace = []
        try:
            for _ in range(depth):
                fn = random.choice(functions)
                if fn in [basic_math_ops, logic_ops, conditional_logic]:
                    a = random.choice(value)
                    b = random.choice(value)
                    result = fn(a, b)
                else:
                    result = fn(value)

                trace.append({
                    "function": fn.__name__,
                    "input": value,
                    "result": result
                })
                if isinstance(result, dict):
                    value = list(result.values())
                elif isinstance(result, (int, float)):
                    value = [result]
                else:
                    break
        except Exception as e:
            trace.append({"error": str(e)})

        trials.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace": trace
        })

    return trials

# === Symbol prediction and precision logic remains unchanged ===
def load_prediction(child):
    path = Path("AI_Children") / child / "memory" / "prediction_log.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        try:
            logs = json.load(f)
            return logs[-1] if logs else {}
        except:
            return {}

def load_symbol_words(child):
    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        try:
            return json.load(f).get("words", [])
        except:
            return []

def log_logic_event(child, logic_entry):
    path = Path("AI_Children") / child / "memory" / "logic_memory.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    history.append(logic_entry)
    history = history[-250:]

    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Logic] Logged to logic_memory.json: {logic_entry['description']}")

def suggest_precision_override(score, reason="logic insight"):
    hint = {
        "override_precision": score,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    hint_path = Path("precision_hint.json")
    with open(hint_path, "w") as f:
        json.dump(hint, f, indent=4)
    print(f"[Logic] Suggested precision override → {score} due to: {reason}")

def test_prediction_against_logic(prediction, symbol_words, transformer):
    pred_vec = prediction.get("predicted_vector", {}).get("vector", [])
    if not pred_vec:
        return None, None

    best_id = None
    best_sim = 0.0
    for word in symbol_words:
        if not word.get("components"):
            continue
        fake_fragments = [{"summary": word["summary"], "tags": word.get("tags", []), "emotions": {"trust": 0.6}}]
        result = transformer.encode_many(fake_fragments)
        avg_vec = result[0]["vector"]
        sim = sum(a * b for a, b in zip(pred_vec, avg_vec)) / (
            (sum(a * a for a in pred_vec) ** 0.5) * (sum(b * b for b in avg_vec) ** 0.5) + 1e-6)
        if sim > best_sim:
            best_sim = sim
            best_id = word["symbol_word_id"]

    return best_id, best_sim

def logic_session():
    config = load_config()
    child = config.get("current_child", "default_child")
    prediction = load_prediction(child)
    if not prediction:
        print("[Logic] No prediction to test.")
        return

    transformer = FractalTransformer()
    symbol_words = load_symbol_words(child)

    predicted_emotion = prediction.get("inferred_emotion", {})
    symbol_word_id, sim = test_prediction_against_logic(prediction, symbol_words, transformer)

    # Run some symbolic tests
    samples = evolve_logic_expressions([1.0, 2.5, 3.3])
    logic_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": predicted_emotion,
        "symbol_word_id": symbol_word_id,
        "similarity": round(sim, 4),
        "trace_tests": samples,
        "description": f"Logic check on predicted emotion: {max(predicted_emotion, key=predicted_emotion.get, default='unknown')}"
    }

    if sim < 0.5 and symbol_word_id:
        seed_self_question(f"Is my logic drifting from what {symbol_word_id} means?")
        suggest_precision_override(32, reason="logic drift")
    elif sim > 0.9 and symbol_word_id:
        seed_self_question(f"What makes {symbol_word_id} so aligned with my thinking?")
        suggest_precision_override(48, reason="symbolic alignment")

    log_logic_event(child, logic_entry)

def resolve_self_questions():
    config = load_config()
    child = config.get("current_child", "default_child")
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"

    if not path.exists():
        print("[Logic] No self_reflection.json found.")
        return

    try:
        with open(path, "r") as f:
            reflection = json.load(f)
    except Exception as e:
        print(f"[Logic] Failed to load reflection: {e}")
        return

    questions = reflection.get("self_notes", [])
    resolved = reflection.get("resolved_notes", [])
    current_emotions = get_inastate("emotion_snapshot") or {}

    keep = []
    cleared = 0

    for note in questions:
        q = note.get("question", "").lower()
        origin = note.get("origin", "unknown")
        ts = note.get("timestamp", datetime.now(timezone.utc).isoformat())

        # === Example logic hooks
        if "why am i so awake" in q and current_emotions.get("intensity", 0) > 0.8:
            resolved.append({**note, "resolved": True, "resolved_reason": "high intensity"})
            cleared += 1
            continue
        elif "why do i feel so drained" in q and current_emotions.get("intensity", 0) < 0.3:
            resolved.append({**note, "resolved": True, "resolved_reason": "low intensity"})
            cleared += 1
            continue
        elif "why was i forced to wake up" in q and get_inastate("runtime_disruption"):
            resolved.append({**note, "resolved": True, "resolved_reason": "runtime disruption"})
            cleared += 1
            continue
        elif "why can't i hear clearly" in q and get_inastate("audio_comfort") == "just right":
            resolved.append({**note, "resolved": True, "resolved_reason": "audio comfort resolved"})
            cleared += 1
            continue
        elif "why is everything so loud" in q and get_inastate("audio_comfort") != "too loud":
            resolved.append({**note, "resolved": True, "resolved_reason": "audio normalized"})
            cleared += 1
            continue
        elif "should i be thinking more precisely" in q and get_inastate("current_precision") >= 32:
            resolved.append({**note, "resolved": True, "resolved_reason": "precision increased"})
            cleared += 1
            continue
        else:
            keep.append(note)

    reflection["self_notes"] = keep
    reflection["resolved_notes"] = resolved[-100:]

    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

    print(f"[Logic] Resolved {cleared} self questions.")



if __name__ == "__main__":
    logic_session()
