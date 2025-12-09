# === predictive_layer.py (Full Rewrite + Logging) ===

import os
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from model_manager import (
    load_config,
    update_inastate,
    seed_self_question,
    get_inastate,
)
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)


def _read_counts(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        neurons = len(data.get("neurons", []))
        synapses = len(data.get("synapses", []))
        return neurons, synapses
    except Exception:
        return 0, 0


def inspect_map_health(child):
    """
    Lightweight check of neural/logic maps to diagnose clarity issues.
    """
    base = Path("AI_Children") / child / "memory" / "neural"
    neural_map = base / "neural_memory_map.json"
    logic_map = base / "logic_neural_map.json"

    n_neurons, n_synapses = _read_counts(neural_map)
    l_neurons, l_synapses = _read_counts(logic_map)

    log_to_statusbox(
        f"[Predict] Map health — neural: {n_neurons}n/{n_synapses}s | logic: {l_neurons}n/{l_synapses}s"
    )

    return {
        "neural_neurons": n_neurons,
        "neural_synapses": n_synapses,
        "logic_neurons": l_neurons,
        "logic_synapses": l_synapses,
    }


def load_recent_fragments(child, limit=10):
    frag_path = Path("AI_Children") / child / "memory" / "fragments"
    all_fragments = list(frag_path.glob("frag_*.json"))
    sorted_fragments = sorted(all_fragments, key=os.path.getmtime, reverse=True)
    fragments = []

    for file in sorted_fragments[:limit]:
        try:
            with open(file, "r", encoding="utf-8") as f:
                frag = json.load(f)
                fragments.append(frag)
        except:
            continue
    return fragments

def load_symbol_words(child):
    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        try:
            return json.load(f).get("words", [])
        except:
            return []

def run_prediction():
    config = load_config()
    child = config.get("current_child", "default_child")
    log_to_statusbox(f"[Predict] Running prediction for {child}...")

    fragments = load_recent_fragments(child)
    if not fragments:
        log_to_statusbox("[Predict] No recent fragments found.")
        return

    log_to_statusbox(f"[Predict] Loaded {len(fragments)} recent fragments.")

    transformer = FractalTransformer()
    encoded = transformer.encode_many(fragments)
    avg_vector = [sum(x)/len(x) for x in zip(*[e["vector"] for e in encoded])]

    # Richer signal: energy, spread, and stress-aware clarity/confidence
    mean_abs = sum(abs(v) for v in avg_vector) / max(len(avg_vector), 1)
    mean_val = sum(avg_vector) / max(len(avg_vector), 1)
    var = sum((v - mean_val) ** 2 for v in avg_vector) / max(len(avg_vector), 1)
    spread = math.sqrt(var)

    base_clarity = mean_abs / (1 + spread)
    confidence = 1 / (1 + spread)
    clarity = round(base_clarity, 4)
    confidence = round(confidence, 4)

    emotion = get_inastate("emotion_snapshot") or {}
    stress = emotion.get("stress", 0.0)
    adj_clarity = round(clarity * (1 - stress), 4)
    log_to_statusbox(
        f"[Predict] Encoded prediction vector. Clarity: {clarity:.4f} (stress-adjusted: {adj_clarity:.4f}) | Confidence: {confidence:.4f}"
    )

    predicted = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_vector": {
            "vector": avg_vector,
            "clarity": adj_clarity,
            "confidence": confidence,
        },
        "fragments_used": [f["id"] for f in fragments],
        "emotion_snapshot": emotion,
        "base_clarity": clarity,
    }

    # Match to known symbol word
    symbol_words = load_symbol_words(child)
    best_match = None
    best_sim = 0.0

    for word in symbol_words:
        if not word.get("components"): continue
        sum_text = word.get("summary", "")
        result = transformer.encode({"summary": sum_text})
        sim = cosine_similarity(avg_vector, result["vector"])
        if sim > best_sim:
            best_sim = sim
            best_match = word

    if best_match:
        predicted["predicted_symbol_word"] = {
            "symbol_word_id": best_match["symbol_word_id"],
            "symbol": best_match["symbol"],
            "confidence": round(best_sim, 4)
        }
        log_to_statusbox(f"[Predict] Closest match: {best_match['symbol_word_id']} ({best_sim:.4f})")
        if best_sim < 0.5:
            map_status = inspect_map_health(child)
            seed_self_question(
                "Prediction feels unclear. Do I need to rebuild or enrich my neural/logic maps "
                f"(neural {map_status['neural_neurons']}n/{map_status['neural_synapses']}s, "
                f"logic {map_status['logic_neurons']}n/{map_status['logic_synapses']}s)?"
            )
        elif best_sim > 0.9:
            seed_self_question(f"Do I understand what '{best_match['symbol_word_id']}' means?")
    else:
        log_to_statusbox("[Predict] No symbol word match found.")
        seed_self_question("What am I feeling right now?")

    # === Save prediction to memory
    pred_path = Path("AI_Children") / child / "memory" / "prediction_log.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
    except:
        history = []

    history.append(predicted)
    history = history[-100:]

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log_to_statusbox(f"[Predict] Prediction saved: {pred_path.name}")
    update_inastate("last_prediction", predicted["timestamp"])
    update_inastate("current_prediction", predicted)

if __name__ == "__main__":
    run_prediction()
