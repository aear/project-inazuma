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
    clarity = round(sum(avg_vector) / len(avg_vector), 4)

    emotion = get_inastate("emotion_snapshot") or {}
    stress = emotion.get("stress", 0.0)
    adj_clarity = round(clarity * (1 - stress), 4)
    log_to_statusbox(
        f"[Predict] Encoded prediction vector. Clarity: {clarity:.4f} (stress-adjusted: {adj_clarity:.4f})"
    )

    predicted = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_vector": {
            "vector": avg_vector,
            "clarity": adj_clarity,
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
            seed_self_question("Why does my prediction feel unclear?")
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
