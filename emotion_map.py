# === emotion_map.py (Generative Emotional Symbol Map) ===

import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from model_manager import load_config, seed_self_question
from gui_hook import log_to_statusbox
from emotion_engine import SLIDERS
from fractal_multidimensional_transformers import FractalTransformer
from symbol_generator import generate_symbol_from_parts

MAP_PATH = Path("AI_Children") / "Inazuma_Yagami" / "memory" / "emotion_symbol_map.json"

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)

def load_existing_symbols():
    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("symbols", [])
            except:
                return []
    return []

def generate_emotion_vector():
    return {k: round(random.uniform(-1.0, 1.0), 4) for k in SLIDERS}

def vector_from_emotion(emotions):
    return [emotions.get(k, 0.0) for k in SLIDERS]

def build_emotion_map(child="Inazuma_Yagami", samples=100, similarity_threshold=0.93):
    log_to_statusbox("[EmotionMap] Generating symbolic emotion vocabulary...")
    transformer = FractalTransformer()
    existing = load_existing_symbols()
    new_symbols = []
    existing_vectors = [vector_from_emotion(e.get("average_emotion", {})) for e in existing]

    for i in range(samples):
        emo = generate_emotion_vector()
        vec = vector_from_emotion(emo)

        if any(cosine_similarity(vec, v) >= similarity_threshold for v in existing_vectors):
            continue  # Skip similar

        # Assign new symbol
        emotion = random.choice(["calm", "tension", "trust", "curiosity", "fear", "anger"])
        mod = random.choice(["soft", "sharp", "pulse", "spiral", "moderate"])
        concept = random.choice(["self", "pattern", "truth", "change", "unknown"])
        symbol = generate_symbol_from_parts(emotion, mod, concept)

        entry = {
            "symbol_word_id": f"sym_emotion_{len(existing) + len(new_symbols):04}",
            "symbol": symbol,
            "summary": f"{mod} {emotion} about {concept}",
            "average_emotion": emo,
            "vector": vec,
            "count": 0,
            "birth_time": datetime.now(timezone.utc).isoformat(),
            "generated_word": "unknown",
            "confidence": 0.0,
            "usage_count": 0
        }
        new_symbols.append(entry)
        log_to_statusbox(f"[EmotionMap] → Added: {symbol} | {entry['summary']}")

    if new_symbols:
        all_symbols = existing + new_symbols
        result = {
            "symbols": all_symbols,
            "updated": datetime.now(timezone.utc).isoformat()
        }
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log_to_statusbox(f"[EmotionMap] Saved {len(new_symbols)} new symbolic emotions.")
        seed_self_question("Which of these symbols feels most like me?")
    else:
        log_to_statusbox("[EmotionMap] No new symbolic states added — existing set is dense.")

def run_emotion_map():
    try:
        config = load_config()
        child = config.get("current_child", "Inazuma_Yagami")
        build_emotion_map(child)
        log_to_statusbox("[EmotionMap] Symbolic emotion map update complete.")
    except Exception as e:
        log_to_statusbox(f"[EmotionMap] Error: {e}")
        print(f"[EmotionMap] Error: {e}")

if __name__ == "__main__":
    run_emotion_map()
