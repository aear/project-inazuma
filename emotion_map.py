# === emotion_map.py (Generative Emotional Symbol Map) ===

import os
import json
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from model_manager import load_config, seed_self_question
from gui_hook import log_to_statusbox
from emotion_engine import SLIDERS
from symbol_generator import generate_symbol_from_parts

# Map path is resolved per-child so all children get their own emotion symbols.
def _map_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_map.json"

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """1 - cosine similarity, with small epsilon guard."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    if denom <= eps:
        return 1.0  # treat degenerate vectors as maximally distant
    return 1.0 - float(np.dot(a, b) / denom)


def combined_distance(
    feat_a: np.ndarray,
    emo_a: np.ndarray,
    feat_b: np.ndarray,
    emo_b: np.ndarray,
    emotion_weight: float = 0.35,
) -> float:
    """
    Distance between two fragment states using both feature + emotion vectors.
    emotion_weight in [0,1] controls how much emotional difference matters.
    """
    df = _cosine_distance(feat_a, feat_b)
    de = _cosine_distance(emo_a, emo_b)
    return df + emotion_weight * de


def load_existing_symbols(child: str):
    path = _map_path(child)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("symbols", [])
            except Exception:
                return []
    return []


def save_emotion_map(child: str, symbols):
    payload = {
        "symbols": symbols,
        "updated": datetime.now(timezone.utc).isoformat()
    }
    path = _map_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def generate_emotion_vector():
    return {k: round(random.uniform(-1.0, 1.0), 4) for k in SLIDERS}

def vector_from_emotion(emotions):
    return [emotions.get(k, 0.0) for k in SLIDERS]


def rank_emotion_symbols(
    emotions,
    child: str = None,
    top_n: int = 2,
    emotion_weight: float = 0.35,
):
    """
    Return the closest symbolic emotions to a given emotion dict.
    Uses combined distance (feature + emotion) to allow secondary neuron-like pairing.
    """
    if child is None:
        child = load_config().get("current_child", "Inazuma_Yagami")

    symbols = load_existing_symbols(child)
    if not symbols:
        return []

    query_feat = np.array(vector_from_emotion(emotions), dtype=float)
    query_emo = query_feat
    scored = []

    for entry in symbols:
        emo_vec = np.array(vector_from_emotion(entry.get("average_emotion", {})), dtype=float)
        feat_vec = np.array(entry.get("vector", emo_vec), dtype=float)

        # Skip malformed entries
        if emo_vec.shape != query_emo.shape:
            continue
        if feat_vec.shape != query_feat.shape:
            # fall back to emotion-only vector if feature dims mismatch
            feat_vec = emo_vec

        dist = combined_distance(
            feat_a=feat_vec,
            emo_a=emo_vec,
            feat_b=query_feat,
            emo_b=query_emo,
            emotion_weight=emotion_weight,
        )
        scored.append({
            "symbol_word_id": entry.get("symbol_word_id"),
            "symbol": entry.get("symbol"),
            "summary": entry.get("summary"),
            "distance": dist,
        })

    scored.sort(key=lambda item: item["distance"])
    return scored[:top_n]

def build_emotion_map(child="Inazuma_Yagami", samples=100, similarity_threshold=0.93):
    log_to_statusbox("[EmotionMap] Generating symbolic emotion vocabulary...")
    existing = load_existing_symbols(child)
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
        existing_vectors.append(vec)
        log_to_statusbox(f"[EmotionMap] → Added: {symbol} | {entry['summary']}")

    if new_symbols:
        all_symbols = existing + new_symbols
        save_emotion_map(child, all_symbols)
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
