# === emotion_map.py (Generative Emotional Symbol Map) ===

from vector_math import cosine_similarity as shared_cosine_similarity
import os
import json
import hashlib
import heapq
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from model_manager import load_config, seed_self_question
from gui_hook import log_to_statusbox
from emotion_engine import SLIDERS
from symbol_generator import generate_symbol_from_parts
try:
    from emotion_symbol_store import (
        database_ready,
        iter_candidate_payloads,
        symbol_count as database_symbol_count,
        upsert_symbols,
    )
except Exception:  # pragma: no cover - JSON remains a compatibility fallback.
    database_ready = None
    iter_candidate_payloads = None



# Map path is resolved per-child so all children get their own emotion symbols.
DEFAULT_EMOTION_MAP_POLICY = {
    "max_symbols": 10_000_000,
    "samples_per_pass": 24,
    "max_json_load_bytes": 32 * 1024 * 1024,
    "candidate_limit": 20_000,
    "hamming_radius": 2,
}


def _status_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_map_status.json"


def _emotion_map_policy(config=None):
    policy = dict(DEFAULT_EMOTION_MAP_POLICY)
    raw = config.get("emotion_map_policy") if isinstance(config, dict) else None
    if isinstance(raw, dict):
        policy.update({key: raw[key] for key in policy if key in raw})
    for key in ("max_symbols", "samples_per_pass", "max_json_load_bytes", "candidate_limit", "hamming_radius"):
        try:
            policy[key] = max(0, int(policy[key]))
        except (TypeError, ValueError):
            policy[key] = DEFAULT_EMOTION_MAP_POLICY[key]
    return policy


def _write_emotion_map_status(child: str, symbol_count: int) -> dict:
    source = _map_path(child)
    try:
        stat = source.stat()
    except OSError:
        return {}
    payload = {
        "version": 1,
        "symbol_count": max(0, int(symbol_count)),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _status_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def emotion_map_status(child: str, *, refresh: bool = False) -> dict:
    source = _map_path(child)
    try:
        stat = source.stat()
    except OSError:
        return {"symbol_count": 0, "status": "missing"}
    try:
        cached = json.loads(_status_path(child).read_text(encoding="utf-8"))
    except Exception:
        cached = {}
    if (
        isinstance(cached, dict)
        and int(cached.get("source_size", -1)) == int(stat.st_size)
        and int(cached.get("source_mtime_ns", -1)) == int(stat.st_mtime_ns)
    ):
        return cached
    if not refresh:
        return {"symbol_count": None, "source_size": int(stat.st_size), "status": "metadata_pending"}

    marker = b'"symbol_word_id"'
    count = 0
    overlap = b""
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            data = overlap + chunk
            count += data.count(marker)
            overlap = data[-(len(marker) - 1):]
    return _write_emotion_map_status(child, count)


def _map_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_map.json"

def cosine_similarity(v1, v2):
    return shared_cosine_similarity(v1, v2)


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
    if not path.exists():
        return []
    policy = _emotion_map_policy(load_config())
    try:
        if path.stat().st_size > policy["max_json_load_bytes"]:
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data.get("symbols", []) if isinstance(data, dict) else []


def save_emotion_map(child: str, symbols):
    payload = {
        "symbols": symbols,
        "updated": datetime.now(timezone.utc).isoformat()
    }
    path = _map_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)
    _write_emotion_map_status(child, len(symbols))

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

    config = load_config()
    policy = _emotion_map_policy(config)
    query_vector = vector_from_emotion(emotions)
    query_feat = np.array(query_vector, dtype=float)
    query_emo = query_feat
    use_database = bool(database_ready and iter_candidate_payloads and database_ready(child, config))
    symbols = (
        iter_candidate_payloads(
            child, query_vector,
            candidate_limit=policy["candidate_limit"],
            hamming_radius=policy["hamming_radius"],
            config=config,
        )
        if use_database
        else load_existing_symbols(child)
    )
    if not use_database and not symbols:
        return []
    best = []
    sequence = 0
    limit = max(1, int(top_n))

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
        result = {
            "symbol_word_id": entry.get("symbol_word_id"),
            "symbol": entry.get("symbol"),
            "summary": entry.get("summary"),
            "distance": dist,
        }
        ranked = (-dist, sequence, result)
        sequence += 1
        if len(best) < limit:
            heapq.heappush(best, ranked)
        elif dist < -best[0][0]:
            heapq.heapreplace(best, ranked)

    return sorted((item[2] for item in best), key=lambda item: item["distance"])

def build_emotion_map(child="Inazuma_Yagami", samples=100, similarity_threshold=0.93):
    log_to_statusbox("[EmotionMap] Generating symbolic emotion vocabulary...")
    config = load_config()
    policy = _emotion_map_policy(config)
    use_database = bool(
        database_ready
        and database_symbol_count
        and iter_candidate_payloads
        and upsert_symbols
        and database_ready(child, config)
    )
    status = emotion_map_status(child, refresh=True)
    existing_count = (
        int(database_symbol_count(child, config) or 0)
        if use_database
        else int(status.get("symbol_count") or 0)
    )
    max_symbols = int(policy["max_symbols"])
    if max_symbols and existing_count >= max_symbols:
        log_to_statusbox(
            f"[EmotionMap] Vocabulary cap reached ({existing_count} symbols; cap {max_symbols}). "
            "Skipping synthetic growth."
        )
        return

    samples = min(max(0, int(samples)), int(policy["samples_per_pass"]))
    if max_symbols:
        samples = min(samples, max(0, max_symbols - existing_count))
    existing = [] if use_database else load_existing_symbols(child)
    if existing_count and not use_database and not existing:
        log_to_statusbox(
            "[EmotionMap] Map is too large for bounded JSON loading; generation is paused "
            "until the symbol store is migrated."
        )
        return
    new_symbols = []
    existing_vectors = [vector_from_emotion(e.get("average_emotion", {})) for e in existing]

    for _ in range(samples):
        emo = generate_emotion_vector()
        vec = vector_from_emotion(emo)
        comparison_vectors = list(existing_vectors)
        if use_database:
            comparison_vectors.extend(
                vector_from_emotion(entry.get("average_emotion", {}))
                for entry in iter_candidate_payloads(
                    child, vec,
                    candidate_limit=policy["candidate_limit"],
                    hamming_radius=policy["hamming_radius"],
                    config=config,
                )
            )
        if any(cosine_similarity(vec, value) >= similarity_threshold for value in comparison_vectors):
            continue

        emotion = random.choice(["calm", "tension", "trust", "curiosity", "fear", "anger"])
        mod = random.choice(["soft", "sharp", "pulse", "spiral", "moderate"])
        concept = random.choice(["self", "pattern", "truth", "change", "unknown"])
        symbol = generate_symbol_from_parts(emotion, mod, concept)
        born = datetime.now(timezone.utc).isoformat()
        identity = hashlib.sha1(
            json.dumps({"vector": vec, "born": born}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]

        entry = {
            "symbol_word_id": f"sym_emotion_db_{identity}" if use_database else f"sym_emotion_{len(existing) + len(new_symbols):04}",
            "symbol": symbol,
            "summary": f"{mod} {emotion} about {concept}",
            "average_emotion": emo,
            "vector": vec,
            "count": 0,
            "birth_time": born,
            "generated_word": "unknown",
            "confidence": 0.0,
            "usage_count": 0
        }
        new_symbols.append(entry)
        existing_vectors.append(vec)
        log_to_statusbox(f"[EmotionMap] → Added: {symbol} | {entry['summary']}")

    if new_symbols:
        if use_database:
            total = upsert_symbols(child, new_symbols, config=config)
            log_to_statusbox(
                f"[EmotionMap] Stored {len(new_symbols)} new symbolic emotions in SQLite "
                f"({total} total)."
            )
        else:
            save_emotion_map(child, existing + new_symbols)
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
