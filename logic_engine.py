
# === logic_engine.py (Intellectual Version) ===
# Includes precision override, symbolic reasoning, and math/logic evolution

import os
import json
import time
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from model_manager import load_config, get_inastate, seed_self_question, mark_self_question_resolved
from transformers.fractal_multidimensional_transformers import FractalTransformer
from logic_map_builder import extract_logic_vector, run_logic_map_builder
from io_utils import atomic_write_json
from streaming_json import iter_selected_array_objects
from ina_ml import cosine_similarity

try:
    from logic_memory_store import store_logic_entry
except Exception:  # pragma: no cover - JSON remains a compatibility fallback.
    store_logic_entry = None


def _json_safe(value):
    """
    Make values safe for json serialization (e.g., complex numbers).
    """
    if isinstance(value, complex):
        return {
            "_type": "complex",
            "real": value.real,
            "imag": value.imag,
            "magnitude": math.hypot(value.real, value.imag),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)

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
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "max": max(values),
        "min": min(values),
        "variance": sum((x - mean) ** 2 for x in values) / len(values),
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
    config = load_config()
    raw_policy = config.get("logic_engine_policy") if isinstance(config, dict) else {}
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    candidate_limit = max(32, int(policy.get("symbol_candidate_limit", 20_000)))
    compact_path = path.with_name("symbol_words.logic_index.json")
    source_mtime_ns = int(path.stat().st_mtime_ns)
    try:
        with compact_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (
            isinstance(cached, dict)
            and int(cached.get("source_mtime_ns", -1)) == source_mtime_ns
            and isinstance(cached.get("words"), list)
        ):
            return cached["words"][:candidate_limit]
    except Exception:
        pass

    # Legacy entries can retain enormous component histories. Logic matching
    # needs only compact semantic fields, so stream past component payloads.
    fields = {"symbol_word_id", "summary", "tags", "vector", "symbol"}
    try:
        words = list(iter_selected_array_objects(path, "words", fields, limit=candidate_limit))
    except Exception as exc:
        print(f"[Logic] Failed to stream symbol candidates: {exc}")
        return []
    try:
        atomic_write_json(
            compact_path,
            {"source_mtime_ns": source_mtime_ns, "candidate_limit": candidate_limit, "words": words},
            indent=2,
            ensure_ascii=True,
        )
    except Exception:
        pass
    return words


def log_logic_event(child, logic_entry):
    safe_entry = _json_safe(logic_entry)
    if store_logic_entry is not None:
        try:
            store_logic_entry(child, safe_entry, extract_logic_vector(safe_entry))
        except Exception as exc:
            print(f"[Logic] SQLite store unavailable; retaining JSON fallback: {exc}")

    path = Path("AI_Children") / child / "memory" / "logic_memory.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    history.append(safe_entry)
    history = history[-250:]

    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Logic] Logged to logic_memory.json: {logic_entry['description']}")
    try:
        run_logic_map_builder()
    except Exception as e:
        print(f"[Logic] Failed to update logic map: {e}")

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

def rank_prediction_against_logic(prediction, symbol_words, transformer, limit=3):
    """Return calibrated alternatives so ambiguous meanings remain visible."""
    pred_vec = prediction.get("predicted_vector", {}).get("vector", [])
    if not pred_vec:
        return []

    candidates = []
    for word in symbol_words:
        if not isinstance(word, dict):
            continue
        existing = word.get("vector")
        if isinstance(existing, list) and existing:
            candidates.append((word, existing, None))
            continue
        summary = str(word.get("summary") or "").strip()
        if summary:
            candidates.append((word, None, {
                "summary": summary,
                "tags": word.get("tags", []),
                "emotions": {"trust": 0.6},
            }))

    missing = [fragment for _word, vector, fragment in candidates if vector is None]
    encoded_missing = iter(transformer.encode_many(missing) if missing else [])
    ranked = []
    for word, existing, _fragment in candidates:
        avg_vec = existing if existing is not None else next(encoded_missing).get("vector", [])
        similarity = cosine_similarity(pred_vec, avg_vec, epsilon=0.0)
        ranked.append({
            "symbol_word_id": word.get("symbol_word_id"),
            "summary": str(word.get("summary") or ""),
            "similarity": float(similarity),
        })
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[:max(1, int(limit))]


def test_prediction_against_logic(prediction, symbol_words, transformer):
    ranked = rank_prediction_against_logic(prediction, symbol_words, transformer, limit=1)
    if not ranked:
        return None, 0.0 if prediction.get("predicted_vector", {}).get("vector") else None
    return ranked[0]["symbol_word_id"], ranked[0]["similarity"]

def logic_session():
    config = load_config()
    child = config.get("current_child", "default_child")
    prediction = load_prediction(child)
    if not prediction:
        print("[Logic] No prediction to test.")
        return

    transformer = FractalTransformer()
    symbol_words = load_symbol_words(child)

    # Prefer a structured emotion vector; fall back to the raw prediction vector
    predicted_emotion = prediction.get("emotion_snapshot", {}).get("values", {})
    if not predicted_emotion:
        pv = prediction.get("predicted_vector", {}).get("vector", [])
        predicted_emotion = {f"dim_{i}": v for i, v in enumerate(pv)} if pv else {}
    predicted_vector = prediction.get("predicted_vector", {}).get("vector", [])
    ranked_matches = rank_prediction_against_logic(prediction, symbol_words, transformer, limit=3)
    best_match = ranked_matches[0] if ranked_matches else {}
    symbol_word_id = best_match.get("symbol_word_id")
    sim = float(best_match.get("similarity") or 0.0)
    second_sim = float(ranked_matches[1].get("similarity") or 0.0) if len(ranked_matches) > 1 else 0.0
    match_margin = max(0.0, sim - second_sim)

    # Run some symbolic tests
    samples = evolve_logic_expressions([1.0, 2.5, 3.3])
    logic_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": predicted_emotion,
        "prediction_vector": predicted_vector,
        "symbol_word_id": symbol_word_id,
        "similarity": round(float(sim or 0.0), 4),
        "similarity_margin": round(match_margin, 4),
        "symbol_alternatives": [
            {"symbol_word_id": item.get("symbol_word_id"), "summary": item.get("summary"),
             "similarity": round(float(item.get("similarity") or 0.0), 4)}
            for item in ranked_matches
        ],
        "trace_tests": samples,
        "description": f"Logic check on predicted emotion: {max(predicted_emotion, key=predicted_emotion.get, default='unknown')}"
    }

    if float(sim or 0.0) < 0.5 and symbol_word_id:
        seed_self_question(f"Is my logic drifting from what {symbol_word_id} means?")
        suggest_precision_override(32, reason="logic drift")
    elif float(sim or 0.0) > 0.9 and match_margin >= 0.08 and symbol_word_id:
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
        resolution_reason = None

        if "why am i so awake" in q and current_emotions.get("intensity", 0) > 0.8:
            resolution_reason = "high intensity"
        elif "why do i feel so drained" in q and current_emotions.get("intensity", 0) < 0.3:
            resolution_reason = "low intensity"
        elif "why was i forced to wake up" in q and get_inastate("runtime_disruption"):
            resolution_reason = "runtime disruption"
        elif "why can't i hear clearly" in q and get_inastate("audio_comfort") == "just right":
            resolution_reason = "audio comfort resolved"
        elif "why is everything so loud" in q and get_inastate("audio_comfort") != "too loud":
            resolution_reason = "audio normalized"
        elif "should i be thinking more precisely" in q and get_inastate("current_precision") >= 32:
            resolution_reason = "precision increased"

        if resolution_reason:
            note_with_status = {**note, "resolved": True, "resolved_reason": resolution_reason}
            resolved.append(note_with_status)
            mark_self_question_resolved(note.get("question"), resolution_reason)
            cleared += 1
        else:
            keep.append(note)

    reflection["self_notes"] = keep
    reflection["resolved_notes"] = resolved[-100:]

    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

    print(f"[Logic] Resolved {cleared} self questions.")



if __name__ == "__main__":
    logic_session()
