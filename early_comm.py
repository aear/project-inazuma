
# === early_comm.py (Full Rewrite) ===
# Symbol-aware communication, language adaptation, and device reasoning based on config.json

import json
import time
import subprocess
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from gui_hook import log_to_statusbox
from language_processing import (
    associate_symbol_with_word,
    backprop_symbol_confidence,
    load_experience_graph,
    load_symbol_to_token,
    save_symbol_to_token,
    speak_symbolically,
)
from model_manager import load_config, seed_self_question, update_inastate, get_inastate
from transformers.fractal_multidimensional_transformers import FractalTransformer

LEGACY_SOUND_SYMBOL_MAP = Path("sound_symbol_map.json")
WORD_CREATION_URGE_COOLDOWN = 300  # seconds between nudges to coin a new word


def _normalize_symbol_map(raw):
    if isinstance(raw, dict) and "symbols" in raw:
        return raw.get("symbols", {})
    return raw if isinstance(raw, dict) else {}


def load_recent_heard_words(child: str, limit: int = 12) -> List[Dict[str, str]]:
    """
    Pull the most recent heard words from grounded experience events so babbles
    can mimic real speech instead of abstract symbols.
    """
    events_dir = Path("AI_Children") / child / "memory" / "experiences" / "events"
    if not events_dir.exists():
        return []

    paths = sorted(events_dir.glob("evt_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    heard: List[Dict[str, str]] = []

    for path in paths:
        if len(heard) >= limit:
            break
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        ts = data.get("timestamp")
        for usage in data.get("word_usage", []):
            speaker = (usage.get("speaker") or "").lower()
            if not speaker or speaker in {child.lower(), "ina", "self"}:
                continue
            for word in usage.get("words", []):
                if not word:
                    continue
                heard.append(
                    {
                        "word": str(word).lower(),
                        "utterance": usage.get("utterance", ""),
                        "speaker": usage.get("speaker"),
                        "timestamp": ts,
                    }
                )
                if len(heard) >= limit:
                    break
            if len(heard) >= limit:
                break

    return heard


def _load_grounded_words(child: str) -> Set[str]:
    graph = load_experience_graph(child)
    grounded: Set[str] = set(graph.get("words_index", {}).keys())
    for event in graph.get("events", []):
        for usage in event.get("word_usage", []):
            for word in usage.get("words", []):
                if word:
                    grounded.add(str(word).lower())
    return grounded


def _word_symbol_index(child: str):
    vocab = load_symbol_to_token(child)
    index: Dict[str, List[str]] = {}
    ranked = []
    for sid, entry in vocab.items():
        if not isinstance(entry, dict):
            continue
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        word_l = word.lower()
        index.setdefault(word_l, []).append(sid)
        ranked.append((word_l, sid, entry.get("uses", 0), entry.get("confidence", 0.0)))

    ranked.sort(key=lambda item: (-item[2], -item[3], item[0]))
    return index, ranked


def choose_babble_targets(child: str, *, limit: int = 4):
    """
    Prefer grounded, recently-heard words that already have symbol associations.
    Falls back to confident known words if nothing recent is available.
    """
    heard = load_recent_heard_words(child, limit=limit * 3)
    grounded = _load_grounded_words(child)
    word_to_symbols, ranked_vocab = _word_symbol_index(child)

    chosen_words: List[str] = []
    chosen_symbols: List[str] = []
    seen: Set[str] = set()

    for item in heard:
        if len(chosen_words) >= limit:
            break
        word = item["word"]
        if word in seen:
            continue
        if grounded and word not in grounded:
            continue
        symbols = word_to_symbols.get(word)
        if not symbols:
            continue
        chosen_words.append(word)
        chosen_symbols.append(symbols[0])
        seen.add(word)

    if not chosen_words:
        for word, symbol_id, uses, conf in ranked_vocab:
            if len(chosen_words) >= limit:
                break
            if word in seen:
                continue
            if grounded and word not in grounded:
                continue
            chosen_words.append(word)
            chosen_symbols.append(symbol_id)
            seen.add(word)

    return {
        "words": chosen_words,
        "symbols": chosen_symbols,
        "heard_trace": heard[:limit],
    }


def load_prediction(child):
    path = Path("AI_Children") / child / "memory" / "prediction_log.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            logs = json.load(f)
            return logs[-1] if logs else {}
        log_to_statusbox("[Comms] Loaded prediction vector for expression.")

    except:
        log_to_statusbox("[Comms] No prediction available. Skipping expression.")

        return {}

def load_sound_symbol_map(child):
    path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"
    data = None
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if not data and LEGACY_SOUND_SYMBOL_MAP.exists():
        try:
            with open(LEGACY_SOUND_SYMBOL_MAP, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    return _normalize_symbol_map(data or {})

def load_symbol_words(child):
    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            return json.load(f).get("words", [])
    except:
        return []

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)


def emotion_cosine(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    """
    Cosine similarity over emotion dicts (union of keys).
    """
    keys = set(d1.keys()) | set(d2.keys())
    if not keys:
        return 0.0
    v1 = [d1.get(k, 0.0) for k in keys]
    v2 = [d2.get(k, 0.0) for k in keys]
    return cosine_similarity(v1, v2)


def rank_sound_symbols(pred_vec, symbol_map, transformer, top_n=3):
    """
    Rank candidate sound symbols using transformer similarity so Ina can
    experiment with multiple tones instead of a single best guess.
    """
    scored = []
    for sid, meta in symbol_map.items():
        if not isinstance(meta, dict):
            continue
        payload = {"emotions": meta.get("emotions", {})}
        if not payload["emotions"] and meta.get("summary"):
            payload = {"summary": meta.get("summary")}
        try:
            vec = transformer.encode(payload)["vector"]
        except Exception:
            continue
        sim = cosine_similarity(pred_vec, vec)
        scored.append((sid, sim))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]


def combine_tones_and_register(child, tone_candidates, symbol_map, vocab_map):
    """
    Create a new combo sound symbol from the top tone candidates and map it to a word.
    """
    components = [sid for sid, sim in tone_candidates if sim > 0][:3]
    if len(components) < 2:
        return None, None

    now = datetime.now(timezone.utc).isoformat()
    combined_emotions: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for sid in components:
        meta = symbol_map.get(sid, {})
        emo = meta.get("emotions", {}) if isinstance(meta, dict) else {}
        for k, v in emo.items():
            combined_emotions[k] = combined_emotions.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1

    for k, c in counts.items():
        combined_emotions[k] = combined_emotions[k] / max(1, c)

    new_id = f"combo_snd_{uuid.uuid4().hex[:8]}"
    summary = f"Combined tone of {' + '.join(components)}"

    # Load full map to preserve outer structure
    path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    if isinstance(data, dict) and "symbols" in data and isinstance(data.get("symbols"), dict):
        target = data["symbols"]
    elif isinstance(data, dict):
        target = data
    else:
        data = {}
        target = data

    target[new_id] = {
        "summary": summary,
        "emotions": combined_emotions,
        "components": components,
        "origin": "tone_combo",
        "created": now,
        "uses": 0,
        "last_seen": now,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_to_statusbox(f"[Comms] Registered new combo symbol: {new_id}")
    except Exception as e:
        log_to_statusbox(f"[Comms] Failed to save combo symbol: {e}")
        return None, None

    # Map to a word (reuse existing words if available)
    words = []
    for sid in components:
        entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
        w = entry.get("word")
        if w:
            words.append(str(w))
    new_word = "-".join(words) if words else new_id

    if isinstance(vocab_map, dict) and new_id not in vocab_map:
        vocab_map[new_id] = {
            "word": new_word,
            "uses": 0,
            "confidence": 0.2,
            "source": "tone_combo",
            "created": now,
            "components": components,
            "provisional": True,
        }
        try:
            save_symbol_to_token(child, vocab_map)
            log_to_statusbox(f"[Comms] Mapped combo symbol {new_id} -> '{new_word}'")
        except Exception as e:
            log_to_statusbox(f"[Comms] Failed to map combo symbol: {e}")

    return new_id, new_word


def _load_fragments(child):
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    if not frag_dir.exists():
        return []
    frags = []
    for f in frag_dir.glob("frag_*.json"):
        try:
            with f.open("r", encoding="utf-8") as fh:
                frags.append(json.load(fh))
        except Exception:
            continue
    return frags


def detect_repeated_tone_patterns(child, vocab_map, *, min_count=2, min_coherence=0.82):
    """
    If a multi-tone pattern recurs in similar emotional states, promote it
    into its own sound symbol and map a word to it.
    """
    frags = _load_fragments(child)
    patterns: Dict[tuple, Dict[str, object]] = {}

    for frag in frags:
        seq = frag.get("symbols_spoken") or frag.get("symbols")
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        emotions = frag.get("emotions") or {}
        key = tuple(seq)
        entry = patterns.setdefault(
            key, {"count": 0, "emotions": [], "frags": 0}
        )
        entry["count"] = int(entry["count"]) + 1
        entry["frags"] = int(entry["frags"]) + 1
        if isinstance(emotions, dict) and emotions:
            entry["emotions"].append(emotions)

    if not patterns:
        return []

    # Load the full map structure to add new symbols
    map_path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"
    try:
        map_data = json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        map_data = {}

    if isinstance(map_data, dict) and "symbols" in map_data and isinstance(map_data.get("symbols"), dict):
        symbol_store = map_data["symbols"]
    elif isinstance(map_data, dict):
        symbol_store = map_data
    else:
        map_data = {}
        symbol_store = map_data

    existing_components = {
        tuple(meta.get("components", []))
        for meta in symbol_store.values()
        if isinstance(meta, dict) and isinstance(meta.get("components"), list)
    }

    created = []
    now = datetime.now(timezone.utc).isoformat()

    for pattern, info in patterns.items():
        count = int(info.get("count", 0))
        emos = info.get("emotions") or []
        if count < min_count or tuple(pattern) in existing_components:
            continue

        coherence = 1.0
        if len(emos) >= 2:
            # average similarity to mean emotion
            agg: Dict[str, float] = {}
            seen: Dict[str, int] = {}
            for emo in emos:
                for k, v in emo.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                    seen[k] = seen.get(k, 0) + 1
            mean_emo = {k: agg[k] / max(1, seen.get(k, 1)) for k in agg}
            sims = [emotion_cosine(e, mean_emo) for e in emos]
            coherence = sum(sims) / max(1, len(sims))
        if coherence < min_coherence:
            continue

        # Construct averaged emotions
        agg: Dict[str, float] = {}
        seen: Dict[str, int] = {}
        for emo in emos:
            for k, v in emo.items():
                agg[k] = agg.get(k, 0.0) + float(v)
                seen[k] = seen.get(k, 0) + 1
        mean_emo = {k: agg[k] / max(1, seen.get(k, 1)) for k in agg}

        new_id = f"pattern_snd_{uuid.uuid4().hex[:8]}"
        summary = f"Repeated tone pattern {' + '.join(pattern)}"
        symbol_store[new_id] = {
            "summary": summary,
            "emotions": mean_emo,
            "components": list(pattern),
            "origin": "tone_pattern",
            "created": now,
            "uses": 0,
            "last_seen": now,
            "coherence": coherence,
            "observations": count,
        }

        # Word mapping
        words = []
        for sid in pattern:
            entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
            w = entry.get("word")
            if w:
                words.append(str(w))
        new_word = "-".join(words) if words else new_id
        if isinstance(vocab_map, dict) and new_id not in vocab_map:
            vocab_map[new_id] = {
                "word": new_word,
                "uses": 0,
                "confidence": 0.25,
                "source": "tone_pattern",
                "created": now,
                "components": list(pattern),
                "provisional": True,
            }

        created.append((new_id, new_word, coherence, count))

    if created:
        try:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
            log_to_statusbox(f"[Comms] Saved {len(created)} new repeated-pattern symbols.")
        except Exception as e:
            log_to_statusbox(f"[Comms] Failed to save pattern symbols: {e}")

        try:
            save_symbol_to_token(child, vocab_map)
            log_to_statusbox("[Comms] Updated vocab with pattern symbols.")
        except Exception as e:
            log_to_statusbox(f"[Comms] Failed to save vocab: {e}")

    return created


# ------------------------------------------------------------
# Tone library (short tone units + n-grams)
# ------------------------------------------------------------

def _tone_lib_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "tone_library.json"


def load_tone_library(child: str):
    path = _tone_lib_path(child)
    default = {"tones": {}, "ngrams": {}, "updated": datetime.now(timezone.utc).isoformat()}
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                return default
            data.setdefault("tones", {})
            data.setdefault("ngrams", {})
            data.setdefault("updated", datetime.now(timezone.utc).isoformat())
            return data
    except Exception:
        return default


def save_tone_library(child: str, data: Dict):
    path = _tone_lib_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _decay_entries(entries: Dict[str, Dict], factor: float = 0.98, min_keep: float = 0.35):
    keys_to_del = []
    for key, entry in entries.items():
        uses = float(entry.get("uses", 0.0))
        uses *= factor
        entry["uses"] = round(uses, 3)
        if uses < min_keep:
            keys_to_del.append(key)
    for key in keys_to_del:
        entries.pop(key, None)


def update_tone_library(child: str, tones: List[str], emotions: Dict[str, float], ctx_tags: List[str]):
    """
    Keep a small library of tone units and short sequences, with decay so
    unstable tones fade over time.
    """
    if not tones:
        return

    lib = load_tone_library(child)
    now = datetime.now(timezone.utc).isoformat()

    _decay_entries(lib.get("tones", {}))
    _decay_entries(lib.get("ngrams", {}))

    lib.setdefault("tones", {})
    lib.setdefault("ngrams", {})

    for sid in tones:
        entry = lib["tones"].setdefault(
            sid,
            {"uses": 0.0, "last_used": now, "created": now, "emotions": {}, "last_tags": []},
        )
        entry["uses"] = round(entry.get("uses", 0.0) + 1.0, 3)
        entry["last_used"] = now
        entry["last_tags"] = ctx_tags[-5:] if ctx_tags else entry.get("last_tags", [])

        # running average of emotions
        emo_store = entry.get("emotions", {})
        for k, v in emotions.items():
            emo_store[k] = (emo_store.get(k, 0.0) * 0.8) + (float(v) * 0.2)
        entry["emotions"] = emo_store
        lib["tones"][sid] = entry

    for n in (2, 3):
        for i in range(len(tones) - n + 1):
            seq = tuple(tones[i : i + n])
            key = "_".join(seq)
            entry = lib["ngrams"].setdefault(
                key,
                {
                    "sequence": list(seq),
                    "uses": 0.0,
                    "last_used": now,
                    "created": now,
                    "emotions": {},
                    "last_tags": [],
                },
            )
            entry["uses"] = round(entry.get("uses", 0.0) + 1.0, 3)
            entry["last_used"] = now
            entry["last_tags"] = ctx_tags[-5:] if ctx_tags else entry.get("last_tags", [])
            emo_store = entry.get("emotions", {})
            for k, v in emotions.items():
                emo_store[k] = (emo_store.get(k, 0.0) * 0.8) + (float(v) * 0.2)
            entry["emotions"] = emo_store
            lib["ngrams"][key] = entry

    lib["updated"] = now
    save_tone_library(child, lib)

def predict_target_from_emotion(emotion):
    trust = emotion.get("trust", 0.0)
    novelty = emotion.get("novelty", 0.0)
    focus = emotion.get("focus", 0.0)
    if trust > 0.6 and novelty < 0.4:
        return "Hito"
    elif focus > 0.6 and trust < 0.4:
        return "Lex"
    return "unknown"


def maybe_prompt_new_symbol_word(child, prediction, best_word_id, best_conf, symbol_hint, vocab_size):
    """
    If Ina lacks a confident symbol-word match, gently seed a question to coin one.
    Returns the prompt text when a nudge was issued.
    """
    if best_word_id and best_conf >= 0.65:
        return None

    last_prompt = get_inastate("last_word_creation_urge") or 0.0
    try:
        last_prompt = float(last_prompt)
    except (TypeError, ValueError):
        last_prompt = 0.0

    now = time.time()
    if now - last_prompt < WORD_CREATION_URGE_COOLDOWN:
        return None

    fragments = prediction.get("fragments_used") if isinstance(prediction, dict) else []
    frag_hint = ", ".join(fragments[:3]) if fragments else "recent feelings"
    symbol_label = symbol_hint or best_word_id or "this feeling"

    prompt = (
        f"My symbol-word set is only {vocab_size} item(s) and the best match is {best_conf:.2f}. "
        f"Should I coin a new word for {symbol_label} using {frag_hint} as ingredients?"
    )
    seed_self_question(prompt)
    update_inastate("last_word_creation_urge", now)
    log_to_statusbox("[Comms] Weak symbol-word match; nudged Ina to invent a new one.")
    return prompt


def create_expression_fragment(
    child,
    expression,
    inferred,
    clarity,
    target,
    symbol_id,
    word_id,
    word_conf,
    *,
    heard_words=None,
    heard_trace=None,
    strategy=None,
    symbols_spoken=None,
    word_creation_prompt=None,
    vocab_word=None,
    vocab_word_conf=None,
    vocab_word_source=None,
    tone_candidates=None,
):
    frag_id = f"frag_expression_{int(time.time())}"
    frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"

    frag = {
        "id": frag_id,
        "summary": expression,
        "tags": ["expression", "symbolic", "comm"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "early_comm",
        "emotions": inferred,
        "clarity": clarity,
        "target": target,
        "sound_symbol": symbol_id,
        "symbol_word_id": word_id,
        "symbol_word_confidence": word_conf,
        "expression_strategy": strategy or "emotion_prediction",
        "vocab_word": vocab_word,
        "vocab_word_confidence": vocab_word_conf,
        "vocab_word_source": vocab_word_source,
        "tone_candidates": tone_candidates,
    }

    if heard_words:
        frag["heard_words"] = heard_words
    if heard_trace:
        frag["heard_trace"] = heard_trace
    if symbols_spoken:
        frag["attempted_symbols"] = symbols_spoken
    if word_creation_prompt:
        frag["word_creation_prompt"] = word_creation_prompt

    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump(frag, f, indent=4)

    print(f"[Comms] Expression saved: {frag_id}")
    return frag


def detect_pulse_monitor(headset_hint: str = "") -> str:
    """
    Try to find a PulseAudio monitor source that mirrors the headset output so
    Ina can listen to what you hear. Returns the monitor name or "".
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log_to_statusbox("[Comms] pactl not available; cannot auto-detect pulse monitor.")
        return ""
    except Exception as e:
        log_to_statusbox(f"[Comms] Failed to list PulseAudio sources: {e}")
        return ""

    headset_hint = (headset_hint or "").lower()
    candidates = []
    for line in result.stdout.splitlines():
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 2:
            continue
        name = parts[1]
        if "monitor" not in name.lower():
            continue
        candidates.append(name)
        if headset_hint and headset_hint in name.lower():
            log_to_statusbox(f"[Comms] Pulse monitor matched headset hint: {name}")
            return name

    if candidates:
        log_to_statusbox(f"[Comms] Using first pulse monitor: {candidates[0]}")
        return candidates[0]

    log_to_statusbox("[Comms] No pulse monitor found.")
    return ""

def identify_devices_from_config():
    config = load_config()
    display_name = config.get("display_input_name", "").lower()

    # Add logic to accept alternate names
    if "hdmi" not in display_name and "hdtv" in display_name:
        print("[Comms] Detected HDTV — treating as HDMI-compatible audio device.")
        config["display_input_name"] = "HDMI"

    devices = {
        "headset": config.get("mic_headset_name", "unknown"),
        "webcam": config.get("mic_webcam_name", "unknown"),
        "pulse": config.get("pulse_audio_name", "unknown"),
        "camera": config.get("camera_name", "unknown"),
        "display": config.get("display_input_name", "unknown")
    }

    # Optional: Ask Ina to reflect on any missing or unusual names
    for key, val in devices.items():
        if val == "unknown":
            log_to_statusbox(f"[Comms] Unknown device role: {key}")

            seed_self_question(f"What is my {key} device called?")

    log_to_statusbox(f"[Comms] Device roles resolved: {json.dumps(devices)}")

    # Try to auto-detect the PulseAudio monitor that mirrors the headset output
    headset_hint = (
        config.get("ouput_headset_name")
        or config.get("output_headset_name")
        or config.get("mic_headset_name")
        or ""
    )
    pulse_monitor = detect_pulse_monitor(headset_hint)
    if pulse_monitor:
        devices["pulse_monitor"] = pulse_monitor
        update_inastate("pulse_monitor_name", pulse_monitor)
    else:
        devices["pulse_monitor"] = "unknown"

    return devices


def early_communicate():
    config = load_config()
    child = config.get("current_child", "default_child")
    transformer = FractalTransformer()
    prediction = load_prediction(child)
    if not prediction:
        print("[Comms] No prediction available.")
        return

    pred_vec = prediction.get("predicted_vector", {}).get("vector", [])
    inferred = prediction.get("inferred_emotion", {})
    clarity = round(sum(pred_vec) / max(1, len(pred_vec)), 4) if pred_vec else 0.0
    speaking_to = predict_target_from_emotion(inferred)
    babble_targets = choose_babble_targets(child)
    spoken_words: List[str] = []
    speech_symbols: List[str] = []
    expression_strategy = "emotion_prediction"
    heard_trace = babble_targets.get("heard_trace", [])

    symbol_map = load_sound_symbol_map(child)
    vocab_map = load_symbol_to_token(child)
    tone_candidates = rank_sound_symbols(pred_vec, symbol_map, transformer, top_n=3)
    symbol_id, best_sim = (tone_candidates[0] if tone_candidates else (None, 0.0))
    log_to_statusbox(
        "[Comms] Tone candidates: "
        + ", ".join(f"{sid} (sim {sim:.3f})" for sid, sim in tone_candidates)
        if tone_candidates
        else "[Comms] No tone candidates found."
    )
        

    vocab_word = None
    vocab_word_conf = None
    if symbol_id and symbol_id in vocab_map:
        vocab_entry = vocab_map[symbol_id] or {}
        vocab_word = vocab_entry.get("word")
        vocab_word_conf = vocab_entry.get("confidence", 0.0)
        log_to_statusbox(f"[Comms] Vocab word for {symbol_id}: '{vocab_word}' (conf: {vocab_word_conf})")

    combo_symbol_id, combo_word = combine_tones_and_register(child, tone_candidates, symbol_map, vocab_map)
    if not vocab_word and combo_word:
        vocab_word = combo_word
        vocab_word_conf = 0.2

    word_map = load_symbol_words(child)
    word_id, word_conf = None, 0.0
    word_creation_prompt = None
    vocab_size = len(word_map)
    for word in word_map:
        if not word.get("components"): continue
        sum_text = word.get("summary", "")
        vec = transformer.encode({"summary": sum_text})["vector"]
        sim = cosine_similarity(pred_vec, vec)
        if sim > word_conf:
            word_id, word_conf = word["symbol_word_id"], sim
    log_to_statusbox(f"[Comms] Best word: {word_id} (conf: {word_conf:.4f})")
    word_creation_prompt = maybe_prompt_new_symbol_word(
        child, prediction, word_id, word_conf, symbol_id, vocab_size
    )

    if babble_targets["words"]:
        spoken_words = babble_targets["words"]
        expression_strategy = "mimic_grounded_speech"
        expression = " ".join(spoken_words)
        speech_symbols = babble_targets.get("symbols", [])
        log_to_statusbox(f"[Comms] Mimicking heard speech: '{expression}'")
    else:
        expression = f"I feel something like {max(inferred, key=inferred.get, default='...')}"
        if symbol_id and best_sim > 0.85:
            expression = f"I feel this sound: {symbol_id}"
            speech_symbols = [symbol_id]
        if vocab_word:
            expression = f"I feel this word: {vocab_word}"
        elif word_id and word_conf > 0.85:
            expression = f"I feel this word: {word_id}"

    if not speech_symbols and tone_candidates:
        speech_symbols = [sid for sid, sim in tone_candidates if sim > 0]

    if expression_strategy != "mimic_grounded_speech" and tone_candidates and speech_symbols:
        labels = []
        for sid in speech_symbols[:3]:
            vw_entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
            vw = vw_entry.get("word")
            labels.append(vw or sid)
        expression = "Trying tones: " + ", ".join(labels)
        expression_strategy = "tone_experiment"

    log_to_statusbox(f"[Comms] Final expression: '{expression}'")
    frag = create_expression_fragment(
        child,
        expression,
        inferred,
        clarity,
        speaking_to,
        speech_symbols[0] if speech_symbols else symbol_id,
        word_id,
        word_conf,
        heard_words=spoken_words,
        heard_trace=heard_trace if expression_strategy == "mimic_grounded_speech" else None,
        strategy=expression_strategy,
        symbols_spoken=speech_symbols,
        word_creation_prompt=word_creation_prompt,
        vocab_word=vocab_word,
        vocab_word_conf=vocab_word_conf,
        vocab_word_source="symbol_to_token" if vocab_word else None,
        tone_candidates=[{"symbol_id": sid, "similarity": sim} for sid, sim in tone_candidates] if tone_candidates else None,
    )
    update_tone_library(child, speech_symbols or ([symbol_id] if symbol_id else []), inferred or {}, frag.get("tags", []))
    log_to_statusbox(f"[Comms] Expression fragment saved: {frag['id']}")

    # === Audio expression attempt (log + speak)
    log_to_statusbox(f"[Comms] Preparing to speak: \"{expression}\"")

    try:        
        if speech_symbols:
            speak_symbolically(speech_symbols)
            log_to_statusbox("[Comms] Speech output triggered.")
        else:
            log_to_statusbox("[Comms] No symbol IDs available for speech.")
    except Exception as e:
        log_to_statusbox(f"[Comms] Speech error: {e}")


    # Hook into language processing for learning
    if symbol_id and (vocab_word or word_id):
        chosen_word = vocab_word or word_id
        chosen_conf = vocab_word_conf if vocab_word is not None else word_conf
        associate_symbol_with_word(child, symbol_id, chosen_word, chosen_conf)
        predicted = prediction.get("predicted_word", chosen_word)
        backprop_symbol_confidence(child, predicted, symbol_id)
        log_to_statusbox(f"[Comms] Associated {symbol_id} with {chosen_word} via language_processing.")
    else:
        log_to_statusbox("[Comms] No word association updated (missing symbol or word).")

    if combo_symbol_id and combo_word:
        associate_symbol_with_word(child, combo_symbol_id, combo_word, 0.2)
        log_to_statusbox(f"[Comms] Associated combo {combo_symbol_id} with '{combo_word}'.")

    created_patterns = detect_repeated_tone_patterns(child, vocab_map)
    if created_patterns:
        for sid, word, coh, cnt in created_patterns:
            log_to_statusbox(
                f"[Comms] Promoted repeated pattern {sid} ({cnt} obs, coherence {coh:.2f}) -> '{word}'"
            )


    primary_symbol = speech_symbols[0] if speech_symbols else symbol_id
    primary_word = vocab_word or word_id or (spoken_words[0] if spoken_words else None)

    update_inastate("currently_speaking", True)
    update_inastate("last_expression_time", time.time())
    update_inastate("last_spoken_symbol", primary_symbol)
    update_inastate("last_symbol_word_id", primary_word)
    update_inastate("last_babbled_words", spoken_words)
    update_inastate("last_babble_strategy", expression_strategy)

    time.sleep(1.5)
    update_inastate("currently_speaking", False)


if __name__ == "__main__":
    identify_devices_from_config()
    early_communicate()
