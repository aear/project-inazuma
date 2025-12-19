
# === early_comm.py (Full Rewrite) ===
# Symbol-aware communication, language adaptation, and device reasoning based on config.json

import json
import math
import time
import subprocess
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from embedding_stack import MultimodalEmbedder, guess_language_code
from gui_hook import log_to_statusbox
from language_processing import (
    associate_symbol_with_word,
    backprop_symbol_confidence,
    load_experience_graph,
    load_symbol_to_token,
    save_symbol_to_token,
    speak_symbolically,
)
from model_manager import load_config, seed_self_question, update_inastate, get_inastate, append_typed_outbox_entry
from social_map import get_high_trust_contacts, get_owner_user_id
from transformers.fractal_multidimensional_transformers import FractalTransformer
from symbol_generator import generate_symbol_from_parts

LEGACY_SOUND_SYMBOL_MAP = Path("sound_symbol_map.json")
WORD_CREATION_URGE_COOLDOWN = 300  # seconds between nudges to coin a new word
TYPE_CONTACT_COOLDOWN = 180  # seconds between proactive typed contacts
EMBEDDER = MultimodalEmbedder(dim=128)


def _normalize_symbol_map(raw):
    if isinstance(raw, dict) and "symbols" in raw:
        return raw.get("symbols", {})
    return raw if isinstance(raw, dict) else {}


def _proto_confidence(uses: int, base: float = 0.2) -> float:
    uses = max(1, int(uses))
    return round(min(0.9, base + math.log1p(uses) / 5.0), 3)


def _ensure_vocab_embeddings(vocab_map: Dict[str, Any], language_hint: Optional[str] = None) -> bool:
    """
    Attach text embeddings and language hints to vocab entries if missing.
    Returns True if any entry was modified.
    """
    updated = False
    for entry in vocab_map.values():
        if not isinstance(entry, dict):
            continue
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        emb = entry.get("embedding")
        if isinstance(emb, list) and emb:
            continue
        lang = entry.get("language") or language_hint or guess_language_code(word)
        entry["language"] = lang
        entry["embedding"] = EMBEDDER.embed_text(word, language=lang)
        updated = True
    return updated


def _discord_voice_preference(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return a compact preference block when Discord voice should be used for sound play.
    """
    discord_cfg = config.get("discord") if isinstance(config, dict) else None
    if not isinstance(discord_cfg, dict):
        return None
    if not discord_cfg.get("enabled"):
        return None
    if not discord_cfg.get("prefer_voice_for_sounds"):
        return None

    channel_name = discord_cfg.get("voice_channel_name")
    channel_id = discord_cfg.get("voice_channel_id")
    if not channel_name and not channel_id:
        return None

    return {
        "backend": "discord",
        "channel_name": channel_name,
        "channel_id": channel_id,
        "label": discord_cfg.get("voice_label", "discord_voice"),
    }


def load_symbol_word_state(child: str):
    """
    Load full symbol_word state, including proto and multi-symbol pairs.
    """
    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    default = {"words": [], "proto_words": {}, "multi_symbol_words": {}}
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return default

    if not isinstance(data, dict):
        return default

    data.setdefault("words", [])
    data.setdefault("proto_words", {})
    data.setdefault("multi_symbol_words", {})
    return data


def select_proto_pair_word(tone_candidates, proto_words, multi_symbol_words):
    """
    Pick a multi-symbol word candidate using top tone symbols to encourage paired words.
    """
    symbols = [sid for sid, sim in tone_candidates if sid]
    best = None
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        key_core = "_".join(pair)
        pair_key = f"pair:{key_core}"

        entry = multi_symbol_words.get(pair_key) if isinstance(multi_symbol_words, dict) else None
        source = "multi_symbol_words"
        if not isinstance(entry, dict):
            entry = proto_words.get(key_core) if isinstance(proto_words, dict) else None
            source = "proto_words"

        if isinstance(entry, dict):
            uses = int(entry.get("uses", 0))
            confidence = float(entry.get("confidence", _proto_confidence(uses, base=0.18)))
            if confidence < 0.1:
                continue
            flexible = bool(entry.get("flexible", uses < 5))
            candidate = {
                "key": pair_key,
                "sequence": list(pair),
                "confidence": confidence,
                "flexible": flexible,
                "source": source,
            }
            if not best or candidate["confidence"] > best["confidence"]:
                best = candidate

    if best:
        return best

    if len(symbols) >= 2:
        pair = symbols[:2]
        return {
            "key": f"pair:{'_'.join(pair)}",
            "sequence": pair,
            "confidence": 0.12,
            "flexible": True,
            "source": "new_pair",
        }
    return None


def record_proto_pair_usage(child: str, pair_info: Dict[str, object]):
    if not pair_info:
        return

    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    proto_store = data.setdefault("proto_words", {})
    multi_store = data.setdefault("multi_symbol_words", {})
    now = datetime.now(timezone.utc).isoformat()

    seq = pair_info.get("sequence") or []
    if not isinstance(seq, list) or len(seq) < 2:
        return
    proto_key = "_".join(seq)
    pair_key = f"pair:{proto_key}"

    proto_entry = proto_store.get(proto_key, {"sequence": list(seq), "created": now})
    proto_uses = int(proto_entry.get("uses", 0)) + 1
    proto_entry.update(
        {
            "sequence": list(seq),
            "uses": proto_uses,
            "last_seen": now,
            "confidence": _proto_confidence(proto_uses, base=0.18),
            "flexible": proto_uses < 5,
            "stability": round(min(1.0, proto_uses / 10.0), 3),
            "length": len(seq),
        }
    )
    proto_store[proto_key] = proto_entry

    multi_entry = multi_store.get(pair_key, {"sequence": list(seq), "created": now, "source": "expression_pair"})
    multi_uses = int(multi_entry.get("uses", 0)) + 1
    multi_entry.update(
        {
            "sequence": list(seq),
            "uses": multi_uses,
            "last_seen": now,
            "confidence": _proto_confidence(multi_uses, base=0.24),
            "flexible": multi_uses < 4,
            "stability": round(min(1.0, multi_uses / 8.0), 3),
            "source": multi_entry.get("source", "expression_pair"),
        }
    )
    multi_store[pair_key] = multi_entry
    data["updated_at"] = now

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_to_statusbox(f"[Comms] Paired word '{pair_key}' updated (uses={multi_uses}, flexible={multi_entry['flexible']}).")
    except Exception as e:
        log_to_statusbox(f"[Comms] Failed to record proto pair usage: {e}")


def _update_tone_diversity_metrics(
    child: str,
    symbols: List[str],
    *,
    window_seconds: int = 3600,
    max_entries: int = 60,
):
    """
    Track recent spoken symbols so Ina can notice repetition patterns herself.
    Stores both the rolling history and summary stats in inastate.
    """
    clean_symbols = [sid for sid in symbols if sid]
    if not clean_symbols:
        return

    now = time.time()
    raw_history = get_inastate("tone_voice_history")
    history: List[Dict[str, float]] = []
    if isinstance(raw_history, list):
        for entry in raw_history:
            sym = entry.get("symbol")
            ts = entry.get("ts")
            if not sym or not isinstance(ts, (int, float)):
                continue
            if now - float(ts) <= window_seconds:
                history.append({"symbol": sym, "ts": float(ts)})

    for sym in clean_symbols:
        history.append({"symbol": sym, "ts": now})

    history = history[-max_entries:]
    if not history:
        update_inastate("tone_voice_history", [])
        update_inastate("tone_voice_metrics", None)
        return

    counts: Dict[str, int] = {}
    for entry in history:
        sym = entry["symbol"]
        counts[sym] = counts.get(sym, 0) + 1

    total = len(history)
    unique_symbols = len(counts)
    dominant_symbol, dominant_count = max(counts.items(), key=lambda kv: kv[1])

    streak_symbol = history[-1]["symbol"]
    streak_len = 0
    for entry in reversed(history):
        if entry["symbol"] == streak_symbol:
            streak_len += 1
        else:
            break

    metrics = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "window_seconds": window_seconds,
        "samples": total,
        "unique_symbols": unique_symbols,
        "unique_ratio": round(unique_symbols / max(1, total), 3),
        "dominant_symbol": dominant_symbol,
        "dominant_ratio": round(dominant_count / max(1, total), 3),
        "current_streak_symbol": streak_symbol,
        "current_streak_length": streak_len,
        "recent_symbols": [entry["symbol"] for entry in history[-6:]],
    }
    if total >= 5 and metrics["dominant_ratio"] >= 0.8:
        metrics["observation"] = "single_symbol_dominant"

    update_inastate("tone_voice_history", history)
    update_inastate("tone_voice_metrics", metrics)


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
                lang = guess_language_code(str(word))
                heard.append(
                    {
                        "word": str(word).lower(),
                        "utterance": usage.get("utterance", ""),
                        "speaker": usage.get("speaker"),
                        "timestamp": ts,
                        "language": lang,
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
        ranked.append(
            (
                word_l,
                sid,
                entry.get("uses", 0),
                entry.get("confidence", 0.0),
                entry.get("language"),
            )
        )

    ranked.sort(key=lambda item: (-item[2], -item[3], item[0]))
    return index, ranked


def choose_babble_targets(
    child: str,
    *,
    limit: int = 4,
    language_hint: Optional[str] = None,
    heard: Optional[List[Dict[str, str]]] = None,
):
    """
    Prefer grounded, recently-heard words that already have symbol associations.
    Falls back to confident known words if nothing recent is available.
    """
    heard = heard if heard is not None else load_recent_heard_words(child, limit=limit * 3)
    lang_target = language_hint if language_hint not in {None, "und", ""} else None
    grounded = _load_grounded_words(child)
    word_to_symbols, ranked_vocab = _word_symbol_index(child)

    chosen_words: List[str] = []
    chosen_symbols: List[str] = []
    seen: Set[str] = set()

    for item in heard:
        if len(chosen_words) >= limit:
            break
        word = item["word"]
        lang = item.get("language") or guess_language_code(word)
        if lang_target and lang and lang != lang_target:
            continue
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
        for word, symbol_id, uses, conf, lang in ranked_vocab:
            if len(chosen_words) >= limit:
                break
            if lang_target and lang and lang != lang_target:
                continue
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
        "language_hint": lang_target or language_hint,
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


def combine_tones_and_register(child, tone_candidates, symbol_map, vocab_map, *, language_hint=None):
    """
    Create a new combo sound symbol from the top tone candidates and map it to a word.
    """
    components = [sid for sid, sim in tone_candidates if sim > 0][:3]
    if len(components) < 2:
        return None, None, None

    now = datetime.now(timezone.utc).isoformat()
    combined_emotions: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    component_langs: List[str] = []
    for sid in components:
        meta = symbol_map.get(sid, {})
        emo = meta.get("emotions", {}) if isinstance(meta, dict) else {}
        for k, v in emo.items():
            combined_emotions[k] = combined_emotions.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
        entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
        lang = entry.get("language")
        if lang:
            component_langs.append(str(lang))

    for k, c in counts.items():
        combined_emotions[k] = combined_emotions[k] / max(1, c)

    lang_choice = component_langs[0] if component_langs else (language_hint or "und")
    symbol_embedding = EMBEDDER.embed_symbol_sequence(components, language=lang_choice)

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
        "language": lang_choice,
        "symbol_embedding": symbol_embedding,
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
            "language": lang_choice,
            "embedding": EMBEDDER.embed_text(new_word, language=lang_choice),
            "symbol_embedding": symbol_embedding,
        }
        try:
            save_symbol_to_token(child, vocab_map)
            log_to_statusbox(f"[Comms] Mapped combo symbol {new_id} -> '{new_word}'")
        except Exception as e:
            log_to_statusbox(f"[Comms] Failed to map combo symbol: {e}")

    return new_id, new_word, lang_choice


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

        component_langs = []
        for sid in pattern:
            entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
            lang = entry.get("language")
            if lang:
                component_langs.append(str(lang))
        lang_choice = component_langs[0] if component_langs else "und"
        symbol_embedding = EMBEDDER.embed_symbol_sequence(list(pattern), language=lang_choice)

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
            "language": lang_choice,
            "symbol_embedding": symbol_embedding,
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
                "language": lang_choice,
                "embedding": EMBEDDER.embed_text(new_word, language=lang_choice),
                "symbol_embedding": symbol_embedding,
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


def _tone_alignment_score(current_emotions: Dict[str, float], tone_entry: Dict[str, object]) -> float:
    emotions = tone_entry.get("emotions", {}) if isinstance(tone_entry, dict) else {}
    if not isinstance(emotions, dict) or not emotions or not current_emotions:
        return 0.0
    return emotion_cosine(current_emotions, emotions)


def select_tone_riff(tone_library: Dict[str, Dict], current_emotions: Dict[str, float], *, min_uses: float = 0.5):
    """
    Surface an n-gram riff from the tone library so Ina can replay short tone phrases.
    """
    ngrams = tone_library.get("ngrams", {}) if isinstance(tone_library, dict) else {}
    candidates = []

    if isinstance(ngrams, dict):
        for key, entry in ngrams.items():
            seq = entry.get("sequence") if isinstance(entry, dict) else None
            if not seq and isinstance(key, str):
                seq = key.split("_")
            if not isinstance(seq, list) or len(seq) < 2:
                continue
            uses = float(entry.get("uses", 0.0)) if isinstance(entry, dict) else 0.0
            if uses < min_uses:
                continue
            alignment = _tone_alignment_score(current_emotions, entry if isinstance(entry, dict) else {})
            candidates.append(
                {
                    "sequence": seq,
                    "uses": uses,
                    "alignment": alignment,
                    "last_used": entry.get("last_used") if isinstance(entry, dict) else None,
                    "tags": entry.get("last_tags") if isinstance(entry, dict) else [],
                    "kind": "riff",
                    "key": key,
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item["uses"], item["alignment"], item.get("last_used") or ""), reverse=True)
    return candidates[0]


def select_rare_tone(tone_library: Dict[str, Dict], current_emotions: Dict[str, float]):
    """
    Pick an underused tone so Ina can stretch beyond her preferred sounds.
    """
    tones = tone_library.get("tones", {}) if isinstance(tone_library, dict) else {}
    candidates = []

    if isinstance(tones, dict):
        for sid, entry in tones.items():
            if not isinstance(entry, dict):
                continue
            uses = float(entry.get("uses", 0.0))
            alignment = _tone_alignment_score(current_emotions, entry)
            candidates.append(
                {
                    "symbol_id": sid,
                    "uses": uses,
                    "alignment": alignment,
                    "last_used": entry.get("last_used"),
                    "tags": entry.get("last_tags", []),
                    "kind": "rare_tone",
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item["uses"], -item["alignment"], item.get("last_used") or ""))
    return candidates[0]

def predict_target_from_emotion(emotion):
    trust = float(emotion.get("trust", 0.0) or 0.0)
    novelty = float(emotion.get("novelty", 0.0) or 0.0)
    attention = float(emotion.get("attention", 0.0) or emotion.get("focus", 0.0) or 0.0)
    connection = float(emotion.get("connection", 0.0) or 0.0)

    if trust > 0.6 and novelty < 0.4:
        return generate_symbol_from_parts("trust", "soft", "self")
    if attention > 0.6 and trust < 0.4:
        return generate_symbol_from_parts("focus", "sharp", "pattern")
    if connection > 0.55:
        return generate_symbol_from_parts("care", "moderate", "connection")
    mood = "curiosity" if novelty >= 0.3 else "calm"
    return generate_symbol_from_parts(mood, "spiral", "unknown")


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
    proto_word=None,
    tone_explorations=None,
    speech_suppressed=None,
    voice_target=None,
    language_hint=None,
    embedding=None,
    symbol_embedding=None,
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
    if proto_word:
        frag["proto_word"] = proto_word
    if tone_explorations:
        frag["tone_explorations"] = tone_explorations
    if speech_suppressed:
        frag["speech_suppressed"] = speech_suppressed
    if voice_target:
        frag["voice_target"] = voice_target
    if language_hint:
        frag["language_hint"] = language_hint
    if embedding:
        frag["embedding"] = embedding
    if symbol_embedding:
        frag["symbol_embedding"] = symbol_embedding

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
    voice_pref = _discord_voice_preference(config)
    voice_delivery = "discord_voice" if voice_pref else "local_audio"
    prev_voice_pref = get_inastate("preferred_voice_output") or {}
    if voice_pref:
        slim_prev = {k: prev_voice_pref.get(k) for k in ("backend", "channel_name", "channel_id", "label")}
        slim_new = {k: voice_pref.get(k) for k in ("backend", "channel_name", "channel_id", "label")}
        if slim_prev != slim_new:
            update_inastate(
                "preferred_voice_output",
                {**voice_pref, "updated": datetime.now(timezone.utc).isoformat()},
            )
            label = voice_pref.get("channel_name") or voice_pref.get("channel_id") or voice_pref.get("label")
            log_to_statusbox(f"[Comms] Preferring Discord voice channel '{label}' for sound experiments.")
    elif prev_voice_pref:
        update_inastate("preferred_voice_output", None)

    min_urge_to_speak = float(config.get("min_urge_to_speak", 0.25))
    voice_urge_state = get_inastate("urge_to_voice") or get_inastate("urge_to_communicate") or {}
    try:
        voice_urge_level = float(voice_urge_state.get("level", 0.0))
    except Exception:
        voice_urge_level = 0.0
    allow_speech = voice_urge_level >= min_urge_to_speak or bool(config.get("ignore_urge_for_speech", False))
    min_urge_to_type = float(config.get("min_urge_to_type", 0.35))
    type_contact_cooldown = int(config.get("type_contact_cooldown", TYPE_CONTACT_COOLDOWN))
    type_urge_state = get_inastate("urge_to_type") or {}
    try:
        type_urge_level = float(type_urge_state.get("level", 0.0))
    except Exception:
        type_urge_level = 0.0
    last_typed_contact = get_inastate("last_typed_contact")
    since_last_typed = (time.time() - last_typed_contact) if last_typed_contact else None
    allow_high_trust_dm = bool(config.get("allow_high_trust_dm", True))
    try:
        owner_user_id = int(get_owner_user_id(config) or 0)
    except Exception:
        owner_user_id = None
    owner_user_id_str = str(owner_user_id) if owner_user_id else None
    humor_invite = get_inastate("humor_expression_invite") or {}
    humor_ready = bool(humor_invite.get("ready"))
    transformer = FractalTransformer()
    prediction = load_prediction(child)
    if not prediction:
        print("[Comms] No prediction available.")
        return

    pred_vec = prediction.get("predicted_vector", {}).get("vector", [])
    inferred = (
        prediction.get("inferred_emotion")
        or prediction.get("emotion_snapshot", {}).get("values", {})
        or {}
    )
    clarity = round(sum(pred_vec) / max(1, len(pred_vec)), 4) if pred_vec else 0.0
    speaking_to = predict_target_from_emotion(inferred)
    recent_heard = load_recent_heard_words(child, limit=12)
    context_language = guess_language_code(" ".join(item.get("word", "") for item in recent_heard))
    babble_targets = choose_babble_targets(child, language_hint=context_language, heard=recent_heard)
    spoken_words: List[str] = []
    speech_symbols: List[str] = []
    expression_strategy = "emotion_prediction"
    heard_trace = babble_targets.get("heard_trace", [])

    symbol_map = load_sound_symbol_map(child)
    vocab_map = load_symbol_to_token(child)
    if _ensure_vocab_embeddings(vocab_map, language_hint=context_language):
        save_symbol_to_token(child, vocab_map)

    tone_candidates = rank_sound_symbols(pred_vec, symbol_map, transformer, top_n=3)
    symbol_id, best_sim = (tone_candidates[0] if tone_candidates else (None, 0.0))
    tone_library = load_tone_library(child)
    tone_riff_option = select_tone_riff(tone_library, inferred or {})
    rare_tone_option = select_rare_tone(tone_library, inferred or {})
    alt_candidate = tone_candidates[1] if len(tone_candidates) > 1 else None
    sound_explorations: List[Dict[str, object]] = []

    if tone_riff_option:
        sound_explorations.append(
            {
                "strategy": "tone_riff",
                "symbols": tone_riff_option.get("sequence", []),
                "uses": tone_riff_option.get("uses", 0.0),
                "alignment": tone_riff_option.get("alignment", 0.0),
                "last_used": tone_riff_option.get("last_used"),
                "tags": tone_riff_option.get("tags", []),
                "selected": False,
                "delivery": voice_delivery,
            }
        )
    if rare_tone_option:
        sound_explorations.append(
            {
                "strategy": "rare_tone",
                "symbols": [rare_tone_option.get("symbol_id")],
                "uses": rare_tone_option.get("uses", 0.0),
                "alignment": rare_tone_option.get("alignment", 0.0),
                "last_used": rare_tone_option.get("last_used"),
                "tags": rare_tone_option.get("tags", []),
                "selected": False,
                "delivery": voice_delivery,
            }
        )
    if alt_candidate:
        sound_explorations.append(
            {
                "strategy": "adjacent_tone",
                "symbols": [alt_candidate[0]],
                "similarity": alt_candidate[1],
                "selected": False,
                "delivery": voice_delivery,
            }
        )

    if sound_explorations:
        summary = "; ".join(
            f"{opt['strategy']}->{','.join(str(s) for s in (opt.get('symbols') or []))}"
            for opt in sound_explorations
        )
        delivery_note = f" (target: {voice_delivery})" if voice_pref else ""
        log_to_statusbox(f"[Comms] Sound explorations queued{delivery_note}: {summary}")

    def label_for_symbol(sid: str) -> str:
        entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
        return str(entry.get("word") or sid)

    word_state = load_symbol_word_state(child)
    proto_words = word_state.get("proto_words", {})
    multi_symbol_words = word_state.get("multi_symbol_words", {})
    pair_choice = select_proto_pair_word(tone_candidates, proto_words, multi_symbol_words) if tone_candidates else None
    proto_word_used = None
    tone_exploration_used = None
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
        entry_lang = vocab_entry.get("language")
        if context_language not in {None, "", "und"} and entry_lang and entry_lang != context_language:
            vocab_word_conf = vocab_word_conf * 0.8
            log_to_statusbox(
                f"[Comms] Language mismatch for {symbol_id}: vocab '{entry_lang}' vs ctx '{context_language}'. Confidence softened."
            )
        log_to_statusbox(f"[Comms] Vocab word for {symbol_id}: '{vocab_word}' (conf: {vocab_word_conf})")

    combo_symbol_id, combo_word, combo_lang = combine_tones_and_register(
        child,
        tone_candidates,
        symbol_map,
        vocab_map,
        language_hint=context_language,
    )
    if not vocab_word and combo_word:
        vocab_word = combo_word
        vocab_word_conf = 0.2

    word_map = word_state.get("words", [])
    word_id, word_conf = None, 0.0
    word_creation_prompt = None
    vocab_size = len(word_map) + len(proto_words) + len(multi_symbol_words)
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
        if pair_choice and expression_strategy == "emotion_prediction" and (not vocab_word) and (not speech_symbols or best_sim < 0.85):
            speech_symbols = list(pair_choice["sequence"])
            expression_strategy = "proto_pair_word"
            word_id = pair_choice["key"]
            word_conf = pair_choice["confidence"]
            proto_word_used = pair_choice
            expression = f"I feel this paired word: {word_id}"
            if pair_choice.get("flexible"):
                expression += " (flexible)"
            log_to_statusbox(f"[Comms] Trying paired symbol word {word_id} (conf {word_conf:.3f}).")

        if expression_strategy == "emotion_prediction" and not speech_symbols:
            if tone_riff_option and (best_sim < 0.9 or tone_riff_option.get("alignment", 0.0) >= 0.25):
                speech_symbols = list(tone_riff_option.get("sequence", []))
                expression_strategy = "tone_riff"
                tone_exploration_used = {
                    "strategy": "tone_riff",
                    "symbols": speech_symbols,
                    "alignment": tone_riff_option.get("alignment"),
                    "uses": tone_riff_option.get("uses"),
                    "delivery": voice_delivery,
                }
                labels = [label_for_symbol(sid) for sid in speech_symbols[:3]]
                suffix = " via Discord voice" if voice_pref else ""
                expression = f"Exploring a tone riff{suffix}: " + ", ".join(labels)
                log_to_statusbox(
                    f"[Comms] Exploring riff ({tone_riff_option.get('uses', 0.0):.2f} uses, align {tone_riff_option.get('alignment', 0.0):.2f})."
                )
            elif rare_tone_option and (best_sim < 0.9 or rare_tone_option.get("uses", 0.0) < 1.0):
                chosen = rare_tone_option.get("symbol_id")
                if chosen:
                    speech_symbols = [chosen]
                    expression_strategy = "rare_tone"
                    tone_exploration_used = {
                        "strategy": "rare_tone",
                        "symbols": speech_symbols,
                        "alignment": rare_tone_option.get("alignment"),
                        "uses": rare_tone_option.get("uses"),
                        "delivery": voice_delivery,
                    }
                    suffix = " via Discord voice" if voice_pref else ""
                    expression = f"Exploring an underused tone{suffix}: {label_for_symbol(chosen)}"
                    log_to_statusbox(
                        f"[Comms] Surfacing rare tone {chosen} (uses {rare_tone_option.get('uses', 0.0):.2f}, align {rare_tone_option.get('alignment', 0.0):.2f})."
                    )
            elif alt_candidate and best_sim < 0.88:
                alt_sid, alt_sim = alt_candidate
                speech_symbols = [alt_sid]
                expression_strategy = "adjacent_tone"
                tone_exploration_used = {
                    "strategy": "adjacent_tone",
                    "symbols": speech_symbols,
                    "similarity": alt_sim,
                    "delivery": voice_delivery,
                }
                suffix = " via Discord voice" if voice_pref else ""
                expression = f"Trying a nearby tone{suffix}: {label_for_symbol(alt_sid)}"
                log_to_statusbox(f"[Comms] Exploring adjacent tone {alt_sid} (sim {alt_sim:.3f}).")

    if not speech_symbols and tone_candidates:
        speech_symbols = [sid for sid, sim in tone_candidates if sim > 0]

    if expression_strategy not in {"mimic_grounded_speech", "proto_pair_word"} and tone_candidates and speech_symbols:
        labels = []
        for sid in speech_symbols[:3]:
            vw_entry = vocab_map.get(sid, {}) if isinstance(vocab_map, dict) else {}
            vw = vw_entry.get("word")
            labels.append(vw or sid)
        suffix = " (Discord voice)" if voice_pref else ""
        expression = f"Trying tones{suffix}: " + ", ".join(labels)
        expression_strategy = "tone_experiment"

    if tone_exploration_used:
        for opt in sound_explorations:
            if (
                opt.get("strategy") == tone_exploration_used.get("strategy")
                and opt.get("symbols") == tone_exploration_used.get("symbols")
            ):
                opt["selected"] = True
                break
    update_inastate("sound_exploration_options", sound_explorations)

    log_to_statusbox(f"[Comms] Final expression: '{expression}'")
    symbol_seq_for_embed = speech_symbols or ([symbol_id] if symbol_id else [])
    symbol_embedding = (
        EMBEDDER.embed_symbol_sequence(symbol_seq_for_embed, language=context_language)
        if symbol_seq_for_embed
        else None
    )
    text_embedding = EMBEDDER.embed_text(expression, language=context_language) if expression else None
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
        proto_word=proto_word_used,
        tone_explorations=sound_explorations,
        speech_suppressed=(
            {
                "reason": "low_urge",
                "urge_level": voice_urge_level,
                "threshold": min_urge_to_speak,
            }
            if not allow_speech
            else None
        ),
        voice_target=voice_pref,
        language_hint=context_language,
        embedding=text_embedding,
        symbol_embedding=symbol_embedding,
    )
    if proto_word_used:
        record_proto_pair_usage(child, proto_word_used)
    update_tone_library(child, speech_symbols or ([symbol_id] if symbol_id else []), inferred or {}, frag.get("tags", []))
    log_to_statusbox(f"[Comms] Expression fragment saved: {frag['id']}")

    # === Optional typed contact (volitional, minimal payload)
    allow_symbol_autotype = bool(config.get("allow_symbol_autotype", True))
    typed_payload = get_inastate("typed_contact_payload") or {}
    payload_text = typed_payload.get("text") if isinstance(typed_payload, dict) else None
    payload_kind = typed_payload.get("kind") if isinstance(typed_payload, dict) else None
    payload_allow_empty = bool(typed_payload.get("allow_empty")) if isinstance(typed_payload, dict) else False
    ready_to_type = type_urge_level >= min_urge_to_type or bool(config.get("ignore_urge_for_typing", False))
    cooled_down = since_last_typed is None or since_last_typed >= type_contact_cooldown
    payload_target_user = typed_payload.get("target_user_id") if isinstance(typed_payload, dict) else None
    payload_target_label = typed_payload.get("target") if isinstance(typed_payload, dict) else None
    high_trust_contacts = get_high_trust_contacts(config=config, min_level="high", limit=3) if allow_high_trust_dm else []
    last_heard_contact = get_inastate("last_heard_contact")
    last_heard_user_id = None
    last_heard_is_dm = False
    if isinstance(last_heard_contact, dict):
        raw_last_user = last_heard_contact.get("user_id")
        if raw_last_user is not None:
            last_heard_user_id = str(raw_last_user)
        last_heard_is_dm = bool(last_heard_contact.get("is_dm"))

    # Prefer human-friendly labels over raw symbol ids when crafting typed text.
    symbol_labels = [label_for_symbol(sid) for sid in speech_symbols[:3] if sid]
    symbol_text = " ".join(symbol_labels) if symbol_labels else None
    fallback_text = vocab_word or word_id or symbol_text
    queued_id = None
    audio_clip_path = None
    if allow_symbol_autotype and speech_symbols:
        try:
            audio_dm_dir = Path("AI_Children") / child / "memory" / "comm_output" / "typed_audio"
            audio_dm_dir.mkdir(parents=True, exist_ok=True)
            audio_clip_path = audio_dm_dir / f"ina_sound_{uuid.uuid4().hex[:8]}.opus"
            speak_symbolically(
                speech_symbols,
                child=child,
                record_path=audio_clip_path,
                playback=False,
                record_format="opus",
            )
            if not audio_clip_path.exists():
                audio_clip_path = None
        except Exception as exc:
            audio_clip_path = None
            log_to_statusbox(f"[Comms] Failed to render DM audio clip: {exc}")

    if ready_to_type and cooled_down:
        chosen_text = None
        chosen_source = None
        target_user_id = None
        target_label = payload_target_label or "owner_dm"

        if isinstance(payload_text, str):
            chosen_text = payload_text
            chosen_source = payload_kind or "typed_payload"
        elif allow_symbol_autotype and fallback_text:
            chosen_text = fallback_text
            chosen_source = "symbol_sequence" if speech_symbols else "word_hint"

        if payload_target_user:
            target_user_id = str(payload_target_user)
            target_label = payload_target_label or "user_dm"
        elif last_heard_user_id and last_heard_is_dm:
            if owner_user_id_str and last_heard_user_id == owner_user_id_str:
                target_user_id = owner_user_id_str
                target_label = "owner_dm"
            elif allow_high_trust_dm and high_trust_contacts:
                matched = next(
                    (
                        contact
                        for contact in high_trust_contacts
                        if str(contact.get("user_id")) == last_heard_user_id
                    ),
                    None,
                )
                if matched:
                    target_user_id = last_heard_user_id
                    target_label = "trusted_dm"
        elif allow_high_trust_dm and high_trust_contacts:
            if owner_user_id_str:
                candidates = [
                    contact
                    for contact in high_trust_contacts
                    if str(contact.get("user_id")) != owner_user_id_str
                ] or high_trust_contacts
            else:
                candidates = high_trust_contacts
            if candidates:
                target_user_id = str(candidates[0].get("user_id"))
                target_label = "trusted_dm"
        elif owner_user_id_str:
            target_user_id = owner_user_id_str
            target_label = "owner_dm"

        rationale = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urge_level": type_urge_level,
            "urge_threshold": min_urge_to_type,
            "cooled_down": cooled_down,
            "target_label": target_label,
            "target_user_id": target_user_id,
            "selected_source": chosen_source or "unspecified",
            "symbol_autotype_enabled": allow_symbol_autotype,
            "payload_kind": payload_kind,
            "audio_clip_path": str(audio_clip_path) if audio_clip_path else None,
            "last_heard_contact": last_heard_contact if isinstance(last_heard_contact, dict) else None,
            "candidates": {
                "speech_symbols": speech_symbols[:3] if speech_symbols else None,
                "vocab_word": vocab_word,
                "word_id": word_id,
                "high_trust_ids": [c.get("user_id") for c in high_trust_contacts] if high_trust_contacts else None,
                "owner_user_id": owner_user_id,
                "last_heard_user_id": last_heard_user_id,
            },
        }
        if humor_invite:
            rationale["humor_bridge"] = {
                "ready": humor_ready,
                "level": humor_invite.get("level"),
                "note": humor_invite.get("note"),
                "expires_at": humor_invite.get("expires_at"),
            }

        if chosen_text is not None and (chosen_text.strip() or payload_allow_empty):
            queued_id = append_typed_outbox_entry(
                chosen_text,
                target=target_label,
                user_id=target_user_id,
                metadata={
                    "source": chosen_source or "unspecified",
                    "strategy": expression_strategy,
                    "urge_to_type": type_urge_level,
                    "urge_to_voice": voice_urge_level,
                },
                allow_empty=payload_allow_empty,
                attachment_path=str(audio_clip_path) if audio_clip_path else None,
            )
            if queued_id:
                update_inastate("last_typed_contact", time.time())
                update_inastate(
                    "last_contact_rationale",
                    {
                        **rationale,
                        "queued_id": queued_id,
                        "payload_text_preview": (chosen_text or "")[:160],
                        "allow_empty": payload_allow_empty,
                    },
                )
                if payload_text is not None:
                    update_inastate("typed_contact_payload", None)
                log_to_statusbox(f"[Comms] Queued typed contact ({chosen_source or 'freeform'}): {queued_id}")
        elif ready_to_type and chosen_text is None:
            update_inastate(
                "typing_contact_intent",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "urge_level": type_urge_level,
                    "note": "urge high; waiting for volitional text (silence is okay)",
                    "candidates": {
                        "symbols": speech_symbols[:3] if speech_symbols else None,
                        "word": vocab_word or word_id,
                        "trusted_user_ids": [c.get("user_id") for c in high_trust_contacts] if high_trust_contacts else None,
                    },
                    "last_heard_contact": last_heard_contact if isinstance(last_heard_contact, dict) else None,
                },
            )

    # === Audio expression attempt (log + speak)
    if not allow_speech:
        log_to_statusbox(
            f"[Comms] Staying quiet (urge {voice_urge_level:.2f} < {min_urge_to_speak}). Expression logged only."
        )
    else:
        channel_hint = " via Discord voice" if voice_pref else ""
        log_to_statusbox(f"[Comms] Preparing to speak{channel_hint}: \"{expression}\"")
        try:
            if speech_symbols:
                speak_symbolically(speech_symbols)
                log_to_statusbox("[Comms] Speech output triggered.")
            else:
                log_to_statusbox("[Comms] No symbol IDs available for speech.")
        except Exception as e:
            log_to_statusbox(f"[Comms] Speech error: {e}")

    # Hook into language processing for learning
    if proto_word_used:
        log_to_statusbox("[Comms] Skipped single-symbol association because a paired proto-word was used.")
    elif symbol_id and (vocab_word or word_id):
        chosen_word = vocab_word or word_id
        chosen_conf = vocab_word_conf if vocab_word is not None else word_conf
        associate_symbol_with_word(child, symbol_id, chosen_word, chosen_conf, language=context_language)
        predicted = prediction.get("predicted_word", chosen_word)
        backprop_symbol_confidence(child, predicted, symbol_id)
        log_to_statusbox(f"[Comms] Associated {symbol_id} with {chosen_word} via language_processing.")
    else:
        log_to_statusbox("[Comms] No word association updated (missing symbol or word).")

    if combo_symbol_id and combo_word:
        associate_symbol_with_word(child, combo_symbol_id, combo_word, 0.2, language=combo_lang)
        log_to_statusbox(f"[Comms] Associated combo {combo_symbol_id} with '{combo_word}'.")

    created_patterns = detect_repeated_tone_patterns(child, vocab_map)
    if created_patterns:
        for sid, word, coh, cnt in created_patterns:
            log_to_statusbox(
                f"[Comms] Promoted repeated pattern {sid} ({cnt} obs, coherence {coh:.2f}) -> '{word}'"
            )


    primary_symbol = speech_symbols[0] if speech_symbols else symbol_id
    primary_word = vocab_word or word_id or (spoken_words[0] if spoken_words else None)

    if allow_speech:
        update_inastate("currently_speaking", True)
        update_inastate("last_expression_time", time.time())
        update_inastate("last_spoken_symbol", primary_symbol)
        update_inastate("last_symbol_word_id", primary_word)
        update_inastate("last_babbled_words", spoken_words)
        update_inastate("last_babble_strategy", expression_strategy)
        time.sleep(1.5)
        update_inastate("currently_speaking", False)
    else:
        update_inastate("last_expression_time", time.time())
        update_inastate("last_spoken_symbol", primary_symbol)
        update_inastate("last_symbol_word_id", primary_word)
        update_inastate("last_babbled_words", spoken_words)
        update_inastate("last_babble_strategy", expression_strategy)
        update_inastate("currently_speaking", False)

    _update_tone_diversity_metrics(child, symbol_seq_for_embed)


if __name__ == "__main__":
    identify_devices_from_config()
    early_communicate()
