import os
import re
import json
import hashlib
import tempfile
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
from embedding_stack import MultimodalEmbedder, guess_language_code
from runtime_state import load_config, seed_self_question
from experience_logger import ExperienceLogger
from symbol_generator import (
    ACCENT_GLYPHS,
    ALPHANUMERIC_GLYPHS,
)
from symbol_glyphs import get_symbol_glyph_maps

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None

if TYPE_CHECKING:  # pragma: no cover
    from transformers.fractal_multidimensional_transformers import FractalTransformer


def _memory_root(child: str, base_path: Optional[Path] = None) -> Path:
    base = Path(base_path) if base_path else Path("AI_Children")
    return base / child / "memory"

LEGACY_SOUND_SYMBOL_MAP = Path("sound_symbol_map.json")
_EMBEDDER = MultimodalEmbedder(dim=128)
SOUND_FEATURE_CONFIDENCE_THRESHOLD = 0.5
SOUND_FEATURE_PROMOTION_THRESHOLD = 3
NEUTRAL_SOUND_FINGERPRINT = {
    "pitch_mean": 440,
    "dominant_freq": 440,
    "volume_db": -22,
    "silence_ratio": 0.1,
}


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_bin_frequencies(n_mels: int, sr: int) -> List[float]:
    if n_mels <= 0:
        return []
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sr / 2.0)
    mel_points = [
        mel_min + (mel_max - mel_min) * i / (n_mels + 1)
        for i in range(n_mels + 2)
    ]
    return [_mel_to_hz(mel) for mel in mel_points[1:-1]]


def _infer_sound_features_from_centroid(
    centroid,
    *,
    sample_rate: int = 44100,
    texture: Optional[Dict[str, Any]] = None,
    uses: Optional[int] = None,
):
    if not isinstance(centroid, (list, tuple)) or len(centroid) < 4:
        return None, None

    values: List[Optional[float]] = []
    for val in centroid:
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = None
        if num is not None and not math.isfinite(num):
            num = None
        values.append(num)

    if not any(v is not None for v in values):
        return None, None

    freqs = _mel_bin_frequencies(len(values), sample_rate)
    if len(freqs) != len(values):
        freqs = freqs[: len(values)]

    weights = [
        pow(10.0, v / 10.0) if v is not None else 0.0
        for v in values[: len(freqs)]
    ]
    total = sum(weights)
    if total <= 0:
        return None, None

    pitch_mean = sum(f * w for f, w in zip(freqs, weights)) / total
    dominant_idx = max(range(len(weights)), key=lambda i: weights[i])
    dominant_freq = freqs[dominant_idx]

    top_idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:4]
    max_w = weights[top_idx[0]] or 1.0
    partials = [
        {"freq": round(float(freqs[i]), 2), "gain": round(float(weights[i] / max_w), 3)}
        for i in top_idx
        if weights[i] > 0
    ]

    volume_db = None
    silence_ratio = 0.12
    if isinstance(texture, dict):
        rms = texture.get("rms")
        try:
            rms = float(rms)
        except (TypeError, ValueError):
            rms = None
        if rms is not None:
            volume_db = 20 * math.log10(max(1e-4, rms))
            volume_db = max(-60.0, min(-6.0, volume_db))
            silence_ratio = max(0.05, min(0.5, 0.45 - rms * 1.8))

    if volume_db is None:
        numeric_vals = [v for v in values if v is not None]
        avg_db = sum(numeric_vals) / max(1, len(numeric_vals))
        volume_db = max(-60.0, min(-6.0, avg_db))
        if avg_db < -40.0:
            silence_ratio = 0.35

    confidence = None
    if uses is not None:
        try:
            uses_val = max(0, int(uses))
            confidence = 0.28 + min(0.45, math.log1p(uses_val) / 7.0)
            confidence = min(0.8, confidence)
        except (TypeError, ValueError):
            confidence = None

    fingerprint = {
        "pitch_mean": round(float(pitch_mean), 2),
        "dominant_freq": round(float(dominant_freq), 2),
        "volume_db": round(float(volume_db), 2),
        "silence_ratio": round(float(silence_ratio), 3),
    }
    if partials:
        fingerprint["partials"] = partials

    return fingerprint, confidence


def _load_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_symbol_map(raw):
    """
    Merge legacy top-level symbol entries into the canonical {"symbols": {...}} shape.
    """
    if not isinstance(raw, dict):
        return {}

    symbols = {}
    if isinstance(raw.get("symbols"), dict):
        symbols.update(raw["symbols"])

    for key, val in raw.items():
        if key == "symbols":
            continue
        if isinstance(val, dict) and (
            key.startswith(("sound_symbol_", "sym_snd_", "combo_snd_"))
        ):
            symbols[key] = val

    return symbols


@contextmanager
def _json_lock(path: Path):
    if not fcntl:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _atomic_write_json(path: Path, payload: Any, *, indent: int = 4, ensure_ascii: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=indent, ensure_ascii=ensure_ascii)
    finally:
        os.replace(tmp_path, path)

def load_generated_symbols(child: str, base_path: Optional[Path] = None):
    """
    Load Ina's self-generated symbols with any available vision features.
    Prefers the rendered manifest under vision_session/generated_symbols and
    falls back to identity/self_reflection.json entries.
    """
    base = Path(base_path) if base_path else Path("AI_Children")
    vision_dir = base / child / "memory" / "vision_session" / "generated_symbols"
    manifest_path = vision_dir / "manifest.json"
    identity_path = base / child / "identity" / "self_reflection.json"

    symbols: Dict[str, Dict[str, Any]] = {}

    def _attach_image_features(entry: Dict[str, Any]) -> Dict[str, Any]:
        img_name = entry.get("image")
        if not img_name:
            return entry
        img_path = vision_dir / img_name
        if not img_path.exists():
            return entry
        try:
            import cv2  # Lazy import so modules without CV2 still import.

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                flat = img.flatten().tolist()
                entry["image_features"] = flat[:512]
        except Exception:
            # Best-effort; keep entry even if feature extraction fails.
            pass
        return entry

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or []
            for row in manifest:
                symbol = row.get("symbol")
                if not symbol:
                    continue
                sid = row.get("id") or row.get("symbol_id") or symbol
                entry = {
                    "id": sid,
                    "symbol": symbol,
                    "image": row.get("image"),
                    "summary": row.get("summary"),
                    "meaning": row.get("summary"),
                    "timestamp": row.get("timestamp"),
                    "source": row.get("source", "vision_generated"),
                }
                symbols.setdefault(sid, _attach_image_features(entry))
        except Exception:
            pass

    if identity_path.exists():
        try:
            with open(identity_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("self_generated_symbols", []):
                symbol = row.get("symbol")
                if not symbol:
                    continue
                sid = row.get("id") or symbol
                entry = symbols.get(sid, {"id": sid, "symbol": symbol})
                entry.setdefault("summary", row.get("meaning"))
                entry.setdefault("meaning", row.get("meaning"))
                entry.setdefault("timestamp", row.get("timestamp"))
                entry.setdefault("emotions", row.get("emotions"))
                entry.setdefault("clarity", row.get("clarity"))
                entry.setdefault("tags", row.get("tags"))
                entry.setdefault("components", row.get("components"))
                entry.setdefault("transformer_insights", row.get("transformer_insights"))
                entry.setdefault("source", row.get("origin", "self_generated"))
                if "image_features" not in entry and "image_features" in row:
                    entry["image_features"] = row["image_features"]
                symbols[sid] = entry
        except Exception:
            pass

    return list(symbols.values())

def _stable_symbol_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _default_symbol_for_id(symbol_id: str) -> str:
    """
    Generate a compact, deterministic symbol string for a given id.
    Length varies (2–5 chars) so symbols are not locked to 3 characters.
    """
    seed = _stable_symbol_seed(symbol_id)
    glyphs = get_symbol_glyph_maps()
    emotion_values = list(glyphs["emotion"].values()) or [symbol_id]
    modulation_values = list(glyphs["modulation"].values()) or [symbol_id]
    concept_values = list(glyphs["concept"].values()) or [symbol_id]
    emo = emotion_values[seed % len(emotion_values)]
    mod = modulation_values[(seed // 7) % len(modulation_values)]
    concept = concept_values[(seed // 13) % len(concept_values)]
    target_len = 2 + (seed % 4)  # 2–5 characters

    symbol = emo + concept if target_len == 2 else emo + mod + concept
    filler_pool = ACCENT_GLYPHS + ALPHANUMERIC_GLYPHS
    while len(symbol) < target_len:
        accent_idx = (seed // (len(symbol) + 3)) % len(filler_pool)
        symbol += filler_pool[accent_idx]

    return symbol[:target_len]


def _iter_sound_symbols_for_vocab(symbol_map: Any):
    if not isinstance(symbol_map, dict):
        return []
    raw = symbol_map.get("symbols")
    if isinstance(raw, dict):
        symbol_dict = raw
    else:
        symbol_dict = symbol_map

    return [
        (sid, data) for sid, data in symbol_dict.items() if isinstance(data, dict)
    ]


def _bootstrap_symbol_vocabulary(child: str, vocab: Dict[str, Any], base_path: Optional[Path]):
    """
    Ensure every sound/emotion symbol has a word entry so each symbol id
    can be treated as a distinct word token.
    """
    updated = False
    now = datetime.now(timezone.utc).isoformat()

    # Sound symbols → vocab
    sound_map = load_sound_symbol_map(child, base_path)
    for sym_id, sym_data in _iter_sound_symbols_for_vocab(sound_map):
        if sym_id in vocab:
            continue
        vocab[sym_id] = {
            "word": sym_data.get("symbol") or _default_symbol_for_id(sym_id),
            "uses": sym_data.get("uses", 0),
            "confidence": 0.25,
            "summary": sym_data.get("summary"),
            "source": "sound_symbol_map",
            "created": sym_data.get("timestamp", now),
        }
        updated = True

    # Emotion symbols → vocab
    emo_path = _memory_root(child, base_path) / "emotion_symbol_map.json"
    emo_map = _load_json(emo_path) or {}
    emo_symbols = emo_map.get("symbols") if isinstance(emo_map, dict) else []
    if isinstance(emo_symbols, list):
        for entry in emo_symbols:
            if not isinstance(entry, dict):
                continue
            sym_id = entry.get("symbol_word_id") or entry.get("symbol")
            if not sym_id or sym_id in vocab:
                continue
            vocab[sym_id] = {
                "word": entry.get("symbol") or _default_symbol_for_id(sym_id),
                "uses": entry.get("usage_count", entry.get("count", 0)),
                "confidence": max(entry.get("confidence", 0.15), 0.15),
                "summary": entry.get("summary"),
                "source": "emotion_symbol_map",
                "created": entry.get("birth_time", now),
            }
            updated = True

    return updated


def load_sound_symbol_map(child, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "sound_symbol_map.json"
    data = _load_json(path) if path.exists() else None
    if data is None and LEGACY_SOUND_SYMBOL_MAP.exists():
        data = _load_json(LEGACY_SOUND_SYMBOL_MAP)
    return _normalize_symbol_map(data or {})

def _save_sound_symbol_map(child: str, symbol_map: Dict[str, Any], base_path: Optional[Path] = None):
    """
    Persist the normalized symbol map back to disk, preserving any extra
    top-level keys that might be present in the existing file.
    """
    path = _memory_root(child, base_path) / "sound_symbol_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_json(path) if path.exists() else {}
    payload = existing if isinstance(existing, dict) else {}
    if isinstance(payload.get("symbols"), dict):
        payload["symbols"] = symbol_map
    else:
        payload = {"symbols": symbol_map}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def persist_sound_features(
    child: str,
    symbol_id: str,
    fingerprint: Dict[str, Any],
    source: str = "fallback",
    base_path: Optional[Path] = None,
    log_fn=None,
    feature_confidence: Optional[float] = None,
    evidence_increment: int = 0,
) -> bool:
    """
    Cache newly inferred sound features so Ina can reuse or refine them later.
    Only writes when the stored fingerprint differs.
    """
    if not fingerprint:
        return False

    symbol_map = load_sound_symbol_map(child, base_path)
    entry = symbol_map.get(symbol_id, {})
    if not isinstance(entry, dict):
        entry = {}

    now = datetime.now(timezone.utc).isoformat()
    current_conf = entry.get("feature_confidence")
    new_conf = feature_confidence if feature_confidence is not None else current_conf
    if new_conf is None:
        new_conf = 0.25

    evidence = int(entry.get("evidence", 0)) + max(0, evidence_increment)
    status = entry.get("status", "draft")

    changed = (
        entry.get("sound_features") != fingerprint
        or entry.get("feature_source") != source
        or entry.get("feature_confidence") != new_conf
        or entry.get("evidence") != evidence
    )

    entry["sound_features"] = fingerprint
    entry["feature_source"] = source
    entry["feature_confidence"] = new_conf
    entry["evidence"] = evidence
    entry["updated"] = now

    if evidence >= SOUND_FEATURE_PROMOTION_THRESHOLD and status != "promoted":
        status = "promoted"
        entry["promoted_at"] = now
        entry["promotion_reason"] = f"evidence>={SOUND_FEATURE_PROMOTION_THRESHOLD} via {source}"
        if log_fn:
            log_fn(f"[Voice] Promoted sound features for {symbol_id} after {evidence} evidence samples.")

    entry["status"] = status
    symbol_map[symbol_id] = entry

    if changed:
        _save_sound_symbol_map(child, symbol_map, base_path)
        if log_fn:
            log_fn(f"[Voice] Cached sound features for {symbol_id} ({source}).")
    return changed

def load_fragments(child, base_path: Optional[Path] = None):
    frag_path = _memory_root(child, base_path) / "fragments"
    fragments = []
    for f in frag_path.glob("frag_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "sound_symbol" in data:
                    fragments.append(data)
        except:
            continue
    return fragments

def extract_sound_features_from_summary(summary):
    import re
    match = re.search(r"pitch ([\d.]+) Hz, vol (-?[\d.]+) dB, dom freq ([\d.]+) Hz", summary)
    if match:
        pitch, volume, dom_freq = match.groups()
        return {
            "pitch_mean": float(pitch),
            "volume_db": float(volume),
            "dominant_freq": float(dom_freq),
        }
    return {}

def load_symbol_to_token(child, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "symbol_to_token.json"
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except Exception:
            data = {}

    if _bootstrap_symbol_vocabulary(child, data, base_path):
        save_symbol_to_token(child, data, base_path)

    return data

def save_symbol_to_token(child, data, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "symbol_to_token.json"
    with _json_lock(path):
        _atomic_write_json(path, data, indent=4, ensure_ascii=True)


def load_text_vocab_links(child, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "text_vocab_links.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, (dict, list)) else {}
    except Exception:
        return {}


def _candidate_word(candidate: Any) -> str:
    if isinstance(candidate, (str, int, float)):
        return str(candidate).strip()
    if not isinstance(candidate, dict):
        return ""
    for key in ("word", "token", "text", "vocab_word", "label", "value"):
        value = candidate.get(key)
        if isinstance(value, (str, int, float)):
            word = str(value).strip()
            if word:
                return word
    return ""


def _candidate_symbols(candidate: Any) -> List[str]:
    if not isinstance(candidate, dict):
        return []
    symbols: List[str] = []
    for key in ("symbol", "symbol_id", "sid", "native", "symbol_word_id"):
        value = candidate.get(key)
        if isinstance(value, (str, int, float)):
            symbols.append(str(value).strip())
    raw_symbols = candidate.get("symbols")
    if isinstance(raw_symbols, dict):
        symbols.extend(str(sym).strip() for sym in raw_symbols.keys())
    elif isinstance(raw_symbols, (list, tuple, set)):
        symbols.extend(str(sym).strip() for sym in raw_symbols)
    return [sym for sym in symbols if sym]


def _iter_word_candidates(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_word_candidates(item)
    elif isinstance(value, dict):
        if _candidate_word(value):
            yield value
            return
        for word, item in value.items():
            if isinstance(item, dict):
                candidate = dict(item)
                candidate.setdefault("word", word)
                yield candidate
            elif isinstance(item, (str, int, float)):
                yield {"word": word, "score": item}
    elif isinstance(value, (str, int, float)):
        yield {"word": str(value)}


def _iter_text_vocab_link_candidates(links_payload: Any, symbol: str) -> Iterable[Dict[str, Any]]:
    if isinstance(links_payload, list):
        for entry in links_payload:
            if symbol in _candidate_symbols(entry):
                yield entry
        return

    if not isinstance(links_payload, dict):
        return

    for key in ("links", "items", "candidates"):
        entries = links_payload.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if symbol in _candidate_symbols(entry):
                    yield entry

    symbol_map = links_payload.get("symbols")
    if isinstance(symbol_map, dict) and symbol in symbol_map:
        yield from _iter_word_candidates(symbol_map.get(symbol))

    if symbol in links_payload:
        yield from _iter_word_candidates(links_payload.get(symbol))

    vocab = links_payload.get("vocab")
    if isinstance(vocab, dict):
        for word, entry in vocab.items():
            if symbol not in _candidate_symbols(entry):
                continue
            candidate = dict(entry) if isinstance(entry, dict) else {}
            candidate.setdefault("word", word)
            yield candidate


def _words_from_value(value: Any) -> List[str]:
    words: List[str] = []
    if isinstance(value, str):
        words.extend(tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", value))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            words.extend(_words_from_value(item))
    return [word for word in words if word]


def _symbol_lookup_context(context: Optional[Dict[str, Any]]) -> Dict[str, set]:
    words = set()
    tags = set()
    if isinstance(context, dict):
        for key in ("tokens", "words", "input_tokens", "source_tokens"):
            words.update(_words_from_value(context.get(key)))
        for key in ("text", "source_text", "prompt", "utterance", "user_text"):
            words.update(_words_from_value(context.get(key)))
        for key in ("tags", "source", "channel", "adapter"):
            tags.update(_words_from_value(context.get(key)))
    elif context:
        words.update(_words_from_value(context))
    return {"words": words, "tags": tags}


def _candidate_tags(candidate: Dict[str, Any]) -> set:
    tags = set()
    for key in ("tags", "sources", "contexts", "source", "channel", "adapter"):
        tags.update(_words_from_value(candidate.get(key)))
    return tags


def _numeric_candidate_value(candidate: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            return float(candidate.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _score_text_vocab_candidate(candidate: Dict[str, Any], lookup_context: Dict[str, set]) -> float:
    word = _candidate_word(candidate).lower()
    score = 0.0
    if word and word in lookup_context["words"]:
        score += 100.0

    tag_overlap = _candidate_tags(candidate) & lookup_context["tags"]
    score += len(tag_overlap) * 20.0

    score += _numeric_candidate_value(candidate, "similarity", "score") * 10.0
    score += _numeric_candidate_value(candidate, "symbol_confidence", "confidence") * 5.0

    count = _numeric_candidate_value(candidate, "count", "uses", "use_count")
    if count > 0:
        score += math.log10(count + 1.0)
    return score


def _candidate_native_word(candidate: Any) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("symbol_word", "glyph", "native_word", "native_text"):
        value = candidate.get(key)
        if isinstance(value, (str, int, float)):
            word = str(value).strip()
            if word:
                return word
    return ""


def _use_native_glyphs(native_style: str) -> bool:
    return str(native_style or "symbols").strip().lower() in {
        "glyph",
        "glyphs",
        "symbol_word",
        "native_glyphs",
    }


def _lookup_text_vocab_word(
    symbol: str,
    links_payload: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    lookup_context = _symbol_lookup_context(context)
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for candidate in _iter_text_vocab_link_candidates(links_payload, symbol):
        if not isinstance(candidate, dict):
            continue
        word = _candidate_word(candidate)
        if not word:
            continue
        score = _score_text_vocab_candidate(candidate, lookup_context)
        if score > best_score:
            best = {**candidate, "word": word, "score": round(score, 4)}
            best_score = score
    return best


def _build_text_vocab_word_symbol_index(links_payload: Any) -> Dict[str, str]:
    index: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    context = {"words": set(), "tags": set()}

    def consider(candidate: Any):
        if not isinstance(candidate, dict):
            return
        word = _candidate_word(candidate).lower()
        symbols = _candidate_symbols(candidate)
        if not word or not symbols:
            return
        score = _score_text_vocab_candidate(candidate, context)
        if score > scores.get(word, float("-inf")):
            index[word] = symbols[0]
            scores[word] = score

    if isinstance(links_payload, list):
        for entry in links_payload:
            consider(entry)
    elif isinstance(links_payload, dict):
        for key in ("links", "items", "candidates"):
            entries = links_payload.get(key)
            if isinstance(entries, list):
                for entry in entries:
                    consider(entry)
        symbol_map = links_payload.get("symbols")
        if isinstance(symbol_map, dict):
            for symbol, value in symbol_map.items():
                for candidate in _iter_word_candidates(value):
                    candidate = dict(candidate)
                    candidate.setdefault("symbol", symbol)
                    consider(candidate)
        for symbol, value in links_payload.items():
            if not str(symbol).startswith("sym_"):
                continue
            for candidate in _iter_word_candidates(value):
                candidate = dict(candidate)
                candidate.setdefault("symbol", symbol)
                consider(candidate)
        vocab = links_payload.get("vocab")
        if isinstance(vocab, dict):
            for word, entry in vocab.items():
                candidate = dict(entry) if isinstance(entry, dict) else {}
                candidate.setdefault("word", word)
                consider(candidate)
    return index


def build_dual_symbolic_message(
    symbols,
    *,
    child: str = "Inazuma_Yagami",
    base_path: Optional[Path] = None,
    human_text: Optional[str] = None,
    fallback_human_text: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    fallback_to_symbol_to_token: bool = True,
    native_style: str = "symbols",
    native_label: str = "Ina native",
    human_label: str = "Human guess",
) -> Optional[Dict[str, Any]]:
    if isinstance(symbols, str):
        normalized = [symbols.strip()] if symbols.strip() else []
    else:
        normalized = [str(sym).strip() for sym in (symbols or []) if str(sym).strip()]
    if not normalized:
        return None

    explicit_human_text = str(human_text or "").strip()
    native_tokens: List[str] = list(normalized)
    guessed_words: List[str] = list(normalized)
    unresolved_indexes: List[tuple] = []
    native_sources: Dict[str, str] = {}
    gloss_sources: Dict[str, str] = {}
    use_glyphs = _use_native_glyphs(native_style)

    links_payload = load_text_vocab_links(child, base_path=base_path) if (use_glyphs or not explicit_human_text) else {}
    for idx, sym in enumerate(normalized):
        link = _lookup_text_vocab_word(sym, links_payload, context=context) if links_payload else None
        if link and use_glyphs:
            native_word = _candidate_native_word(link)
            if native_word:
                native_tokens[idx] = native_word
                native_sources[sym] = "text_vocab_links"

        if explicit_human_text:
            continue
        if link:
            guessed_words[idx] = str(link.get("word") or sym)
            gloss_sources[sym] = "text_vocab_links"
        else:
            unresolved_indexes.append((idx, sym))

    if not explicit_human_text and unresolved_indexes and fallback_to_symbol_to_token:
        vocab = load_symbol_to_token(child, base_path=base_path)
        still_unresolved: List[tuple] = []
        for idx, sym in unresolved_indexes:
            entry = vocab.get(sym) if isinstance(vocab, dict) else {}
            word = str(entry.get("word") or "").strip() if isinstance(entry, dict) else ""
            if word:
                guessed_words[idx] = word
                gloss_sources[sym] = "symbol_to_token"
                if use_glyphs and native_tokens[idx] == sym:
                    native_tokens[idx] = word
                    native_sources[sym] = "symbol_to_token"
            else:
                still_unresolved.append((idx, sym))
        unresolved_indexes = still_unresolved

    unresolved_symbols: List[str] = [sym for _, sym in unresolved_indexes]
    native_text = " ".join(native_tokens)
    fallback_text = str(fallback_human_text or "").strip()
    gloss_text = explicit_human_text or " ".join(guessed_words)
    if fallback_text and gloss_text == native_text:
        gloss_text = fallback_text
    if not gloss_text:
        gloss_text = native_text

    if gloss_text == native_text:
        combined = native_text
    else:
        combined = f"{native_label}: {native_text}\n{human_label}: {gloss_text}"

    return {
        "text": combined,
        "native_text": native_text,
        "gloss_text": gloss_text,
        "unresolved_symbols": unresolved_symbols,
        "native_sources": native_sources,
        "gloss_sources": gloss_sources,
    }


def _ensure_vocab_embeddings(vocab: Dict[str, Any], language_hint: Optional[str] = None) -> bool:
    """
    Attach deterministic text embeddings and language hints to vocab entries
    when missing. Returns True if any entry was updated.
    """
    updated = False
    for entry in vocab.values():
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
        entry["embedding"] = _EMBEDDER.embed_text(word, language=lang)
        updated = True
    return updated

def match_sound_symbol_to_input(input_vector, symbol_map, transformer):
    best_match = None
    best_sim = 0.0
    for symbol_id, entry in symbol_map.items():
        emotions = entry.get("emotions", {})
        result = transformer.encode({"emotions": emotions})
        sim = cosine_similarity(input_vector, result["vector"])
        if sim > best_sim:
            best_sim = sim
            best_match = symbol_id
    return best_match, best_sim

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)

def associate_symbol_with_word(
    child,
    symbol_id,
    word,
    confidence=0.35,
    grounding: Optional[Dict[str, str]] = None,
    language: Optional[str] = None,
    base_path: Optional[Path] = None,
):
    vocab = load_symbol_to_token(child, base_path)
    lang = language or guess_language_code(word)
    text_embedding = _EMBEDDER.embed_text(word, language=lang)
    if symbol_id not in vocab:
        vocab[symbol_id] = {
            "word": word,
            "uses": 1,
            "confidence": round(min(0.9, confidence), 2),
            "language": lang,
            "embedding": text_embedding,
        }
    else:
        entry = vocab[symbol_id]
        if entry["word"].lower() != word.lower():
            seed_self_question(f"Was '{entry['word']}' the wrong word for {symbol_id}? Now seen as '{word}'?")
            entry["confidence"] = round(entry.get("confidence", 0.5) - 0.1, 2)
            entry["language"] = lang
            entry["embedding"] = text_embedding
        else:
            entry["uses"] += 1
            entry["confidence"] = round(min(0.9, entry.get("confidence", 0.35) + 0.05), 2)
            entry.setdefault("language", lang)
            if not entry.get("embedding"):
                entry["embedding"] = text_embedding
        vocab[symbol_id] = entry
    save_symbol_to_token(child, vocab, base_path)
    print(f"[LangLearn] Associated {symbol_id} → '{word}' with confidence {vocab[symbol_id]['confidence']}")
    if grounding and grounding.get("event_id"):
        logger = ExperienceLogger(child=child, base_path=base_path)
        logger.attach_word_usage(
            grounding["event_id"],
            speaker=grounding.get("speaker", "system"),
            utterance=grounding.get("utterance", word),
            words=[word],
            entity_links=grounding.get("entity_links"),
        )
        print(
            f"[LangLearn] Grounded '{word}' in experience event {grounding['event_id']} (speaker: {grounding.get('speaker', 'system')})."
        )
    else:
        seed_self_question(
            f"What experience grounds the word '{word}' for symbol {symbol_id}?"
        )

def backprop_symbol_confidence(child, predicted_word, expressed_symbol, base_path: Optional[Path] = None):
    vocab = load_symbol_to_token(child, base_path)
    if expressed_symbol not in vocab:
        return

    entry = vocab[expressed_symbol]
    current_word = entry["word"]
    if current_word.lower() != predicted_word.lower():
        entry["confidence"] = round(entry.get("confidence", 0.5) - 0.15, 2)
        entry["flagged"] = True
        seed_self_question(f"Was '{current_word}' incorrect for {expressed_symbol}? Prediction suggested '{predicted_word}'.")
    else:
        entry["confidence"] = round(min(0.9, entry.get("confidence", 0.5) + 0.1), 2)

    vocab[expressed_symbol] = entry
    save_symbol_to_token(child, vocab, base_path)
    print(f"[LangLearn] Updated confidence for {expressed_symbol} → '{entry['word']}' to {entry['confidence']}")

def speak_symbolically(
    symbols,
    child="Inazuma_Yagami",
    *,
    record_path: Optional[str | Path] = None,
    playback: bool = True,
    record_format: str = "wav",
):
    import numpy as np
    import sounddevice as sd
    from gui_hook import log_to_statusbox

    if isinstance(symbols, str):
        symbols = [symbols]

    config = load_config()
    polyphonic = bool(config.get("allow_polyphonic_voice", True))
    sample_rate = int(config.get("voice_sample_rate", 22050))
    feedback_heard_voice = bool(config.get("feedback_heard_voice", True))
    freq_rep_cfg = config.get("voice_frequency_replication")
    if not isinstance(freq_rep_cfg, dict):
        freq_rep_cfg = {}

    def _harmonic_summary(audio: np.ndarray, sr: int, top_k: int = 5):
        if audio.size == 0:
            return []
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
        if spec.size == 0:
            return []
        top_idx = np.argpartition(spec, -top_k)[-top_k:]
        peaks = sorted(zip(freqs[top_idx], spec[top_idx]), key=lambda x: -x[1])
        return [
            {"freq_hz": round(float(f), 2), "mag": round(float(m), 4)}
            for f, m in peaks[:top_k]
        ]

    symbol_map = load_sound_symbol_map(child)
    waveform = []

    for sid in symbols:
        entry = symbol_map.get(sid) or {}

        fingerprint = entry.get("sound_features")
        feature_confidence = entry.get("feature_confidence")
        feature_source = None
        if not fingerprint and "summary" in entry:
            fingerprint = extract_sound_features_from_summary(entry["summary"])
            feature_source = "summary_reconstruction" if fingerprint else None
            if fingerprint:
                log_to_statusbox(f"[Voice] Reconstructed sound features from summary for {sid}.")
        if not fingerprint:
            sample_rate = entry.get("sample_rate") or 44100
            try:
                sample_rate = int(sample_rate)
            except (TypeError, ValueError):
                sample_rate = 44100
            inferred, inferred_conf = _infer_sound_features_from_centroid(
                entry.get("centroid"),
                sample_rate=sample_rate,
                texture=entry.get("texture"),
                uses=entry.get("uses"),
            )
            if inferred:
                fingerprint = inferred
                feature_source = "centroid_infer"
                if inferred_conf is not None:
                    feature_confidence = inferred_conf
                log_to_statusbox(f"[Voice] Derived sound features from centroid for {sid}.")
        if not fingerprint:
            # Seed a simple tone from the symbol id so it's never silent
            seed = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
            base_freq = 300 + (seed % 600)  # 300–900 Hz
            fingerprint = {
                "pitch_mean": base_freq,
                "dominant_freq": base_freq,
                "volume_db": -24 + (seed % 12),  # -24 to -13 dB
                "silence_ratio": 0.1,
            }
            feature_source = "fallback_hash_seed"
            log_to_statusbox(f"[Voice] Synthesizing fallback tone for {sid}.")

        if not fingerprint:
            log_to_statusbox(f"[Voice] Symbol {sid} missing usable sound features.")
            continue

        playback_fingerprint = fingerprint
        if feature_source:
            # Gate low-confidence guesses to a neutral tone unless promoted.
            confidence = feature_confidence if feature_confidence is not None else (0.35 if feature_source == "summary_reconstruction" else 0.25)
            status = entry.get("status", "draft")
            if status != "promoted" and confidence < SOUND_FEATURE_CONFIDENCE_THRESHOLD:
                log_to_statusbox(
                    f"[Voice] Neutralized low-confidence tone for {sid} "
                    f"(conf {confidence:.2f} < {SOUND_FEATURE_CONFIDENCE_THRESHOLD})."
                )
                playback_fingerprint = NEUTRAL_SOUND_FINGERPRINT

            persist_sound_features(
                child,
                sid,
                fingerprint,
                source=feature_source,
                log_fn=log_to_statusbox,
                feature_confidence=confidence,
                evidence_increment=1,
            )

        chunk = synthesize_from_fingerprint(
            playback_fingerprint,
            sr=sample_rate,
            symbol_id=sid,
            replication_cfg=freq_rep_cfg,
        )
        waveform.append(chunk)

    if waveform:
        if len(waveform) == 1 or not polyphonic:
            audio = np.concatenate(waveform)
            log_to_statusbox(f"[Voice] Synthesizing {len(waveform)} sound symbols (stacked).")
        else:
            max_len = max(w.shape[0] for w in waveform)
            padded = []
            for w in waveform:
                if w.shape[0] < max_len:
                    pad = np.zeros(max_len, dtype=np.float32)
                    pad[: w.shape[0]] = w
                    padded.append(pad)
                else:
                    padded.append(w)
            mix = np.sum(np.vstack(padded), axis=0)
            peak = np.max(np.abs(mix)) or 1.0
            mix = (mix / peak) * 0.85  # prevent clipping while keeping some headroom
            audio = mix.astype(np.float32)
            log_to_statusbox(f"[Voice] Synthesizing {len(waveform)} sound symbols (polyphonic mix).")

        if record_path:
            try:
                rec_path = Path(record_path)
                rec_path.parent.mkdir(parents=True, exist_ok=True)
                pcm = np.clip(audio, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                if record_format.lower() == "mp3":
                    try:
                        from pydub import AudioSegment

                        seg = AudioSegment(
                            pcm16.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=1,
                        )
                        seg.export(str(rec_path), format="mp3")
                    except Exception:
                        log_to_statusbox("[Voice] MP3 export failed; falling back to WAV.")
                        record_format = "wav"
                if record_format.lower() != "mp3":
                    import wave
                    with wave.open(str(rec_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm16.tobytes())
                log_to_statusbox(f"[Voice] Saved synthesized audio to {rec_path} ({record_format.upper()}).")
            except Exception as exc:
                log_to_statusbox(f"[Voice] Failed to save synthesized audio: {exc}")

        if playback:
            sd.play(audio, samplerate=sample_rate)
        if feedback_heard_voice:
            try:
                logger = ExperienceLogger(child=child)
                harmonic_profile = _harmonic_summary(audio, sample_rate)
                event_id = logger.log_event(
                    situation_tags=["self_voice", "audio", "feedback"],
                    perceived_entities=[
                        {
                            "type": "self_voice",
                            "symbols": symbols,
                            "polyphonic": polyphonic and len(waveform) > 1,
                            "harmonics": harmonic_profile,
                        }
                    ],
                    internal_state={"sample_rate": sample_rate, "polyphonic": polyphonic},
                    narrative="Ina listened to her own synthesized voice.",
                )
                logger.attach_word_usage(
                    event_id,
                    speaker=child,
                    utterance=" ".join(str(s) for s in symbols),
                    words=[str(s) for s in symbols],
                )
                log_to_statusbox(f"[Voice] Logged self-voice feedback as event {event_id}.")
            except Exception as e:
                log_to_statusbox(f"[Voice] Failed to log self-voice feedback: {e}")
    else:
        log_to_statusbox("[Voice] No audio generated from symbolic request.")


# === New: Symbol Image Training ===
def train_from_symbol_images(child):
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    import cv2

    transformer = FractalTransformer()
    symbol_dir = Path("AI_Children") / child / "memory" / "vision_session" / "generated_symbols"
    manifest_path = symbol_dir / "manifest.json"
    if not manifest_path.exists():
        print("[LangTrain] No symbol manifest found.")
        return

    with open(manifest_path, "r") as f:
        entries = json.load(f)

    for entry in entries:
        symbol = entry.get("symbol")
        path = symbol_dir / entry.get("image")
        if not path.exists():
            continue

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        flat = img.flatten().tolist()
        fragment = {
            "modality": "image",
            "image_features": flat[:512],
            "tags": ["symbolic", "visual", "generated"],
            "summary": symbol,
            "emotions": {"focus": 0.3, "novelty": 0.6}
        }
        vec = transformer.encode_image_fragment(fragment)
        print(f"[LangTrain] Trained on symbol image: {symbol} | Importance: {vec['importance']}")

def _resolve_frequency_layers(fingerprint, base_freq, replication_cfg, symbol_id):
    freq_seq = fingerprint.get("frequency_layers")
    if not isinstance(freq_seq, list):
        freq_seq = fingerprint.get("partials")

    layers = []
    if isinstance(freq_seq, list):
        for idx, entry in enumerate(freq_seq):
            freq_val = None
            gain = None
            if isinstance(entry, dict):
                freq_val = entry.get("freq") or entry.get("frequency")
                gain = entry.get("gain") or entry.get("weight") or entry.get("amp")
            elif isinstance(entry, (int, float)):
                freq_val = float(entry)
                gain = 1.0 / (idx + 1)

            if freq_val is None:
                continue
            try:
                freq_val = float(freq_val)
            except (TypeError, ValueError):
                continue
            if freq_val <= 0:
                continue

            if gain is None:
                gain = 1.0 / (idx + 1)
            try:
                gain = float(gain)
            except (TypeError, ValueError):
                gain = 1.0 / (idx + 1)

            layers.append((freq_val, max(0.05, min(1.0, gain))))

    if layers:
        return layers

    cfg = replication_cfg if isinstance(replication_cfg, dict) else {}
    if not cfg.get("enabled"):
        return [(base_freq, 1.0)]

    ratios = cfg.get("ratios")
    if not isinstance(ratios, list) or not ratios:
        ratios = [1.0]

    try:
        amp_decay = float(cfg.get("amplitude_decay", 0.8))
    except (TypeError, ValueError):
        amp_decay = 0.8
    amp_decay = min(0.95, max(0.3, amp_decay))

    try:
        replicas = int(cfg.get("replicas_per_layer", 1))
    except (TypeError, ValueError):
        replicas = 1
    replicas = max(1, replicas)

    detune_cents = cfg.get("detune_cents", 0.0)
    try:
        detune_cents = float(detune_cents)
    except (TypeError, ValueError):
        detune_cents = 0.0
    detune_cents = max(0.0, detune_cents)

    min_freq = cfg.get("min_frequency", 55.0)
    max_freq = cfg.get("max_frequency", 12000.0)
    try:
        min_freq = float(min_freq)
    except (TypeError, ValueError):
        min_freq = 55.0
    try:
        max_freq = float(max_freq)
    except (TypeError, ValueError):
        max_freq = 12000.0

    seed = _stable_symbol_seed(symbol_id) if symbol_id else 0

    def _clamp(freq):
        return min(max_freq, max(min_freq, freq))

    layers = []
    for idx, raw_ratio in enumerate(ratios):
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio == 0:
            ratio = 1.0
        freq_val = _clamp(base_freq * ratio)
        gain = 1.0 if idx == 0 else pow(amp_decay, idx)
        layers.append((freq_val, gain))

        if replicas > 1 and detune_cents > 0:
            for rep in range(1, replicas):
                cents = detune_cents * rep
                bit_idx = (idx * 7) + rep
                sign = -1 if ((seed >> bit_idx) & 1) == 0 else 1
                detune_ratio = pow(2.0, (sign * cents) / 1200.0)
                freq_detuned = _clamp(freq_val * detune_ratio)
                layers.append((freq_detuned, gain * pow(amp_decay, rep * 0.5)))

    return layers


def synthesize_from_fingerprint(
    fingerprint,
    duration_ms=1500,
    sr=22050,
    *,
    symbol_id: Optional[str] = None,
    replication_cfg: Optional[Dict[str, Any]] = None,
):
    import numpy as np

    pitch = fingerprint.get("pitch_mean", 440)
    freq = fingerprint.get("dominant_freq", pitch)
    volume = min(1.0, max(0.1, (fingerprint.get("volume_db", -40) + 60) / 60))
    silence = fingerprint.get("silence_ratio", 0.1)

    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sr * duration_s), False)
    freq_layers = _resolve_frequency_layers(fingerprint, freq, replication_cfg, symbol_id)

    waveform = np.zeros_like(t)
    for layer_freq, layer_gain in freq_layers:
        waveform += np.sin(2 * np.pi * layer_freq * t) * layer_gain

    peak = np.max(np.abs(waveform)) or 1.0
    waveform = (waveform / peak) * volume

    # Optional: shape with envelope
    envelope = np.linspace(0, 1, len(t))
    waveform = waveform * envelope

    # Simulate silence
    silence_len = int(len(waveform) * silence)
    waveform[:silence_len] = 0.0

    return waveform.astype(np.float32)


# === New: Book Text Training ===
def train_from_books(child):
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    import fitz  # PyMuPDF

    config = load_config()
    book_path = Path(config.get("book_folder_path", "books/"))
    if not book_path.exists():
        print("[LangTrain] Book folder not found.")
        return

    transformer = FractalTransformer()
    all_texts = []

    for file in book_path.glob("*.pdf"):
        try:
            with fitz.open(file) as doc:
                text = "".join(page.get_text() for page in doc)
            all_texts.append(text)
        except Exception as e:
            print(f"[LangTrain] Failed to read {file}: {e}")

    for file in book_path.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8")
            all_texts.append(text)
        except Exception as e:
            print(f"[LangTrain] Failed to read {file}: {e}")

    count = 0
    for text in all_texts:
        parts = [text[i:i+300] for i in range(0, len(text), 300)]
        for chunk in parts[:10]:
            fragment = {
                "summary": chunk,
                "tags": ["text", "book", "read"],
                "emotions": {"curiosity": 0.5, "focus": 0.4}
            }
            vec = transformer.encode(fragment)
            print(f"[LangTrain] Book fragment score: {vec['importance']}")
            count += 1

    print(f"[LangTrain] Total book chunks processed: {count}")

def summarize_known_words(child):
    vocab_path = Path("AI_Children") / child / "memory" / "symbol_to_token.json"
    if not vocab_path.exists():
        print("[LangLearn] No known vocabulary yet.")
        return
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    print("[LangLearn] Known vocabulary:")
    for sym, entry in vocab.items():
        print(f" - {entry['word']} (symbol: {sym}, uses: {entry['uses']})")
        groundings = describe_word_grounding(child, entry["word"])
        for grounding in groundings:
            tags = ", ".join(grounding.get("situation_tags", []))
            print(
                f"    ↳ grounded in event {grounding['event_id']} ({tags or 'no tags'})"
                f" — narrative: {grounding.get('narrative', '')}"
            )

def respond_to_word(child, word, *, base_path: Optional[Path] = None):
    vocab_path = _memory_root(child, base_path) / "symbol_to_token.json"
    symbol_map = load_sound_symbol_map(child, base_path)

    if not os.path.exists(vocab_path) or not symbol_map:
        print("[LangLearn] No vocab or symbol map found.")
        return

    if not ensure_word_grounded(child, word, base_path=base_path):
        print(f"[LangLearn] Cannot respond with '{word}' without experiential grounding.")
        return

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    for sym, entry in vocab.items():
        if entry.get("word", "").lower() == word.lower():
            clip = symbol_map.get(sym, {}).get("clip")
            if clip:
                clip_path = _memory_root(child, base_path) / "audio_session" / clip
                if clip_path.exists():
                    print(f"[LangLearn] Responding with: {word} → {clip_path.name}")
                    try:
                        from pydub import AudioSegment
                        from pydub.playback import play

                        audio = AudioSegment.from_file(clip_path)
                        play(audio)
                    except Exception as exc:
                        print(f"[LangLearn] Failed to play {clip_path.name}: {exc}")
            describe_word_grounding(child, word, base_path=base_path, verbose=True)
            return
    print(f"[LangLearn] No audio response found for: '{word}'")


def _build_reply_transformer_insights(
    symbols: List[str],
    vocab: Dict[str, Any],
    unknown: List[str],
) -> Optional[Dict[str, Any]]:
    known_words = [
        str((vocab.get(sym, {}) or {}).get("word") or sym).strip()
        for sym in symbols
        if str((vocab.get(sym, {}) or {}).get("word") or sym).strip()
    ]
    seed_inputs = []
    for value in known_words + list(unknown[:4]) + list(symbols[:4]):
        token = str(value or "").strip()
        if token and token not in seed_inputs:
            seed_inputs.append(token)

    insights: Dict[str, Any] = {}

    if seed_inputs:
        try:
            from transformers.seedling_transformer import SeedlingTransformer

            germinated = SeedlingTransformer(seed=_stable_symbol_seed("|".join(seed_inputs))).germinate(seed_inputs[:8])
            clusters = germinated.get("clusters") if isinstance(germinated, dict) else {}
            seeds = germinated.get("seeds") if isinstance(germinated, dict) else {}
            seed_suggestions = []
            if isinstance(seeds, dict):
                for value in seeds.values():
                    token = str(value or "").strip()
                    if token and token not in seed_suggestions:
                        seed_suggestions.append(token)
            insights["seedling"] = {
                "cluster_count": len(clusters) if isinstance(clusters, dict) else 0,
                "seed_suggestions": seed_suggestions[:4],
            }
        except Exception:
            pass

    if symbols or known_words or unknown:
        try:
            from transformers.mycelial_transformer import MycelialTransformer

            result = MycelialTransformer(max_links=2).weave(
                {
                    "tags": list(symbols[:4]),
                    "fragments": list(unknown[:4]),
                    "text": list(known_words[:4]),
                }
            )
            pathways = result.get("pathways") if isinstance(result, dict) else []
            pathways = pathways if isinstance(pathways, list) else []
            insights["mycelial"] = {
                "pathway_count": len(pathways),
                "sample_pathways": [item for item in pathways[:4] if isinstance(item, dict)],
            }
        except Exception:
            pass

    return insights or None


def generate_symbolic_reply_from_text(
    text: str,
    *,
    child: str = "Inazuma_Yagami",
    base_path: Optional[Path] = None,
    max_symbols: int = 4,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Try to reply to a text prompt using Ina's known symbol vocabulary.
    Returns a dict with {"text", "symbols", "unknown"} or None if no match found.
    """
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text)]
    if not tokens:
        return None

    links_payload = load_text_vocab_links(child, base_path=base_path)
    linked_word_to_symbol = _build_text_vocab_word_symbol_index(links_payload) if links_payload else {}
    lang_hint = guess_language_code(text)
    vocab: Dict[str, Any] = {}
    word_to_symbol: Dict[str, str] = {}
    embedding_index = []
    vocab_loaded = False

    def ensure_vocab_loaded():
        nonlocal vocab, word_to_symbol, embedding_index, vocab_loaded
        if vocab_loaded:
            return
        vocab = load_symbol_to_token(child, base_path=base_path)
        if _ensure_vocab_embeddings(vocab, language_hint=lang_hint):
            save_symbol_to_token(child, vocab, base_path)
        word_to_symbol = {
            (entry.get("word") or "").lower(): symbol
            for symbol, entry in vocab.items()
            if isinstance(entry, dict) and entry.get("word")
        }
        embedding_index = []
        for sym, entry in vocab.items():
            if not isinstance(entry, dict):
                continue
            emb = entry.get("embedding")
            if isinstance(emb, list) and emb:
                embedding_index.append((sym, emb, entry.get("language")))
        vocab_loaded = True

    matched: List[str] = []
    unknown: List[str] = []
    seen = set()

    for tok in tokens:
        sym = linked_word_to_symbol.get(tok)
        if sym:
            if sym not in seen:
                matched.append(sym)
                seen.add(sym)
            continue

        ensure_vocab_loaded()
        sym = word_to_symbol.get(tok)
        if sym:
            if sym not in seen:
                matched.append(sym)
                seen.add(sym)
            continue

        tok_emb = _EMBEDDER.embed_text(tok, language=lang_hint)
        best_sym = None
        best_sim = 0.0
        for sym_id, emb, lang in embedding_index:
            if sym_id in seen:
                continue
            if lang_hint != "und" and lang and lang != lang_hint:
                continue
            sim = _EMBEDDER.cosine(tok_emb, emb)
            if sim > 0.42 and sim > best_sim:
                best_sim = sim
                best_sym = sym_id
        if best_sym:
            matched.append(best_sym)
            seen.add(best_sym)
        elif tok not in unknown:
            unknown.append(tok)

    if unknown:
        seed_self_question(
            "What symbols align with the word(s): " + ", ".join(unknown[:6])
        )

    if not matched:
        return None

    symbols_to_speak = matched[:max_symbols]
    try:
        speak_symbolically(symbols_to_speak, child=child)
    except Exception:
        pass

    reply_context: Dict[str, Any] = dict(context) if isinstance(context, dict) else {}
    reply_context.setdefault("source_text", text)
    reply_context.setdefault("tokens", tokens)
    reply_tags = ["symbolic_reply"]
    for tag in _words_from_value(reply_context.get("tags")):
        if tag not in reply_tags:
            reply_tags.append(tag)
    reply_context["tags"] = reply_tags

    dual_message = build_dual_symbolic_message(
        symbols_to_speak,
        child=child,
        base_path=base_path,
        context=reply_context,
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
    )
    fallback_text = " ".join(symbols_to_speak)
    transformer_insights = _build_reply_transformer_insights(symbols_to_speak, vocab, unknown)
    return {
        "text": (dual_message or {}).get("text") or fallback_text,
        "symbols": symbols_to_speak,
        "unknown": unknown,
        "native_text": (dual_message or {}).get("native_text"),
        "gloss_text": (dual_message or {}).get("gloss_text"),
        "unresolved_symbols": (dual_message or {}).get("unresolved_symbols") or [],
        "native_sources": (dual_message or {}).get("native_sources") or {},
        "gloss_sources": (dual_message or {}).get("gloss_sources") or {},
        "transformer_insights": transformer_insights,
    }


def ensure_word_grounded(
    child: str,
    word: str,
    *,
    base_path: Optional[Path] = None,
) -> bool:
    """Ensure that a word is backed by experiential memory before use."""

    if is_word_grounded(child, word, base_path=base_path):
        return True

    groundings = describe_word_grounding(child, word, base_path=base_path)
    if groundings:
        context = ", ".join(g.get("event_id", "?") for g in groundings)
        seed_self_question(
            f"I recall events {context} but they do not ground '{word}' sufficiently."
        )
    else:
        seed_self_question(
            f"What experience grounds the word '{word}'? I should ask for clarification."
        )
    return False


# === Experience Graph Queries ===
def load_experience_graph(child: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(base_path) if base_path else Path("AI_Children")
    path = base / child / "memory" / "experiences" / "experience_graph.json"
    if not path.exists():
        return {"events": [], "edges": [], "words_index": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"events": [], "edges": [], "words_index": {}}


def describe_word_grounding(
    child: str,
    word: str,
    *,
    base_path: Optional[Path] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Return the experiences that ground a given word."""

    graph = load_experience_graph(child, base_path)
    grounded_events: List[Dict[str, Any]] = []
    target = word.lower()
    for event in graph.get("events", []):
        for usage in event.get("word_usage", []):
            if target in [w.lower() for w in usage.get("words", [])]:
                result = {
                    "event_id": event.get("id"),
                    "timestamp": event.get("timestamp"),
                    "situation_tags": event.get("situation_tags", []),
                    "narrative": event.get("narrative", ""),
                    "speaker": usage.get("speaker"),
                    "utterance": usage.get("utterance"),
                    "words": usage.get("words", []),
                }
                grounded_events.append(result)
                if verbose:
                    tags = ", ".join(result["situation_tags"])
                    print(
                        f"[LangLearn] Experience grounding for '{word}': event {result['event_id']}"
                        f" ({tags or 'no tags'}) — utterance '{usage.get('utterance')}'"
                    )
    return grounded_events


def suggest_words_for_context(
    child: str,
    *,
    situation_tags: Optional[Iterable[str]] = None,
    entity_labels: Optional[Iterable[str]] = None,
    base_path: Optional[Path] = None,
) -> List[str]:
    """Suggest vocabulary grounded in experiences matching the context."""

    graph = load_experience_graph(child, base_path)
    tags = set(tag.lower() for tag in (situation_tags or []))
    entities = set(label.lower() for label in (entity_labels or []))
    candidates: Dict[str, int] = {}

    for event in graph.get("events", []):
        event_tags = {tag.lower() for tag in event.get("situation_tags", [])}
        event_entities = {ent.lower() for ent in event.get("entities", [])}
        if tags and not tags.intersection(event_tags):
            continue
        if entities and not entities.intersection(event_entities):
            continue
        for usage in event.get("word_usage", []):
            for word in usage.get("words", []):
                candidates[word] = candidates.get(word, 0) + 1

    return sorted(candidates, key=lambda w: (-candidates[w], w))


def is_word_grounded(child: str, word: str, base_path: Optional[Path] = None) -> bool:
    """Check whether a vocabulary item is backed by at least one experience."""

    graph = load_experience_graph(child, base_path)
    target = word.lower()
    index = graph.get("words_index", {})
    if target in index and index[target]:
        return True
    for event in graph.get("events", []):
        for usage in event.get("word_usage", []):
            if target in [w.lower() for w in usage.get("words", [])]:
                return True
    return False

if __name__ == "__main__":
    config = load_config()
    child = config.get("current_child", "default_child")

    from transformers.fractal_multidimensional_transformers import FractalTransformer

    transformer = FractalTransformer()
    symbol_map = load_sound_symbol_map(child)
    fragments = load_fragments(child)

    print(f"[LangLearn] Loaded {len(fragments)} fragments and {len(symbol_map)} sound symbols.")
    summarize_known_words(child)
