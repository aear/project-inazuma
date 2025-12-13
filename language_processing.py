import os
import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
from embedding_stack import MultimodalEmbedder, guess_language_code
from model_manager import load_config, seed_self_question
from experience_logger import ExperienceLogger
from symbol_generator import (
    ACCENT_GLYPHS,
    ALPHANUMERIC_GLYPHS,
    CONCEPT_GLYPHS,
    EMOTION_GLYPHS,
    MODULATION_GLYPHS,
)

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
    emo = list(EMOTION_GLYPHS.values())[seed % len(EMOTION_GLYPHS)]
    mod = list(MODULATION_GLYPHS.values())[(seed // 7) % len(MODULATION_GLYPHS)]
    concept = list(CONCEPT_GLYPHS.values())[(seed // 13) % len(CONCEPT_GLYPHS)]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


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

        chunk = synthesize_from_fingerprint(playback_fingerprint, sr=sample_rate)
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

def synthesize_from_fingerprint(fingerprint, duration_ms=1500, sr=22050):
    import numpy as np

    pitch = fingerprint.get("pitch_mean", 440)
    freq = fingerprint.get("dominant_freq", pitch)
    volume = min(1.0, max(0.1, (fingerprint.get("volume_db", -40) + 60) / 60))
    silence = fingerprint.get("silence_ratio", 0.1)

    t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), False)
    waveform = np.sin(2 * np.pi * freq * t) * volume

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
    symbol_map_path = _memory_root(child, base_path) / "sound_symbol_map.json"

    if not os.path.exists(vocab_path) or not os.path.exists(symbol_map_path):
        print("[LangLearn] No vocab or symbol map found.")
        return

    if not ensure_word_grounded(child, word, base_path=base_path):
        print(f"[LangLearn] Cannot respond with '{word}' without experiential grounding.")
        return

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(symbol_map_path, "r") as f:
        symbol_map = json.load(f)

    for sym, entry in vocab.items():
        if entry["word"].lower() == word.lower():
            clip = symbol_map.get(sym, {}).get("clip")
            if clip:
                clip_path = _memory_root(child, base_path) / "audio_session" / clip
                if clip_path.exists():
                    print(f"[LangLearn] Responding with: {word} → {clip_path.name}")
                    from pydub import AudioSegment
                    from pydub.playback import play
                    audio = AudioSegment.from_mp3(clip_path)
                    play(audio)
            describe_word_grounding(child, word, base_path=base_path, verbose=True)
            return
    print(f"[LangLearn] No audio response found for: '{word}'")


def generate_symbolic_reply_from_text(
    text: str,
    *,
    child: str = "Inazuma_Yagami",
    base_path: Optional[Path] = None,
    max_symbols: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Try to reply to a text prompt using Ina's known symbol vocabulary.
    Returns a dict with {"text", "symbols", "unknown"} or None if no match found.
    """
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text)]
    if not tokens:
        return None

    vocab = load_symbol_to_token(child, base_path=base_path)
    lang_hint = guess_language_code(text)
    if _ensure_vocab_embeddings(vocab, language_hint=lang_hint):
        save_symbol_to_token(child, vocab, base_path)

    word_to_symbol = {
        (entry.get("word") or "").lower(): symbol
        for symbol, entry in vocab.items()
        if entry.get("word")
    }

    embedding_index = []
    for sym, entry in vocab.items():
        if not isinstance(entry, dict):
            continue
        emb = entry.get("embedding")
        if isinstance(emb, list) and emb:
            embedding_index.append((sym, emb, entry.get("language")))

    matched: List[str] = []
    unknown: List[str] = []
    seen = set()

    for tok in tokens:
        sym = word_to_symbol.get(tok)
        if sym and sym not in seen:
            matched.append(sym)
            seen.add(sym)
        else:
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

    labels = [(vocab.get(sym, {}) or {}).get("word") or sym for sym in symbols_to_speak]
    return {"text": " ".join(labels), "symbols": symbols_to_speak, "unknown": unknown}


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
