import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
from model_manager import load_config, seed_self_question
from experience_logger import ExperienceLogger

if TYPE_CHECKING:  # pragma: no cover
    from transformers.fractal_multidimensional_transformers import FractalTransformer


def _memory_root(child: str, base_path: Optional[Path] = None) -> Path:
    base = Path(base_path) if base_path else Path("AI_Children")
    return base / child / "memory"

LEGACY_SOUND_SYMBOL_MAP = Path("sound_symbol_map.json")


def _load_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_symbol_map(raw):
    if isinstance(raw, dict) and "symbols" in raw:
        return raw.get("symbols", {})
    return raw if isinstance(raw, dict) else {}


def load_sound_symbol_map(child, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "sound_symbol_map.json"
    data = _load_json(path) if path.exists() else None
    if data is None and LEGACY_SOUND_SYMBOL_MAP.exists():
        data = _load_json(LEGACY_SOUND_SYMBOL_MAP)
    return _normalize_symbol_map(data or {})

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
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_symbol_to_token(child, data, base_path: Optional[Path] = None):
    path = _memory_root(child, base_path) / "symbol_to_token.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

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
    confidence=1.0,
    grounding: Optional[Dict[str, str]] = None,
    base_path: Optional[Path] = None,
):
    vocab = load_symbol_to_token(child, base_path)
    if symbol_id not in vocab:
        vocab[symbol_id] = {"word": word, "uses": 1, "confidence": round(confidence, 2)}
    else:
        entry = vocab[symbol_id]
        if entry["word"].lower() != word.lower():
            seed_self_question(f"Was '{entry['word']}' the wrong word for {symbol_id}? Now seen as '{word}'?")
            entry["confidence"] = round(entry.get("confidence", 0.5) - 0.1, 2)
        else:
            entry["uses"] += 1
            entry["confidence"] = round(min(1.0, entry.get("confidence", 0.5) + 0.05), 2)
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
        entry["confidence"] = round(min(1.0, entry.get("confidence", 0.5) + 0.1), 2)

    vocab[expressed_symbol] = entry
    save_symbol_to_token(child, vocab, base_path)
    print(f"[LangLearn] Updated confidence for {expressed_symbol} → '{entry['word']}' to {entry['confidence']}")

def speak_symbolically(symbols, child="Inazuma_Yagami"):
    import numpy as np
    import sounddevice as sd
    from gui_hook import log_to_statusbox

    if isinstance(symbols, str):
        symbols = [symbols]

    symbol_map = load_sound_symbol_map(child)
    waveform = []

    for sid in symbols:
        entry = symbol_map.get(sid)
        if not entry:
            log_to_statusbox(f"[Voice] Symbol {sid} not found in sound symbol map.")
            continue

        fingerprint = entry.get("sound_features")
        if not fingerprint and "summary" in entry:
            fingerprint = extract_sound_features_from_summary(entry["summary"])
            log_to_statusbox(f"[Voice] Reconstructed sound features from summary for {sid}.")

        if not fingerprint:
            log_to_statusbox(f"[Voice] Symbol {sid} missing usable sound features.")
            continue

        chunk = synthesize_from_fingerprint(fingerprint)
        waveform.append(chunk)

    if waveform:
        audio = np.concatenate(waveform)
        log_to_statusbox(f"[Voice] Synthesizing {len(waveform)} sound symbols...")
        sd.play(audio, samplerate=22050)
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
