import os
import json
import cv2
import fitz  # PyMuPDF
from pathlib import Path
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question


def load_sound_symbol_map(child):
    path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def load_fragments(child):
    frag_path = Path("AI_Children") / child / "memory" / "fragments"
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

def load_symbol_to_token(child):
    path = Path("AI_Children") / child / "memory" / "symbol_to_token.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)
    
def save_symbol_to_token(child, data):
    path = Path("AI_Children") / child / "memory" / "symbol_to_token.json"
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

def associate_symbol_with_word(child, symbol_id, word, confidence=1.0):
    vocab = load_symbol_to_token(child)
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
    save_symbol_to_token(child, vocab)
    print(f"[LangLearn] Associated {symbol_id} → '{word}' with confidence {vocab[symbol_id]['confidence']}")

def backprop_symbol_confidence(child, predicted_word, expressed_symbol):
    vocab = load_symbol_to_token(child)
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
    save_symbol_to_token(child, vocab)
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

def respond_to_word(child, word):
    vocab_path = Path("AI_Children") / child / "memory" / "symbol_to_token.json"
    symbol_map_path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"

    if not os.path.exists(vocab_path) or not os.path.exists(symbol_map_path):
        print("[LangLearn] No vocab or symbol map found.")
        return

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(symbol_map_path, "r") as f:
        symbol_map = json.load(f)

    for sym, entry in vocab.items():
        if entry["word"].lower() == word.lower():
            clip = symbol_map.get(sym, {}).get("clip")
            if clip:
                clip_path = Path("AI_Children") / child / "memory" / "audio_session" / clip
                if clip_path.exists():
                    print(f"[LangLearn] Responding with: {word} → {clip_path.name}")
                    from pydub import AudioSegment
                    from pydub.playback import play
                    audio = AudioSegment.from_mp3(clip_path)
                    play(audio)
                    return
    print(f"[LangLearn] No audio response found for: '{word}'")

if __name__ == "__main__":
    config = load_config()
    child = config.get("current_child", "default_child")

    transformer = FractalTransformer()
    symbol_map = load_sound_symbol_map(child)
    fragments = load_fragments(child)

    print(f"[LangLearn] Loaded {len(fragments)} fragments and {len(symbol_map)} sound symbols.")
    summarize_known_words(child)
