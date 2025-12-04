# audio_digest.py — FULL REWRITE FOR BIOLOGICAL COGNITION PIPELINE
#
# This module replaces the old digest logic with a genuine perceptual → symbolic
# audio pipeline suitable for Ina's brain architecture.
#
import json
import uuid
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from emotion_engine import get_current_emotion_state     # soft dependency
from gui_hook import log_to_statusbox
from fragmentation_engine import fragment_audio_digest
from model_manager import load_config

try:  # optional meaning map hook
    from meaning_map import update_proto_word_stats  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    update_proto_word_stats = None


# Legacy fallbacks (old location in repo root)
SYMBOL_MAP_PATH = Path("sound_symbol_map.json")
SYMBOL_WORDS_PATH = Path("symbol_words.json")

# Sample rate used across the pipeline
TARGET_SR = 44100

# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------

def _current_child(child=None):
    if child:
        return child
    try:
        cfg = load_config()
        return cfg.get("current_child", "default_child")
    except Exception:
        return "default_child"


def _memory_root(child):
    root = Path("AI_Children") / child / "memory"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _symbol_paths(child):
    mem_root = _memory_root(child)
    return mem_root / "sound_symbol_map.json", mem_root / "symbol_words.json"

# ------------------------------------------------------------
# Utility: Load + Save Symbol Maps
# ------------------------------------------------------------

def _read_json(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log_to_statusbox(f"[AudioDigest] Failed to read {path}: {e}")
    return default


def load_sound_symbol_map(child=None):
    """
    Load the sound symbol map from the child's memory folder, falling back
    to the legacy repo-root file if necessary.
    """
    child = _current_child(child)
    primary_path, _ = _symbol_paths(child)
    data = _read_json(primary_path, default={})

    # Backwards compatibility with legacy map location/shape
    if not data:
        legacy = _read_json(SYMBOL_MAP_PATH, default={})
        data = legacy

    if not isinstance(data, dict):
        data = {}

    if "symbols" not in data:
        data = {"symbols": data if isinstance(data, dict) else {}}

    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    return data, primary_path


def save_sound_symbol_map(data, child=None):
    child = _current_child(child)
    path, _ = _symbol_paths(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_symbol_words(child=None):
    """
    Load proto-word candidates discovered from audio symbols.
    """
    child = _current_child(child)
    _, words_path = _symbol_paths(child)
    data = _read_json(words_path, default={})

    # Backwards compatibility with legacy location
    if not data:
        legacy = _read_json(SYMBOL_WORDS_PATH, default={})
        data = legacy

    if not isinstance(data, dict):
        data = {}
    if "proto_words" not in data or not isinstance(data.get("proto_words"), dict):
        data["proto_words"] = {}
    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    return data, words_path


def save_symbol_words(data, child=None):
    child = _current_child(child)
    _, path = _symbol_paths(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------------------------
# 1. COCHLEAR FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_cochlear_features(wave, sr):
    """
    Biological-ish front-end:
    - Mel filterbank (20–40 bands)
    - Energy envelope
    - Frame-level perceptual vectors
    """
    try:
        mel = librosa.feature.melspectrogram(
            y=wave, sr=sr, n_mels=32, hop_length=256, n_fft=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db.T.tolist()  # frame-major

        return mel_db
    except Exception as e:
        log_to_statusbox(f"[AudioDigest] Feature extraction failed: {e}")
        return []


# ------------------------------------------------------------
# 2. FRAME → SYMBOL ASSIGNMENT
# ------------------------------------------------------------

def assign_sound_symbols(feature_frames, symbol_map):
    """
    A simple online clustering approach.
    - Compares frame to existing centroids
    - Creates new symbol if far from all existing ones
    """
    symbols = []
    threshold = 50.0  # distance threshold for new cluster; tunable

    seen_ts = datetime.now(timezone.utc).isoformat()

    if "symbols" not in symbol_map or not isinstance(symbol_map.get("symbols"), dict):
        symbol_map["symbols"] = {}

    for frame in feature_frames:
        f = np.array(frame)
        best_id = None
        best_dist = float("inf")

        for sym_id, sym_data in symbol_map["symbols"].items():
            centroid = np.array(sym_data["centroid"])
            dist = np.linalg.norm(f - centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = sym_id

        # If far from all existing centroids → create new symbol
        if best_dist > threshold or best_id is None:
            new_id = f"sym_snd_{uuid.uuid4().hex[:8]}"
            symbol_map["symbols"][new_id] = {
                "centroid": frame,
                "uses": 1,
                "last_seen": seen_ts,
            }
            symbols.append(new_id)
        else:
            # Update centroid (online mean)
            sym = symbol_map["symbols"][best_id]
            old = np.array(sym.get("centroid", frame))
            new = (old * sym.get("uses", 1) + f) / (sym.get("uses", 1) + 1)
            sym["centroid"] = new.tolist()
            sym["uses"] = sym.get("uses", 1) + 1
            sym["last_seen"] = seen_ts
            symbol_map["symbols"][best_id] = sym
            symbols.append(best_id)

    return symbols, symbol_map


# ------------------------------------------------------------
# 3. SYMBOL SEQUENCES → PROTO-WORDS
# ------------------------------------------------------------

def derive_proto_words(symbol_sequence, symbol_words):
    """
    Detect frequent n-grams (n=2–4). Lightweight proto-word discovery.
    """
    proto_store = symbol_words.setdefault("proto_words", {})
    if not isinstance(proto_store, dict):
        proto_store = {}
        symbol_words["proto_words"] = proto_store

    candidates = []

    for n in [2, 3, 4]:
        for i in range(len(symbol_sequence) - n + 1):
            seq = tuple(symbol_sequence[i : i + n])
            key = "_".join(seq)

            if key not in proto_store:
                proto_store[key] = {
                    "sequence": list(seq),
                    "uses": 1,
                }
            else:
                proto_store[key]["uses"] = proto_store.get(key, {}).get("uses", 0) + 1

            candidates.append(key)

    return candidates, symbol_words


# ------------------------------------------------------------
# 4. MAIN ANALYSIS STEP
# ------------------------------------------------------------

def analyze_audio_clip(clip_path, transformer=None, *, child=None, label="unknown"):
    """
    Converts an audio file into:
      - cochlear features
      - sound symbols
      - proto-words
    """
    child = _current_child(child)
    try:
        wave, sr = librosa.load(clip_path, sr=TARGET_SR)
        duration = librosa.get_duration(y=wave, sr=sr)
        features = extract_cochlear_features(wave, sr)

        symbol_map, symbol_map_path = load_sound_symbol_map(child)
        symbol_sequence, symbol_map = assign_sound_symbols(features, symbol_map)

        symbol_words, symbol_words_path = load_symbol_words(child)
        proto_word_candidates, symbol_words = derive_proto_words(symbol_sequence, symbol_words)

        save_sound_symbol_map(symbol_map, child)
        save_symbol_words(symbol_words, child)

        if update_proto_word_stats:
            try:
                update_proto_word_stats(proto_word_candidates)  # type: ignore[arg-type]
            except Exception as hook_err:
                log_to_statusbox(f"[AudioDigest] Proto-word hook failed: {hook_err}")

        clarity = round(float(np.mean(np.abs(wave))), 4)
        summary = (
            f"{label} audio: {len(symbol_sequence)} symbols, "
            f"{len(proto_word_candidates)} proto-words, {duration:.1f}s"
        )

        return {
            "duration": duration,
            "frames": features,
            "symbols": symbol_sequence,
            "proto_words": proto_word_candidates,
            "summary": summary,
            "clarity": clarity,
            "tags": ["audio", "digest", label] if label else ["audio", "digest"],
            "symbol_map_path": str(symbol_map_path),
            "symbol_words_path": str(symbol_words_path),
            "child": child,
        }

    except Exception as e:
        log_to_statusbox(f"[AudioDigest] analyze failed: {e}")
        return None


# ------------------------------------------------------------
# 5. FRAGMENT GENERATION
# ------------------------------------------------------------

def generate_fragment(clip_path, analysis, child=None, label="unknown"):
    """
    Creates a memory fragment compatible with:
      - memory_graph.py
      - training pipeline
      - emotional tagging
    """

    child = _current_child(child)

    # Emotion snapshot (soft dependency)
    try:
        emotion_snapshot = get_current_emotion_state()
    except Exception:
        emotion_snapshot = {}

    tags = analysis.get("tags", []) if isinstance(analysis, dict) else []
    if label and label not in tags:
        tags.append(label)
    if "audio" not in tags:
        tags.append("audio")
    if "live" not in tags:
        tags.append("live")

    summary = analysis.get("summary") if isinstance(analysis, dict) else None
    if not summary:
        summary = f"{label} audio fragment ({analysis.get('duration', 0):.1f}s)" if isinstance(analysis, dict) else "audio fragment"

    frag = fragment_audio_digest(
        clip_path=str(clip_path),
        label=label or "audio",
        analysis=analysis or {},
        tags=tags,
        importance=analysis.get("clarity", 0.1) if isinstance(analysis, dict) else 0.1,
        emotions=emotion_snapshot,
        symbols=analysis.get("symbols") if isinstance(analysis, dict) else None,
        store=True,
    )

    # Attach richer payload expected by downstream systems
    frag.update({
        "fragment_type": "audio",
        "source_file": str(clip_path),
        "symbols": analysis.get("symbols", []) if isinstance(analysis, dict) else [],
        "proto_words": analysis.get("proto_words", []) if isinstance(analysis, dict) else [],
        "features": analysis.get("frames", []) if isinstance(analysis, dict) else [],
        "duration": analysis.get("duration", 0) if isinstance(analysis, dict) else 0,
        "emotion": emotion_snapshot,
        "child": child,
    })
    frag.setdefault("summary", summary)

    # Legacy disk write (ensures downstream tools scanning fragments/ can see audio)
    try:
        legacy_dir = Path("AI_Children") / child / "memory" / "fragments"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = legacy_dir / f"{frag.get('id', uuid.uuid4().hex)}.json"
        with legacy_path.open("w", encoding="utf-8") as f:
            json.dump(frag, f, indent=2)
    except Exception as e:
        log_to_statusbox(f"[AudioDigest] Legacy fragment save failed: {e}")

    log_to_statusbox(f"[AudioDigest] Fragment saved: {frag.get('id')}")

    return frag
