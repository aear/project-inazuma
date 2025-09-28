
# === raw_file_manager.py (Multimodal Self-Read) ===

import os
import json
import wave
import contextlib
import sys
import itertools
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox

_AUDIO_DIGEST_IMPORT_ERROR = None
try:
    from audio_digest import analyze_audio_clip, generate_fragment
except Exception as e:  # pragma: no cover - import guard
    analyze_audio_clip = None
    generate_fragment = None
    _AUDIO_DIGEST_IMPORT_ERROR = e


FRAG_LIMIT = 1000
ALLOWED_TEXT_EXT = {".txt", ".md", ".json", ".py"}
ALLOWED_MEDIA_EXT = {".png", ".jpg", ".jpeg", ".wav", ".mp3"}

# === Core Config and State ===
def load_config():
    path = Path("config.json")
    if not path.exists():
        log_to_statusbox("[Pretrain] config.json not found.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

def _load_path_from_config(key):
    value = config.get(key)
    if not value:
        return None
    try:
        return Path(value).expanduser()
    except TypeError:
        return None

book_folder_path = _load_path_from_config("book_folder_path")
music_folder_path = _load_path_from_config("music_folder_path")

def get_child():
    log_to_statusbox("[RawFileManager] Attempting to retrieve 'child'...")

    # First try to get from environment variable
    child = os.getenv("CHILD")
    if child:
        log_to_statusbox(f"[RawFileManager] Found 'child' in environment: {child}")
        return child

    # If not found, try to get from command line argument
    if len(sys.argv) > 1:
        child = sys.argv[1]
        log_to_statusbox(f"[RawFileManager] Found 'child' in command line args: {child}")
        return child

    # Fallback to config.json if not set by environment or args
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                child = config.get("current_child", "Inazuma_Yagami")
                log_to_statusbox(f"[RawFileManager] Found 'child' in config.json: {child}")
                return child
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Error loading config.json: {e}")
            return "Inazuma_Yagami"

    # If nothing works, return the default child
    log_to_statusbox("[RawFileManager] No valid 'child' found, using default: Inazuma_Yagami")
    return "Inazuma_Yagami"

child = get_child()

log_to_statusbox(f"[RawFileManager] Final child: {child}")


def is_readable_file(path):
    return (
        path.suffix.lower() in set(ALLOWED_TEXT_EXT).union(ALLOWED_MEDIA_EXT)
        and path.stat().st_size < 5 * 1024 * 1024  # < 5MB
    )

def load_history(child):
    path = Path("AI_Children") / child / "memory" / "read_history.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_history(child, history):
    path = Path("AI_Children") / child / "memory" / "read_history.json"
    with open(path, "w") as f:
        json.dump(history[-250:], f, indent=4)

def log_reflection(child, fragment):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "r") as f:
            reflection = json.load(f)
    except:
        reflection = {}

    history = reflection.get("self_read_fragments", [])
    history.append({
        "timestamp": fragment["timestamp"],
        "summary": fragment.get("summary", "")[:60],
        "filename": fragment.get("source")
    })
    reflection["self_read_fragments"] = history[-100:]

    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

def fragment_text(text, source, transformer):
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    fragments = []
    for chunk in chunks[:5]:
        frag = {
            "summary": chunk,
            "tags": ["self_read", "code" if source.endswith(".py") else "text"],
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"curiosity": 0.6, "focus": 0.4}
        }
        vec = transformer.encode(frag)
        frag["importance"] = vec["importance"]
        fragments.append(frag)
    return fragments

def fragment_image(image_path, transformer):
    try:
        with Image.open(image_path) as img:
            array = np.array(img.convert("L")).flatten().tolist()
        frag = {
            "modality": "image",
            "image_features": array[:512],
            "summary": f"Visual symbol or artifact from {image_path.name}",
            "tags": ["self_read", "image"],
            "source": str(image_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"focus": 0.3, "novelty": 0.5}
        }
        vec = transformer.encode_image_fragment(frag)
        frag["importance"] = vec["importance"]
        return [frag]
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to process image {image_path}: {e}")
        return []

def fragment_audio(audio_path, transformer):
    ext = audio_path.suffix.lower()

    if ext == ".wav":
        try:
            with contextlib.closing(wave.open(str(audio_path), "r")) as wf:
                frames = wf.readframes(wf.getnframes())
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Failed to process WAV {audio_path}: {e}")
            return []

        audio_data = list(frames[:1024])
        frag = {
            "modality": "audio",
            "audio_features": [x / 255.0 for x in audio_data],
            "summary": f"Sound fragment from {audio_path.name}",
            "tags": ["self_read", "audio"],
            "source": str(audio_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"attention": 0.5, "novelty": 0.6},
        }
        vec = transformer.encode_audio_fragment(frag)
        frag["importance"] = vec["importance"]
        return [frag]

    if ext == ".mp3":
        if analyze_audio_clip is None:
            log_to_statusbox(
                "[RawFileManager] MP3 decoding unavailable: "
                f"{_AUDIO_DIGEST_IMPORT_ERROR}"
            )
            return []
        try:
            analysis = analyze_audio_clip(audio_path, transformer)
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Failed to decode MP3 {audio_path}: {e}")
            return []

        if not analysis:
            log_to_statusbox(
                f"[RawFileManager] MP3 analysis returned no data for {audio_path.name}."
            )
            return []

        tags = ["self_read", "audio"]
        for tag in analysis.get("tags", []):
            if tag not in tags:
                tags.append(tag)

        frag = {
            "modality": "audio",
            "summary": analysis.get(
                "summary", f"Sound fragment from {audio_path.name}"
            ),
            "tags": tags,
            "source": str(audio_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": analysis.get("emotions", {"attention": 0.5}),
        }

        vec = transformer.encode_audio_fragment(frag)
        clarity = analysis.get("clarity")
        try:
            frag["importance"] = (
                round(float(clarity), 4) if clarity is not None else vec["importance"]
            )
        except (TypeError, ValueError):
            frag["importance"] = vec["importance"]

        return [frag]

    log_to_statusbox(
        f"[RawFileManager] Unsupported audio format for {audio_path.name}: {ext}"
    )
    return []

def self_read_and_train():
    child = get_child()
    default_root = Path.home() / "Projects" / "Project Inazuma"

    raw_history = load_history(child)
    history = {entry for entry in raw_history if "/" in entry}
    legacy_history = {entry for entry in raw_history if "/" not in entry}
    new_fragments = []

    roots = []
    seen_roots = set()

    def add_root(path, audio_only=False):
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return
        if resolved in seen_roots:
            return
        seen_roots.add(resolved)
        roots.append((path, audio_only))

    if default_root.exists():
        add_root(default_root, audio_only=False)
    else:
        log_to_statusbox(f"[SelfRead] Project root not found: {default_root}")

    if book_folder_path and book_folder_path.exists():
        add_root(book_folder_path, audio_only=False)
    elif book_folder_path:
        log_to_statusbox(f"[SelfRead] Book folder not found: {book_folder_path}")

    if music_folder_path and music_folder_path.exists():
        add_root(music_folder_path, audio_only=True)
    elif music_folder_path:
        log_to_statusbox(f"[SelfRead] Music folder not found: {music_folder_path}")

    log_to_statusbox(f"[SelfRead] Child set to: {child}")
    if roots:
        log_to_statusbox("[SelfRead] Roots to scan: " + ", ".join(str(path) for path, _ in roots))
    else:
        log_to_statusbox("[SelfRead] No available roots to scan.")
        return
    log_to_statusbox(f"[SelfRead] Loaded {len(history) + len(legacy_history)} previously seen files.")

    transformer = FractalTransformer()
    count = 0

    audio_patterns = ("*.wav", "*.mp3")

    for base_root, audio_only in roots:
        log_to_statusbox(f"[SelfRead] Scanning: {base_root}")

        if audio_only:
            file_iter = itertools.chain.from_iterable(base_root.rglob(pattern) for pattern in audio_patterns)
        else:
            file_iter = base_root.rglob("*")

        for path in file_iter:
            if not path.is_file():
                continue

            try:
                relative_path = path.relative_to(base_root)
            except ValueError:
                relative_path = path.name

            rel_str = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path)
            history_key = f"{base_root.name}/{rel_str}"


            elif ext in [".wav", ".mp3"]:
                result = fragment_audio(path, transformer)

            else:
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — unrecognized extension.")

            log_to_statusbox(f"[SelfRead] Inspecting: {path}")

            if history_key in history or (base_root == default_root and path.name in legacy_history):
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — already seen.")
                continue

            if not is_readable_file(path):
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — not a supported format or too large.")
                continue

            ext = path.suffix.lower()
            log_to_statusbox(f"[SelfRead] PROCESSING {path.name} (.{ext})")

            try:
                if ext in ALLOWED_TEXT_EXT:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    result = fragment_text(text, path.name, transformer)

                elif ext in [".png", ".jpg", ".jpeg"]:
                    result = fragment_image(path, transformer)

                elif ext in [".wav"]:
                    result = fragment_audio(path, transformer)

                elif ext in [".mp3"]:
                    log_to_statusbox(f"[SelfRead] NOTE: {path.name} is mp3 — audio_digest handles those. Skipping.")
                    continue

                else:
                    log_to_statusbox(f"[SelfRead] SKIP {path.name} — unrecognized extension.")
                    continue

                if result:
                    for frag in result:
                        frag_id = f"frag_selfread_{abs(hash(frag['summary'])) % 10**12}"
                        frag["id"] = frag_id

                        frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"
                        frag_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(frag_path, "w", encoding="utf-8") as f:
                            json.dump(frag, f, indent=4)

                        log_to_statusbox(f"[SelfRead] + Fragment saved: {frag_id} from {path.name}")
                        log_reflection(child, frag)
                        new_fragments.append(frag)

                    if base_root == default_root:
                        legacy_history.discard(path.name)
                    history.add(history_key)
                    count += len(result)

            except Exception as e:
                log_to_statusbox(f"[SelfRead] ERROR processing {path.name}: {e}")

            if count >= FRAG_LIMIT:
                log_to_statusbox("[SelfRead] Fragment limit reached — stopping scan.")
                break

        if count >= FRAG_LIMIT:
            break

    combined_history = list(history.union(legacy_history))
    save_history(child, combined_history)
    log_to_statusbox(f"[SelfRead] Done. {count} new fragments saved.")

    if count > 0:
        log_to_statusbox("[SelfRead] Calling training pipeline...")
        os.system("python train_fragments.py")
    else:
        log_to_statusbox("[SelfRead] No new fragments to train on.")


def pretrain_audio_digest(paths, child):
    log_to_statusbox(f"[PretrainDigest] Starting digest on {len(paths)} file(s) for {child}")
    transformer = FractalTransformer()

    if analyze_audio_clip is None or generate_fragment is None:
        log_to_statusbox(
            "[PretrainDigest] Audio digest unavailable: "
            f"{_AUDIO_DIGEST_IMPORT_ERROR}"
        )
        return

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            log_to_statusbox(f"[PretrainDigest] File not found: {path}")
            continue

        if not path.suffix.lower() in [".mp3", ".wav"]:
            log_to_statusbox(f"[PretrainDigest] Skipping unsupported file: {path.name}")
            continue

        try:
            log_to_statusbox(f"[PretrainDigest] Analyzing {path.name}...")
            result = analyze_audio_clip(path, transformer)
            if result:
                generate_fragment(path, result, child)
                log_to_statusbox(f"[PretrainDigest] + Fragment created for: {path.name}")
            else:
                log_to_statusbox(f"[PretrainDigest] Failed to analyze: {path.name}")
        except Exception as e:
            log_to_statusbox(f"[PretrainDigest] ERROR on {path.name}: {e}")


if __name__ == "__main__":
    self_read_and_train()
