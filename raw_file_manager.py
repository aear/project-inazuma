
# === raw_file_manager.py (Multimodal Self-Read) ===

import os
import json
import wave
import contextlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
from fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config
from gui_hook import log_to_statusbox


FRAG_LIMIT = 1000
ALLOWED_TEXT_EXT = {".txt", ".md", ".json", ".py"}
ALLOWED_MEDIA_EXT = {".png", ".jpg", ".jpeg", ".wav", ".mp3"}

# === Core Config and State ===
def load_config():
    path = Path("config.json")
    if not path.exists():
        log_to_statusbox(f"[Pretrain] config.json not found.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

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
        img = Image.open(image_path).convert("L")
        array = np.array(img).flatten().tolist()
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
    except:
        return []

def fragment_audio(audio_path, transformer):
    try:
        with contextlib.closing(wave.open(str(audio_path), 'r')) as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = list(frames[:1024])
            frag = {
                "modality": "audio",
                "audio_features": [x / 255.0 for x in audio_data],
                "summary": f"Sound fragment from {audio_path.name}",
                "tags": ["self_read", "audio"],
                "source": str(audio_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emotions": {"attention": 0.5, "novelty": 0.6}
            }
            vec = transformer.encode_audio_fragment(frag)
            frag["importance"] = vec["importance"]
            return [frag]
    except:
        return []

def self_read_and_train():
    child = get_child()
    root = Path.home() / "Projects" / "Project Inazuma"
    history = set(load_history(child))
    new_fragments = []

    log_to_statusbox(f"[SelfRead] Child set to: {child}")
    log_to_statusbox(f"[SelfRead] Scanning: {root}")
    log_to_statusbox(f"[SelfRead] Loaded {len(history)} previously seen files.")

    transformer = FractalTransformer()
    count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        log_to_statusbox(f"[SelfRead] Inspecting: {path}")

        if path.name in history:
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

                history.add(path.name)
                count += len(result)

        except Exception as e:
            log_to_statusbox(f"[SelfRead] ERROR processing {path.name}: {e}")

        if count >= FRAG_LIMIT:
            log_to_statusbox("[SelfRead] Fragment limit reached — stopping scan.")
            break

    save_history(child, list(history))
    log_to_statusbox(f"[SelfRead] Done. {count} new fragments saved.")
    
    if count > 0:
        log_to_statusbox("[SelfRead] Calling training pipeline...")
        os.system("python train_fragments.py")
    else:
        log_to_statusbox("[SelfRead] No new fragments to train on.")


from audio_digest import analyze_audio_clip, generate_fragment

def pretrain_audio_digest(paths, child):
    log_to_statusbox(f"[PretrainDigest] Starting digest on {len(paths)} file(s) for {child}")
    transformer = FractalTransformer()

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
