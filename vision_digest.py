# vision_digest.py

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question
from gui_hook import log_to_statusbox
from language_processing import load_generated_symbols, cosine_similarity

def log_vision_digest(file_path, frag_id, matched_tags):
    config = load_config()
    child = config.get("current_child", "default_child")
    log_path = Path("AI_Children") / child / "memory" / "vision_digest_log.json"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": str(file_path.name),
        "fragment_id": frag_id,
        "matched_tags": matched_tags
    }

    try:
        if log_path.exists():
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []
    except:
        log = []

    log.append(entry)
    log = log[-150:]  # Trim to last 150 entries

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    log_to_statusbox(f"[VisionDigest] Logged: {frag_id} from {file_path.name}")


def run_text_recognition(image, child="Inazuma_Yagami"):
    # Convert input image to features
    if image is None:
        return []

    flat = image.flatten().tolist()
    image_features = flat[:512]

    symbol_map = load_generated_symbols(child)
    matches = []

    for entry in symbol_map:
        stored = entry.get("image_features")
        if not stored:
            continue
        sim = cosine_similarity(image_features, stored)
        if sim > 0.93:
            matches.append(entry["id"])

    return matches

def generate_fragment(path, features, modality, child, summary="vision digest", tags=None):
    frag_id = f"frag_vision_digest_{int(time.time())}"
    vision_symbol = "vision_symbol_" + frag_id[-6:]

    fragment = {
        "id": frag_id,
        "summary": summary,
        "tags": tags or ["vision", "symbolic", "digest"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "vision_digest",
        "clarity": 0.5,
        "modality": modality,
        "image_path" if modality == "image" else "video_path": path.name,
        f"{modality}_features": features,
        "vision_symbol": vision_symbol,
        "emotions": {"curiosity": 0.4, "focus": 0.3}
    }

    # Save to fragments
    frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"
    with open(frag_path, "w") as f:
        json.dump(fragment, f, indent=4)

    log_to_statusbox(f"[VisionDigest] Fragment saved: {frag_id}")
    return fragment

def match_known_symbols(features, symbol_map):
    best = None
    best_sim = 0.0
    for sym in symbol_map:
        vec = sym.get("image_features")
        if not vec:
            continue
        sim = cosine_similarity(features, vec)
        if sim > best_sim:
            best_sim = sim
            best = sym
    return best, best_sim

def process_image(path, transformer, child, symbol_map):
    try:
        image = cv2.imread(str(path))
        if image is None:
            log_to_statusbox(f"[VisionDigest] Failed to load image: {path.name}")
            return

        flat = image.flatten().tolist()
        features = flat[:512]

        tags = ["vision", "digest", "image"]
        best, sim = match_known_symbols(features, symbol_map)
        if best and sim > 0.9:
            tags.append(best["id"])
            log_to_statusbox(f"[VisionDigest] Symbol match: {best['id']} (sim {sim:.4f})")

        texts = run_text_recognition(image, child)
        if texts:
            tags += [f"text:{t}" for t in texts]

        frag = generate_fragment(path, features, "image", child, summary="image memory", tags=tags)
        log_vision_digest(path, frag["id"], tags)
        os.remove(path)

    except Exception as e:
        log_to_statusbox(f"[VisionDigest] Error on image {path.name}: {e}")

def process_video(path, transformer, child, symbol_map):
    try:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return
        last = frames[-1]
        flat = last.flatten().tolist()
        features = flat[:512]

        tags = ["vision", "digest", "video"]
        best, sim = match_known_symbols(features, symbol_map)
        if best and sim > 0.9:
            tags.append(best["id"])
            log_to_statusbox(f"[VisionDigest] Symbol match: {best['id']} (sim {sim:.4f})")

        texts = run_text_recognition(last)
        if texts:
            tags += [f"text:{t}" for t in texts]

        frag = generate_fragment(path, features, "video", child, summary="motion memory", tags=tags)
        log_vision_digest(path, frag["id"], tags)
        os.remove(path)

    except Exception as e:
        log_to_statusbox(f"[VisionDigest] Error on video {path.name}: {e}")

def run_vision_digest():
    config = load_config()
    child = config.get("current_child", "default_child")
    session_dir = Path("AI_Children") / child / "memory" / "vision_session"
    transformer = FractalTransformer()

    symbol_map = load_generated_symbols(child)
    images = list(session_dir.glob("*.jpg"))
    videos = list(session_dir.glob("*.mp4"))

    if not images and not videos:
        log_to_statusbox("[VisionDigest] No files found.")
        return

    for img in images:
        process_image(img, transformer, child, symbol_map)

    for vid in videos:
        process_video(vid, transformer, child, symbol_map)

if __name__ == "__main__":
    run_vision_digest()
