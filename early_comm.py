
# === early_comm.py (Full Rewrite) ===
# Symbol-aware communication, language adaptation, and device reasoning based on config.json

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from model_manager import (
    load_config, update_inastate, get_inastate, seed_self_question
)
from transformers.fractal_multidimensional_transformers import FractalTransformer
from language_processing import associate_symbol_with_word, backprop_symbol_confidence, synthesize_from_fingerprint
from gui_hook import log_to_statusbox


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
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

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

def predict_target_from_emotion(emotion):
    trust = emotion.get("trust", 0.0)
    novelty = emotion.get("novelty", 0.0)
    focus = emotion.get("focus", 0.0)
    if trust > 0.6 and novelty < 0.4:
        return "Hito"
    elif focus > 0.6 and trust < 0.4:
        return "Lex"
    return "unknown"
    
from pydub import AudioSegment
from pydub.playback import play

from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path
import json
from language_processing import speak_symbolically


def create_expression_fragment(child, expression, inferred, clarity, target, symbol_id, word_id, word_conf):
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
        "symbol_word_confidence": word_conf
    }

    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump(frag, f, indent=4)

    print(f"[Comms] Expression saved: {frag_id}")
    return frag

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
    clarity = round(sum(pred_vec) / len(pred_vec), 4)
    speaking_to = predict_target_from_emotion(inferred)

    symbol_map = load_sound_symbol_map(child)
    symbol_id, best_sim = None, 0.0
    for sid, meta in symbol_map.items():
        vec = transformer.encode({"emotions": meta.get("emotions", {})})["vector"]
        sim = cosine_similarity(pred_vec, vec)
        if sim > best_sim:
            symbol_id, best_sim = sid, sim
    log_to_statusbox(f"[Comms] Best sound symbol: {symbol_id} (sim: {best_sim:.4f})")
        

    word_map = load_symbol_words(child)
    word_id, word_conf = None, 0.0
    for word in word_map:
        if not word.get("components"): continue
        sum_text = word.get("summary", "")
        vec = transformer.encode({"summary": sum_text})["vector"]
        sim = cosine_similarity(pred_vec, vec)
        if sim > word_conf:
            word_id, word_conf = word["symbol_word_id"], sim
    log_to_statusbox(f"[Comms] Best word: {word_id} (conf: {word_conf:.4f})")
            

    expression = f"I feel something like {max(inferred, key=inferred.get, default='...')}"
    if symbol_id and best_sim > 0.85:
        expression = f"I feel this sound: {symbol_id}"
    if word_id and word_conf > 0.85:
        expression = f"I feel this word: {word_id}"

    log_to_statusbox(f"[Comms] Final expression: '{expression}'")
    frag = create_expression_fragment(child, expression, inferred, clarity, speaking_to, symbol_id, word_id, word_conf)
    log_to_statusbox(f"[Comms] Expression fragment saved: {frag['id']}")

    # === Audio expression attempt (log + speak)
    log_to_statusbox(f"[Comms] Preparing to speak: \"{expression}\"")

    try:        
        if symbol_id:
            speak_symbolically(symbol_id)
            log_to_statusbox("[Comms] Speech output triggered.")
        else:
            log_to_statusbox("[Comms] No symbol ID available for speech.")
    except Exception as e:
        log_to_statusbox(f"[Comms] Speech error: {e}")


    # Hook into language processing for learning
    if symbol_id and word_id:
        associate_symbol_with_word(child, symbol_id, word_id, word_conf)
        predicted = prediction.get("predicted_word", word_id)
        backprop_symbol_confidence(child, predicted, symbol_id)
    log_to_statusbox(f"[Comms] Associated {symbol_id} with {word_id} via language_processing.")


    update_inastate("currently_speaking", True)
    update_inastate("last_expression_time", time.time())
    update_inastate("last_spoken_symbol", symbol_id)
    update_inastate("last_symbol_word_id", word_id)

    time.sleep(1.5)
    update_inastate("currently_speaking", False)


if __name__ == "__main__":
    identify_devices_from_config()
    early_communicate()
