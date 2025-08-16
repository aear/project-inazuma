
import os
import json
import time
import hashlib
import numpy as np
import librosa
from datetime import datetime, timezone
from pathlib import Path
from pydub import AudioSegment
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question

def analyze_audio_clip(path, transformer):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        volume = 20 * np.log10(rms + 1e-6)

        # Fundamental frequency and pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size > 0 else 0.0
        pitch_var = float(np.var(pitch_vals)) if pitch_vals.size > 0 else 0.0

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()

        # Silence and zero-crossings
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        silence_threshold = 0.01
        silent = np.sum(np.abs(y) < silence_threshold)
        silence_ratio = silent / len(y)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc[1:], axis=1)
        mfcc_var = np.var(mfcc[1:], axis=1)

        # FFT dominant frequency
        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        dominant_freq = float(freqs[np.argmax(fft)]) if len(freqs) > 0 else 0.0

        # === Emotion encoding (clarity vector only uses emotional-feel features)
        clarity_input = {
            "duration": duration / 60,
            "volume": (volume + 60) / 60,
            "pitch": pitch_mean / 500,
            "pitch_var": pitch_var / 5000,
            "brightness": spectral_centroid / 5000,
            "bandwidth": spectral_bandwidth / 5000,
            "flatness": spectral_flatness,
            "zcr": zcr,
            "silence_ratio": silence_ratio,
            "mfcc_1": mfcc_mean[0] / 100,
            "mfcc_2": mfcc_mean[1] / 100,
            "dominant_freq": dominant_freq / 10000,
            "timestamp_hour": datetime.now().hour / 24.0
        }

        result = transformer.encode({"emotions": clarity_input})
        clarity = round(sum(result["vector"]) / len(result["vector"]), 4)

        summary = f"Sound with pitch {round(pitch_mean,1)} Hz, vol {round(volume,1)} dB, dom freq {round(dominant_freq,1)} Hz"

        tags = ["audio", "symbolic", "digest"]

        # === Auto tags
        if silence_ratio > 0.5:
            tags.append("quiet")
        elif volume > -30:
            tags.append("intense")

        if pitch_mean > 800:
            tags.append("high_pitch")
        elif pitch_mean < 200:
            tags.append("low_pitch")

        if spectral_flatness > 0.3:
            tags.append("harsh")
        elif spectral_centroid < 1000:
            tags.append("soft")

        if zcr > 0.1:
            tags.append("noisy")

        return {
            "summary": summary,
            "clarity": float(clarity),
            "duration": float(duration),
            "volume": float(volume),
            "emotions": {k: float(v) for k, v in clarity_input.items()},
            "tags": tags,
            "sound_features": {
                "pitch_mean": float(pitch_mean),
                "pitch_var": float(pitch_var),
                "dominant_freq": float(dominant_freq),
                "mfcc": [float(x) for x in mfcc_mean.tolist()],
                "mfcc_var": [float(x) for x in mfcc_var.tolist()],
                "zcr": float(zcr),
                "silence_ratio": float(silence_ratio),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "spectral_flatness": float(spectral_flatness),
                "volume_db": float(volume)
            },
            "symbolic_subspace": "audio_fingerprint"
        }


    except Exception as e:
        print(f"[Digest] Failed to analyze {path.name}: {e}")
        return None
def assign_sound_symbol(analysis, clip_name):
    base_string = f"{analysis['summary']}_{analysis['clarity']:.3f}"
    symbol_id = "sound_symbol_" + hashlib.sha1(base_string.encode()).hexdigest()[:12]
    return symbol_id


def generate_fragment(path, analysis, child):
    frag_id = f"frag_audio_digest_{int(time.time())}"
    symbol_id = assign_sound_symbol(analysis, path.name)

    fragment = {
        "id": frag_id,
        "summary": analysis["summary"],
        "tags": ["audio", "symbolic", "digest"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "audio_digest",
        "clarity": analysis["clarity"],
        "duration": analysis["duration"],
        "emotions": analysis["emotions"],
        "audio_path": path.name,
        "sound_symbol": symbol_id
    }

    frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"
    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump(fragment, f, indent=4)
    print(f"[Digest] Fragment saved: {frag_id}")

    # Log to symbol map
    symbol_map_path = Path("AI_Children") / child / "memory" / "sound_symbol_map.json"
    try:
        if symbol_map_path.exists():
            with open(symbol_map_path, "r") as f:
                symbol_map = json.load(f)
        else:
            symbol_map = {}

        symbol_map[symbol_id] = {
            "summary": analysis["summary"],
            "timestamp": fragment["timestamp"],
            "clip": path.name,
            "clarity": analysis["clarity"],
            "emotions": analysis["emotions"]
        }

        with open(symbol_map_path, "w", encoding="utf-8") as f:
            json.dump(symbol_map, f, indent=4)
        print(f"[Digest] Sound symbol assigned: {symbol_id}")
    except Exception as e:
        print(f"[Digest] Failed to update sound symbol map: {e}")

def run_audio_digest():
    config = load_config()
    child = config.get("current_child", "default_child")
    session_dir = Path("AI_Children") / child / "memory" / "audio_session"
    transformer = FractalTransformer()

    clips = list(session_dir.glob("*.mp3"))
    if not clips:
        print("[Digest] No audio clips to process.")
        return

    for clip in clips:
        result = analyze_audio_clip(clip, transformer)
        if result:
            generate_fragment(clip, result, child)
            try:
                os.remove(clip)
                print(f"[Digest] Removed processed clip: {clip.name}")
            except:
                print(f"[Digest] Failed to delete {clip.name}")

if __name__ == "__main__":
    run_audio_digest()
