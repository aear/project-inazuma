
# === fractal_multidimensional_transformers.py (Multimodal Upgrade) ===

import math
import hashlib
import numpy as np


class FractalLayer:
    def __init__(self, length=7):
        self.length = length

    def process(self, input_values, precision=0.5):
        output = []
        for i in range(self.length):
            mod = (i + 1) * precision
            val = sum(math.sin(mod * x + i) for x in input_values)
            output.append(val / len(input_values))
        return output

class FractalTransformer:
    def __init__(self, depth=3, length=7):
        self.structure = [FractalLayer(length) for _ in range(depth)]
        self.depth = depth
        self.length = length
        self.precision = 0.5

    def encode(self, fragment):
        if "image_features" in fragment:
            return self.encode_image_fragment(fragment)
        elif "audio_features" in fragment:
            return self.encode_audio_fragment(fragment)
        else:
            return self.encode_symbolic_fragment(fragment)
    
    def encode_fragment(self, fragment):
        """
        Alias for encode(). Provided for compatibility with modules expecting encode_fragment.
        """
        return self.encode(fragment)


    def encode_many(self, fragment_list):
        batch_vectors = []
        for frag in fragment_list:
            inputs = self.process_inputs(frag)
            state = inputs
            for layer in self.structure:
                state = layer.process(state, self.precision)

            avg = sum(state) / len(state)
            encoded = {
                "id": frag.get("id"),
                "vector": state,
                "precision": self.precision,
                "symbolic": "symbolic" in frag.get("tags", []),
                "importance": round(avg, 4),
                "tags": frag.get("tags", []),
                "timestamp": frag.get("timestamp", ""),
                "summary": frag.get("summary", "")
            }
            batch_vectors.append(encoded)
        return batch_vectors

    # === New: Specific encoders ===
    def encode_audio_fragment(self, fragment):
        base = fragment.get("summary") or fragment.get("id") or str(fragment)
        digest = hashlib.sha256(base.encode()).digest()
        vec = [((b - 128) / 128.0) * self.precision for b in digest[:64]]
        return {
            "vector": vec,
            "importance": round(np.mean(np.abs(vec)), 4),
            "source": "FractalTransformer"
        }

    def encode_image_fragment(self, fragment):
        fragment["modality"] = "image"
        return self.encode(fragment)

    def encode_video_fragment(self, fragment):
        fragment["modality"] = "video"
        return self.encode(fragment)

    def encode_symbolic_fragment(self, fragment):
        return self.encode_audio_fragment(fragment)  # for now, same logic
    
    def process_inputs(self, fragment):
        # Determine modality and fallback
        if fragment.get("modality") == "audio":
            features = fragment.get("audio_features", [])
        elif fragment.get("modality") == "image":
            features = fragment.get("image_features", [])
        elif fragment.get("modality") == "video":
            features = fragment.get("video_features", [])
        else:
            features = self._default_text_emotion_input(fragment)

        return features if features else [0.0]

    def _default_text_emotion_input(self, fragment):
        emotions = fragment.get("emotions", {})
        summary = fragment.get("summary", "")
        text_features = [ord(c) / 255 for c in summary[:64]]
        emotion_values = [float(v or 0.0) for v in emotions.values()]
        if not emotion_values:
            emotion_values = [0.0] * 3
        return emotion_values + text_features
    
    def load_precision_profile(self, child="Inazuma_Yagami"):
        import os
        import json
        from pathlib import Path
        from gui_hook import log_to_statusbox

        profile_path = Path("AI_Children") / child / "memory" / "precision_profile.json"
        config_path = Path("precision_config.json")

        precision = 0.5  # fallback

        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    profile = json.load(f)
                    precision = float(profile.get("max_precision", 0.5)) / 64.0
            except Exception as e:
                log_to_statusbox(f"[Precision] Failed to load profile: {e}")

        elif config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    precision = float(config.get("max_precision", 64)) / 64.0
            except Exception as e:
                log_to_statusbox(f"[Precision] Failed to load config: {e}")

        self.precision = round(precision, 4)
        log_to_statusbox(f"[Precision] Applied precision: {self.precision:.4f} ({int(self.precision * 64)}-bit)")

        return True


