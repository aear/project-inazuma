# === fractal_multidimensional_transformers.py (Multimodal Upgrade) ===

import math
import hashlib
from statistics import median
from typing import Iterable, List, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    # Lightweight fallback to keep transforms working without numpy
    class _NPFallback:
        @staticmethod
        def abs(seq):
            return [abs(x) for x in seq]

        @staticmethod
        def mean(seq):
            return sum(seq) / len(seq) if seq else 0.0

        @staticmethod
        def std(seq):
            mu = _NPFallback.mean(seq)
            return math.sqrt(sum((x - mu) ** 2 for x in seq) / max(len(seq), 1))

        @staticmethod
        def percentile(seq, q):
            if not seq:
                return 0.0
            data = sorted(seq)
            k = int((q / 100.0) * (len(data) - 1))
            return data[k]

    np = _NPFallback()

def load_precision_profile(child="Inazuma_Yagami"):
    """Return stored precision profile for a given child."""
    import json
    from pathlib import Path
    from gui_hook import log_to_statusbox

    profile_path = Path("AI_Children") / child / "memory" / "precision_profile.json"
    config_path = Path("precision_config.json")
    try:
        if profile_path.exists():
            with open(profile_path, "r") as f:
                return json.load(f)
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception as e:
        log_to_statusbox(f"[Precision] Failed to load profile: {e}")
    return {"max_precision": 64}



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
    def __init__(self, depth=3, length=7, embed_dim=64):
        self.structure = [FractalLayer(length) for _ in range(depth)]
        self.depth = depth
        self.length = length
        self.precision = 0.5
        self.embed_dim = embed_dim

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
                "summary": frag.get("summary", ""),
                "features_used": len(inputs),
            }
            batch_vectors.append(encoded)
        return batch_vectors

    # === New: Specific encoders ===
    def encode_audio_fragment(self, fragment):
        vec = self._numeric_embedding(fragment.get("audio_features", []))
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
        vec = self._text_emotion_embedding(fragment)
        return {
            "vector": vec,
            "importance": round(np.mean(np.abs(vec)), 4),
            "source": "FractalTransformer"
        }
    
    def process_inputs(self, fragment):
        # Determine modality and fallback
        if fragment.get("modality") == "audio":
            features = self._numeric_embedding(fragment.get("audio_features", []))
        elif fragment.get("modality") == "image":
            features = self._numeric_embedding(fragment.get("image_features", []))
        elif fragment.get("modality") == "video":
            features = self._numeric_embedding(fragment.get("video_features", []))
        elif "audio_features" in fragment:
            features = self._numeric_embedding(fragment.get("audio_features", []))
        elif "image_features" in fragment:
            features = self._numeric_embedding(fragment.get("image_features", []))
        else:
            features = self._text_emotion_embedding(fragment)

        return features if features else [0.0]

    def _text_emotion_embedding(self, fragment):
        summary = str(fragment.get("summary") or fragment.get("id") or "")
        tags = fragment.get("tags", [])
        emotions = fragment.get("emotions", {})

        # Character trigram hashing (deterministic, order-aware)
        clean = summary.lower()
        trigram_vec = [0.0] * self.embed_dim
        for i in range(len(clean) - 2):
            tri = clean[i : i + 3]
            h = int(hashlib.sha256(tri.encode()).hexdigest()[:8], 16)
            trigram_vec[h % self.embed_dim] += 1.0

        # Emotion sliders as context
        emo_values = [float(v or 0.0) for v in emotions.values()]
        if emo_values:
            emo_stats = self._describe_numeric(emo_values)
        else:
            emo_stats = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Tag presence hashed
        tag_vec = [0.0] * (self.embed_dim // 4)
        for tag in tags:
            h = int(hashlib.sha256(str(tag).encode()).hexdigest()[:8], 16)
            tag_vec[h % len(tag_vec)] += 1.0

        combined = trigram_vec + emo_stats + tag_vec
        return self._normalize_vector(combined)

    def _numeric_embedding(self, values: Sequence[float] | None) -> List[float]:
        seq = [float(v) for v in (values or [])]
        if not seq:
            return [0.0] * self.embed_dim

        stats = self._describe_numeric(seq)
        hashed = self._hash_project(seq, self.embed_dim - len(stats))
        return self._normalize_vector(stats + hashed)

    def _describe_numeric(self, seq: Sequence[float]) -> List[float]:
        if not seq:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        mean = np.mean(seq)
        std = np.std(seq)
        mn = min(seq)
        mx = max(seq)
        med = median(seq)
        return [float(round(x, 6)) for x in (mean, std, mn, mx, med)]

    def _hash_project(self, seq: Sequence[float], dim: int) -> List[float]:
        if dim <= 0:
            return []
        projected = [0.0] * dim
        for i, v in enumerate(seq):
            idx = (i * 1315423911) % dim
            projected[idx] += float(v)
        return projected

    def _normalize_vector(self, vec: Iterable[float]) -> List[float]:
        vec = list(vec)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round((v / norm) * self.precision, 6) for v in vec]
    
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
