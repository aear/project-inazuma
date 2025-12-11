"""
Lightweight multimodal embedding stack to align audio, text, and symbol traces.

Goals:
- Stay dependency-light (pure Python + optional NumPy).
- Provide deterministic hashing so embeddings are stable across runs.
- Carry lightweight language hints derived from scripts/characters.
"""

import hashlib
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# Basic script heuristics to label text fragments with a language hint.
_SCRIPT_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"[ぁ-ゖァ-ヺー]"), "ja"),
    (re.compile(r"[一-鿿]"), "zh"),
    (re.compile(r"[가-힣]"), "ko"),
    (re.compile(r"[А-Яа-яЁё]"), "ru"),
    (re.compile(r"[α-ωΑ-Ω]"), "el"),
    (re.compile(r"[א-ת]"), "he"),
    (re.compile(r"[ا-ي]"), "ar"),
    (re.compile(r"[ऀ-ॿ]"), "hi"),
]


def guess_language_code(text: str, default: str = "und") -> str:
    """
    Heuristic language hint based on Unicode ranges.
    Returns ISO-like short codes or 'und'.
    """
    if not text:
        return default

    for pattern, code in _SCRIPT_HINTS:
        if pattern.search(text):
            return code

    # Default to English-like if mostly ASCII letters/spaces.
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    if ascii_ratio > 0.85:
        return "en"

    return default


def _safe_norm(vec: Iterable[float]) -> List[float]:
    lst = list(float(v) for v in vec)
    norm = math.sqrt(sum(v * v for v in lst)) or 1.0
    return [v / norm for v in lst]


def _hash_project(tokens: Iterable[str], dim: int) -> List[float]:
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    return vec


def _stats(values: Sequence[float], target_dim: int) -> List[float]:
    seq = [float(v) for v in (values or [])]
    if not seq:
        return [0.0] * target_dim

    if _np is not None:
        arr = _np.array(seq, dtype=float)
        stats = [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(_np.median(arr)),
            float(_np.percentile(arr, 25)),
            float(_np.percentile(arr, 75)),
        ]
    else:
        mean = sum(seq) / len(seq)
        var = sum((v - mean) ** 2 for v in seq) / max(len(seq), 1)
        stats = [
            mean,
            math.sqrt(var),
            min(seq),
            max(seq),
            sorted(seq)[len(seq) // 2],
            0.0,
            0.0,
        ]

    if len(stats) >= target_dim:
        return [float(v) for v in stats[:target_dim]]

    padded = list(stats)
    padded.extend([0.0] * (target_dim - len(padded)))
    return padded


class MultimodalEmbedder:
    """
    Deterministic embedding utility for:
      - text (character trigrams + tags + language hint)
      - symbol sequences (sound symbol IDs)
      - audio frames (mel-like frame matrices + texture hints)
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        # Allocate subspaces
        self.text_dim = dim
        self.symbol_dim = dim
        self.audio_dim = dim

    def embed_text(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[float]:
        lang = language or guess_language_code(text)
        clean = (text or "").lower()
        trigrams = [clean[i : i + 3] for i in range(len(clean) - 2)]
        trigram_vec = _hash_project(trigrams, max(8, self.text_dim - 16))

        tag_vec = _hash_project((tags or []), 8)
        lang_vec = _hash_project([f"lang:{lang}"], 8)
        combined = trigram_vec + tag_vec + lang_vec
        return _safe_norm(combined)[: self.text_dim]

    def embed_symbol_sequence(
        self,
        sequence: Sequence[str],
        *,
        language: Optional[str] = None,
    ) -> List[float]:
        seq = [str(sid) for sid in sequence if sid]
        if not seq:
            return [0.0] * self.symbol_dim

        hash_vec = _hash_project(seq, max(16, self.symbol_dim - 8))
        lang_vec = _hash_project([f"lang:{language}"] if language else [], 8)
        combined = hash_vec + lang_vec
        return _safe_norm(combined)[: self.symbol_dim]

    def embed_audio_frames(
        self,
        frames: Sequence[Sequence[float]],
        *,
        texture: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """
        frames: list of mel/feature frames (frame-major).
        texture: optional coarse stats from audio_digest.
        """
        if not frames:
            return [0.0] * self.audio_dim

        # Aggregate along time: per-bin mean and std.
        if _np is not None:
            arr = _np.array(frames, dtype=float)
            mean = arr.mean(axis=0).tolist()
            std = arr.std(axis=0).tolist()
        else:
            # Fallback without NumPy
            transposed = list(zip(*frames))
            mean = [sum(col) / max(len(col), 1) for col in transposed]
            std = [
                math.sqrt(sum((v - m) ** 2 for v in col) / max(len(col), 1))
                for col, m in zip(transposed, mean)
            ]

        stats_vec = _stats(mean + std, target_dim=max(16, self.audio_dim // 2))

        texture_terms = []
        if isinstance(texture, dict):
            for key in sorted(texture.keys()):
                try:
                    texture_terms.append(float(texture[key]))
                except Exception:
                    continue
        texture_vec = _stats(texture_terms, target_dim=8)

        combined = stats_vec + texture_vec
        if len(combined) < self.audio_dim:
            combined.extend([0.0] * (self.audio_dim - len(combined)))

        return _safe_norm(combined)[: self.audio_dim]

    def blend(self, *vectors: Sequence[float]) -> List[float]:
        valid = [list(map(float, v)) for v in vectors if v]
        if not valid:
            return [0.0] * self.dim

        length = min(len(v) for v in valid)
        summed = [0.0] * length
        for vec in valid:
            for i in range(length):
                summed[i] += vec[i]

        avg = [v / len(valid) for v in summed]
        return _safe_norm(avg)

    def cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        va = [float(x) for x in (a or [])]
        vb = [float(x) for x in (b or [])]
        if not va or not vb:
            return 0.0
        length = min(len(va), len(vb))
        dot = sum(va[i] * vb[i] for i in range(length))
        norm_a = math.sqrt(sum(x * x for x in va[:length])) or 1.0
        norm_b = math.sqrt(sum(x * x for x in vb[:length])) or 1.0
        return dot / (norm_a * norm_b)
