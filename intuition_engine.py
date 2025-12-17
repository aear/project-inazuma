"""
intuition_engine.py
-------------------

Bridges Ina's symbolic memory (Akashic Reservoir) with the quantum transformer
so intuition can collapse multiple possible states into a single “felt” hint.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gui_hook import log_to_statusbox
from transformers.fractal_multidimensional_transformers import FractalTransformer
from transformers.QTransformer import QTransformer

# Local copy of the 24D emotion axes so we avoid importing emotion_engine
EMOTION_SLIDERS: Tuple[str, ...] = (
    "intensity",
    "attention",
    "trust",
    "care",
    "curiosity",
    "novelty",
    "familiarity",
    "stress",
    "risk",
    "negativity",
    "positivity",
    "simplicity",
    "complexity",
    "interest",
    "clarity",
    "fuzziness",
    "alignment",
    "safety",
    "threat",
    "presence",
    "isolation",
    "connection",
    "ownership",
    "externality",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = (norm_a * norm_b) or 1e-6
    return dot / denom


def _normalize(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _emotion_vector(snapshot: Dict[str, Any]) -> List[float]:
    """
    Extract a deterministic 24D vector in slider order from an emotion snapshot.
    """
    if snapshot is None:
        return [0.0] * len(EMOTION_SLIDERS)

    values = snapshot.get("values") if isinstance(snapshot, dict) else snapshot
    if not isinstance(values, dict):
        return [0.0] * len(EMOTION_SLIDERS)

    vector = [_normalize(float(values.get(axis, 0.0))) for axis in EMOTION_SLIDERS]
    return vector


def _tag_overlap_score(tags_a: Iterable[str], tags_b: Iterable[str]) -> float:
    a = {t.lower() for t in tags_a if t}
    b = {t.lower() for t in tags_b if t}
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


@dataclass
class ReservoirEntry:
    fragment_id: str
    summary: str
    tags: List[str]
    vector: List[float]
    emotion_vector: List[float]
    timestamp: Optional[str]
    recency_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "summary": self.summary,
            "tags": self.tags,
            "vector": self.vector,
            "emotion_vector": self.emotion_vector,
            "timestamp": self.timestamp,
            "recency_weight": self.recency_weight,
        }


class AkashicMemory:
    """
    Lightweight, high-association cache of recent fragments.  Acts as a
    reservoir Ina can consult when logic/prediction falter.
    """

    def __init__(self, child: str, *, limit: int = 256):
        self.child = child
        self.limit = limit
        self.memory_root = Path("AI_Children") / child / "memory"
        self.fragments_root = self.memory_root / "fragments"
        self.state_path = self.memory_root / "akashic_memory.json"
        self.transformer = FractalTransformer(depth=2, length=5, embed_dim=32)
        self._reservoir: List[ReservoirEntry] = []
        self._last_refresh = 0.0
        self._refresh_interval = 600.0  # seconds

    # -------------------------------------------------------------- public API
    def ensure_loaded(self, force: bool = False) -> None:
        now = time.time()
        if self._reservoir and not force and (now - self._last_refresh) < self._refresh_interval:
            return

        disk_state = self._load_disk_state()
        if disk_state and not force and (now - disk_state.get("timestamp", 0.0)) < self._refresh_interval:
            self._reservoir = disk_state.get("entries", [])
            self._last_refresh = disk_state.get("timestamp", now)
            return

        self._reservoir = self._rebuild_from_fragments()
        self._last_refresh = now
        self._save_disk_state(self._reservoir)

    def contextual_sample(
        self,
        context_tags: Iterable[str],
        emotion_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        self.ensure_loaded()
        if not self._reservoir:
            return []

        tagset = {t.lower() for t in context_tags if t}
        candidates = []
        for entry in self._reservoir:
            tag_score = _tag_overlap_score(tagset, entry.tags)
            emo_score = _cosine_similarity(emotion_vector, entry.emotion_vector)
            weight = (
                0.5 * tag_score
                + 0.35 * emo_score
                + 0.15 * entry.recency_weight
            )
            candidates.append(
                {
                    "fragment_id": entry.fragment_id,
                    "summary": entry.summary,
                    "tags": entry.tags,
                    "vector": entry.vector,
                    "emotion_vector": entry.emotion_vector,
                    "timestamp": entry.timestamp,
                    "score": round(weight, 4),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

    # -------------------------------------------------------------- internals
    def _load_disk_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return {}
        entries = []
        for raw in data.get("reservoir", []):
            if not isinstance(raw, dict):
                continue
            vector = [float(v) for v in raw.get("vector", [])]
            emo_vec = [float(v) for v in raw.get("emotion_vector", [])]
            entries.append(
                ReservoirEntry(
                    fragment_id=str(raw.get("fragment_id")),
                    summary=str(raw.get("summary") or "")[:160],
                    tags=[str(t) for t in raw.get("tags", [])[:8]],
                    vector=vector,
                    emotion_vector=emo_vec,
                    timestamp=raw.get("timestamp"),
                    recency_weight=float(raw.get("recency_weight", 0.5)),
                )
            )
        return {"entries": entries, "timestamp": data.get("epoch", 0.0)}

    def _save_disk_state(self, entries: List[ReservoirEntry]) -> None:
        payload = {
            "reservoir": [entry.to_dict() for entry in entries],
            "epoch": time.time(),
            "updated": _now_iso(),
        }
        try:
            with self.state_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            log_to_statusbox(f"[Akasha] Failed to persist reservoir: {exc}")

    def _recent_fragment_paths(self) -> List[Path]:
        if not self.fragments_root.exists():
            return []
        entries: List[Tuple[float, Path]] = []
        for frag in self.fragments_root.rglob("frag_*.json"):
            try:
                entries.append((frag.stat().st_mtime, frag))
            except OSError:
                continue
        entries.sort(reverse=True)
        return [path for _, path in entries[: self.limit]]

    def _rebuild_from_fragments(self) -> List[ReservoirEntry]:
        entries: List[ReservoirEntry] = []
        paths = self._recent_fragment_paths()
        if not paths:
            return entries

        newest_mtime = paths[0].stat().st_mtime if paths else 0.0
        oldest_mtime = paths[-1].stat().st_mtime if paths else newest_mtime
        span = max(newest_mtime - oldest_mtime, 1.0)

        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    frag = json.load(fh)
            except Exception:
                continue

            frag_id = str(frag.get("id") or path.stem)
            summary = str(frag.get("summary") or "")[:160]
            tags = [str(t) for t in frag.get("tags", [])[:8]]
            emotions = frag.get("emotions", {})
            emo_vec = self._emotion_from_fragment(emotions)
            embedding = self._encode_fragment(frag)
            if not embedding:
                continue

            mtime = path.stat().st_mtime
            recency = (mtime - oldest_mtime) / span if span else 0.5
            entries.append(
                ReservoirEntry(
                    fragment_id=frag_id,
                    summary=summary,
                    tags=tags,
                    vector=embedding[:32],
                    emotion_vector=emo_vec,
                    timestamp=frag.get("timestamp"),
                    recency_weight=round(recency, 4),
                )
            )

        return entries

    def _encode_fragment(self, fragment: Dict[str, Any]) -> List[float]:
        try:
            embedding = self.transformer.encode_fragment(fragment)
        except Exception:
            return []
        vector = embedding.get("vector") if isinstance(embedding, dict) else None
        if not vector:
            return []
        return [float(v) for v in vector]

    def _emotion_from_fragment(self, emotions: Any) -> List[float]:
        if not isinstance(emotions, dict):
            return [0.0] * len(EMOTION_SLIDERS)
        sliders = emotions.get("sliders") if isinstance(emotions.get("sliders"), dict) else emotions
        if not isinstance(sliders, dict):
            return [0.0] * len(EMOTION_SLIDERS)
        return [_normalize(float(sliders.get(axis, 0.0))) for axis in EMOTION_SLIDERS]


class QuantumIntuitionEngine:
    """
    Interfaces directly with QTransformer.  Uses AkashicMemory to assemble a
    context window, feeds it into the quantum transformer, then produces
    intent-tagged telemetry.
    """

    def __init__(self, child: str):
        self.child = child
        self.akashic = AkashicMemory(child)
        self.quantum = QTransformer()

    def probe(
        self,
        *,
        context_tags: Iterable[str],
        emotion_snapshot: Dict[str, Any],
        fuzz_level: float = 0.0,
    ) -> Dict[str, Any]:
        emotion_vector = _emotion_vector(emotion_snapshot)
        context_list = [t for t in context_tags if t]
        reservoir = self.akashic.contextual_sample(context_list, emotion_vector, top_k=5)
        symbol_seed = self._compose_symbol_seed(context_list, reservoir)
        quantum_read = self.quantum.dream(symbol_seed, emotion_vector or [0.0] * self.quantum.qubit_count)

        collapse = self._contextual_collapse(reservoir, emotion_vector)
        emotion_bias = self._derive_emotion_bias(quantum_read.get("tags", []), fuzz_level=fuzz_level)

        insight = {
            "timestamp": _now_iso(),
            "context_tags": context_list,
            "symbol_seed": symbol_seed,
            "quantum_probe": quantum_read,
            "collapse": collapse,
            "emotion_bias": emotion_bias,
            "reservoir_sample": reservoir,
        }
        return insight

    # ------------------------------------------------------------ internals
    def _compose_symbol_seed(self, context_tags: List[str], reservoir: List[Dict[str, Any]]) -> str:
        tag_str = "|".join(sorted(context_tags)) if context_tags else "self"
        top_summaries = " ".join(entry.get("summary", "")[:32] for entry in reservoir[:3])
        return f"{self.child}:{tag_str}:{top_summaries[:120]}"

    def _contextual_collapse(
        self,
        reservoir: List[Dict[str, Any]],
        emotion_vector: List[float],
    ) -> Dict[str, Any]:
        if not reservoir:
            return {"collapsed": None, "candidates": []}

        # Treat candidate scores as amplitudes, then normalize into probabilities.
        weights = []
        for entry in reservoir:
            base_score = float(entry.get("score", 0.0))
            emo_score = _cosine_similarity(emotion_vector, entry.get("emotion_vector", []))
            weights.append(max(base_score * 0.7 + emo_score * 0.3, 1e-4))

        total = sum(weights) or 1e-4
        probs = [w / total for w in weights]

        candidates = []
        for entry, prob in zip(reservoir, probs):
            candidates.append(
                {
                    "fragment_id": entry.get("fragment_id"),
                    "summary": entry.get("summary"),
                    "tags": entry.get("tags"),
                    "probability": round(prob, 4),
                }
            )

        collapsed = max(candidates, key=lambda item: item["probability"])
        return {"collapsed": collapsed, "candidates": candidates}

    def _derive_emotion_bias(self, tags: Iterable[str], *, fuzz_level: float) -> Dict[str, float]:
        """
        Map quantum tags to gentle emotion slider nudges.  This provides the
        context-dependent bias described in the MainNotes.
        """
        tagset = {t.lower() for t in tags if t}
        bias: Dict[str, float] = {}

        if {"calm", "clarity"} & tagset:
            bias["stress"] = -0.05 * (1.0 + fuzz_level)
            bias["fuzziness"] = -0.03
            bias["clarity"] = 0.04
        if {"trust", "glow"} & tagset:
            bias["trust"] = 0.05
            bias["connection"] = 0.03
        if {"loss", "betrayal"} & tagset:
            bias["isolation"] = 0.04
            bias["connection"] = -0.03
        if {"curiosity", "spark"} & tagset:
            bias["curiosity"] = 0.04
            bias["attention"] = 0.02
        if {"tension", "fire"} & tagset:
            bias["intensity"] = 0.03
            bias["risk"] = 0.02

        if not bias and tagset:
            # Default: gentle reduction of fuzz when tags exist but no mapping.
            bias["fuzziness"] = -0.015

        return {k: round(v, 4) for k, v in bias.items()}
