"""Grounded clustering with intentionally strange seed generation."""
from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Dict, Iterable, List, Mapping

from gui_hook import log_to_statusbox
from origin_record import make_origin


def _lexical_geometry(symbol: str) -> Dict[str, float]:
    text = str(symbol).lower()
    features: Dict[str, float] = {}
    for char in text:
        if char.isalnum():
            features[f"char:{char}"] = features.get(f"char:{char}", 0.0) + 1.0
    for index in range(max(0, len(text) - 1)):
        gram = text[index:index + 2]
        features[f"bigram:{gram}"] = features.get(f"bigram:{gram}", 0.0) + 1.5
    features[f"length:{min(8, len(text) // 2)}"] = 1.0
    return features


def _profile_geometry(symbol: str, profile: Mapping[str, Any] | None) -> Dict[str, float]:
    if not profile:
        return _lexical_geometry(symbol)
    features: Dict[str, float] = {}
    vector = profile.get("vector")
    if isinstance(vector, (list, tuple)):
        for index, value in enumerate(vector[:64]):
            try:
                features[f"vector:{index}"] = float(value)
            except (TypeError, ValueError):
                pass
    emotions = profile.get("emotions") or profile.get("emotional_profile")
    if isinstance(emotions, Mapping):
        for key, value in list(emotions.items())[:32]:
            try:
                features[f"emotion:{key}"] = float(value)
            except (TypeError, ValueError):
                pass
    for key in ("modality", "origin"):
        if profile.get(key):
            features[f"{key}:{profile[key]}"] = 1.0
    cooccurrence = profile.get("cooccurrence") or ()
    if isinstance(cooccurrence, (list, tuple, set)):
        for value in list(cooccurrence)[:32]:
            features[f"co:{value}"] = 1.0
    return features or _lexical_geometry(symbol)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class SeedlingTransformer:
    VERSION = "V2"

    def __init__(self, seed: int | None = None, similarity_threshold: float = 0.22) -> None:
        self._rng = random.Random(seed)
        self.similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))

    def germinate(
        self, symbols: Iterable[str], *, symbol_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        symbol_list = [str(symbol) for symbol in symbols]
        profiles = symbol_profiles or {}
        feature_rows = {symbol: _profile_geometry(symbol, profiles.get(symbol)) for symbol in symbol_list}
        groups: List[List[str]] = []
        centroids: List[Dict[str, float]] = []
        for symbol in symbol_list:
            features = feature_rows[symbol]
            best_index = -1
            best_score = self.similarity_threshold
            for index, centroid in enumerate(centroids):
                score = _cosine(features, centroid)
                if score >= best_score:
                    best_index, best_score = index, score
            if best_index < 0:
                groups.append([symbol]); centroids.append(dict(features))
                continue
            groups[best_index].append(symbol)
            centroid = centroids[best_index]
            size = len(groups[best_index])
            keys = set(centroid) | set(features)
            centroids[best_index] = {key: ((centroid.get(key, 0.0) * (size - 1)) + features.get(key, 0.0)) / size for key in keys}

        clusters: Dict[str, List[str]] = {}
        symbol_clusters: Dict[str, str] = {}
        seeds: Dict[str, str] = {}
        for group, centroid in zip(groups, centroids):
            signature = "|".join(f"{key}:{centroid[key]:.3f}" for key in sorted(centroid))
            key = "geometry_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
            clusters[key] = group
            for symbol in group:
                symbol_clusters[symbol] = key
            shuffled = group[:]
            self._rng.shuffle(shuffled)
            parts = [value[:max(1, len(value) // 2)] for value in shuffled[:2]]
            seeds[key] = "".join(parts)

        origin = make_origin(
            self.__class__.__name__, self.VERSION, inputs={"symbols": symbol_list[:32]},
            trigger="germination", metadata={
                "cluster_basis": "profile_geometry" if profiles else "lexical_geometry",
                "similarity_threshold": self.similarity_threshold, "cluster_count": len(clusters),
            },
        )
        log_to_statusbox(f"[Seedling] Germinated {len(seeds)} grounded seeds from {len(symbol_list)} symbols.")
        return {"clusters": clusters, "symbol_clusters": symbol_clusters, "seeds": seeds, "origins": [origin]}
