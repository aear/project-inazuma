"""Bounded, ranked cross-domain lateral association."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from gui_hook import log_to_statusbox
from origin_record import make_origin

try:  # pragma: no cover
    from meaning_map import get_symbol_neighbors
except Exception:  # pragma: no cover
    def get_symbol_neighbors(symbol_id: str | None = None, tags: Iterable[str] | None = None, k: int = 5) -> List[str]:
        return []


def _tokens(value: str) -> set[str]:
    text = str(value).lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    chars = {text[index:index + 2] for index in range(max(0, len(text) - 1))}
    return words | chars


@dataclass
class Pathway:
    source: str
    target: str
    score: float
    factors: Dict[str, float] = field(default_factory=dict)
    relation: str = "lateral"

    def as_dict(self) -> Dict[str, Any]:
        return {"from": self.source, "to": self.target, "relation": self.relation, "score": self.score, "factors": self.factors}


class MycelialTransformer:
    VERSION = "V2"

    def __init__(self, max_links: int = 3, max_items: int = 256) -> None:
        self.max_links = max(1, int(max_links))
        self.max_items = max(2, min(2048, int(max_items)))

    def _expand_tags(self, tag: str, k: int) -> List[str]:
        try:
            return get_symbol_neighbors(tags=[tag], k=k)
        except Exception:
            return []

    @staticmethod
    def _score(
        left: str, right: str, emotional_vector: Mapping[str, float],
        usefulness: Mapping[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        left_tokens, right_tokens = _tokens(left), _tokens(right)
        union = left_tokens | right_tokens
        overlap = len(left_tokens & right_tokens) / max(1, len(union))
        semantic_distance = max(0.05, 1.0 - overlap)
        novelty = max(0.05, 1.0 - min(1.0, overlap * 1.5))
        emotion_keys = {str(key).lower() for key, value in emotional_vector.items() if abs(float(value or 0.0)) >= 0.2}
        emotion_hit = bool((left_tokens | right_tokens) & emotion_keys)
        emotional_relevance = 1.0 if emotion_hit else max(0.1, sum(abs(float(v or 0.0)) for v in emotional_vector.values()) / max(1, len(emotional_vector)))
        history_key = f"{left}->{right}"
        historical_usefulness = max(0.05, min(1.0, float(usefulness.get(history_key, usefulness.get(right, 0.5)) or 0.0)))
        score = novelty * semantic_distance * emotional_relevance * historical_usefulness
        factors = {
            "novelty": round(novelty, 4), "semantic_distance": round(semantic_distance, 4),
            "emotional_relevance": round(emotional_relevance, 4),
            "historical_usefulness": round(historical_usefulness, 4),
        }
        return round(score, 6), factors

    def weave(
        self, data: Dict[str, Iterable[str]], emotional_vector: Dict[str, float] | None = None,
        historical_usefulness: Mapping[str, float] | None = None,
    ) -> Dict[str, Any]:
        emotional_vector = emotional_vector or {}
        historical_usefulness = historical_usefulness or {}
        domains = ("tags", "fragments", "visuals", "audio", "text")
        items: List[Tuple[str, str]] = []
        seen = set()
        for domain in domains:
            for value in data.get(domain, []) or []:
                value_str = str(value)
                for candidate in (value_str, *self._expand_tags(value_str, self.max_links)):
                    key = (domain, str(candidate))
                    if key not in seen:
                        seen.add(key); items.append(key)
                    if len(items) >= self.max_items:
                        break
                if len(items) >= self.max_items:
                    break
            if len(items) >= self.max_items:
                break

        pathways = []
        for index, (left_domain, left) in enumerate(items):
            candidates = []
            for right_domain, right in items[index + 1:]:
                if left_domain == right_domain:
                    continue
                score, factors = self._score(left, right, emotional_vector, historical_usefulness)
                candidates.append(Pathway(f"{left_domain}:{left}", f"{right_domain}:{right}", score, factors))
            candidates.sort(key=lambda item: (-item.score, item.target))
            pathways.extend(candidates[:self.max_links])
        origin = make_origin(
            self.__class__.__name__, self.VERSION,
            inputs={domain: list(data.get(domain, []) or [])[:32] for domain in domains},
            trigger="cross_domain_association", metadata={
                "candidate_items": len(items), "retained_links": len(pathways),
                "max_items": self.max_items, "input_truncated": len(items) >= self.max_items,
            },
        )
        log_to_statusbox(f"[Mycelial] Retained {len(pathways)} ranked lateral pathways.")
        return {"pathways": [pathway.as_dict() for pathway in pathways], "origins": [origin]}
