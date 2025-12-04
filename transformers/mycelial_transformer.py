"""mycelial_transformer.py

Mycelial Transformer
--------------------
A lightweight cross-domain inference engine inspired by the lateral growth of
mycelium.  Instead of building deep hierarchical representations it links
"distant cousins" of symbolic memory sideways, encouraging non linear pathways
useful for creative leaps and symbolic healing.

The transformer accepts fragments from different modalities and attempts to
weave small associative networks between them.  Integrations with optional
subsystems are kept minimal so the component can operate in isolation during
unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from gui_hook import log_to_statusbox

# Optional dependency to fetch symbol neighbours -------------------------------
try:  # pragma: no cover - optional dependency
    from meaning_map import get_symbol_neighbors
except Exception:  # pragma: no cover - fallback used in tests
    def get_symbol_neighbors(symbol_id: str | None = None,
                             tags: Iterable[str] | None = None,
                             k: int = 5) -> List[str]:
        return []


@dataclass
class Pathway:
    """Represents a sideways association between two symbolic items."""

    source: str
    target: str
    relation: str = "lateral"

    def as_dict(self) -> Dict[str, str]:
        return {"from": self.source, "to": self.target, "relation": self.relation}


class MycelialTransformer:
    """Build lateral symbolic pathways across modalities.

    Parameters
    ----------
    max_links:
        Maximum number of lateral links each item may form.  Keeps the network
        small and manageable.
    """

    def __init__(self, max_links: int = 3):
        self.max_links = max_links

    # ------------------------------------------------------------------ utils
    def _expand_tags(self, tag: str, k: int) -> List[str]:
        """Expand a tag sideways using the meaning map if available."""
        try:
            return get_symbol_neighbors(tags=[tag], k=k)
        except Exception:  # pragma: no cover - extreme edge case
            return []

    # ----------------------------------------------------------------- public
    def weave(self, data: Dict[str, Iterable[str]],
              emotional_vector: Dict[str, float] | None = None) -> Dict[str, List[Dict[str, str]]]:
        """Link fragments from different domains into lateral pathways.

        Parameters
        ----------
        data:
            Mapping containing any of ``tags``, ``fragments``, ``visuals``,
            ``audio`` or ``text``.  Each value should be an iterable of strings.
        emotional_vector:
            Optional mapping of emotion names to values.  When supplied, the
            average is logged but otherwise not used.
        """

        domains = ["tags", "fragments", "visuals", "audio", "text"]
        items: List[Tuple[str, str]] = []
        for domain in domains:
            for value in data.get(domain, []) or []:
                value_str = str(value)
                items.append((domain, value_str))
                # sideways growth: include neighbours that are not part of the
                # provided data
                for neigh in self._expand_tags(value_str, self.max_links):
                    items.append((domain, neigh))

        pathways: List[Pathway] = []
        for i, (d1, v1) in enumerate(items):
            links = 0
            for d2, v2 in items[i + 1:]:
                if d1 == d2:
                    continue  # sideways only across domains
                pathways.append(Pathway(f"{d1}:{v1}", f"{d2}:{v2}"))
                links += 1
                if links >= self.max_links:
                    break

        # optional emotional summary
        if emotional_vector:
            avg = round(sum(emotional_vector.values()) / len(emotional_vector), 4)
            log_to_statusbox(f"[Mycelial] Emotional resonance average: {avg}")
        log_to_statusbox(f"[Mycelial] Built {len(pathways)} lateral pathways.")

        return {"pathways": [p.as_dict() for p in pathways]}
