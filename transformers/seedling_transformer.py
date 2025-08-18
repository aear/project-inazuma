"""seedling_transformer.py

Seedling Transformer
--------------------
Early-stage ideation — pre-conceptual thinking.  Takes messy symbolic
fragments and attempts to coax nascent structures.  It may produce nonsense or
unexpected gems, serving as a germination stage rather than optimisation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List

from gui_hook import log_to_statusbox


class SeedlingTransformer:
    """Germinate new symbolic seeds from noisy fragments.

    Parameters
    ----------
    seed:
        Optional random seed to make behaviour deterministic for tests.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ----------------------------------------------------------------- public
    def germinate(self, symbols: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
        """Form clusters and generate emergent seed strings.

        Parameters
        ----------
        symbols:
            Iterable of raw symbol fragments.

        Returns
        -------
        dict
            Mapping with ``clusters`` and ``seeds`` keys. ``clusters`` maps the
            cluster key to the list of original symbols while ``seeds`` maps the
            same key to a newly combined fragment.
        """

        symbol_list = [str(s) for s in symbols]

        clusters: Dict[str, List[str]] = defaultdict(list)
        for sym in symbol_list:
            key = sym[0] if sym else self._rng.choice("abcdefghijklmnopqrstuvwxyz")
            clusters[key].append(sym)

        seeds: Dict[str, str] = {}
        for key, group in clusters.items():
            if not group:
                continue
            shuffled = group[:]
            self._rng.shuffle(shuffled)
            parts = [g[: max(1, len(g) // 2)] for g in shuffled[:2]]
            seeds[key] = "".join(parts)

        log_to_statusbox(
            f"[Seedling] Germinated {len(seeds)} seeds from {len(symbol_list)} symbols."
        )
        return {"clusters": dict(clusters), "seeds": seeds}
