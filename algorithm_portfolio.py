"""Tiered exact vector search plus a benchmark-driven cost model.

The names describe working-set scale, not Ina's retention tiers: local is a
small scalar scan, solar is a packed standard-library scan, and galactic uses
bounded packed batches. All three preserve the same exact cosine top-k contract.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import homo_silicus_numeric as hs

SearchHit = Tuple[str, float]
VectorRecord = Tuple[str, Sequence[float]]


def _top_k(scored: Iterable[SearchHit], k: int) -> List[SearchHit]:
    return heapq.nlargest(max(0, int(k)), scored, key=lambda item: (item[1], item[0]))


def _cosine(query: Sequence[float], vector: Sequence[float]) -> float:
    size = min(len(query), len(vector))
    if not size:
        return 0.0
    dot = sum(float(query[i]) * float(vector[i]) for i in range(size))
    qnorm = math.sqrt(sum(float(query[i]) ** 2 for i in range(size)))
    vnorm = math.sqrt(sum(float(vector[i]) ** 2 for i in range(size)))
    return dot / (qnorm * vnorm + 1e-8)


class LocalSearch:
    name = "local"

    def search(self, records: Iterable[VectorRecord], query: Sequence[float], k: int = 10) -> List[SearchHit]:
        return _top_k(((str(item_id), _cosine(query, vector)) for item_id, vector in records), k)


class SolarSearch:
    """Scan one compact contiguous float64 matrix without third-party code."""
    name = "solar"

    def search(self, records: Iterable[VectorRecord], query: Sequence[float], k: int = 10) -> List[SearchHit]:
        rows = list(records)
        if not rows:
            return []
        dimensions = min(len(query), len(rows[0][1]))
        if dimensions <= 0:
            return []
        # Unequal vectors use legacy per-pair truncation, so retain the scalar
        # path rather than silently changing its result contract.
        if any(min(len(query), len(vector)) != dimensions for _, vector in rows):
            return LocalSearch().search(rows, query, k)
        matrix = hs.array([vector[:dimensions] for _, vector in rows])
        scores = hs.cosine_rows(matrix, query[:dimensions])
        return _top_k(((str(rows[index][0]), score) for index, score in enumerate(scores)), k)


class GalacticSearch:
    name = "galactic"

    def __init__(self, batch_size: int = 8192):
        self.batch_size = max(1, int(batch_size))

    def search(self, records: Iterable[VectorRecord], query: Sequence[float], k: int = 10) -> List[SearchHit]:
        iterator = iter(records)
        best: List[SearchHit] = []
        while True:
            batch = []
            try:
                for _ in range(self.batch_size):
                    batch.append(next(iterator))
            except StopIteration:
                pass
            if not batch:
                break
            best = _top_k((*best, *SolarSearch().search(batch, query, k)), k)
            if len(batch) < self.batch_size:
                break
        return best


@dataclass(frozen=True)
class CostEstimate:
    tier: str
    milliseconds: float
    peak_rss_mb: float
    source: str


class EmpiricalCostModel:
    """Interpolate benchmark observations and choose within a memory budget."""

    def __init__(self, observations: Optional[Dict[str, List[dict]]] = None):
        self.observations = observations or {}

    @classmethod
    def from_report(cls, path: Path) -> "EmpiricalCostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        grouped: Dict[str, List[dict]] = {}
        for row in payload.get("results", []):
            grouped.setdefault(str(row["tier"]), []).append(row)
        return cls(grouped)

    def estimate(self, tier: str, items: int, dimensions: int) -> CostEstimate:
        rows = self.observations.get(tier, [])
        if rows:
            nearest = min(rows, key=lambda row: abs(int(row["items"]) - items))
            scale = max(1.0, items * dimensions) / max(1.0, float(nearest["items"]) * float(nearest["dimensions"]))
            return CostEstimate(tier, float(nearest["median_ms"]) * scale,
                                float(nearest["peak_rss_mb"]) * max(1.0, scale), "benchmark")
        work = max(1, items) * max(1, dimensions)
        priors = {
            "local": (work / 18_000.0, 8.0 + work * 0.000004),
            "solar": (0.25 + work / 350_000.0, 24.0 + work * 0.000008),
            "galactic": (0.8 + work / 250_000.0, 20.0 + min(work, 8192 * dimensions) * 0.000008),
        }
        milliseconds, memory = priors[tier]
        return CostEstimate(tier, milliseconds, memory, "prior")

    def choose(self, items: int, dimensions: int, memory_budget_mb: float = float("inf")) -> CostEstimate:
        tiers = ["local", "solar", "galactic"]
        estimates = [self.estimate(tier, items, dimensions) for tier in tiers]
        eligible = [item for item in estimates if item.peak_rss_mb <= memory_budget_mb]
        return min(eligible or estimates, key=lambda item: (item.milliseconds, item.peak_rss_mb))


class AlgorithmPortfolio:
    def __init__(self, cost_model: Optional[EmpiricalCostModel] = None, galactic_batch_size: int = 8192):
        self.cost_model = cost_model or EmpiricalCostModel()
        self.algorithms = {"local": LocalSearch(), "solar": SolarSearch(),
                           "galactic": GalacticSearch(galactic_batch_size)}

    def search(self, records: Iterable[VectorRecord], query: Sequence[float], *, count: int,
               k: int = 10, memory_budget_mb: float = float("inf"), tier: Optional[str] = None):
        estimate = (self.cost_model.estimate(tier, count, len(query)) if tier else
                    self.cost_model.choose(count, len(query), memory_budget_mb))
        return self.algorithms[estimate.tier].search(records, query, k), estimate
