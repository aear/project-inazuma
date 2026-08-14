"""Shared, deterministic ML primitives with no mandatory third-party deps."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from statistics import median
from typing import Any, Iterable, Optional, Sequence


def coerce_vector(values: Iterable[Any] | None, *, invalid: float = 0.0) -> list[float]:
    output = []
    for value in values or ():
        try:
            output.append(float(value))
        except (TypeError, ValueError, OverflowError):
            output.append(float(invalid))
    return output


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def normalize_vector(
    vector: Iterable[Any] | None,
    *,
    scale: float = 1.0,
    digits: Optional[int] = None,
) -> list[float]:
    values = coerce_vector(vector)
    norm = vector_norm(values) or 1.0
    normalized = [(value / norm) * float(scale) for value in values]
    return [round(value, digits) for value in normalized] if digits is not None else normalized


def normalize_distribution(values: Iterable[Any] | None) -> list[float]:
    """Normalize non-negative weights by mass, falling back to uniform."""
    vector = coerce_vector(values)
    if not vector:
        return []
    total = sum(vector)
    if total <= 0.0:
        return [1.0 / len(vector)] * len(vector)
    return [value / total for value in vector]


def shannon_entropy(values: Iterable[Any] | None) -> float:
    """Return natural-log entropy while ignoring zero/negative mass."""
    return -sum(value * math.log(value) for value in coerce_vector(values) if value > 0.0)


def mean_center(values: Iterable[Any] | None) -> list[float]:
    """Subtract the arithmetic mean from a vector."""
    vector = coerce_vector(values)
    if not vector:
        return []
    mean = sum(vector) / len(vector)
    return [value - mean for value in vector]


def cosine_similarity(
    left: Sequence[float], right: Sequence[float], *,
    epsilon: float = 1e-8, overlap_norms: bool = False,
) -> float:
    """Use the project's established unequal-length semantics."""
    if not left or not right:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    if overlap_norms:
        length = min(len(left), len(right))
        left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(length)))
        right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(length)))
    else:
        left_norm = vector_norm(left)
        right_norm = vector_norm(right)
    denominator = left_norm * right_norm + float(epsilon)
    return dot / denominator if denominator > 0.0 else 0.0


@lru_cache(maxsize=65_536)
def deterministic_hash_bucket(value: str, dimension: int) -> int:
    if int(dimension) <= 0:
        raise ValueError("dimension must be positive")
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % int(dimension)


def hash_project(tokens: Iterable[Any], dimension: int) -> list[float]:
    if int(dimension) <= 0:
        return []
    result = [0.0] * int(dimension)
    for token in tokens:
        result[deterministic_hash_bucket(str(token), int(dimension))] += 1.0
    return result


def numeric_summary(values: Iterable[Any] | None) -> list[float]:
    seq = coerce_vector(values)
    if not seq:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    mean = sum(seq) / len(seq)
    variance = sum((value - mean) ** 2 for value in seq) / len(seq)
    return [mean, math.sqrt(variance), min(seq), max(seq), float(median(seq))]
