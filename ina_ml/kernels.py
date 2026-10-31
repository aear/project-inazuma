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
    values = vector if hasattr(vector, "__len__") else tuple(vector)
    try:
        return math.hypot(*values)
    except TypeError:
        return math.hypot(*(float(value) for value in values))


def normalize_vector(
    vector: Iterable[Any] | None,
    *,
    scale: float = 1.0,
    digits: Optional[int] = None,
) -> list[float]:
    values = coerce_vector(vector)
    raw_norm = vector_norm(values)
    norm = raw_norm or 1.0
    if values and all(math.isfinite(value) for value in values) and (
        any(values) and (not math.isfinite(raw_norm) or raw_norm < 1e-150)
    ):
        # A finite vector can overflow/underflow intermediate squared norms.
        peak = max(abs(value) for value in values)
        scaled = [value / peak for value in values]
        length = math.hypot(*scaled)
        normalized = [(value / length) * float(scale) for value in scaled]
    else:
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
    padding = float(epsilon)
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if overlap_norms:
        length = min(len(left), len(right))
        left, right = left[:length], right[:length]
    a_norm, b_norm = vector_norm(left), vector_norm(right)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    if math.isfinite(a_norm) and math.isfinite(b_norm):
        # Finite hypot norms also establish finite coordinates. Keep the common
        # path allocation-light; scaled copies are only needed at numeric limits.
        dot = sum(float(x) * float(y) for x, y in zip(left, right))
        mass = a_norm * b_norm
        denominator = mass + padding
        if math.isfinite(dot) and math.isfinite(denominator) and mass > 1e-280:
            return max(-1.0, min(1.0, dot / denominator))
    a, b = [float(value) for value in left], [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*a, *b)):
        return 0.0  # Invalid coordinates cannot establish a similarity match.
    a_peak, b_peak = max(map(abs, a), default=0.0), max(map(abs, b), default=0.0)
    if not a_peak or not b_peak:
        return 0.0
    a_scaled, b_scaled = [x / a_peak for x in a], [y / b_peak for y in b]
    scaled_dot = math.fsum(x * y for x, y in zip(a_scaled, b_scaled))
    # Preserve epsilon's original absolute damping, including on tiny vectors.
    if padding:
        logarithm = math.log(padding) - math.log(a_peak) - math.log(b_peak)
        correction = math.inf if logarithm > 709.0 else math.exp(logarithm)
    else:
        correction = 0.0
    stable = scaled_dot / (math.hypot(*a_scaled) * math.hypot(*b_scaled) + correction)
    return max(-1.0, min(1.0, stable))


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
