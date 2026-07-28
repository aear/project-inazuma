"""Small dependency-free vector operations shared by cognitive graph modules."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Preserve the project's legacy unequal-length cosine semantics."""
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (vector_norm(left) * vector_norm(right) + 1e-8)


