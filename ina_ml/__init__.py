"""Canonical dependency-light machine-learning kernels for Project Inazuma."""
from .numerics import RGBFrame
from .kernels import (
    coerce_vector,
    cosine_similarity,
    deterministic_hash_bucket,
    hash_project,
    normalize_vector,
    numeric_summary,
    vector_norm,
)

__all__ = [
    "RGBFrame", "coerce_vector", "cosine_similarity", "deterministic_hash_bucket",
    "hash_project", "normalize_vector", "numeric_summary", "vector_norm",
]
