"""Canonical dependency-light machine-learning kernels for Project Inazuma."""
from .numerics import RGBFrame
from .kernels import (
    coerce_vector,
    cosine_similarity,
    deterministic_hash_bucket,
    hash_project,
    mean_center,
    normalize_distribution,
    normalize_vector,
    numeric_summary,
    shannon_entropy,
    vector_norm,
)

__all__ = [
    "RGBFrame", "coerce_vector", "cosine_similarity", "deterministic_hash_bucket",
    "hash_project", "mean_center", "normalize_distribution", "normalize_vector",
    "numeric_summary", "shannon_entropy", "vector_norm",
]
