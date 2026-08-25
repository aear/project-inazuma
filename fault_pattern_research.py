"""Deterministic, dependency-free tools for synthetic fault-pattern research.

This module models and measures bit faults.  It deliberately does not provide
encryption: experiments must treat their payload as already authenticated
ciphertext and must not claim that sparse XOR flips provide confidentiality.
"""
from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Iterable, Mapping, Sequence


MODEL_VERSION = "V1"
MAX_BITS = 8 * 1024 * 1024
MAX_EVENTS = 4096


def _positions(values: Iterable[int], bit_count: int) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if len(result) > MAX_EVENTS:
        raise ValueError(f"fault map exceeds {MAX_EVENTS} positions")
    if any(value < 0 or value >= bit_count for value in result):
        raise ValueError("fault position outside carrier")
    return result


def generate_fault_map(bit_count: int, event_count: int, *, seed: int,
                       cluster_probability: float = 0.12,
                       max_cluster_size: int = 4) -> tuple[int, ...]:
    """Generate a bounded synthetic map; event_count counts event anchors."""
    bit_count = int(bit_count)
    event_count = int(event_count)
    if not 8 <= bit_count <= MAX_BITS:
        raise ValueError(f"bit_count must be 8..{MAX_BITS}")
    if not 0 <= event_count <= min(MAX_EVENTS, bit_count):
        raise ValueError("invalid event_count")
    if not 0.0 <= float(cluster_probability) <= 1.0:
        raise ValueError("cluster_probability must be 0..1")
    if not 1 <= int(max_cluster_size) <= 16:
        raise ValueError("max_cluster_size must be 1..16")
    rng = random.Random(int(seed))
    faults: set[int] = set()
    for _ in range(event_count):
        anchor = rng.randrange(bit_count)
        faults.add(anchor)
        if rng.random() < cluster_probability:
            size = rng.randint(2, max_cluster_size)
            for offset in range(1, size):
                candidate = anchor + offset
                if candidate < bit_count:
                    faults.add(candidate)
    return _positions(faults, bit_count)


def apply_fault_map(carrier: bytes, positions: Iterable[int]) -> bytes:
    bit_count = len(carrier) * 8
    if bit_count > MAX_BITS:
        raise ValueError(f"carrier exceeds {MAX_BITS} bits")
    changed = bytearray(carrier)
    for position in _positions(positions, bit_count):
        changed[position // 8] ^= 1 << (position % 8)
    return bytes(changed)


def extract_fault_map(reference: bytes, observed: bytes) -> tuple[int, ...]:
    if len(reference) != len(observed):
        raise ValueError("reference and observed carriers must have equal length")
    if len(reference) * 8 > MAX_BITS:
        raise ValueError(f"carrier exceeds {MAX_BITS} bits")
    faults = []
    for byte_index, (left, right) in enumerate(zip(reference, observed)):
        difference = left ^ right
        for bit_index in range(8):
            if difference & (1 << bit_index):
                faults.append(byte_index * 8 + bit_index)
    return tuple(faults)


def fault_features(positions: Iterable[int], bit_count: int, *, bank_bits: int = 4096) -> dict[str, Any]:
    faults = _positions(positions, int(bit_count))
    gaps = [right - left for left, right in zip(faults, faults[1:])]
    adjacent = sum(gap == 1 for gap in gaps)
    bit_lanes = Counter(position % 8 for position in faults)
    banks = Counter(position // max(8, int(bank_bits)) for position in faults)
    return {
        "fault_count": len(faults),
        "density": len(faults) / bit_count,
        "adjacent_fraction": adjacent / max(1, len(gaps)),
        "mean_gap": sum(gaps) / len(gaps) if gaps else float(bit_count),
        "bit_lane_counts": [bit_lanes[index] for index in range(8)],
        "occupied_banks": len(banks),
        "max_bank_faults": max(banks.values(), default=0),
    }


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("distributions must have the same non-zero length")
    left_total, right_total = float(sum(left)), float(sum(right))
    if left_total <= 0.0 or right_total <= 0.0:
        raise ValueError("distributions must have positive mass")
    return 0.5 * sum(abs(a / left_total - b / right_total) for a, b in zip(left, right))


def position_capacity_bits(bit_count: int, fault_count: int) -> float:
    """Upper bound for choosing exactly fault_count positions among bit_count."""
    n, k = int(bit_count), int(fault_count)
    if n < 1 or not 0 <= k <= n:
        raise ValueError("require bit_count >= 1 and 0 <= fault_count <= bit_count")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def score_candidate(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Transparent starter score; Ina is expected to challenge and replace it."""
    lane_tv = total_variation(candidate["bit_lane_counts"], reference["bit_lane_counts"])
    density_scale = max(float(reference["density"]), 1e-12)
    density_error = abs(float(candidate["density"]) - float(reference["density"])) / density_scale
    cluster_error = abs(float(candidate["adjacent_fraction"]) - float(reference["adjacent_fraction"]))
    return {
        "bit_lane_total_variation": lane_tv,
        "relative_density_error": density_error,
        "adjacent_fraction_error": cluster_error,
        "starter_detectability": min(1.0, (lane_tv + min(1.0, density_error) + cluster_error) / 3.0),
    }


def create_challenge_experiment(lab: Any, *, hypothesis: str, code: str,
                                dataset: Any, autonomous_continuation_budget: int = 0) -> dict[str, Any]:
    """Copy these instruments into one immutable, isolated experiment artifact."""
    support_source = __import__(__name__).__loader__.get_source(__name__)
    if not support_source:
        raise RuntimeError("fault-pattern support source is unavailable")
    return lab.create(
        question=("Can an encoded payload remain exactly recoverable while its held-out "
                  "fault maps resemble the declared synthetic reference model?"),
        hypothesis=hypothesis,
        code=code,
        dataset=dataset,
        room="python-scratch",
        support_files={"fault_pattern_research.py": support_source},
        autonomous_continuation_budget=autonomous_continuation_budget,
    )


__all__ = [
    "MODEL_VERSION", "MAX_BITS", "MAX_EVENTS", "apply_fault_map", "extract_fault_map",
    "create_challenge_experiment", "fault_features", "generate_fault_map", "position_capacity_bits", "score_candidate",
    "total_variation",
]
