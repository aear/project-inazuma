"""Deterministic microbenchmarks for Project Inazuma's CPU-heavy kernels.

This benchmark uses generated inputs only.  It does not read Ina's memories and
does not write project state.  Run it from the repository root:

    python3 -m benchmarks.benchmark_compute_hotspots

The cases intentionally avoid matches so clustering and graph construction scan
their full candidate sets.  This makes changes in quadratic hot paths visible.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List

import memory_graph
import meaning_map


def _vectors(count: int, dimensions: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        result.append([value / norm for value in vector])
    return result


def _measure(call: Callable[[], Any], repeats: int) -> Dict[str, float]:
    samples = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": round(statistics.median(samples), 6),
        "min_seconds": round(min(samples), 6),
        "max_seconds": round(max(samples), 6),
    }


def _synapse_case(count: int, dimensions: int, repeats: int) -> Dict[str, Any]:
    vectors = _vectors(count, dimensions, seed=count)
    neurons = [
        {"id": f"n{index}", "vector": vector}
        for index, vector in enumerate(vectors)
    ]
    result = _measure(
        lambda: memory_graph.build_synaptic_links(
            neurons,
            threshold=1.1,
            include_direction=False,
            compact_records=True,
        ),
        repeats,
    )
    result.update(
        {
            "case": "memory_graph.build_synaptic_links",
            "items": count,
            "dimensions": dimensions,
            "comparisons": count * (count - 1) // 2,
        }
    )
    return result


def _memory_cluster_case(count: int, dimensions: int, repeats: int) -> Dict[str, Any]:
    vectors = _vectors(count, dimensions, seed=count + 10_000)
    fragments = [
        {"id": f"f{index}", "tags": [f"tag-{index}"]}
        for index in range(count)
    ]
    cache = {fragment["id"]: vector for fragment, vector in zip(fragments, vectors)}
    result = _measure(
        lambda: memory_graph.cluster_fragments(
            fragments, cache, threshold=1.1, tag_weight=0.0
        ),
        repeats,
    )
    result.update(
        {
            "case": "memory_graph.cluster_fragments",
            "items": count,
            "dimensions": dimensions,
            "worst_case_comparisons": count * (count - 1) // 2,
        }
    )
    return result


def _meaning_cluster_case(count: int, dimensions: int, repeats: int) -> Dict[str, Any]:
    vectors = _vectors(count, dimensions, seed=count + 20_000)
    encoded = [
        {"id": f"s{index}", "tags": [f"tag-{index}"], "vector": vector}
        for index, vector in enumerate(vectors)
    ]
    result = _measure(
        lambda: meaning_map._cluster_encoded(encoded, threshold=1.1),
        repeats,
    )
    result.update(
        {
            "case": "meaning_map._cluster_encoded",
            "items": count,
            "dimensions": dimensions,
            "worst_case_comparisons": count * (count - 1) // 2,
        }
    )
    return result


def run(sizes: Iterable[int], dimensions: int, repeats: int) -> Dict[str, Any]:
    results = []
    for count in sizes:
        results.append(_synapse_case(count, dimensions, repeats))
        results.append(_memory_cluster_case(count, dimensions, repeats))
        results.append(_meaning_cluster_case(count, dimensions, repeats))
    return {
        "benchmark": "Project Inazuma compute hotspots",
        "input": "deterministic synthetic normalized vectors; no project memory read",
        "repeats": repeats,
        "results": results,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,250,500,1000")
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    sizes = [max(2, int(value)) for value in args.sizes.split(",") if value.strip()]
    payload = run(
        sizes,
        dimensions=max(2, args.dimensions),
        repeats=max(1, args.repeats),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
