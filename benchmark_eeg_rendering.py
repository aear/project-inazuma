"""Benchmark complete EEG render-buffer preparation with synthetic graphs.

This deliberately does not read Ina's memory. It uses only the standard library,
so it can run on development machines that do not have the Qt EEG stack loaded.

    python3 benchmark_eeg_rendering.py --sizes 10000,50000 --edge-ratio 5
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from typing import Any, Dict, Iterable, List

from eeg_rendering import RENDER_PROFILES, buffer_bytes, estimated_gl_vertices, pack_edges, pack_nodes


COLORS = {
    "memory_graph": (0.33, 0.82, 1.0),
    "logic": (0.52, 1.0, 0.76),
    "meaning_map": (0.98, 0.78, 0.36),
}


def _color(item: Dict[str, Any]):
    return COLORS.get(item.get("network_type"), (0.7, 0.78, 0.86))


def _graph(node_count: int, edge_ratio: int):
    rng = random.Random(node_count * 97 + edge_ratio)
    networks = tuple(COLORS)
    nodes = []
    positions = {}
    for index in range(node_count):
        pos = (rng.uniform(-12, 12), rng.uniform(-9, 9), rng.uniform(-7, 7))
        node = {
            "id": index,
            "pos": pos,
            "activation": rng.random(),
            "network_type": networks[index % len(networks)],
        }
        nodes.append(node)
        positions[index] = pos
    edge_count = node_count * edge_ratio
    edges = [
        {
            "source": index % node_count,
            "target": (index * 17 + 1) % node_count,
            "weight": rng.random(),
            "network_type": networks[index % len(networks)],
        }
        for index in range(edge_count)
    ]
    return nodes, edges, positions


def _case(node_count: int, edge_ratio: int, repeats: int) -> Dict[str, Any]:
    nodes, edges, positions = _graph(node_count, edge_ratio)
    samples: List[float] = []
    last_nodes = last_edges = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        last_nodes = pack_nodes(nodes, _color)
        last_edges = pack_edges(edges, positions, _color)
        samples.append(time.perf_counter() - started)
    assert last_nodes is not None and last_edges is not None
    profiles = {
        name: {
            "estimated_gl_vertices": estimated_gl_vertices(node_count, len(edges), name),
            "antialias": options["antialias"],
            "glow_pass": options["glow"],
            "pixel_sized_nodes": options["px_mode"],
        }
        for name, options in RENDER_PROFILES.items()
    }
    return {
        "neurons": node_count,
        "synapses": len(edges),
        "retained_neurons": last_nodes["count"],
        "retained_synapses": last_edges["count"],
        "median_prepare_ms": round(statistics.median(samples) * 1000.0, 3),
        "min_prepare_ms": round(min(samples) * 1000.0, 3),
        "buffer_mib": round(buffer_bytes(last_nodes, last_edges) / (1024 * 1024), 3),
        "profiles": profiles,
    }


def run(sizes: Iterable[int], edge_ratio: int, repeats: int) -> Dict[str, Any]:
    return {
        "benchmark": "EEG complete-graph render preparation",
        "input": "deterministic synthetic graph; no project memory read",
        "note": "CPU preparation is shared; profile vertex counts expose the relative GL workload.",
        "results": [_case(size, edge_ratio, repeats) for size in sizes],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000,50000")
    parser.add_argument("--edge-ratio", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    sizes = [max(1, int(value)) for value in args.sizes.split(",") if value.strip()]
    print(json.dumps(run(sizes, max(0, args.edge_ratio), max(1, args.repeats)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
