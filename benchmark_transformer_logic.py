"""Synthetic, memory-safe benchmark for transformer encoding and logic matching."""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time

import logic_engine
from transformers.fractal_multidimensional_transformers import FractalTransformer


def _measure(function, repeats: int) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def run(count: int = 2000, repeats: int = 7, seed: int = 1729) -> dict:
    rng = random.Random(seed)
    fragments = [{
        "id": f"f-{index}",
        "summary": "Ina notices a recurring pattern and considers its meaning " + str(index % 37),
        "tags": ["symbolic", f"tag-{index % 23}", f"context-{index % 11}"],
        "emotions": {
            "trust": rng.random(), "curiosity": rng.random(), "intensity": rng.random(),
        },
    } for index in range(max(1, count))]
    transformer = FractalTransformer()
    transformer._precision_next_refresh = float("inf")
    encoded = transformer.encode_many(fragments)
    prediction = {"predicted_vector": {"vector": encoded[0]["vector"]}}
    stored_words = [{
        "symbol_word_id": f"s-{index}",
        "summary": fragment["summary"],
        "tags": fragment["tags"],
        "vector": item["vector"],
    } for index, (fragment, item) in enumerate(zip(fragments, encoded))]
    missing_words = [{key: value for key, value in word.items() if key != "vector"}
                     for word in stored_words]
    return {
        "benchmark": "Project Inazuma transformer and logic kernels",
        "input": "deterministic synthetic fragments; no Ina memory read",
        "items": len(fragments),
        "repeats": repeats,
        "results": {
            "fractal_encode_many": _measure(lambda: transformer.encode_many(fragments), repeats),
            "logic_match_missing_vectors": _measure(
                lambda: logic_engine.test_prediction_against_logic(
                    prediction, missing_words, transformer
                ), repeats,
            ),
            "logic_match_stored_vectors": _measure(
                lambda: logic_engine.test_prediction_against_logic(
                    prediction, stored_words, transformer
                ), repeats,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(json.dumps(run(max(1, args.items), max(1, args.repeats), args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
