"""Synthetic benchmark for modular cognition dispatch; reads no Ina memory."""
from __future__ import annotations

import argparse
import json
import statistics
import time

from cognition_runtime import CapabilityRegistry, CapabilitySpec, CognitionRuntime, CognitiveContext, ResultBus


def _measure(fn, repeats):
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1000.0)
    return {"median_ms": round(statistics.median(values), 4), "min_ms": round(min(values), 4), "max_ms": round(max(values), 4)}


def run(iterations=1000, repeats=7):
    names = ("logic", "math", "prediction")
    handlers = {name: (lambda context, payload, name=name: {"capability": name, "value": payload}) for name in names}
    context = CognitiveContext.build(observations=[{"signal": 0.5}], goals=["interpret"], provenance=["synthetic"])
    registry = CapabilityRegistry([CapabilitySpec(name=name, description=name) for name in names])
    runtime = CognitionRuntime(registry, ResultBus(max_contributions=iterations * len(names)), max_parallel=1)
    for name, handler in handlers.items():
        runtime.install_handler(name, handler)
    def direct():
        for _ in range(iterations):
            for name in names:
                handlers[name](context, 1)
    def modular():
        for _ in range(iterations):
            for name in names:
                runtime.route(name, context, payload=1)
    return {
        "benchmark": "cognition runtime dispatch", "input": "synthetic; no Ina memory read",
        "iterations": iterations, "capabilities_per_iteration": len(names),
        "direct_reference": _measure(direct, repeats),
        "modular_runtime": _measure(modular, repeats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(max(1, args.iterations), max(1, args.repeats)), indent=2))


if __name__ == "__main__":
    main()
