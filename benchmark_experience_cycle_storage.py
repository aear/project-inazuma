"""Explicit bounded historical/V2 Experience Engine storage benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import time
from typing import Any, Callable

from config_layers import load_config
from experience_engine import ExperienceCycleEngine
from historical_source import historical_module, resolve_revision
from module_benchmarks import TRANSFORMER_V1_REVISION


def _flush_files(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total += path.stat().st_size
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    return total


def _measure(root: Path, samples: int, operation: Callable[[Path, int], None]) -> dict[str, Any]:
    latencies, sizes = [], []
    for index in range(samples):
        with tempfile.TemporaryDirectory(prefix=".ina_experience_benchmark_", dir=root) as directory:
            sample_root = Path(directory)
            started = time.perf_counter()
            operation(sample_root, index)
            sizes.append(_flush_files(sample_root))
            latencies.append(time.perf_counter() - started)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95)))
    return {
        "samples": samples,
        "median_latency_ms": round(median(latencies) * 1000.0, 4),
        "p95_latency_ms": round(ordered[p95_index] * 1000.0, 4),
        "mean_storage_bytes": round(sum(sizes) / len(sizes), 2),
    }


def run_benchmark(*, hdd_root: Path, nvme_root: Path, samples: int = 9) -> dict[str, Any]:
    bounded = max(3, min(25, int(samples)))
    historical = historical_module("experience_logger.py", TRANSFORMER_V1_REVISION)

    def v1(sample_root: Path, index: int) -> None:
        logger = historical.ExperienceLogger(child=f"V1_{index}", base_path=sample_root)
        logger.log_event(
            situation_tags=["benchmark"], actions=[{"type": "attempt", "reference": f"payload-{index}"}],
            outcome={"observed": True}, narrative="one bounded attempt",
        )

    def v2(sample_root: Path, index: int) -> None:
        engine = ExperienceCycleEngine(child=f"V2_{index}", base_path=sample_root)
        cycle = engine.start_cycle("one bounded attempt", domain="benchmark", payload_references=[f"payload-{index}"])
        engine.complete_attempt(
            cycle["cycle_id"], attempt_reference=f"attempt-{index}",
            observation_references=[f"observation-{index}"], evaluation={"observed": True}, choice="stop",
        )

    devices = {"hdd": Path(hdd_root), "nvme": Path(nvme_root)}
    results: dict[str, Any] = {}
    for name, root in devices.items():
        if not root.is_dir():
            results[name] = {"available": False, "root": str(root)}
            continue
        results[name] = {
            "available": True, "root": str(root),
            "V1": _measure(root, bounded, v1), "V2": _measure(root, bounded, v2),
        }
    if results.get("hdd", {}).get("available") and results.get("nvme", {}).get("available"):
        results["comparison"] = {
            version: {
                "nvme_vs_hdd_median_ratio": round(
                    results["nvme"][version]["median_latency_ms"]
                    / max(0.0001, results["hdd"][version]["median_latency_ms"]), 4
                )
            }
            for version in ("V1", "V2")
        }
    return {
        "benchmark": "experience_cycle_storage", "benchmark_version": "V1",
        "historical_revision": resolve_revision(TRANSFORMER_V1_REVISION),
        "run_at": datetime.now(timezone.utc).isoformat(), "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    storage = config.get("storage_layout") if isinstance(config.get("storage_layout"), dict) else {}
    result = run_benchmark(
        hdd_root=Path(storage.get("durable_project_root") or "."),
        nvme_root=Path(storage.get("fast_root") or "/missing-fast-storage"),
        samples=args.samples,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
