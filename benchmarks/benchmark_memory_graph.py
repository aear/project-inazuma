"""Small real-I/O benchmark for Ina's mirror/catalogue pipeline."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

import memory_mirror_db as mirror


SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _duration_projection(total: int, rate: float) -> dict:
    seconds = total / rate if rate > 0 else 0.0
    return {
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600.0, 2),
        "days": round(seconds / 86400.0, 3),
        "years": round(seconds / SECONDS_PER_YEAR, 4),
    }


def _years(total: int, rate: float) -> Optional[float]:
    if rate <= 0:
        return None
    return total / rate / SECONDS_PER_YEAR


def run_benchmark(root: Path, *, records: int, payload_bytes: int, target: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ina_graph_bench_", dir=str(root)) as temp:
        base = Path(temp)
        source_dir = base / "events"
        source_dir.mkdir()
        config = {
            "memory_mirror_policy": {
                "enabled": True,
                "mirror_on_read": True,
                "db_root": str(base / "db"),
                "db_filename": "benchmark.sqlite3",
                "max_record_bytes": 25 * 1024 * 1024,
                "batch_records": 256,
                "batch_bytes": 16 * 1024 * 1024,
                "batch_seconds": 2.0,
                "wal_autocheckpoint_pages": 4096,
                "synchronous": "NORMAL",
                "remove_json_after_verified": False,
                "quarantine_json_after_verified": False,
            }
        }
        padding = "x" * max(0, payload_bytes)
        paths = []
        payloads = []
        create_started = time.perf_counter()
        for index in range(records):
            item_id = f"evt_benchmark_{index:09d}"
            payload = {
                "id": item_id,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "narrative": padding,
                "importance": 0.1,
            }
            path = source_dir / f"{item_id}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
            payloads.append(payload)
        create_seconds = time.perf_counter() - create_started

        ingest_started = time.perf_counter()
        for path, payload in zip(paths, payloads):
            mirror.mirror_json_file(
                "Benchmark", "experience_event", path, payload=payload, config=config
            )
        mirror.flush_mirror_writes(
            mirror.mirror_db_path("Benchmark", config), close=True
        )
        ingest_seconds = time.perf_counter() - ingest_started
        catalog_status = mirror.mirror_status("Benchmark", config)
        verified = sum(
            int(item.get("removal_eligible") or 0)
            for item in catalog_status.get("kinds", {}).values()
        )

        mirror._SESSION_CACHE.clear()
        replay_started = time.perf_counter()
        replay_hits = 0
        for path, payload in zip(paths, payloads):
            result = mirror.mirror_json_file(
                "Benchmark", "experience_event", path, payload=payload, config=config
            )
            replay_hits += result.get("status") == "cached_verified"
        mirror.flush_mirror_writes(close=True)
        replay_seconds = time.perf_counter() - replay_started

        ingest_rate = records / ingest_seconds if ingest_seconds else 0.0
        replay_rate = records / replay_seconds if replay_seconds else 0.0
        return {
            "root": str(root),
            "records": records,
            "payload_bytes": payload_bytes,
            "create_seconds": round(create_seconds, 4),
            "fresh_ingest_seconds": round(ingest_seconds, 4),
            "fresh_records_per_second": round(ingest_rate, 2),
            "verified_records": verified,
            "restart_skip_seconds": round(replay_seconds, 4),
            "restart_skip_records_per_second": round(replay_rate, 2),
            "restart_cache_hits": replay_hits,
            "target_records": target,
            "fresh_target_lower_bound": _duration_projection(target, ingest_rate),
            "restart_audit_lower_bound": _duration_projection(target, replay_rate),
            "warning": "Synthetic small-directory rates are optimistic; legacy flat-directory enumeration will be slower.",
        }


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Ina memory catalogue throughput.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--target", type=int, default=150_000_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_benchmark(
        args.root,
        records=max(1, args.records),
        payload_bytes=max(0, args.payload_bytes),
        target=max(1, args.target),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
