"""Benchmark local/solar/galactic crossovers on synthetic or bounded Ina data."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import threading
import time
from pathlib import Path

from algorithm_portfolio import GalacticSearch, LocalSearch, SolarSearch
from embedding_stack import MultimodalEmbedder

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _rss_bytes():
    if psutil is not None:
        return psutil.Process().memory_info().rss
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def normalized_vectors(count, dimensions, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        rows.append((f"synthetic-{index}", [value / norm for value in vector]))
    return rows


def _fragment_vector(payload, embedder):
    for key in ("embedding", "vector", "latent_vector"):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
            return value
    text = next((payload.get(key) for key in ("text", "narrative", "summary", "content", "transcript")
                 if isinstance(payload.get(key), str) and payload.get(key)), "")
    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    return embedder.embed_text((text or " ".join(map(str, tags)))[:8192], tags=tags[:32])


def sample_real_fragments(root: Path, limit: int, scan_limit: int, seed: int):
    """Bounded reservoir sample; never recurse or enumerate the whole store."""
    candidates = []
    directories = [root, *(root / tier for tier in ("short", "working", "long", "cold"))]
    per_dir = max(1, scan_limit // len(directories))
    for directory in directories:
        if not directory.is_dir():
            continue
        with os.scandir(directory) as entries:
            for offset, entry in enumerate(entries):
                if offset >= per_dir:
                    break
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
                    candidates.append(Path(entry.path))
    random.Random(seed).shuffle(candidates)
    embedder = MultimodalEmbedder()
    records = []
    for path in candidates:
        if len(records) >= limit:
            break
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            vector = list(map(float, _fragment_vector(payload, embedder)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if vector:
            records.append((str(payload.get("id") or path.stem), vector))
    return records


def measure(fn, repeats):
    peak = _rss_bytes()
    stop = threading.Event()

    def watch():
        nonlocal peak
        while not stop.wait(0.01):
            try:
                peak = max(peak, _rss_bytes())
            except Exception:
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    samples = []
    try:
        for _ in range(repeats):
            gc.collect()
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
    finally:
        stop.set()
        watcher.join(timeout=1.0)
    return {"median_ms": round(statistics.median(samples), 4), "min_ms": round(min(samples), 4),
            "peak_rss_mb": round(peak / (1024 * 1024), 3)}


def crossovers(results):
    output, previous = [], None
    for size in sorted({int(row["items"]) for row in results}):
        winner = min((row for row in results if int(row["items"]) == size),
                     key=lambda row: (float(row["median_ms"]), float(row["peak_rss_mb"])))
        if winner["tier"] != previous:
            output.append({"from_items": size, "preferred_tier": winner["tier"]})
            previous = winner["tier"]
    return output


def run(records, sizes, repeats, source):
    results = []
    for size in sizes:
        subset = list(records[:size])
        if not subset:
            continue
        query = subset[len(subset) // 2][1]
        for runner in (LocalSearch(), SolarSearch(), GalacticSearch()):
            metrics = measure(lambda r=runner: r.search(subset, query, 10), repeats)
            metrics.update({"tier": runner.name, "items": len(subset), "dimensions": len(query)})
            results.append(metrics)
    return {"benchmark": "Ina tiered exact-search portfolio", "source": source,
            "memory_metric": "peak process RSS sampled through psutil or /proc (includes dataset)",
            "results": results, "crossovers": crossovers(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,1000,10000")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--child", default="Inazuma_Yagami")
    parser.add_argument("--real-limit", type=int, default=10000)
    parser.add_argument("--scan-limit", type=int, default=50000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = sorted({max(1, int(value)) for value in args.sizes.split(",") if value.strip()})
    if args.real:
        root = Path("AI_Children") / args.child / "memory" / "fragments"
        records = sample_real_fragments(root, max(args.real_limit, max(sizes)), args.scan_limit, args.seed)
        source = f"bounded sample of {root}"
    else:
        records = normalized_vectors(max(sizes), max(2, args.dimensions), args.seed)
        source = "deterministic synthetic normalized vectors"
    report = run(records, sizes, max(1, args.repeats), source)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
