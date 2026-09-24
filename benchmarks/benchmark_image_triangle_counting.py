"""Run the reusable image-triangle counting benchmark.

Usage: python -m benchmarks.benchmark_image_triangle_counting MANIFEST IMAGE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from image_triangle_counting import count_image_regions


BENCHMARK_SCHEMA = "ina.benchmark.image_triangle_counting/V1"


def evaluate(manifest_path: str | Path, image_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("unsupported image counting benchmark manifest")
    result = count_image_regions(image_path, manifest.get("regions") or ())
    expected = {row["name"]: row for row in manifest.get("regions") or ()}
    cases = []
    for row in result["regions"]:
        target = expected[row["name"]].get("expected") or {}
        actual = {"total": row["count"]["value"], **row["count"]["groups"]}
        matches = {key: actual.get(key, 0) == int(value) for key, value in target.items()}
        cases.append({"region": row["name"], "expected": target, "actual": actual,
                      "signals": matches, "correct": bool(matches) and all(matches.values()),
                      "observations": row["observations"]})
    return {
        "schema": BENCHMARK_SCHEMA, "benchmark_id": manifest.get("benchmark_id"),
        "detector_schema": result["schema"], "image": str(image_path),
        "correct": sum(case["correct"] for case in cases), "total": len(cases),
        "passed": bool(cases) and all(case["correct"] for case in cases),
        "ocr_used": result["ocr_used"], "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.image)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
