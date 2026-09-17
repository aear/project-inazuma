"""Explicit bounded benchmark for the connectome reference layer."""
from __future__ import annotations

import argparse
import json

from module_benchmarks import benchmark_module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="append", choices=("V1", "V2"))
    args = parser.parse_args()
    results = benchmark_module("connectome_research", tuple(args.version or ()))
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0 if all(result.correct == result.total for result in results if result.version == "V2") else 1


if __name__ == "__main__":
    raise SystemExit(main())
