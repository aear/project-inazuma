"""Explicit read-only research discovery command."""
from __future__ import annotations

import argparse
import json

from research_capability import search_wikipedia


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(search_wikipedia(args.query, limit=args.limit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
