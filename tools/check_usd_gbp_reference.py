"""Explicitly invoked USD/GBP reference-rate check."""
from __future__ import annotations

import json

from fiat_exchange import fetch_usd_gbp_reference


def main() -> int:
    print(json.dumps(fetch_usd_gbp_reference(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
