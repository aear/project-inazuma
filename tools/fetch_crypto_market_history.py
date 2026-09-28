"""Explicitly invoked acquisition of the Observatory's crypto history cohort."""
from __future__ import annotations

import argparse
from io import BytesIO
import subprocess
from urllib import parse

from crypto_market_learning import DEFAULT_ASSETS, acquire_snapshot
from external_access import resolve_public_addresses


class _CurlResponse(BytesIO):
    def __init__(self, payload, *, address, addresses):
        super().__init__(payload)
        self.destination_verified = True
        self.connected_address = address
        self.resolved_addresses = addresses
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.close()
        return False


def _curl_opener(req, timeout):
    """CLI transport for providers that reject Python urllib's TLS fingerprint."""
    parsed = parse.urlsplit(req.full_url)
    addresses = resolve_public_addresses(parsed.hostname or "")
    address = addresses[0]
    completed = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--max-time", str(int(timeout)),
         "--resolve", f"{parsed.hostname}:443:{address}", req.full_url],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return _CurlResponse(completed.stdout, address=address, addresses=addresses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument("--transport", choices=("urllib", "curl"), default="urllib")
    args = parser.parse_args()
    kwargs = {"opener": _curl_opener} if args.transport == "curl" else {}
    print(acquire_snapshot(root=args.root, assets=args.assets, **kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
