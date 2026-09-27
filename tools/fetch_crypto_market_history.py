"""Explicitly invoked acquisition of the Observatory's crypto history cohort."""
from __future__ import annotations

import argparse
from io import BytesIO
import subprocess

from crypto_market_learning import DEFAULT_ASSETS, acquire_snapshot


class _CurlResponse(BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.close()
        return False


def _curl_opener(req, timeout):
    """CLI transport for providers that reject Python urllib's TLS fingerprint."""
    completed = subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", str(int(timeout)), req.full_url],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return _CurlResponse(completed.stdout)


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
