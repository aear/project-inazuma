"""Immutable daily-price snapshots and leakage-safe crypto learning windows.

This module acquires observations and prepares chronological learning material.
It deliberately contains no order, portfolio, paper-trading, or live-trading API.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Mapping
from urllib import parse, request

from io_utils import flush_for_durability
from external_access import ExternalPolicy, ExternalSession


SCHEMA = "ina.crypto_market_snapshot/V1"
WINDOW_SCHEMA = "ina.crypto_market_learning_window/V1"
SOURCE_NAME = "Coin Metrics Community API"
SOURCE_ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
DEFAULT_ASSETS = ("btc", "eth", "ltc", "xrp", "doge", "ada", "bch", "xmr")
USER_AGENT = "Project-Inazuma-Developmental-Observatory/1.0"
COIN_METRICS_POLICY = ExternalPolicy(
    "coin_metrics_history", ("community-api.coinmetrics.io",),
    max_response_bytes=2 * 1024 * 1024, timeout_seconds=30, max_requests=4,
    allowed_content_types=("application/json",),
)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        flush_for_durability(handle, path)
    temporary.replace(path)


def _source_url(asset: str) -> str:
    query = parse.urlencode({
        "assets": asset, "metrics": "PriceUSD", "frequency": "1d",
        "page_size": 10000,
    })
    return f"{SOURCE_ENDPOINT}?{query}"


def fetch_daily_prices(
    asset: str, *, opener: Callable[..., Any] = request.urlopen, timeout: float = 30.0,
) -> tuple[list[dict[str, Any]], str]:
    symbol = str(asset).strip().lower()
    if symbol not in DEFAULT_ASSETS:
        raise ValueError(f"asset must be one of: {', '.join(DEFAULT_ASSETS)}")
    url = _source_url(symbol)
    rows: list[dict[str, Any]] = []
    session = ExternalSession(COIN_METRICS_POLICY, opener=opener)
    while url:
        response = session.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        payload = json.loads(response["body"].decode("utf-8"))
        for item in payload.get("data") or ():
            if str(item.get("asset", "")).lower() != symbol:
                continue
            price = float(item["PriceUSD"])
            if not math.isfinite(price) or price <= 0:
                continue
            rows.append({"time": str(item["time"]), "price_usd": price})
        next_url = payload.get("next_page_url")
        url = str(next_url) if next_url else ""
    rows.sort(key=lambda row: row["time"])
    if not rows:
        raise ValueError(f"no valid daily PriceUSD observations returned for {symbol}")
    if len({row["time"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate timestamps returned for {symbol}")
    return rows, _source_url(symbol)


def acquire_snapshot(
    *, root: Path | str = ".", assets: Iterable[str] = DEFAULT_ASSETS,
    opener: Callable[..., Any] = request.urlopen, observed_at: str | None = None,
) -> Path:
    requested = tuple(dict.fromkeys(str(asset).strip().lower() for asset in assets))
    if not 5 <= len(requested) <= 10:
        raise ValueError("a snapshot must contain between 5 and 10 distinct assets")
    fetched_at = observed_at or datetime.now(timezone.utc).isoformat()
    snapshot_id = fetched_at.replace(":", "-").replace("+", "_")
    target = Path(root) / "benchmark_results" / "crypto_market" / "snapshots" / snapshot_id
    manifest_assets = []
    for asset in requested:
        rows, source_url = fetch_daily_prices(asset, opener=opener)
        payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"{asset}.jsonl"
        _atomic_bytes(target / relative, payload)
        manifest_assets.append({
            "asset": asset, "observations": len(rows), "first_time": rows[0]["time"],
            "last_time": rows[-1]["time"], "sha256": digest, "path": relative,
            "source_url": source_url,
        })
    manifest = {
        "schema": SCHEMA, "snapshot_id": snapshot_id, "fetched_at": fetched_at,
        "source": SOURCE_NAME, "frequency": "1d", "metric": "PriceUSD",
        "coverage_claim": "full_available_source_history_not_asset_lifetime",
        "selection_warning": "Current named assets are a survivorship-biased cohort, not a historical market census.",
        "license_review_required_before_redistribution": True,
        "assets": manifest_assets,
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(target / "manifest.json", encoded)
    return target / "manifest.json"


def load_asset(manifest_path: Path | str, asset: str) -> list[dict[str, Any]]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entry = next((row for row in manifest.get("assets", ()) if row.get("asset") == asset), None)
    if not entry:
        raise KeyError(f"asset not present in snapshot: {asset}")
    source = manifest_file.parent / entry["path"]
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"snapshot hash mismatch: {asset}")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def chronological_split(rows: Iterable[Mapping[str, Any]], *, train: float = .70, validation: float = .15) -> dict[str, list[dict[str, Any]]]:
    ordered = [dict(row) for row in rows]
    if len(ordered) < 20:
        raise ValueError("at least 20 daily observations are required")
    if ordered != sorted(ordered, key=lambda row: str(row["time"])):
        raise ValueError("observations must already be chronological")
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("train and validation fractions must leave a test partition")
    first = max(1, min(len(ordered) - 2, int(len(ordered) * train)))
    second = max(first + 1, min(len(ordered) - 1, int(len(ordered) * (train + validation))))
    return {"train": ordered[:first], "validation": ordered[first:second], "test": ordered[second:]}


def build_learning_windows(
    asset: str, rows: Iterable[Mapping[str, Any]], *, lookback: int = 30, horizon: int = 7,
) -> list[dict[str, Any]]:
    ordered = [dict(row) for row in rows]
    if lookback < 2 or horizon < 1:
        raise ValueError("lookback must be >= 2 and horizon must be >= 1")
    windows = []
    for target_index in range(lookback, len(ordered) - horizon + 1):
        history = ordered[target_index - lookback:target_index]
        future = ordered[target_index:target_index + horizon]
        prices = [float(row["price_usd"]) for row in history]
        returns = [prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))]
        mean = sum(returns) / len(returns)
        volatility = math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))
        future_return = float(future[-1]["price_usd"]) / float(history[-1]["price_usd"]) - 1.0
        windows.append({
            "schema": WINDOW_SCHEMA, "asset": asset,
            "observed_start": history[0]["time"], "observed_end": history[-1]["time"],
            "target_start": future[0]["time"], "target_end": future[-1]["time"],
            "features": {"return": prices[-1] / prices[0] - 1.0, "volatility": volatility,
                         "maximum_drawdown": _maximum_drawdown(prices)},
            "label": {"future_return": future_return, "direction": "up" if future_return > 0 else "down_or_flat"},
            "trade_action": None, "paper_trade_authorized": False, "live_trade_authorized": False,
        })
    return windows


def _maximum_drawdown(prices: list[float]) -> float:
    peak = prices[0]
    worst = 0.0
    for price in prices:
        peak = max(peak, price)
        worst = min(worst, price / peak - 1.0)
    return worst


__all__ = [
    "DEFAULT_ASSETS", "SCHEMA", "WINDOW_SCHEMA", "acquire_snapshot",
    "build_learning_windows", "chronological_split", "fetch_daily_prices", "load_asset",
]
