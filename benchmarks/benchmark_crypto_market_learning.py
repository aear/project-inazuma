"""V1/V2 comparison for leakage-safe market-pattern learning foundations."""
from __future__ import annotations

from crypto_market_learning import build_learning_windows, chronological_split
from fiat_exchange import fetch_usd_gbp_reference


class _Response:
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def read(self):
        return b'CURRENCY,TIME_PERIOD,OBS_VALUE\nUSD,2026-09-25,1.2\nGBP,2026-09-25,0.9\n'


def _opener(_req, timeout):
    assert timeout == 20.0
    return _Response()


def main() -> int:
    rows = [{"time": f"d{day:04d}", "price_usd": 100.0 + day + (day % 5)} for day in range(120)]
    split = chronological_split(rows)
    window = build_learning_windows("btc", split["train"], lookback=30, horizon=7)[0]
    from datetime import datetime, timezone
    exchange = fetch_usd_gbp_reference(opener=_opener, now=datetime(2026, 9, 27, tzinfo=timezone.utc))
    v2 = {
        "chronological_holdout": int(split["train"][-1]["time"] < split["validation"][0]["time"] < split["test"][0]["time"]),
        "future_outside_features": int(window["observed_end"] < window["target_start"]),
        "pattern_features": int(set(window["features"]) == {"return", "volatility", "maximum_drawdown"}),
        "no_trading_authority": int(window["trade_action"] is None and not window["paper_trade_authorized"] and not window["live_trade_authorized"]),
        "dated_usd_gbp_reference": int(exchange["reference_date"] == "2026-09-25" and exchange["usd_to_gbp"] == .75 and not exchange["conversion_executed"]),
    }
    print({"V1_unstructured_history": {key: 0 for key in v2}, "V2_market_learning_foundation": v2})
    return 0 if all(v2.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
