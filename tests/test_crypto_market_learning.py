from __future__ import annotations

import json

import pytest

from crypto_market_learning import (
    DEFAULT_ASSETS, acquire_snapshot, build_learning_windows, chronological_split,
    load_asset,
)


class Response:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def read(self): return json.dumps(self.payload).encode()


def fixture_opener(req, timeout):
    del timeout
    asset = req.full_url.split("assets=")[1].split("&", 1)[0]
    return Response({"data": [
        {"asset": asset, "time": f"2026-01-{day:02d}T00:00:00Z", "PriceUSD": str(100 + day)}
        for day in range(1, 29)
    ]})


def test_snapshot_is_hash_verified_and_records_coverage_limits(tmp_path):
    manifest_path = acquire_snapshot(root=tmp_path, assets=DEFAULT_ASSETS[:5], opener=fixture_opener, observed_at="2026-02-01T00:00:00+00:00")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["coverage_claim"] == "full_available_source_history_not_asset_lifetime"
    assert "survivorship" in manifest["selection_warning"]
    assert manifest["license_review_required_before_redistribution"] is True
    assert len(load_asset(manifest_path, "btc")) == 28
    source = manifest_path.parent / "btc.jsonl"
    source.write_text(source.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_asset(manifest_path, "btc")


def test_chronological_split_has_no_overlap_or_future_leakage():
    rows = [{"time": f"2026-01-{day:02d}", "price_usd": day} for day in range(1, 29)]
    split = chronological_split(rows)
    assert split["train"][-1]["time"] < split["validation"][0]["time"]
    assert split["validation"][-1]["time"] < split["test"][0]["time"]
    assert sum(map(len, split.values())) == len(rows)


def test_learning_windows_keep_future_labels_outside_observed_range():
    rows = [{"time": f"d{day:03d}", "price_usd": 100 + day} for day in range(60)]
    window = build_learning_windows("btc", rows, lookback=30, horizon=7)[0]
    assert window["observed_end"] < window["target_start"]
    assert window["features"]["return"] > 0
    assert window["label"]["direction"] == "up"
    assert window["trade_action"] is None
    assert window["paper_trade_authorized"] is False
    assert window["live_trade_authorized"] is False


def test_snapshot_requires_a_bounded_five_to_ten_asset_cohort(tmp_path):
    with pytest.raises(ValueError, match="between 5 and 10"):
        acquire_snapshot(root=tmp_path, assets=("btc",), opener=fixture_opener)
