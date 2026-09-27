from datetime import datetime, timezone

import pytest

from fiat_exchange import fetch_usd_gbp_reference


CSV = b'''CURRENCY,TIME_PERIOD,OBS_VALUE\nUSD,2026-09-24,1.1800\nGBP,2026-09-24,0.8600\nUSD,2026-09-25,1.2000\nGBP,2026-09-25,0.9000\n'''


class Response:
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def read(self): return CSV


def opener(req, timeout):
    assert "data-api.ecb.europa.eu" in req.full_url
    assert timeout == 20.0
    return Response()


def test_usd_gbp_cross_uses_same_latest_ecb_reference_date():
    result = fetch_usd_gbp_reference(opener=opener, now=datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert result["reference_date"] == "2026-09-25"
    assert result["usd_to_gbp"] == pytest.approx(.75)
    assert result["gbp_to_usd"] == pytest.approx(4 / 3)
    assert result["stale"] is False
    assert result["conversion_executed"] is False
    assert result["paper_trade_authorized"] is False
    assert result["live_trade_authorized"] is False


def test_exchange_reference_discloses_staleness():
    result = fetch_usd_gbp_reference(
        opener=opener, now=datetime(2026, 10, 5, tzinfo=timezone.utc), stale_after_days=7,
    )
    assert result["age_days"] == 10
    assert result["stale"] is True


def test_exchange_reference_requires_timezone_aware_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        fetch_usd_gbp_reference(opener=opener, now=datetime(2026, 9, 27))
