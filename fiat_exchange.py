"""On-demand, provenance-bearing USD/GBP reference-rate lookup.

Rates are informational observations.  This module does not exchange currency,
place orders, value a portfolio, or run on a timer.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from io import StringIO
import math
from typing import Any, Callable
from urllib import parse, request

from external_access import ExternalPolicy, ExternalSession


SCHEMA = "ina.fiat_exchange_reference/V1"
SOURCE_NAME = "European Central Bank reference rates"
SOURCE_ENDPOINT = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD+GBP.EUR.SP00.A"
USER_AGENT = "Project-Inazuma-Developmental-Observatory/1.0"
ECB_POLICY = ExternalPolicy(
    "ecb_reference_rates", ("data-api.ecb.europa.eu",), max_response_bytes=1024 * 1024,
    timeout_seconds=20, max_requests=1, allowed_content_types=("text/csv",),
)


def _url(start: date) -> str:
    return f"{SOURCE_ENDPOINT}?{parse.urlencode({'startPeriod': start.isoformat(), 'format': 'csvdata'})}"


def fetch_usd_gbp_reference(
    *, opener: Callable[..., Any] | None = None,
    now: datetime | None = None, lookback_days: int = 14,
    stale_after_days: int = 7, timeout: float = 20.0,
) -> dict[str, Any]:
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if lookback_days < 7 or stale_after_days < 1:
        raise ValueError("lookback_days must be >= 7 and stale_after_days must be >= 1")
    url = _url(observed_now.date() - timedelta(days=lookback_days))
    session = ExternalSession(ECB_POLICY, opener=opener)
    response = session.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"})
    body = response["body"].decode("utf-8-sig")
    by_date: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(StringIO(body)):
        currency = str(row.get("CURRENCY") or "").upper()
        if currency not in {"USD", "GBP"}:
            continue
        value = float(row["OBS_VALUE"])
        if math.isfinite(value) and value > 0:
            by_date.setdefault(str(row["TIME_PERIOD"]), {})[currency] = value
    common = sorted(day for day, rates in by_date.items() if set(rates) == {"USD", "GBP"})
    if not common:
        raise ValueError("ECB response contains no common USD/GBP reference date")
    reference_date = common[-1]
    rates = by_date[reference_date]
    usd_to_gbp = rates["GBP"] / rates["USD"]
    age_days = (observed_now.date() - date.fromisoformat(reference_date)).days
    return {
        "schema": SCHEMA, "source": SOURCE_NAME, "source_url": url,
        "reference_date": reference_date,
        "checked_at": observed_now.isoformat(), "age_days": age_days,
        "stale": age_days > stale_after_days,
        "ecb_currency_per_eur": {"USD": rates["USD"], "GBP": rates["GBP"]},
        "usd_to_gbp": usd_to_gbp, "gbp_to_usd": 1.0 / usd_to_gbp,
        "informational_only": True, "conversion_executed": False,
        "paper_trade_authorized": False, "live_trade_authorized": False,
    }


__all__ = ["SCHEMA", "SOURCE_ENDPOINT", "fetch_usd_gbp_reference"]
