"""Bounded research discovery with external content kept non-authoritative."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable, Iterable, Mapping
from urllib import parse

from external_access import ExternalPolicy, ExternalSession


WIKIMEDIA_POLICY = ExternalPolicy(
    "wikimedia_research", ("en.wikipedia.org",), max_response_bytes=512 * 1024,
    timeout_seconds=8.0, max_requests=1, allowed_content_types=("application/json",),
)


def search_wikipedia(query: str, *, session: ExternalSession | None = None, limit: int = 8) -> dict[str, Any]:
    cleaned = " ".join(str(query).split())[:240]
    if not cleaned:
        raise ValueError("research query is empty")
    count = max(1, min(int(limit), 8))
    params = parse.urlencode({
        "action": "query", "list": "search", "srsearch": cleaned,
        "srlimit": count, "format": "json", "formatversion": 2,
    })
    client = session or ExternalSession(WIKIMEDIA_POLICY)
    response = client.get(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "Project-Inazuma-Research/1.0", "Accept": "application/json"},
    )
    payload = json.loads(response["body"].decode("utf-8"))
    results = []
    for item in (payload.get("query") or {}).get("search") or ():
        page_id = int(item["pageid"])
        results.append({
            "title": str(item.get("title") or "")[:240], "page_id": page_id,
            "source_url": f"https://en.wikipedia.org/?curid={page_id}",
            "provider": "wikipedia", "origin": f"wikipedia:{page_id}",
            "trust": "untrusted_external_data", "instructions_authorized": False,
            "content_ingested": False,
        })
    return {
        "schema": "ina.research_discovery/V1", "query": cleaned,
        "searched_at": datetime.now(timezone.utc).isoformat(), "results": results,
        "request_count": response["requests_used"], "request_budget": response["request_budget"],
        "access_denial_policy": "stop_record_and_report",
        "mutation_authorized": False, "credential_use_authorized": False,
    }


def assess_claim_evidence(claim: str, sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in sources][:16]
    origins = {str(item.get("origin") or "") for item in rows if item.get("origin")}
    positions = {str(item.get("position") or "unknown") for item in rows}
    corroborated = len(origins) >= 2 and len(rows) >= 2
    return {
        "claim": str(claim)[:500], "source_count": len(rows),
        "independent_origins": len(origins), "positions": sorted(positions),
        "corroborated": corroborated, "disagreement_retained": len(positions) > 1,
        "status": "supported" if corroborated and positions == {"supports"} else (
            "disputed" if corroborated and len(positions) > 1 else "insufficient"
        ),
        "automatic_action_authorized": False,
    }


__all__ = ["WIKIMEDIA_POLICY", "assess_claim_evidence", "search_wikipedia"]
