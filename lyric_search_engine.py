"""Bounded, provenance-first lyric candidate search for Ina."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
USER_AGENT = "Project-Inazuma-LyricSearch/1.0 (local research tool)"
MAX_RESULTS = 8


def _clean(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


def search_local_music_manifest(root: Path, query: str, *, limit: int = MAX_RESULTS) -> List[dict]:
    path = root / "ina_public_music_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    terms = [term.casefold() for term in _clean(query).split() if len(term) > 1]
    ranked = []
    for relative, record in (manifest.get("tracks") or {}).items():
        if not isinstance(record, dict):
            continue
        haystack = " ".join((str(record.get("title") or ""), str(record.get("creator") or ""))).casefold()
        score = sum(1 for term in terms if term in haystack)
        if not score:
            continue
        ranked.append((score, str(record.get("title") or Path(relative).stem), relative, record))
    ranked.sort(key=lambda row: (-row[0], row[1].casefold(), row[2]))
    return [{
        "provider": "local_public_music",
        "title": title,
        "creator": _clean(record.get("creator"), 160),
        "audio_relative_path": relative,
        "source_url": _clean(record.get("source_url"), 500),
        "license_id": _clean(record.get("license_id"), 80),
        "rights_status": "audio_verified_lyrics_not_present",
        "lyrics_ingestion_allowed": False,
    } for _score, title, relative, record in ranked[:max(1, min(limit, MAX_RESULTS))]]


def search_wikisource(
    query: str,
    *,
    limit: int = 5,
    opener: Optional[Callable[..., Any]] = None,
) -> List[dict]:
    """Discover likely pages; page text remains blocked pending rights review."""
    params = urllib.parse.urlencode({
        "action": "opensearch", "search": f"{_clean(query)} song lyrics",
        "limit": max(1, min(int(limit), MAX_RESULTS)), "namespace": 0, "format": "json",
    })
    request = urllib.request.Request(f"{WIKISOURCE_API}?{params}", headers={"User-Agent": USER_AGENT})
    response = (opener or urllib.request.urlopen)(request, timeout=8.0)
    with response:
        payload = json.loads(response.read(512 * 1024).decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 4:
        return []
    titles, descriptions, urls = payload[1:4]
    results = []
    for title, description, url in zip(titles, descriptions, urls):
        results.append({
            "provider": "wikisource",
            "title": _clean(title, 240),
            "description": _clean(description, 400),
            "source_url": _clean(url, 500),
            "provider_policy_url": "https://en.wikisource.org/wiki/Wikisource:Copyright_policy",
            "rights_status": "candidate_requires_page_license_review",
            "lyrics_ingestion_allowed": False,
        })
    return results


def search_lyrics(
    query: str,
    *,
    public_music_root: Path,
    include_web: bool = True,
    web_search: Callable[..., List[dict]] = search_wikisource,
) -> Dict[str, Any]:
    normalized = _clean(query)
    if not normalized:
        return {"status": "blocked", "reason": "empty_query", "results": []}
    local_limit = MAX_RESULTS if not include_web else MAX_RESULTS // 2
    results = search_local_music_manifest(public_music_root, normalized, limit=local_limit)
    errors = []
    if include_web:
        try:
            results.extend(web_search(normalized, limit=MAX_RESULTS - local_limit))
        except Exception as exc:
            errors.append({"provider": "wikisource", "error": type(exc).__name__})
    return {
        "schema": "ina.lyric_search/V1",
        "status": "complete" if not errors else "partial",
        "query": normalized,
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "results": results[:MAX_RESULTS],
        "errors": errors,
        "ingestion_policy": "discovery_is_not_permission",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search bounded public lyric candidates.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--public-music-root", type=Path)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--runtime-request", action="store_true")
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.runtime_request:
        from config_layers import load_config
        from runtime_state import get_inastate, update_inastate
        child = args.child or str(load_config().get("current_child") or "Inazuma_Yagami")
        request = get_inastate("lyric_search_request", {}, child=child)
        request = request if isinstance(request, dict) else {}
        query = _clean(request.get("query"))
        root = Path(load_config().get("public_music_folder_path") or "")
        result = search_lyrics(query, public_music_root=root, include_web=not args.local_only)
        result["request_id"] = request.get("id")
        update_inastate("lyric_search_result", result, child=child)
        update_inastate("lyric_search_request", {**request, "requested": False, "status": result["status"]}, child=child)
    else:
        if not args.query or args.public_music_root is None:
            parser.error("query and --public-music-root are required outside runtime-request mode")
        result = search_lyrics(
            args.query, public_music_root=args.public_music_root, include_web=not args.local_only
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
