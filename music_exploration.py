"""Provider-neutral plans for Ina's bounded music listening choices."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


SUPPORTED_PROVIDERS = {"local_library", "discord_music_bot"}


def build_listening_plan(
    request: Dict[str, Any], policy: Dict[str, Any], *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Validate one chosen listening attempt without selecting music for Ina."""
    provider = str(request.get("provider") or policy.get("default_provider") or "local_library").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        return {"status": "blocked", "reason": "unsupported_provider", "provider": provider}
    if not bool(policy.get("enabled", False)):
        return {"status": "blocked", "reason": "music_exploration_disabled", "provider": provider}
    query = str(request.get("query") or "").strip()[:240]
    if not query:
        return {"status": "blocked", "reason": "missing_music_choice", "provider": provider}
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    plan = {
        "status": "ready",
        "provider": provider,
        "query": query,
        "requested_at": stamp,
        "purpose": str(request.get("purpose") or "explore_other_people_music")[:120],
        "one_bounded_attempt": True,
    }
    if provider == "discord_music_bot":
        channel_id = str(policy.get("discord_request_channel_id") or "").strip()
        template = str(policy.get("discord_request_template") or "Please play: {query}")
        if not channel_id:
            return {**plan, "status": "blocked", "reason": "discord_music_channel_not_configured"}
        plan.update({"channel_id": channel_id, "request_text": template.replace("{query}", query)[:400]})
    else:
        plan["action"] = "browse_verified_local_library"
    return plan


def artist_identity(policy: Dict[str, Any]) -> Dict[str, str]:
    """Expose Ina's own artist identity separately from listening history."""
    raw = policy.get("artist_identity") if isinstance(policy.get("artist_identity"), dict) else {}
    return {
        "name": str(raw.get("name") or "Inazuma").strip()[:120],
        "spotify_url": str(raw.get("spotify_url") or "").strip()[:500],
    }
