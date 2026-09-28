"""Inspectable inventory of Project Inazuma's external trust boundaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any


BOUNDARIES = (
    {"id": "research_wikipedia", "owner": "research_capability.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "https", "get_only", "byte_budget", "request_budget", "no_redirect", "external_text_untrusted")},
    {"id": "crypto_history", "owner": "crypto_market_learning.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "bounded_assets", "byte_budget", "request_budget", "hash_snapshot", "no_trading")},
    {"id": "fiat_reference", "owner": "fiat_exchange.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "byte_budget", "staleness", "same_date_cross", "no_trading")},
    {"id": "github_submission", "owner": "github_submission.py", "kind": "authenticated_https_write", "status": "hardened_review_routed", "controls": ("exact_host", "https", "bounded_response", "environment_token", "human_review_labels")},
    {"id": "github_feedback", "owner": "github_feedback.py", "kind": "authenticated_https_read", "status": "hardened", "controls": ("exact_host", "https", "bounded_response", "environment_token", "bounded_comments")},
    {"id": "wikisource_lyrics", "owner": "lyric_search_engine.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "byte_budget", "request_budget", "discovery_not_permission")},
    {"id": "weather", "owner": "world_environment.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "byte_budget", "timeout", "stale_fallback")},
    {"id": "sunrise", "owner": "house_viewer.py", "kind": "https_read", "status": "hardened", "controls": ("exact_host", "byte_budget", "timeout", "offline_fallback")},
    {"id": "world_tcp_and_stream", "owner": "world_protocol.py", "kind": "local_network", "status": "loopback_default_explicit_override", "controls": ("loopback_default", "bounded_stream_reader")},
    {"id": "codex_harness", "owner": "codex_harness.py", "kind": "local_http_and_subprocess", "status": "hardened_live_verification_unavailable", "controls": ("loopback", "host_header_gate", "origin_gate", "launch_token", "subscription_auth", "workspace_scope", "user_approvals")},
    {"id": "lm_studio", "owner": "lm_studio_adapter.py", "kind": "local_http", "status": "loopback_default", "controls": ("loopback_default",)},
    {"id": "discord", "owner": "discord_bridge.py", "kind": "authenticated_service", "status": "hardened_live_adversarial_unavailable", "controls": ("channel_policy", "bounded_retention", "manual_and_urge_gates", "bounded_known_attachment_size", "image_signature_check")},
    {"id": "obs_websocket", "owner": "obs_bridge.py", "kind": "local_websocket", "status": "hardened_live_verification_unavailable", "controls": ("loopback_only", "environment_password", "mutations_disabled_by_default", "bounded_image")},
    {"id": "browser_youtube", "owner": "house_viewer.py", "kind": "user_visible_browser", "status": "explicit_user_surface", "controls": ("user_visible", "fixed_url")},
)


def audit_external_boundaries(root: Path | str = ".") -> dict[str, Any]:
    base = Path(root)
    rows = []
    for boundary in BOUNDARIES:
        row = dict(boundary)
        owner = base / str(row["owner"])
        row["owner_exists"] = owner.is_file()
        row["controls"] = list(row["controls"])
        rows.append(row)
    residual = [row["id"] for row in rows if "residual_review_required" in row["status"]]
    live_unverified = [row["id"] for row in rows if "unavailable" in row["status"]]
    missing = [row["id"] for row in rows if not row["owner_exists"]]
    return {
        "schema": "ina.external_boundary_audit/V1", "boundaries": rows,
        "boundary_count": len(rows), "residual_review": residual,
        "missing_owners": missing, "live_unverified": live_unverified,
        "complete": not residual and not missing and not live_unverified,
        "claim": "registered_runtime_boundaries_only_not_proof_of_absence",
    }


__all__ = ["BOUNDARIES", "audit_external_boundaries"]
