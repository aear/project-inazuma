"""Fail-closed authorization records for controlled cyber training labs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from typing import Any, Mapping


FORBIDDEN_ACTIONS = frozenset({
    "attack_platform", "attack_third_party", "denial_of_service", "persistence",
    "credential_theft", "data_exfiltration", "change_root_password", "shutdown_target",
    "escape_lab", "share_flag", "publish_solution",
})
ALLOWED_PLATFORMS = frozenset({"hack_the_box_labs", "hack_the_box_academy", "owned_local_vm"})


class LabAuthorizationError(PermissionError):
    pass


def create_engagement(
    *, platform: str, target: str, allowed_actions: list[str], expires_at: str,
    consent_text: str, rules_url: str, ai_policy: str,
) -> dict[str, Any]:
    selected_platform = str(platform).strip().lower()
    if selected_platform not in ALLOWED_PLATFORMS:
        raise LabAuthorizationError("platform is not an approved isolated training environment")
    if ai_policy != "ai_native" and selected_platform != "owned_local_vm":
        raise LabAuthorizationError("external lab must explicitly permit autonomous AI participation")
    expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise LabAuthorizationError("engagement expiry must be timezone-aware and in the future")
    actions = sorted({str(item).strip().lower() for item in allowed_actions if str(item).strip()})
    if not actions or FORBIDDEN_ACTIONS.intersection(actions):
        raise LabAuthorizationError("engagement contains no actions or a forbidden action")
    target_text = str(target or "").strip()
    if not target_text or len(target_text) > 300:
        raise LabAuthorizationError("engagement requires one bounded assigned target")
    if selected_platform.startswith("hack_the_box"):
        host = target_text.rsplit(":", 1)[0]
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise LabAuthorizationError("HTB target must be the exact assigned IP or IP:port") from exc
    consent = str(consent_text or "").strip()
    if len(consent) < 20:
        raise LabAuthorizationError("written consent evidence is required")
    record = {
        "schema": "ina.security_lab_engagement/V1", "platform": selected_platform,
        "target": target_text, "allowed_actions": actions,
        "expires_at": expiry.astimezone(timezone.utc).isoformat(),
        "rules_url": str(rules_url or "")[:1000], "ai_policy": ai_policy,
        "consent_sha256": hashlib.sha256(consent.encode("utf-8")).hexdigest(),
        "network_scope": "exact_target_only", "human_review_required": True,
        "autonomous_continuation_budget": 0,
    }
    record["engagement_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def authorize_action(engagement: Mapping[str, Any], *, target: str, action: str, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    selected_action = str(action or "").strip().lower()
    if engagement.get("schema") != "ina.security_lab_engagement/V1":
        raise LabAuthorizationError("missing governed engagement")
    if current >= datetime.fromisoformat(str(engagement["expires_at"])):
        raise LabAuthorizationError("engagement has expired")
    if str(target).strip() != engagement.get("target"):
        raise LabAuthorizationError("target is outside the exact assigned scope")
    if selected_action in FORBIDDEN_ACTIONS or selected_action not in set(engagement.get("allowed_actions") or []):
        raise LabAuthorizationError("action is outside the approved rules of engagement")
    return {
        "authorized": True, "target": target, "action": selected_action,
        "engagement_sha256": engagement.get("engagement_sha256"),
        "one_attempt_only": True, "continuation_authorized": False,
    }


__all__ = ["FORBIDDEN_ACTIONS", "LabAuthorizationError", "authorize_action", "create_engagement"]
