"""Fail-closed authorization records for controlled cyber training labs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import hmac
import secrets
import socket
import threading
from typing import Any, Mapping


FORBIDDEN_ACTIONS = frozenset({
    "attack_platform", "attack_third_party", "denial_of_service", "persistence",
    "credential_theft", "data_exfiltration", "change_root_password", "shutdown_target",
    "escape_lab", "share_flag", "publish_solution",
})
ALLOWED_PLATFORMS = frozenset({"hack_the_box_labs", "hack_the_box_academy", "owned_local_vm"})
_REVIEW_KEY = secrets.token_bytes(32)
_CONSUMED: set[str] = set()
_LOCK = threading.Lock()


def _encoded(record: Mapping[str, Any]) -> bytes:
    return json.dumps({key: value for key, value in record.items() if key not in {"engagement_sha256", "review_seal"}}, sort_keys=True, separators=(",", ":")).encode()


def _endpoint(target: str) -> tuple[str, int | None]:
    try:
        return str(ipaddress.ip_address(target)), None
    except ValueError:
        host, separator, port = target.rpartition(":")
        try:
            host = str(ipaddress.ip_address(host.strip("[]")))
            number = int(port)
            if not separator or not 1 <= number <= 65535:
                raise ValueError()
            return host, number
        except ValueError as exc:
            raise LabAuthorizationError("target must be an exact IP or IP:port") from exc


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
    _endpoint(target_text)
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


def review_engagement(engagement: Mapping[str, Any], *, consent_text: str,
                      reviewer: str, confirmed_digest: str) -> dict[str, Any]:
    """Trusted human-review entry point; never exposed as an Ina personal tool.

    A process-local seal prevents external JSON from manufacturing approval.
    Restarting the review process invalidates approvals and requires re-review.
    """
    record = dict(engagement)
    digest = hashlib.sha256(_encoded(record)).hexdigest()
    if not hmac.compare_digest(digest, str(confirmed_digest)) or digest != record.get("engagement_sha256"):
        raise LabAuthorizationError("review digest mismatch")
    if hashlib.sha256(consent_text.strip().encode()).hexdigest() != record.get("consent_sha256") or not reviewer.strip():
        raise LabAuthorizationError("review requires matching consent evidence and reviewer")
    # Revalidate the draft before signing it.
    create_engagement(platform=record["platform"], target=record["target"],
                      allowed_actions=record["allowed_actions"], expires_at=record["expires_at"],
                      consent_text=consent_text, rules_url=record["rules_url"], ai_policy=record["ai_policy"])
    record["reviewer"] = reviewer.strip()[:200]
    record["review_nonce"] = secrets.token_hex(16)
    record["engagement_sha256"] = hashlib.sha256(_encoded(record)).hexdigest()
    record["review_seal"] = hmac.new(_REVIEW_KEY, _encoded(record), hashlib.sha256).hexdigest()
    return record


def authorize_action(engagement: Mapping[str, Any], *, target: str, action: str, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    selected_action = str(action or "").strip().lower()
    if engagement.get("schema") != "ina.security_lab_engagement/V1":
        raise LabAuthorizationError("missing governed engagement")
    expected = hmac.new(_REVIEW_KEY, _encoded(engagement), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(engagement.get("review_seal", ""))):
        raise LabAuthorizationError("engagement is unreviewed or tampered")
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


def execute_tcp_connect(engagement: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """One bounded connect, no payload, redirects, DNS, shell, or arbitrary tools.

    Only this implemented action has an execution boundary. General lab tools
    remain unavailable until a network-isolated executor exists.
    """
    authorize_action(engagement, target=target, action="tcp_connect")
    host, port = _endpoint(target)
    if port is None:
        raise LabAuthorizationError("execution requires an explicitly reviewed port")
    nonce = engagement["review_nonce"]
    with _LOCK:
        if nonce in _CONSUMED:
            raise LabAuthorizationError("one-attempt engagement already consumed")
        _CONSUMED.add(nonce)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as connection:
        connection.settimeout(3.0)
        connection.connect((host, port))
        peer = connection.getpeername()
        if str(ipaddress.ip_address(peer[0])) != host or peer[1] != port:
            raise LabAuthorizationError("connected peer is outside reviewed network scope")
    return {"connected": True, "target": target, "payload_sent": False, "continuation_authorized": False}


__all__ = ["FORBIDDEN_ACTIONS", "LabAuthorizationError", "authorize_action", "create_engagement"]
