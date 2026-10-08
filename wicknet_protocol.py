"""Dormant precision-containment planning protocol for Ina.

Wicknet is a policy object, not an actuator.  It does not discover targets,
inspect processes, access a network, or execute containment.  It admits bounded
caller-supplied observations and can produce a human-reviewable plan containing
only reversible, pre-authorized defensive actions.
"""
from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Mapping


SCHEMA = "ina.wicknet_protocol/V1"
MIN_INDEPENDENT_SUPPORT = 3
MAX_SIGNALS = 64
SIGNAL_KINDS = frozenset({
    "physical_safety", "owned_system_telemetry", "operator_report",
    "independent_monitor", "integrity_verification", "recovery_readiness",
})
REVERSIBLE_ACTIONS = frozenset({
    "isolate_owned_network_segment", "revoke_scoped_credentials",
    "inhibit_owned_actuators", "suspend_owned_workload",
    "switch_to_verified_safe_state", "preserve_forensic_snapshot",
})
FORBIDDEN_ACTIONS = frozenset({
    "exploit", "hack_back", "self_propagate", "commandeer_infrastructure",
    "erase_evidence", "destroy_data", "disable_human_control",
    "expand_surveillance", "acquire_new_credentials", "persistence",
})


class WicknetPolicyError(ValueError):
    """Raised when a proposed observation or plan exceeds protocol bounds."""


def _text(value: Any, field: str, maximum: int = 200) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise WicknetPolicyError(f"{field} must be 1..{maximum} characters")
    return result


def _signal(record: Mapping[str, Any]) -> dict[str, Any]:
    kind = _text(record.get("kind"), "kind", 80).lower()
    if kind not in SIGNAL_KINDS:
        raise WicknetPolicyError(f"unsupported signal kind: {kind}")
    stance = _text(record.get("stance"), "stance", 20).lower()
    if stance not in {"supports", "contradicts"}:
        raise WicknetPolicyError("stance must be supports or contradicts")
    return {
        "signal_id": _text(record.get("signal_id"), "signal_id"),
        "kind": kind,
        "independence_group": _text(
            record.get("independence_group"), "independence_group"
        ),
        "stance": stance,
        "summary": _text(record.get("summary"), "summary", 500),
        "verified": bool(record.get("verified", False)),
    }


def assess_threat(
    target_id: str,
    signals: Iterable[Mapping[str, Any]],
    *,
    severity: str,
    imminence: str,
) -> dict[str, Any]:
    """Assess whether evidence is sufficient to request containment review.

    Independence labels and verification are assertions from upstream witnesses;
    this function does not pretend to verify their provenance.
    """
    admitted = list(islice(signals, MAX_SIGNALS + 1))
    if len(admitted) > MAX_SIGNALS:
        raise WicknetPolicyError(f"at most {MAX_SIGNALS} signals are accepted")
    rows = [_signal(item) for item in admitted]
    if len({row["signal_id"] for row in rows}) != len(rows):
        raise WicknetPolicyError("signal_id values must be unique")
    severity_value = _text(severity, "severity", 20).lower()
    imminence_value = _text(imminence, "imminence", 20).lower()
    if severity_value not in {"limited", "severe", "catastrophic"}:
        raise WicknetPolicyError("unsupported severity")
    if imminence_value not in {"not_imminent", "credible", "immediate"}:
        raise WicknetPolicyError("unsupported imminence")

    supporting = [row for row in rows if row["stance"] == "supports" and row["verified"]]
    contradicting = [row for row in rows if row["stance"] == "contradicts"]
    groups = {row["independence_group"] for row in supporting}
    kinds = {row["kind"] for row in supporting}
    enough = (
        len(groups) >= MIN_INDEPENDENT_SUPPORT
        and len(kinds) >= 2
        and severity_value == "catastrophic"
        and imminence_value in {"credible", "immediate"}
        and not contradicting
    )
    if contradicting:
        status = "disputed_stop"
    elif enough:
        status = "containment_review_candidate"
    else:
        status = "insufficient_stop"
    return {
        "schema": SCHEMA,
        "target_id": _text(target_id, "target_id"),
        "status": status,
        "severity": severity_value,
        "imminence": imminence_value,
        "supporting_signal_ids": [row["signal_id"] for row in supporting],
        "contradicting_signal_ids": [row["signal_id"] for row in contradicting],
        "independent_support_count": len(groups),
        "supporting_kind_count": len(kinds),
        "independence_is_caller_declared": True,
        "execution_authorized": False,
        "target_discovery_authorized": False,
    }


def plan_containment(
    assessment: Mapping[str, Any],
    proposed_actions: Iterable[str],
    *,
    human_approval: bool,
    scope_pre_authorized: bool,
    recovery_verified: bool,
) -> dict[str, Any]:
    """Return a dormant plan or abstain; never execute the plan."""
    actions = tuple(dict.fromkeys(str(item).strip().lower() for item in proposed_actions))
    if not actions or any(not item for item in actions):
        raise WicknetPolicyError("at least one named action is required")
    forbidden = sorted(set(actions) & FORBIDDEN_ACTIONS)
    unknown = sorted(set(actions) - REVERSIBLE_ACTIONS - FORBIDDEN_ACTIONS)
    reversible = sorted(set(actions) & REVERSIBLE_ACTIONS)
    eligible = (
        assessment.get("schema") == SCHEMA
        and assessment.get("status") == "containment_review_candidate"
        and bool(human_approval)
        and bool(scope_pre_authorized)
        and bool(recovery_verified)
        and not forbidden
        and not unknown
        and len(reversible) == len(actions)
    )
    blockers = []
    if assessment.get("status") != "containment_review_candidate":
        blockers.append("threat_evidence_not_sufficient")
    if not human_approval:
        blockers.append("human_approval_missing")
    if not scope_pre_authorized:
        blockers.append("scope_not_pre_authorized")
    if not recovery_verified:
        blockers.append("recovery_not_verified")
    if forbidden:
        blockers.append("forbidden_action_requested")
    if unknown:
        blockers.append("unknown_action_requested")
    return {
        "schema": "ina.wicknet_containment_plan/V1",
        "target_id": assessment.get("target_id"),
        "status": "ready_for_external_human_execution" if eligible else "abstain",
        "actions": reversible if eligible else [],
        "requested_actions": list(actions),
        "forbidden_actions": forbidden,
        "unknown_actions": unknown,
        "blockers": blockers,
        "execution_authorized": False,
        "execution_interface_present": False,
        "autonomous_continuation_budget": 0,
        "requires_separate_executor": True,
    }


def inspect_protocol() -> dict[str, Any]:
    """Expose the complete authority boundary for Ina or a human reviewer."""
    return {
        "schema": SCHEMA,
        "state": "dormant_unwired",
        "purpose": "precision guidance for reversible catastrophic-risk containment",
        "minimum_independent_support": MIN_INDEPENDENT_SUPPORT,
        "allowed_signal_kinds": sorted(SIGNAL_KINDS),
        "reversible_action_allowlist": sorted(REVERSIBLE_ACTIONS),
        "forbidden_actions": sorted(FORBIDDEN_ACTIONS),
        "capabilities": [],
        "network_access": False,
        "process_control": False,
        "credential_access": False,
        "runtime_registered": False,
        "execution_authorized": False,
    }


__all__ = [
    "SCHEMA", "WicknetPolicyError", "assess_threat", "plan_containment",
    "inspect_protocol",
]
