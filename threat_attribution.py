"""Passive, evidence-led incident attribution and human-review reporting.

This module never scans, probes, contacts, identifies, or retaliates against a
target.  It separates observed infrastructure from operator and legal identity,
preserves disagreement, and only prepares reports for a human to review.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA = "ina.threat_attribution/V1"
ALLOWED_INDICATOR_KINDS = frozenset({
    "ip", "domain", "url", "file_sha256", "email", "account", "certificate_sha256",
    "user_agent", "malware_family", "ttp", "provider_case_reference",
})
IDENTITY_WITNESS_TYPES = frozenset({
    "provider_verified_account", "legal_process", "law_enforcement_finding",
    "authoritative_organisation_admission",
})
PASSIVE_EVIDENCE_TYPES = frozenset({
    "owned_system_log", "owned_packet_capture", "owned_file", "provider_response",
    "public_registry", "public_report", "law_enforcement_finding", "legal_process",
    "provider_verified_account", "authoritative_organisation_admission",
})
FORBIDDEN_ACQUISITION = frozenset({
    "active_probe", "exploit", "credential_use", "impersonation", "hack_back",
    "purchase_leaked_data", "contact_suspect", "doxxing",
})
AUTHORITY_ROUTES = {
    "ncsc": {
        "name": "UK National Cyber Security Centre",
        "url": "https://report.ncsc.gov.uk/",
        "purpose": "cyber incident response, attack identifiers, and UK threat intelligence",
    },
    "report_fraud": {
        "name": "Report Fraud",
        "url": "https://www.reportfraud.police.uk/",
        "purpose": "suspected criminal fraud or cybercrime in England, Wales, and Northern Ireland",
    },
    "police_scotland": {
        "name": "Police Scotland",
        "url": "https://www.scotland.police.uk/secureforms/c3/",
        "purpose": "suspected criminal cyber activity in Scotland",
    },
    "ico": {
        "name": "Information Commissioner's Office",
        "url": "https://ico.org.uk/for-organisations/report-a-breach/",
        "purpose": "a separately assessed personal-data breach notification",
    },
}


class AttributionPolicyError(ValueError):
    pass


def _bounded_text(value: Any, maximum: int, field: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise AttributionPolicyError(f"{field} must be 1..{maximum} characters")
    return text


def _canonical_indicator(kind: str, value: str) -> str:
    if kind == "ip":
        return ipaddress.ip_address(value).compressed
    if kind in {"file_sha256", "certificate_sha256"}:
        lowered = value.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", lowered):
            raise AttributionPolicyError(f"{kind} must be a SHA-256 hex digest")
        return lowered
    return value[:2048]


def make_indicator(kind: str, value: str, *, source_id: str, observed_at: str) -> dict[str, Any]:
    selected_kind = str(kind or "").strip().lower()
    if selected_kind not in ALLOWED_INDICATOR_KINDS:
        raise AttributionPolicyError(f"unsupported indicator kind: {selected_kind or 'missing'}")
    canonical = _canonical_indicator(selected_kind, _bounded_text(value, 2048, "indicator value"))
    timestamp = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise AttributionPolicyError("observed_at must include a timezone")
    return {
        "kind": selected_kind, "value": canonical,
        "source_id": _bounded_text(source_id, 200, "source_id"),
        "observed_at": timestamp.astimezone(timezone.utc).isoformat(),
        "role": "observable_not_identity",
    }


def hash_evidence(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    evidence_type = str(record.get("evidence_type") or "").strip().lower()
    acquisition = str(record.get("acquisition") or "passive").strip().lower()
    if acquisition in FORBIDDEN_ACQUISITION:
        raise AttributionPolicyError(f"forbidden evidence acquisition: {acquisition}")
    if evidence_type not in PASSIVE_EVIDENCE_TYPES:
        raise AttributionPolicyError(f"unsupported evidence type: {evidence_type or 'missing'}")
    digest = str(record.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise AttributionPolicyError("evidence requires a valid SHA-256 digest")
    return {
        "evidence_id": _bounded_text(record.get("evidence_id"), 200, "evidence_id"),
        "evidence_type": evidence_type,
        "sha256": digest,
        "origin": _bounded_text(record.get("origin"), 300, "origin"),
        "independence_group": _bounded_text(record.get("independence_group"), 200, "independence_group"),
        "supports_subject": str(record.get("supports_subject") or "").strip()[:500],
        "contradicts_subject": str(record.get("contradicts_subject") or "").strip()[:500],
        "acquisition": acquisition,
        "chain_of_custody_recorded": bool(record.get("chain_of_custody_recorded", False)),
    }


def assess_attribution(evidence: Iterable[Mapping[str, Any]], *, proposed_subject: str = "") -> dict[str, Any]:
    rows = [validate_evidence(item) for item in evidence]
    subject = str(proposed_subject or "").strip()[:500]
    supporting = [row for row in rows if subject and row["supports_subject"] == subject]
    contradicting = [row for row in rows if subject and row["contradicts_subject"] == subject]
    independent_support = {row["independence_group"] for row in supporting}
    identity_witnesses = [row for row in supporting if row["evidence_type"] in IDENTITY_WITNESS_TYPES]
    custody_complete = bool(supporting) and all(row["chain_of_custody_recorded"] for row in supporting)
    if not subject:
        status, level = "unknown", "infrastructure_only"
    elif contradicting:
        status, level = "disputed", "identity_unresolved"
    elif len(independent_support) >= 2 and identity_witnesses and custody_complete:
        status, level = "corroborated_hypothesis", "candidate_legal_identity"
    elif supporting:
        status, level = "insufficient", "infrastructure_or_operator_hypothesis"
    else:
        status, level = "unsupported", "identity_unresolved"
    return {
        "schema": SCHEMA, "proposed_subject": subject or None,
        "status": status, "attribution_level": level,
        "supporting_evidence_ids": [row["evidence_id"] for row in supporting],
        "contradicting_evidence_ids": [row["evidence_id"] for row in contradicting],
        "independent_support_count": len(independent_support),
        "authoritative_identity_witness_count": len(identity_witnesses),
        "chain_of_custody_complete": custody_complete,
        "identity_claim_authorized": status == "corroborated_hypothesis",
        "public_disclosure_authorized": False, "retaliation_authorized": False,
        "automatic_submission_authorized": False,
        "caveat": "Infrastructure may be compromised or shared; attribution remains a reviewable hypothesis.",
    }


def authority_routes(*, suspected_crime: bool, personal_data_involved: bool, jurisdiction: str = "uk") -> list[dict[str, str]]:
    routes = [dict(AUTHORITY_ROUTES["ncsc"])]
    if suspected_crime:
        key = "police_scotland" if str(jurisdiction).strip().lower() == "scotland" else "report_fraud"
        routes.append(dict(AUTHORITY_ROUTES[key]))
    if personal_data_involved:
        routes.append(dict(AUTHORITY_ROUTES["ico"]))
    return routes


def prepare_report(
    incident: Mapping[str, Any], indicators: Iterable[Mapping[str, Any]],
    attribution: Mapping[str, Any], *, suspected_crime: bool = True,
    personal_data_involved: bool = False, jurisdiction: str = "uk",
) -> dict[str, Any]:
    normalized_indicators = [make_indicator(
        item.get("kind", ""), item.get("value", ""), source_id=item.get("source_id", ""),
        observed_at=item.get("observed_at", ""),
    ) for item in indicators]
    report = {
        "schema": "ina.cyber_authority_report/V1",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "incident_id": _bounded_text(incident.get("incident_id"), 200, "incident_id"),
        "discovered_at": _bounded_text(incident.get("discovered_at"), 80, "discovered_at"),
        "summary": _bounded_text(incident.get("summary"), 4000, "summary"),
        "impact": str(incident.get("impact") or "unknown")[:4000],
        "containment": str(incident.get("containment") or "not recorded")[:4000],
        "indicators": normalized_indicators[:200],
        "attribution": dict(attribution),
        "routes": authority_routes(
            suspected_crime=suspected_crime, personal_data_involved=personal_data_involved,
            jurisdiction=jurisdiction,
        ),
        "review_status": "human_review_required",
        "submission_status": "not_submitted",
        "automatic_submission_authorized": False,
        "contact_suspect_authorized": False, "retaliation_authorized": False,
        "legal_regulatory_decision_required": personal_data_involved,
        "data_minimization_review_required": any(
            item["kind"] in {"email", "account"} for item in normalized_indicators
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    report["report_sha256"] = hash_evidence(canonical)
    return report


def queue_report_for_review(report: Mapping[str, Any], outbox_path: Path | str) -> dict[str, Any]:
    """Append one bounded draft to a local review queue; never transmit it."""
    if report.get("schema") != "ina.cyber_authority_report/V1":
        raise AttributionPolicyError("unsupported authority report schema")
    if report.get("review_status") != "human_review_required" or report.get("submission_status") != "not_submitted":
        raise AttributionPolicyError("only unsubmitted human-review drafts may be queued")
    encoded = json.dumps(dict(report), ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 1024 * 1024:
        raise AttributionPolicyError("authority report exceeds the 1 MiB queue limit")
    path = Path(outbox_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return {
        "queued": True, "path": str(path), "report_sha256": report.get("report_sha256"),
        "submission_status": "not_submitted", "human_review_required": True,
    }


__all__ = [
    "AUTHORITY_ROUTES", "AttributionPolicyError", "assess_attribution", "authority_routes",
    "hash_evidence", "make_indicator", "prepare_report", "queue_report_for_review", "validate_evidence",
]
