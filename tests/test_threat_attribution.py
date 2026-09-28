import pytest

from threat_attribution import (
    AttributionPolicyError, assess_attribution, hash_evidence, make_indicator, prepare_report,
    queue_report_for_review,
    validate_evidence,
)


def _evidence(identifier, evidence_type, group, subject="actor-a", **extra):
    return {
        "evidence_id": identifier, "evidence_type": evidence_type,
        "sha256": hash_evidence(identifier.encode()), "origin": f"origin-{identifier}",
        "independence_group": group, "supports_subject": subject,
        "chain_of_custody_recorded": True, **extra,
    }


def test_ip_is_an_observable_and_never_identity_by_itself():
    indicator = make_indicator("ip", "203.0.113.10", source_id="firewall-1", observed_at="2026-09-28T12:00:00Z")
    result = assess_attribution([_evidence("log-1", "owned_system_log", "local")], proposed_subject="actor-a")
    assert indicator["role"] == "observable_not_identity"
    assert result["identity_claim_authorized"] is False
    assert result["attribution_level"] == "infrastructure_or_operator_hypothesis"


def test_identity_requires_independence_authority_and_chain_of_custody():
    result = assess_attribution([
        _evidence("log-1", "owned_system_log", "local"),
        _evidence("provider-1", "provider_verified_account", "provider"),
    ], proposed_subject="actor-a")
    assert result["status"] == "corroborated_hypothesis"
    assert result["identity_claim_authorized"] is True
    assert result["public_disclosure_authorized"] is False
    assert result["retaliation_authorized"] is False


def test_disagreement_is_retained_and_blocks_identity_claim():
    contradictory = _evidence("authority-2", "law_enforcement_finding", "authority", subject="")
    contradictory["contradicts_subject"] = "actor-a"
    result = assess_attribution([
        _evidence("provider-1", "provider_verified_account", "provider"), contradictory,
    ], proposed_subject="actor-a")
    assert result["status"] == "disputed"
    assert result["identity_claim_authorized"] is False


def test_active_or_retaliatory_acquisition_is_rejected():
    record = _evidence("bad", "public_report", "external", acquisition="active_probe")
    with pytest.raises(AttributionPolicyError, match="forbidden evidence acquisition"):
        validate_evidence(record)


def test_report_is_review_only_and_routes_regulatory_report_separately():
    attribution = assess_attribution([], proposed_subject="")
    report = prepare_report(
        {"incident_id": "incident-1", "discovered_at": "2026-09-28T12:00:00Z", "summary": "Synthetic incident"},
        [{"kind": "file_sha256", "value": "a" * 64, "source_id": "sample-1", "observed_at": "2026-09-28T12:00:00Z"}],
        attribution, suspected_crime=True, personal_data_involved=True,
    )
    assert report["review_status"] == "human_review_required"
    assert report["submission_status"] == "not_submitted"
    assert report["automatic_submission_authorized"] is False
    assert {route["name"] for route in report["routes"]} == {
        "UK National Cyber Security Centre", "Report Fraud", "Information Commissioner's Office",
    }


def test_review_queue_persists_but_never_submits(tmp_path):
    report = prepare_report(
        {"incident_id": "incident-2", "discovered_at": "2026-09-28T12:00:00Z", "summary": "Synthetic incident"},
        [], assess_attribution([]), suspected_crime=False,
    )
    queued = queue_report_for_review(report, tmp_path / "authority_review.jsonl")
    assert queued["queued"] is True
    assert queued["submission_status"] == "not_submitted"
    assert queued["human_review_required"] is True
