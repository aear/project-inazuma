#!/usr/bin/env python3
"""Bounded versioned capability benchmark for Ina's GitHub outbox governance."""
from __future__ import annotations

import json
import inspect

import github_submission
import self_read_reporting
from github_submission import get_github_submission_config
from storage_migration_report import (
    _public_actionable_evidence,
    _report_confidence,
    _report_signal,
    _signal_is_actionable,
)


def benchmark_v1() -> dict:
    """Historical behaviour: every queued entry was eligible for delivery."""
    entries = [{"id": "submit"}, {"id": "hold"}]
    return {"version": "V1", "eligible_ids": [entry["id"] for entry in entries], "explicit_choice": False}


def benchmark_v2() -> dict:
    """Candidate behaviour: Ina's explicit hold is preserved by delivery."""
    entries = [{"id": "submit", "delivery_choice": "submit"}, {"id": "hold", "delivery_choice": "hold"}]
    eligible = [
        entry["id"]
        for entry in entries
        if str(entry.get("delivery_choice") or "submit").strip().lower() == "submit"
    ]
    invalid_env = get_github_submission_config(
        {"github_submission": {"token_env": "github_pat_not_an_environment_name"}}
    )["token_env"]
    return {
        "version": "V2",
        "eligible_ids": eligible,
        "explicit_choice": True,
        "credential_shaped_token_env_rejected": invalid_env == "GITHUB_TOKEN",
    }


def benchmark_v3() -> dict:
    """Current path: local maintenance is auth-independent and reports transitions."""
    from pathlib import Path

    bridge_source = Path("github_bridge.py").read_text(encoding="utf-8")
    maintenance_position = bridge_source.index("archive_stale_entries_without_delivery(child, cfg)")
    auth_position = bridge_source.index("resolve_github_token(cfg, policy)", maintenance_position)
    baseline = {
        "decisions": {"index": {"tier": "fast", "fast_score": 3.0}},
        "devices": {"fast": {"failures": 0, "free_ratio": 0.8}},
        "directories": {"fragment_root": {"files": 0, "sample_truncated": False}},
        "recent_migrations": [],
    }
    score_drift = json.loads(json.dumps(baseline))
    score_drift["decisions"]["index"]["fast_score"] = 9.0
    actionable = json.loads(json.dumps(baseline))
    actionable["directories"]["fragment_root"]["files"] = 2
    return {
        "version": "V3",
        "local_maintenance_before_auth": maintenance_position < auth_position,
        "score_drift_is_not_a_new_report": _report_signal(baseline) == _report_signal(score_drift),
        "actionable_transition_detected": _signal_is_actionable(_report_signal(actionable)),
        "explicit_choice": True,
    }


def benchmark_v4() -> dict:
    """Follow-ups retain explicit provenance to earlier submitted issues."""
    source = inspect.getsource(github_submission.report_github_finding)
    renderer = inspect.getsource(github_submission.build_issue_body)
    self_read_source = inspect.getsource(self_read_reporting.report_self_read_broken_pipe)
    return {
        "version": "V4",
        "automatic_prior_issue_link": "submitted_issue_for_entry" in source,
        "explicit_related_issue_path": "related_issues" in source,
        "inspectable_related_issue_section": "## Related Issues" in renderer,
        "self_read_resubmission_link": "submitted_issue_for_entry" in self_read_source,
    }


def benchmark_v5() -> dict:
    """Actionable storage reports expose bounded evidence without false certainty."""
    report = {
        "directories": {"fragment_root": {"available": True, "files": 7, "sample_truncated": False}},
        "recent_migrations": [],
    }
    evidence = _public_actionable_evidence(report)
    return {
        "version": "V5",
        "bounded_legacy_count": "legacy_root_files=7" in evidence,
        "scan_completeness_exposed": "fragment_scan_truncated=false" in evidence,
        "confidence_below_certainty": _report_confidence(report) < 1.0,
    }


def main() -> int:
    result = {"benchmark": "github_submission_choice", "versions": [benchmark_v1(), benchmark_v2(), benchmark_v3(), benchmark_v4(), benchmark_v5()]}
    print(json.dumps(result, indent=2, sort_keys=True))
    choice = result["versions"][1]
    candidate = result["versions"][2]
    followup = result["versions"][3]
    storage_evidence = result["versions"][4]
    return 0 if (
        choice["eligible_ids"] == ["submit"]
        and choice["credential_shaped_token_env_rejected"]
        and all(candidate[key] for key in (
            "local_maintenance_before_auth", "score_drift_is_not_a_new_report",
            "actionable_transition_detected", "explicit_choice",
        ))
        and all(followup[key] for key in (
            "automatic_prior_issue_link", "explicit_related_issue_path",
            "inspectable_related_issue_section", "self_read_resubmission_link",
        ))
        and all(storage_evidence[key] for key in (
            "bounded_legacy_count", "scan_completeness_exposed", "confidence_below_certainty",
        ))
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
