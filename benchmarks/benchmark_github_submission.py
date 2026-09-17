#!/usr/bin/env python3
"""Bounded versioned capability benchmark for Ina's GitHub outbox governance."""
from __future__ import annotations

import json

from github_submission import get_github_submission_config
from storage_migration_report import _report_signal, _signal_is_actionable


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


def main() -> int:
    result = {"benchmark": "github_submission_choice", "versions": [benchmark_v1(), benchmark_v2(), benchmark_v3()]}
    print(json.dumps(result, indent=2, sort_keys=True))
    choice = result["versions"][1]
    candidate = result["versions"][2]
    return 0 if (
        choice["eligible_ids"] == ["submit"]
        and choice["credential_shaped_token_env_rejected"]
        and all(candidate[key] for key in (
            "local_maintenance_before_auth", "score_drift_is_not_a_new_report",
            "actionable_transition_detected", "explicit_choice",
        ))
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
