import json
import shutil
from pathlib import Path

import github_submission as gs

def cleanup(child): shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)

def policy(**extra):
    values = {"enabled": True, "finding_cooldown_minutes": 180, "finding_min_confidence": 0.35, "bug_labels": ["bug"], "issue_labels": ["ops"], "feature_labels": ["feature"]}
    values.update(extra)
    return {"github_submission": values}

def test_structured_bug_is_queued_and_deduplicated():
    child = "TestInaFindings"; cleanup(child)
    try:
        first = gs.report_github_finding(child, "Writer fails with errno 5", "Writes stop.", kind="bug", component="memory_writer", confidence=.8, evidence=["errno=5"], reproduction_steps=["Start writer", "Unmount target"], expected="Retry safely", actual="Writer exits", impact="Memory is not persisted", cfg=policy())
        second = gs.report_github_finding(child, "Writer fails with errno 9", "Writes stop again.", kind="bug", component="memory_writer", confidence=.9, cfg=policy())
        assert first["queued"] and not second["queued"] and second["reason"] == "cooldown"
        entry = json.loads(gs.github_outbox_path(child).read_text().splitlines()[0])
        assert entry["kind"] == "bug_report" and entry["labels"] == ["bug"]
        assert "## Reproduction Steps" in entry["body"] and entry["metadata"]["finding_fingerprint"]
        rendered = gs.build_issue_body(entry, policy())
        assert "component: `memory_writer`" in rendered and "severity: `medium`" in rendered
    finally: cleanup(child)

def test_feature_and_operational_issue_have_distinct_labels():
    child = "TestInaFindingKinds"; cleanup(child)
    try:
        feature = gs.report_github_finding(child, "Add storage dashboard", "Show migration health.", kind="feature", confidence=.7, cfg=policy())
        issue = gs.report_github_finding(child, "NVMe temporarily slow", "Latency rose during indexing.", kind="issue", confidence=.7, cfg=policy())
        assert feature["queued"] and issue["queued"]
        entries = [json.loads(line) for line in gs.github_outbox_path(child).read_text().splitlines()]
        assert entries[0]["labels"] == ["feature"] and entries[1]["labels"] == ["ops"]
    finally: cleanup(child)

def test_low_confidence_and_disabled_kind_are_not_queued():
    child = "TestInaFindingGates"; cleanup(child)
    try:
        low = gs.report_github_finding(child, "Maybe broken", "Uncertain symptom.", kind="bug", confidence=.1, cfg=policy())
        disabled = gs.report_github_finding(child, "Useful idea", "Add a view.", kind="feature", confidence=.9, cfg=policy(auto_queue_feature_requests=False))
        assert low["reason"] == "low_confidence" and disabled["reason"] == "kind_disabled"
        assert not gs.github_outbox_path(child).exists()
    finally: cleanup(child)
