import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import github_submission as gs


def _cleanup_child(child: str) -> None:
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_get_github_submission_config_parses_overrides():
    cfg = {
        "github_submission": {
            "enabled": True,
            "delivery_mode": "issues",
            "repo_full_name": "owner/repo",
            "token_env": "ALT_TOKEN",
            "optimization_labels": ["one", "two", "one"],
            "feature_labels": ["feat", "needs-review", "feat"],
            "daily_issue_cap": 7,
            "cooldown_minutes": 45,
            "min_resource_trend_pressure": 0.61,
        }
    }
    policy = gs.get_github_submission_config(cfg)
    assert policy["enabled"] is True
    assert policy["delivery_mode"] == "issues"
    assert policy["repo_full_name"] == "owner/repo"
    assert policy["token_env"] == "ALT_TOKEN"
    assert policy["optimization_labels"] == ["one", "two"]
    assert policy["feature_labels"] == ["feat", "needs-review"]
    assert policy["daily_issue_cap"] == 7
    assert policy["cooldown_minutes"] == 45
    assert policy["min_resource_trend_pressure"] == 0.61


def test_labels_for_kind_select_feature_and_optimization_defaults():
    cfg = {
        "github_submission": {
            "labels": ["base"],
            "optimization_labels": ["opt"],
            "feature_labels": ["feat"],
        }
    }
    assert gs.labels_for_kind("request", cfg) == ["base"]
    assert gs.labels_for_kind("optimization_patch", cfg) == ["opt"]
    assert gs.labels_for_kind("feature_request", cfg) == ["feat"]


def test_append_github_issue_entry_writes_patch_attachment():
    child = "TestGitHubQueue"
    _cleanup_child(child)
    try:
        entry_id = gs.append_github_issue_entry(
            child,
            "Memory graph pressure",
            "Please review the allocator path.",
            kind="patch_attempt",
            labels=["optimization"],
            metadata={"source": "test"},
            patch_text="diff --git a/file.py b/file.py\n+print('x')\n",
        )
        assert entry_id
        queue_path = gs.github_outbox_path(child)
        assert queue_path.exists()
        lines = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        entry = lines[0]
        assert entry["id"] == entry_id
        assert entry["kind"] == "patch_attempt"
        assert entry["labels"] == ["optimization"]
        attachment_path = Path(entry["attachment_path"])
        assert attachment_path.exists()
        assert attachment_path.read_text(encoding="utf-8").startswith("diff --git")
    finally:
        _cleanup_child(child)


def test_build_issue_body_includes_metadata_and_attachment_excerpt():
    child = "TestGitHubBody"
    _cleanup_child(child)
    try:
        attachment_dir = gs.github_attachment_dir(child)
        attachment_dir.mkdir(parents=True, exist_ok=True)
        attachment_path = attachment_dir / "proposal.diff"
        attachment_path.write_text("diff --git a/a.py b/a.py\n+print('hello')\n", encoding="utf-8")
        entry = {
            "id": "github_test_entry",
            "title": "Optimisation request",
            "body": "Observed RAM growth during graph refresh.",
            "kind": "optimization_request",
            "labels": ["ina-suggestion", "optimization"],
            "created_at": "2026-04-04T12:00:00+00:00",
            "attachment_path": str(attachment_path),
            "metadata": {
                "source": "resource_vitals",
                "submission_mode": "both",
                "confidence": 0.83,
                "evidence": ["trend rising", "memory_guard soft"],
                "touched_files": ["memory_graph.py"],
                "review_notes": ["human review required"],
            },
        }
        body = gs.build_issue_body(entry, {"github_submission": {"max_patch_excerpt_chars": 2000, "max_body_chars": 12000}})
        assert "## Ina Submission" in body
        assert "## Summary" in body
        assert "## Evidence" in body
        assert "## Touched Files" in body
        assert "## Review Notes" in body
        assert "## Attachment Excerpt" in body
        assert "submission_mode: `both`" in body
        assert "```diff" in body
        assert "memory_graph.py" in body
    finally:
        _cleanup_child(child)


def test_read_pending_entries_skips_completed_history_ids():
    child = "TestGitHubPending"
    _cleanup_child(child)
    try:
        first_id = gs.append_github_issue_entry(child, "First", "Body one")
        second_id = gs.append_github_issue_entry(child, "Second", "Body two")
        assert first_id and second_id
        gs.log_history(child, first_id, "submitted", issue_number=1)
        pending = gs.read_pending_entries(child, cfg={"github_submission": {"max_batch": 10}}, seen_ids=gs.load_completed_history_ids(child))
        ids = [entry["id"] for entry in pending]
        assert first_id not in ids
        assert second_id in ids
    finally:
        _cleanup_child(child)
